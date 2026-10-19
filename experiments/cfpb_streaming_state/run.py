"""Approved 96-cell pilot driver. No tuning grid or test-evaluation option."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import uuid
from src.cfpb_prepare import check_resources
from .data import ROOT, TASKS, atomic_json, sha256
from .prepare import protected_hashes, verify_frozen


def plan():
    configs = []
    for budget, task, rep, order, method in itertools.product(
        (16384, 131072, 524288), TASKS, ('A_frozen', 'B_causal'),
        ('chronological', 'random'), ('hash_sgd', 'sparse_jl_sgd', 'awm', 'mission')):
        identifier = f'{task}_{rep}_{method}_{order}_b{budget}_s42'
        configs.append(dict(id=identifier, task=task, representation=rep, ordering=order,
                            method=method, budget_bytes=budget, seed=42))
    assert len(configs) == 96 and len({c['id'] for c in configs}) == 96
    return dict(trajectories=configs, max_trajectories=96, max_passes=1,
                benchmark_wall_cap_seconds=3600, output_byte_cap=5*2**30,
                calibration_trajectory=configs[0]['id'], validation_only=True,
                hyperparameters='Prespecified in PROTOCOL.md; no validation tuning',
                checkpoints=[.125, .25, .5, 1.], namespace_seeds=dict(root=42, projection=771, order=619))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', help='Resume an orderly interruption within the original one-hour wall deadline')
    args = parser.parse_args()
    os.chdir(ROOT)
    check_resources(before_stage=True)
    if args.resume:
        output = Path(args.resume)
        manifest = json.loads((output / 'manifest.json').read_text())
        for path, digest in manifest['source_sha256'].items():
            if sha256(ROOT/path) != digest:
                raise ValueError('Cannot resume after changing execution source')
        run_plan = json.loads((output / 'run_plan.json').read_text())
        if run_plan != plan():
            raise ValueError('Cannot change the approved pilot grid')
    else:
        now = datetime.now(timezone.utc)
        run_id = now.strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
        output = Path('results/classical_streaming') / run_id
        output.mkdir(parents=True, exist_ok=False)
        (output/'trials').mkdir(); (output/'data_cache').mkdir()
        run_plan = plan()
        source_paths = sorted(Path('experiments/cfpb_streaming_state').rglob('*.py')) + [Path('experiments/cfpb_streaming_state/PROTOCOL.md')]
        manifest = dict(run_id=run_id, started_utc=now.isoformat(), start_epoch=time.time(),
            status='preparing', root_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
            authorized_dirty_sources=True, source_sha256={str(p):sha256(p) for p in source_paths},
            protected_sha256=protected_hashes(), prior_ledger_sha256=sha256('results/experiment_log.csv'),
            prior_ledger_bytes=Path('results/experiment_log.csv').stat().st_size,
            frozen_artifacts_verified=verify_frozen(), python=sys.version.split()[0], platform=platform.platform(),
            packages={p:importlib.metadata.version(p) for p in ('numpy','scipy','pandas','scikit-learn','psutil','pytest')},
            benchmark_wall_cap_seconds=3600, threads=1, test_access=False,
            upstream=dict(awm='4907851642f520a752c3591612c168da40b03989', mission='03b5bb2bb95a7f7f5ecee3ff4777042d00924b20'),
            trajectory_status={c['id']:'planned' for c in run_plan['trajectories']},
            historical_results='frozen; scientific measurements in this run are classical validation-only pilot measurements')
        atomic_json(output/'run_plan.json', run_plan)
        atomic_json(output/'manifest.json', manifest)
    print('OUTPUT '+str(output), flush=True)
    try:
        if not (output/'preparation.json').exists():
            remaining = 3600 - (time.time()-manifest['start_epoch'])
            command = [sys.executable, '-m', 'experiments.cfpb_streaming_state.prepare', str(output)]
            preparation = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                stdout, stderr = preparation.communicate(timeout=max(.01, remaining))
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                preparation.kill()
                stdout, stderr = preparation.communicate()
                (output/'preparation.log').write_text((stdout+'\n'+stderr).replace(str(ROOT), '<repo>'))
                raise RuntimeError('Preparation interrupted or exceeded original benchmark deadline; no trajectories launched')
            (output/'preparation.log').write_text((stdout+'\n'+stderr).replace(str(ROOT), '<repo>'))
            if preparation.returncode != 0:
                raise RuntimeError('Frozen-data correctness gate failed; see preparation.log; no trajectories launched')
        manifest['status'] = 'calibration'
        for index, config in enumerate(run_plan['trajectories']):
            path = output/'trials'/f"{config['id']}.json"
            if path.exists():
                old = json.loads(path.read_text())
                if old['status'] == 'completed':
                    manifest['trajectory_status'][config['id']] = 'completed'
                    continue
                if old['status'] != 'paused':
                    raise RuntimeError('An earlier trajectory failed; no automatic retries')
            remaining = 3600 - (time.time()-manifest['start_epoch'])
            if remaining <= 10:
                manifest['status'] = 'wall_cap_reached'
                break
            if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > 5*2**30:
                manifest['status'] = 'output_cap_reached'
                break
            check_resources(before_stage=True)
            for source, digest in manifest['source_sha256'].items():
                if sha256(ROOT/source) != digest:
                    raise RuntimeError('Execution source changed during pilot')
            manifest['trajectory_status'][config['id']] = 'running'
            atomic_json(output/'manifest.json', manifest)
            started = time.perf_counter()
            print(f"START {index+1}/96 {config['id']} remaining={remaining:.1f}s", flush=True)
            command = [sys.executable, '-m', 'experiments.cfpb_streaming_state.worker',
                       '--output', str(output), '--trial', config['id'], '--seconds', str(remaining)]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                stdout, stderr = process.communicate(timeout=remaining)
            except KeyboardInterrupt:
                process.send_signal(signal.SIGTERM)
                try:
                    stdout, stderr = process.communicate(timeout=min(5., remaining))
                except subprocess.TimeoutExpired:
                    process.kill(); process.communicate()
                manifest['status'] = 'interrupted'
                if path.exists():
                    manifest['trajectory_status'][config['id']] = json.loads(path.read_text())['status']
                break
            except subprocess.TimeoutExpired:
                process.kill(); stdout, stderr = process.communicate()
                manifest['status'] = 'hard_wall_cap_reached'
                manifest['trajectory_status'][config['id']] = 'failed_hard_deadline'
                break
            elapsed = time.perf_counter()-started
            log = (stdout+'\n'+stderr).replace(str(ROOT), '<repo>')
            (output/'trials'/f"{config['id']}.log").write_text(log)
            if not path.exists():
                raise RuntimeError('Worker failed before writing its state record: '+log[-1500:])
            result = json.loads(path.read_text())
            result['subprocess_wall_seconds'] = elapsed
            atomic_json(path, result)
            manifest['trajectory_status'][config['id']] = result['status']
            print(f"DONE {index+1}/96 {result['status']} {elapsed:.2f}s state={result['peak_model_state_bytes']} bytes", flush=True)
            if process.returncode != 0 or result['status'] == 'failed':
                manifest['status'] = 'correctness_or_resource_failure'
                manifest['failure_reason'] = result.get('failure_reason')
                break
            if result['status'] == 'paused':
                manifest['status'] = 'wall_cap_or_orderly_stop'
                break
            if index == 0:
                atomic_json(output/'calibration.json', dict(trajectory_id=config['id'], status='passed',
                    is_part_of_96_grid=True, examples_processed=result['examples_processed'],
                    training_seconds=result['training_seconds'], peak_rss_bytes=result['peak_rss_bytes'],
                    peak_model_state_bytes=result['peak_model_state_bytes'],
                    total_persistent_bytes=result['peak_total_persistent_bytes'],
                    throughput=result['examples_processed']/result['training_seconds'],
                    primary_validation=result['final_primary'],
                    correctness_basis='pre-run independent update tests; frozen feature and Ridge reference gates'))
                manifest['status'] = 'pilot'
        else:
            manifest['status'] = 'completed'
    except KeyboardInterrupt:
        manifest['status'] = 'interrupted'
    except BaseException as exc:
        manifest['status'] = 'failed'
        manifest['failure_reason'] = type(exc).__name__+': '+str(exc).replace(str(ROOT), '<repo>')
    finally:
        manifest['benchmark_wall_seconds'] = time.time()-manifest['start_epoch']
        for key, value in manifest['trajectory_status'].items():
            if value == 'planned':
                manifest['trajectory_status'][key] = 'skipped_'+manifest['status']
        bad = [p for p,h in manifest['protected_sha256'].items() if sha256(ROOT/p) != h]
        manifest['preservation_mismatches'] = bad
        manifest['frozen_artifacts_verified_after'] = verify_frozen()
        if bad:
            manifest['status'] = 'preservation_failure'
        atomic_json(output/'manifest.json', manifest)
        print('FINAL '+json.dumps({k:manifest.get(k) for k in ('status','benchmark_wall_seconds','failure_reason')}), flush=True)
    if manifest['status'] in ('failed','preservation_failure','correctness_or_resource_failure'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
