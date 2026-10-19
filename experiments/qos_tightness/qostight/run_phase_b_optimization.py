"""One measured gate job per fresh process; never launches parameter sweeps."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import types

import psutil

from .env import REPO, REFERENCE_COMMIT, require_qos_ref_venv
from .resources import gate_ok, measured


def main():
    require_qos_ref_venv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', choices=['baseline', 'cold', 'warm', 'kappa4_cap', 'kappa4'])
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    root = args.directory.resolve()
    if not root.is_relative_to(REPO/'experiments/qos_tightness'):
        raise ValueError('this gate writes only under experiments/qos_tightness')
    out = root/args.job
    out.mkdir(parents=True, exist_ok=False)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    sources = sorted((REPO/'experiments/qos_tightness/qostight').glob('*.py'))
    manifest = dict(job=args.job, date_utc=datetime.now(timezone.utc).isoformat(),
                    base_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                    reference_commit=REFERENCE_COMMIT, status='running',
                    python=sys.version, platform=platform.platform(),
                    versions={k:importlib.metadata.version(k) for k in ['jax','pyqsp','numpy','scipy','psutil']},
                    code_sha256={str(p.relative_to(REPO)):sha(p) for p in sources},
                    interpretation='simulation (expected-unitary QOS, reference convention)',
                    scope='one optimization/correctness gate job; CPU runtime/RSS measured; no full sweeps')
    def save():
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    save()
    print(f'start {args.job}: {out}', flush=True)
    memory = {}
    try:
        if not gate_ok(psutil.virtual_memory().available, psutil.Process().memory_info().rss, before_stage=True):
            raise MemoryError('need >=4 GiB available before job')
        with measured() as memory:
            if args.job == 'baseline':
                # Execute the exact committed prototype, with unchanged shared dependencies.
                for p in sources:
                    if p.name in ('phase_b_prototype.py','phase_cache.py','run_phase_b_optimization.py'):
                        continue
                    rel = str(p.relative_to(REPO))
                    old = subprocess.check_output(['git','show',f'6d25652:{rel}'])
                    if old != p.read_bytes():
                        raise RuntimeError(f'baseline shared dependency differs: {rel}')
                source = subprocess.check_output(['git','show','6d25652:experiments/qos_tightness/qostight/phase_b_prototype.py'])
                (out/'baseline_phase_b_prototype.py').write_bytes(source)
                module = types.ModuleType('qostight.baseline_6d25652')
                exec(compile(source, str(out/'baseline_phase_b_prototype.py'), 'exec'), module.__dict__)
                result = module.calibrate()
            else:
                from .phase_b_prototype import calibrate
                result = calibrate(kappa=4. if args.job.startswith('kappa4') else 2.,
                                   phase_backend='batched', cache_dir=str(root/'phase_cache'),
                                   max_degree=8191 if args.job == 'kappa4' else 4095)
                # Cache state is part of this benchmark's design, not an informal label.
                for field in ('d23_phase_synthesis',):
                    if args.job == 'cold' and result[field]['cache_hit']:
                        raise RuntimeError('cold benchmark unexpectedly hit cache')
                for field in ('inverse','fixed_point'):
                    hit = result[field]['phase_synthesis']['cache_hit']
                    if args.job == 'cold' and hit or args.job == 'warm' and not hit:
                        raise RuntimeError(f'unexpected {args.job} cache state: {field}')
            # Supplement existing calibration gates; do not relax or replace any.
            gates = result['gates']
            for name in ('augmented_gram_error','augmented_lstsq_vs_ridge','readout_formula_error',
                         'd23_unitarity','inverse_unitarity','fixed_point_unitarity','selector_unitarity'):
                if not gates[name] <= 1e-9:
                    raise RuntimeError(f'additional correctness gate failed: {name}={gates[name]}')
        if memory['gate_violation']:
            raise MemoryError(memory['gate_violation'])
        result.update(runtime_seconds=memory['seconds'], peak_rss_bytes=memory['peak_rss_bytes'])
        (out/'result.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
        manifest['status'] = 'completed'
        print(json.dumps(dict(job=args.job, seconds=memory['seconds'], peak_GiB=memory['peak_rss_bytes']/2**30,
                              degrees=result['degrees'], full_state_error=gates['full_state_error'],
                              timings=result['timings'])), flush=True)
    except BaseException as exc:
        manifest.update(status='failed', failure=repr(exc))
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        manifest['measurement'] = memory
        manifest['artifacts_sha256'] = {p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'}
        save()


if __name__ == '__main__':
    main()
