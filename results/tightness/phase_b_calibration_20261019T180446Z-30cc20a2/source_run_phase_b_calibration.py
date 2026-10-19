"""Run one guarded tiny calibration, preserving failures and existing artifacts."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone

import psutil

from .env import REPO, REFERENCE_COMMIT, require_qos_ref_venv
from .resources import gate_ok, measured
from .run_phase1 import INTERPRETATION, LEDGER_COLUMNS, git_commit


def main():
    require_qos_ref_venv()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--N',type=int,default=4)
    parser.add_argument('--s',type=int,default=2)
    parser.add_argument('--kappa',type=float,default=2.)
    parser.add_argument('--eps',type=float,default=.01)
    parser.add_argument('--M',type=int,default=10**8)
    parser.add_argument('--append-ledger',action='store_true')
    args=parser.parse_args()
    if args.N>32:
        raise ValueError('calibration runner limited to N<=32; no full sweeps authorized')
    if not gate_ok(psutil.virtual_memory().available,psutil.Process().memory_info().rss,before_stage=True):
        raise RuntimeError('memory gate: need 4 GiB available')
    now=datetime.now(timezone.utc);run_id=now.strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    out=REPO/'results/tightness'/('phase_b_calibration_'+run_id);out.mkdir()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    sources=sorted((REPO/'experiments/qos_tightness/qostight').glob('*.py'))
    manifest=dict(run_id=run_id,date_utc=now.isoformat(),git_commit=git_commit(),
                  git_dirty_at_start=bool(subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip()),
                  reference_commit=REFERENCE_COMMIT,interpretation=INTERPRETATION,
                  status='running',python=sys.version,platform=platform.platform(),arguments=vars(args),
                  versions={k:importlib.metadata.version(k) for k in ['jax','pyqsp','numpy','scipy','psutil']},
                  lock_sha256=sha(REPO/'experiments/qos_tightness/requirements-lock.txt'),
                  code_sha256={str(p.relative_to(REPO)):sha(p) for p in sources},
                  scope='one fixed-test correctness/calibration instance; no scaling sweep')
    save=lambda:(out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    save(); print('run',out.relative_to(REPO),flush=True)
    try:
        with measured() as memory:
            from .phase_b_prototype import calibrate
            result=calibrate(N=args.N,s=args.s,kappa=args.kappa,eps_state=args.eps,M=args.M)
        result.update(runtime_seconds=memory['seconds'],peak_rss_bytes=memory['peak_rss_bytes'])
        (out/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        with (out/'stage_timings.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['stage','seconds']);w.writerows(result['timings'].items())
        with (out/'correctness.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['metric','value']);w.writerows(result['gates'].items())
        ledger=[]
        for metric,value in [('full_state_l2_error',result['gates']['full_state_error']),
                             ('fixed_test_binomial_correctness_probability',result['classification_probability'])]:
            ledger.append(dict(experiment_id=f'qos-tightness-phase-b-calibration-{run_id}-{metric}',
                               dataset='synthetic signed paired-difference circulant; ideal uniform sparsity',
                               date=now.date().isoformat(),git_commit=manifest['git_commit'],
                               preprocessing='none (synthetic, ||X||=1)',n_samples=args.N,n_features=args.N,
                               max_sparsity=args.s,sketch_method='QOS expected unitary (reference a3fe8d6)',
                               sketch_dimension='',seed=42,model='augmented ridge, QSVT inverse, fixed-point amplification, fixed-test Hadamard',
                               metric=metric,result=value,notes=json.dumps(dict(interpretation=INTERPRETATION,
                               artifact_dir=str(out.relative_to(REPO)),readout='fixed-test; no shadows',
                               kappa=result['kappa_reg'],gamma=result['gamma'],delta=result['delta'],
                               sample_accounting='calculated recipe; nested trace pending'),sort_keys=True)))
        with (out/'ledger_rows.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=LEDGER_COLUMNS);w.writeheader();w.writerows(ledger)
        if args.append_ledger:
            p=REPO/'results/experiment_log.csv';before=p.read_bytes();assert before.endswith(b'\n')
            with p.open('a',newline='') as f:csv.DictWriter(f,fieldnames=LEDGER_COLUMNS).writerows(ledger)
            assert p.read_bytes().startswith(before)
        manifest.update(status='completed',ledger_appended=args.append_ledger,
                        runtime_seconds=memory['seconds'],peak_rss_bytes=memory['peak_rss_bytes'])
        print(json.dumps(dict(degrees=result['degrees'],gates=result['gates'],seconds=memory['seconds'],
                              peak_GiB=memory['peak_rss_bytes']/2**30)),flush=True)
    except BaseException as exc:
        (out/'failure.txt').write_text(traceback.format_exc())
        manifest.update(status='failed',failure=repr(exc));save();raise
    finally:
        manifest['artifacts_sha256']={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}
        save()


if __name__=='__main__':main()
