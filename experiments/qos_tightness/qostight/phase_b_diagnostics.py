"""Polynomial-only preflight, conditional budget model, and calibration plots.

No scaling sweep or stochastic oracle construction is performed here.
"""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import erf, erfcinv

from .env import REPO, require_qos_ref_venv
from .phase_b_prototype import odd_interpolant
from .resources import measured


def degree_preflight():
    import time
    rows=[]
    with measured() as memory:
        for eps,kappa in itertools.product([.01,.001],[2,4,8,16]):
            t=time.perf_counter();a=1/(40*kappa);tol=eps**2/128;k=erfcinv(tol/4)/a
            start=(int(2*k)+1)|1
            row=dict(kappa=kappa,eps_state=eps,starting_degree=start,phase_finding_run=False)
            try:
                _,info=odd_interpolant(lambda x:(1-tol/4)*erf(k*x),start,tol,a,lambda x:np.ones_like(x))
                row.update(info,status='polynomial grid checks passed; angles not generated')
            except RuntimeError as e:
                row.update(status=str(e))
            row['seconds']=time.perf_counter()-t;rows.append(row)
    return dict(rows=rows,memory=memory,interpretation='numerical polynomial preflight; not a scaling sweep')


def budget(result,preflight):
    # 13 unique one-factor settings, both eps* values. No hidden automatic reductions.
    baseline=(4,4.,.1,.1)
    configs={baseline}
    for i,values in enumerate([(2,4,8,16),(2.,4.,8.,16.),(.3,.1,.03,.01),(.2,.1,.05,.02)]):
        for v in values:
            c=list(baseline);c[i]=v;configs.add(tuple(c))
    kernels={(k,min(e,g/(6*np.sqrt(s)))) for s,k,g,d in configs for e in [.01,.001]}
    passed={r['eps_state']:r['degree'] for r in preflight['rows'] if r['kappa']==2 and 'degree' in r}
    d0=passed[.01];d1=passed[.001]
    t0=result['timings']['fixed_point_polynomial_and_angles']
    rows=[]
    for k,e in sorted(kernels):
        # Interpolate/extrapolate polynomial degree in log precision, then linear kappa.
        # This is a forecasting assumption, not measured scaling or a confidence interval.
        degree=(d0+(d1-d0)*math.log10(.01/e))*(k/2)
        rows.append(dict(kappa=k,eps_solver=e,estimated_degree=degree,
                         quadratic_seconds=t0*(degree/d0)**2,cubic_seconds=t0*(degree/d0)**3,
                         exceeds_current_degree_cap=degree>4095))
    prior=REPO/'results/tightness/phase_a_20261019T145729Z-a1644ff3/points.csv'
    points=list(csv.DictReader(prior.open()))
    med={s:float(np.median([float(r['seconds']) for r in points if r['mode']=='primary' and int(r['s'])==s])) for s in [2,4,8,16]}
    seconds=sum(med[s]*5*2*8 for s,k,g,d in configs)
    return dict(unique_configs=len(configs),unique_phase_kernels=len(kernels),evaluations=len(configs)*5*2*8,
                phase_kernel_estimates=rows,setup_hours_quadratic=sum(r['quadratic_seconds'] for r in rows)/3600,
                setup_hours_cubic=sum(r['cubic_seconds'] for r in rows)/3600,
                oracle_build_proxy_hours=seconds/3600,oracle_median_seconds=med,
                assumptions=['single process; no implementation optimization',
                             'phase setup cached once per kappa/actual-solver-epsilon pair',
                             'degree interpolated from two kappa=2 polynomial-only preflights; scales linearly with kappa',
                             'phase time scales as degree squared or cubed, based on loop/Jacobian and solve costs',
                             'oracle timings borrowed from frozen N64 phase (a), fixed primitive epsilon=1e-4',
                             'not a confidence interval; unsupported degrees exceed current cap; no measured large-kappa phases'],
                missing_costs=['reusable Clifford shadows','explicit fresh-sample nested circuits',
                               'higher precision sparse oracles','inverse polynomial phase setup outside calibration',
                               'N128 spot checks','CFPB toys','adaptive grid refinements'],
                full_phase_total_runtime='not yet estimable as a validated finite bound; mandatory paths missing')


def main(directory):
    require_qos_ref_venv()
    run=Path(directory);result=json.loads((run/'result.json').read_text())
    out=run/'diagnostics';out.mkdir()
    preflight=degree_preflight();forecast=budget(result,preflight)
    (out/'degree_preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
    (out/'runtime_forecast.json').write_text(json.dumps(forecast,indent=2)+'\n')
    with (out/'phase_kernel_forecast.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(forecast['phase_kernel_estimates'][0]));w.writeheader();w.writerows(forecast['phase_kernel_estimates'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    stages=result['timings'];axes[0].barh(list(stages),list(stages.values()))
    axes[0].set_xscale('log');axes[0].set_xlabel('Measured wall seconds (one cold tiny instance)')
    axes[0].set_title('Fixed-test expected-unitary calibration')
    keys=['d23_dense_error','inverse_dense_error','fixed_point_dense_error','augmented_lstsq_vs_ridge','full_state_error']
    axes[1].barh(keys,[result['gates'][k] for k in keys]);axes[1].set_xscale('log')
    axes[1].axvline(result['eps_target'],color='red',linestyle='--',label='full-state tolerance only')
    axes[1].set_xlabel('Numerical error (stage metrics differ)');axes[1].legend(fontsize=8)
    fig.savefig(out/'calibration.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for eps,marker in [(.01,'o'),(.001,'s')]:
        r=[x for x in preflight['rows'] if x['eps_state']==eps]
        ax.plot([x['kappa'] for x in r],[x['starting_degree'] for x in r],':',alpha=.5)
        ok=[x for x in r if 'degree' in x]
        ax.scatter([x['kappa'] for x in ok],[x['degree'] for x in ok],marker=marker,label=f'passed polynomial only, eps={eps}')
    ax.axhline(4095,color='red',linestyle='--',label='prototype cap; all kappa>=4 fail')
    ax.set(xlabel='kappa_reg',ylabel='Polynomial degree',title='Preflight: not an end-to-end scaling result')
    ax.legend(fontsize=8);fig.savefig(out/'degree_preflight.png',dpi=160);plt.close(fig)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=dict(calibration_manifest_sha256=sha(run/'manifest.json'),
                  code_sha256={str(p.relative_to(REPO)):sha(p) for p in [Path(__file__).resolve(),Path(__file__).with_name('phase_b_prototype.py').resolve()]},
                  paper_url='https://arxiv.org/html/2604.07639v2',
                  paper_html_sha256=sha(Path('/tmp/qos_paper_v2.html')) if Path('/tmp/qos_paper_v2.html').exists() else 'unknown',
                  artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in forecast.items() if k!='phase_kernel_estimates'},indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_dir');a=p.parse_args();main(a.run_dir)
