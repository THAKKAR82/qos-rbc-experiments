"""Timing only: weighted-index draws and four-observable median-of-means.

This is NOT a Clifford sampler, shadow result, or correctness experiment.
Synthetic lookup weights/values approximate only the work of an enumerated
small-state readout backend, whose mathematical validation remains future work.
"""
import json
from pathlib import Path
import sys
import time

import numpy as np
import psutil
from scipy.stats import binom

from qostight.env import require_qos_ref_venv
from qostight.resources import gate_ok, measured


def main(directory):
    require_qos_ref_venv()
    assert gate_ok(psutil.virtual_memory().available, psutil.Process().memory_info().rss, before_stage=True)
    epsilon = .3/(3*np.sqrt(2)); tolerance = epsilon/2
    # Conditional on correct global-Clifford statistics, Var <=3 tr(O^2)=6.
    # Chebyshev gives bad-block probability <=1/4; a union bound covers m=4.
    block_size = int(np.ceil(4*6/tolerance**2))
    groups = next(k for k in range(1,100,2) if 4*binom.sf(k//2,k,.25) <= .1/2)
    rows = []
    with measured() as memory:
        for record_count in (1080, 36720):
            rng = np.random.default_rng(42)
            weights = rng.uniform(.1, 1, record_count); weights /= weights.sum()
            values = rng.normal(size=(record_count, 4))
            times = []
            for _ in range(5):
                start = time.perf_counter()
                indices = rng.choice(record_count, size=(groups,block_size), p=weights)
                estimates = np.median(values[indices].mean(axis=1), axis=0)
                assert estimates.shape == (4,)
                times.append(time.perf_counter()-start)
            rows.append(dict(synthetic_lookup_record_count=record_count, measured_seconds=times,
                             forecast_200_repetitions_seconds=200*float(np.median(times))))
    if memory['gate_violation']: raise MemoryError(memory['gate_violation'])
    result = dict(interpretation='classical array-throughput proxy ONLY; no Clifford/shadow simulation',
                  seed=42, epsilon=epsilon, estimation_tolerance=tolerance,
                  assumed_variance_upper_bound=6, block_size=block_size, median_groups=groups,
                  records_per_model=block_size*groups,
                  conditional_union_failure_bound=float(4*binom.sf(groups//2,groups,.25)),
                  rows=rows, measurement=memory,
                  excluded_costs=['stabilizer enumeration and validation','Born-weight and observable tables',
                                  'controlled solver or freshly sampled oracle execution'])
    (Path(directory)/'readout_cost_proxy.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
