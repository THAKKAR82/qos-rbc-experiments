# CFPB streaming-state validation pilot

This isolated implementation follows the user-approved bounded pilot described
in [PROTOCOL.md](PROTOCOL.md). It uses the main pinned `.venv`; it does not import
QOS experiment code or require new dependencies. The established CFPB and Bank
implementations and artifacts remain immutable.

Run correctness tests without training CFPB models:

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider experiments/cfpb_streaming_state/tests
```

The completed approved run is under `results/classical_streaming/`. Inspect its
`report.md` before requesting further fits. A fresh invocation of the driver
creates a new experiment and requires authorization; it is not a report command.
The approved invocation uses:

```sh
caffeinate -i env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/tmp/cfpb-streaming-mpl \
  .venv/bin/python -m experiments.cfpb_streaming_state.run
```

An orderly interruption can resume with `--resume results/classical_streaming/<run-id>`
only inside the **original** one-hour wall window, with unchanged source, inputs,
and configuration. Completed trajectories are skipped. Hard failures do not
automatically retry uncertain sample access. Checkpoints and raw replay files
remain in ignored local `data_cache/`; they are required for local resume and
are not publication artifacts. The driver rejects edited input hashes.

Scalar analysis fits no models and reads no source documents or test outcomes:

```sh
.venv/bin/python -m experiments.cfpb_streaming_state.analyze results/classical_streaming/<run-id>
MPLCONFIGDIR=/tmp/cfpb-streaming-mpl .venv/bin/python -m experiments.cfpb_streaming_state.plot results/classical_streaming/<run-id>
```

`analyze --append-ledger` appends only completed trajectories' three validation
metrics, with unique IDs, preserved historical bytes, and conflict checks.
Repeating it with unchanged results adds no duplicate entries. It never records
an incomplete trajectory as a completed score. The run manifest records the
published base commit plus exact execution-source hashes, since pilot execution
precedes the local implementation commit. Analysis-source hashes are separate.

## Files and schema

| Output | Meaning |
| --- | --- |
| `run_plan.json` | Exactly 96 predeclared configurations, seed, checkpoints and caps |
| `manifest.json` | Base commit, execution fingerprints, dependencies, original file/ledger hashes, all trajectory statuses and actual wall time |
| `preparation.json` | Frozen feature checks, batch Ridge references, raw/scaled numerical equivalence, shared preparation resources and input-spool hashes |
| `calibration.json` | First grid trajectory's measured feasibility gate |
| `trials/<id>.json` | Config, optimizer/sketch dimensions, state inventory, samples, prefixes, primary/natural/stratum metrics, resources and resume pointer |
| `trajectory_summary.csv` | One row per planned configuration, including explicit skipped/failed states; missing scores stay empty |
| `checkpoint_metrics.csv` | Scalar performance versus training presentations, by held-out validation population |
| `state_inventory.csv` | Owned capacities and object allocations by category and native dtype; model, sketch, indices, optimizer, projection and preprocessing |
| `method_resources.csv` | Measured runtime, throughput and memory ranges by method/representation |
| `cost_ledger.csv` | Shared preparation, training, validation, worker overhead; no tuning costs concealed |
| `order_differences.csv` | Matched seed/budget random-minus-chronological validation differences |
| `descriptive_retention_crossings.csv` | Individual 95/98/99% point crossings, explicitly not minimum-memory or confidence claims |
| `full_matrix_runtime_projection.csv` | Calculated full-proposal extrapolation from pilot rates; not a measured full run |
| `analysis.json`, `report.md` | Scientific findings, limitations, proposed next stage and stopping point |
| `ledger_append.json` | Appended IDs and historical ledger-prefix verification |

Bytes are allocated component inventories including Python/NumPy overhead;
64-bit-equivalent registers are `ceil(bytes/8)`. Native array register counts
are also recorded by dtype. A 128-bit feature key occupies two uint64 elements.
`peak_model_state_bytes` means the entire algorithm, including sketch/index/
optimizer/projection, not just coefficients. `peak_total_persistent_bytes`
sums algorithm, preprocessing, and stream-access inventories; these separately
charged components can conservatively count small shared immutable Python
objects more than once. Arrays are deduplicated within each inventory. It is
not interpreter RSS or a claim about quantum registers.

Random access owns `4N` permutation bytes plus `8N` offset bytes, object overhead,
and the bounded reader. Its temporary `8N` int64 permutation during construction
is reported separately. Raw disk spool/index bytes are explicit external state.
Current-document and validation-harness inventories are lower bounds on transient
workspace; 50-ms process RSS samples also capture native allocations, with the
usual possibility of missing brief peaks. Driver RSS is not included in worker
RSS. The unpruned final learner is retained for inference; no smaller inference
export is claimed.

Training timing includes reading, encoding, updates and strict per-example state
audits. Evaluation time is separate. Worker and full benchmark wall times include
additional setup/checkpoint costs. Data preparation is measured in a separate
bounded subprocess and cannot retain a free full dataset in learner memory.
The raw export is scanned physically to extract approved train/validation rows;
no test observations are supplied to a learner and no test performance is read.

Only one seed and untuned diagnostic hyperparameters are used. No across-seed
interval, paired prediction bootstrap, full-data SGD convergence control, fitted
causal-representation Ridge control, or confirmed `k_star_grid` is produced.
The toy full-dimensional gradient/sklearn and legacy-JL checks do not replace
those future real-data controls. Frozen preprocessing A sees all January;
causal feature construction B changes the representation. Neither track undoes
the frozen population's offline curation. Balanced training is not run or pooled
with these unweighted results.

## Scoped benchmark (approved 2026-10-20; ~5.7 h projected)

`scoped.py` runs 108 cells: 3 tasks (credit_vs_checking, checking_vs_mortgage, mortgage_vs_rest)
× 2 featurizer tracks (A frozen TF-IDF, B causal) × 3 methods (hashing, sparse JL, AWM) × 6 learner
budgets (4/16/32/64/128/512 KiB), chronological order, validation only. Per cell:

- successive-halving tuning on seed 2026: 8 candidates (eta0 {0.03, 0.1, 0.3, 1} × inverse-time /
  inverse-sqrt, tau = N), 8 at 1 pass → 4 at 2 → 2 at 5, selected by joint validation retention;
- the selected configuration on evaluation seeds 42/123/456 through 5 passes.

`analyze` gives k* at 95%/98% joint retention (paired bootstrap, 2,000 replicates), plus retention
CIs per budget. Learner, featurizer and total state are kept separate.

- **MISSION is excluded**, and the reason is recorded in the manifest.
- **Encoded-row cache:** each (task, track) is encoded once into memory. This is harness
  infrastructure, not learner state.
- **Inverse-sqrt schedule:** added to `Learner`; the default inverse-time behaviour is unchanged.

Run from the repository root under caffeinate, only when no other heavy job is running:

    caffeinate -i .venv/bin/python -m experiments.cfpb_streaming_state.scoped run
    caffeinate -i .venv/bin/python -m experiments.cfpb_streaming_state.scoped run --resume results/classical_streaming/scoped_<id>

A memory or swap gate failure stops the run cleanly. Completed cells are kept, and resume skips them.
