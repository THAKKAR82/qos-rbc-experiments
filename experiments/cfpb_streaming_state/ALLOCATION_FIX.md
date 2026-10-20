# Scoped streaming allocation correction — 2026-10-20

This repair is based on `7df1c3c1d186ec5423dc59dda6d5efcede43b2d0`.
No CFPB benchmark or QOS simulation was run. New training was confined to small
synthetic correctness fixtures. The existing Bank regression suite was run.
Predictive performance of corrected CFPB allocations has **not** been measured.

## Failure and correction

The old allocator maximized width subject to **initialized** reachable state
plus 512 bytes fitting the budget. The final check applied that same inequality
to **trained** state. Deduplicated Python scalar objects can stop sharing identity:
initially `t` and heap occupancy `n` both reference zero; during training they
become independent integers. Occupancy can also stop sharing a cached integer
with capacity. This is a genuine allocation defect, not a reason to increase
the budget or relax the final check.

`methods.py` now uses allocator version `bookkeeping-v2`. It computes fixed
reachable bytes with the changing scalar references absent, then adds independent
maximum sizes for the update counter, floating scale, and (when present) heap
occupancy. Two shallow control-object copies make the calculation without
mutating the original model, its arrays, or RNG. Immutable aliases are retained
in the fixed part. Bounds are calculated using the current interpreter's
`sys.getsizeof`, not an unexplained constant padding increment.

The supported update counter is `0 <= t <= 2**63-1`; overflow is rejected before
an update mutates state. Heap occupancy is bounded by its preallocated capacity.
On the verified interpreter these independent scalar charges are 36 bytes,
24 bytes, and 28 bytes respectively at the approved budgets. This supports far
more than the approved five passes. Allocation selects the largest width with
`allocation_bound(model) + 512 <= budget`.

The original `state_inventory()` and `state_bytes()` measurement, alias
deduplication, byte/register interpretation, active-capacity rule, and final
`state_bytes() + 512 <= budget` requirement are unchanged. Budgets remain
4/16/32/64/128/512 KiB. No preprocessing, labels, splitting, update equations,
regularization, seed lists, hyperparameter grid, pass budgets, scoring,
selection, bootstrap, or scientific plan changed. Different allocated widths
are an intentional consequence of fixing the allocator.

The following are measured local allocation diagnostics using a tiny synthetic
update, not CFPB performance results:

| 64 KiB AWM quantity | Legacy | Corrected |
| --- | ---: | ---: |
| Sketch width per row | 1624 | 1622 |
| Sketch depth | 3 | 3 |
| Active capacity | 512 | 512 |
| Initialized reachable bytes | 65,013 | 64,965 |
| Reachable bytes after synthetic update | 65,041 | 64,993 |
| Trained bytes + unchanged 512 margin | 65,553 | 65,505 |
| Derived supported-state bound | 65,049 | 65,001 |
| Bound + unchanged margin | 65,561 | 65,513 |

The original failure exceeds the final requirement by 17 bytes. The corrected
supported-state bound leaves 23 bytes below 65,536. The extra reserve is 36 bytes
over initialized state here; width discretization removes 48 array bytes.
The bound is an allocation calculation, not a measured peak or process RSS.

## Manifest and resume behavior

`scoped.py` records allocator version and, for every tuning candidate and
evaluation seed, dimensions, initialized bytes, supported-state bound, reserve,
margin, and counter ceiling. These records are harness metadata, not extra
learner state. Existing measured-state fields keep their original meaning.

New manifests include execution-source and numerical-environment fingerprints,
preparation SHA256, and a completed-cell checksum ledger. Cell files also carry
run/source/allocator identity. Resume validates these, the approved plan, replay
spool/offset hashes, and checkpoint allocation provenance **before any write or
model execution**. Analysis applies the same check to prevent combining
incompatible allocations. Unregistered orphan cells are rejected for explicit
recovery review, not silently skipped.

MemoryError is now caught across resource preflight, preparation, encoding,
training/evaluation, and analysis. It records `status="interrupted"`, exception
type, stage, current cell when applicable, and the error message, then re-raises.
The final memory-check error includes cell ID, measured bytes, margin, budget,
and allocator version. Runtime/resource, validation, counter-overflow and
keyboard interruptions are also recorded. Successful matching resumes clear
old interruption details and retain completed cell files unchanged.

## Compatibility of the 15 completed cells

Preserved directory:
`results/classical_streaming/scoped_20261020T142519Z-60752b83/`.
Its 15 cell files contain results/predictions, not serialized learner arrays.
All are `credit_vs_checking / A_frozen`, with 45 completed evaluation seeds.
Its 13 execution-source hashes match the pre-fix source; old dimensions below
were reconstructed from that pinned source, since cells did not record them.
Both schedules and all four learning rates yield the same listed dimensions.

| Method | Budget KiB | Legacy widths → corrected widths |
| --- | --- | --- |
| Hash SGD | 4/16/32/64/128/512 | 355→354 / 1891→1890 / 3939→3938 / 8035→8034 / 16227→16226 / 65379→65378 |
| Sparse JL SGD | 4/16/32/64/128/512 | 116 / 628 / 1311 / 2676 / 5407 / 21791, unchanged |
| AWM | 4/16/32 | 56→54 / 344→342 / 771→769 |

Nine of 15 cells change dimensions, so their predictions cannot stand in for
corrected-allocator results. The six JL cells retain dimensions and unchanged
update equations, but no automatic import or retrospective version relabeling
has been implemented. Treat all 15 as a separate historical partial run.

The three saved AWM cells passed their final checks. That does not establish
intermediate compliance: bookkeeping-only probes at `t=7265, n=7` produce
3,601 / 15,889 / 32,281 bytes at 4/16/32 KiB. Adding the existing margin exceeds
their ceilings by 17 / 17 / 25 bytes. These probes inspect scalar allocation
only and are not purported model trajectories. At full occupancy, small cached
integers can alias capacity again. The saved final measurements cannot certify
the earlier states.

The original manifest still says `running`; it is preserved byte-for-byte.
This report records the stale status externally. All 30 files in that directory,
including its preparation, six spools, six offset files, and 15 cells, remain
unchanged. There is **no safe in-place resume of that directory under this fix**.

## Validation and preservation

The regression tests cover all six budgets, four budgeted methods (including
the excluded MISSION primitive), both schedules, and all four learning rates.
Each of those 192 combinations is checked initially and after 258 real
synthetic updates. Separate probes cover partial/full occupancy, scalar
de-aliasing, `2**30` integer-size growth, and the declared maximum counter.
Tests also check snapshot/RNG preservation, exact fixed-dimension update
equivalence, invalid bookkeeping ranges, counter-overflow rejection without
mutation, the original 64 KiB failure, useful final-check messages, manifest
interruptions, rejected incompatible resumes/analysis without writes, and
matching resumes that skip and preserve completed cells.

Commands use the main `.venv`, with numerical thread counts set to one and
`PYTHONDONTWRITEBYTECODE=1`:

```sh
.venv/bin/python -m pytest -q -p no:cacheprovider experiments/cfpb_streaming_state/tests
.venv/bin/python -m pytest -q -p no:cacheprovider tests --ignore=tests/test_regression.py
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_regression.py
.venv/bin/python -m pip check
git diff --check
```

Results: **306 streaming tests**, **188 repository fast tests**, and **9 Bank
regression tests** pass. `pip check` passes. Python 3.13.9 and all 11 pinned
dependencies match `requirements.txt`; dependencies were not changed. Pip
reported its unwritable user cache and disabled that cache; dependency checks
still succeeded.

Bank regressions verify N=45,211, D=51, maximum row sparsity=16, nnz=723,376,
and the existing theoretical QOS accounting value 41 exactly. They reproduce
the tested paper accuracy 0.8995155858234207, paper balanced accuracy
0.6300640934600773, balanced-model balanced accuracy 0.8256193229214974, CSR JL
endpoint ordinary accuracy 0.899537704823642 and balanced-model balanced accuracy
0.8256067994399695 within `rtol=0, atol=1e-8`. Exact floating deltas were not
exported by the tests; no claim of zero recomputation error is made. The saved
80 sweep rows remain unchanged. A full Bank notebook/sweep rerun was skipped:
changes are confined to the independent streaming experiment; shared `src`,
notebook 05, Bank fixtures, inputs, and numerical dependencies are unchanged.

SHA256 verification covers **830 unchanged pre-existing files**, totaling
1,935,377,625 bytes, including prior results, datasets, notebooks, regression
fixtures and QOS artifacts. The only four modified pre-existing files are
`methods.py`, `scoped.py`, `tests/test_scoped.py`, and this experiment's README.
The **87 frozen CFPB manifest entries** pass their independent checks, as do
the legacy run's 12 spool/offset checks. No saved numerical output or dataset
was updated.

Local audit records:

- [Allocation matrix, environment, checkpoint comparison and preservation summary](validation/allocator_fix_20261020/audit.json).
- [SHA256 preservation manifest](validation/allocator_fix_20261020/preservation_manifest.json).
- [Check commands, counts, measured test durations and outcomes](validation/allocator_fix_20261020/checks.json).

These are separate local validation artifacts; the repository's existing
validation ignore rule applies. No prior source fingerprint or fixture was
refreshed to make a test pass.

## Replacement and safe future resume — approval required

Recommend a complete fresh replacement of the same 108-cell plan using
`bookkeeping-v2`, the same frozen populations/references, chronological order,
tuning seed 2026, evaluation seeds 42/123/456, and unchanged data-access budgets.
Preparation is rebuilt deterministically in the new directory. Do not copy
legacy cell files or alter the old manifest. Record the new run ID alongside
this legacy run ID in the approval/run record.

After approval, start the replacement from the repository root:

```sh
caffeinate -i env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/tmp/cfpb-streaming-mpl \
  .venv/bin/python -m experiments.cfpb_streaming_state.scoped run
```

For a later orderly interruption of that **new corrected run**, with source,
environment and inputs unchanged, the safe resume form is:

```sh
caffeinate -i env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/tmp/cfpb-streaming-mpl \
  .venv/bin/python -m experiments.cfpb_streaming_state.scoped run \
  --resume results/classical_streaming/scoped_<NEW_CORRECTED_RUN_ID>
```

Neither command was executed. No benchmark, new QOS experiment, commit, or push
was performed during this repair. Replacement execution awaits user approval.
