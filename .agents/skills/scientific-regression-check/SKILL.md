---
name: scientific-regression-check
description: Validate this QOS repository after refactors, dependency changes, or experimental code changes; run tests, assess Bank rerun needs, and report scientific drift without modifying scientific code.
---

# Scientific regression check

## Inspect

1. Read repository-root `AGENTS.md` first. All paths below are relative to that root.
2. Inspect changed files, including staged, unstaged, untracked files and any user-specified comparison revision. If root Git metadata is unavailable, use supplied diffs or recorded snapshots and disclose missing history. Never substitute the reference checkout's commit for the project commit.
3. Read `README.md`, root `requirements.txt`, `tests/test_regression.py`, and relevant changed code/tests. Inspect assertion and fixture changes too. Record Python and package versions versus pins, and before/after hashes of notebook 05, data, original results, and frozen test fixtures.

Validate without repairing: do not modify scientific code unless explicitly asked for a fix. Likewise, do not refresh tests, fingerprints, baselines, tolerances, historical ledger entries, or dependencies to conceal failures. Preserve notebook source and outputs; save executed copies separately.

## Required checks

From the root, using the repository environment:

```sh
.venv/bin/python -m pytest -q tests --ignore=tests/test_regression.py
.venv/bin/python -m pytest -q tests/test_regression.py
.venv/bin/python -m pip check
```

Review test discovery if the suite layout changed. Record commands, counts, failures, and warnings; continue independent checks after a failure. Compare installed versions with pins separately: `pip check` only checks dependency consistency. Missing environment/data means validation is incomplete, not permission to silently substitute an interpreter or change dependencies.

## Full Bank rerun decision

- **Required:** notebook 05 or scientific implementation changes could affect preprocessing, representations/dtypes, RNG/sketches, solvers, CV, metrics, aggregation, or accounting; numerical dependencies/Python changed; unexplained numerical drift occurred; or scientific impact cannot be established reliably.
- **Usually skip:** demonstrably documentation, presentation, or test-only changes with unchanged scientific behavior, inputs, and environment. State the evidence. Passing unit tests alone does not waive a required rerun.
- **Intentional methodology changes:** still compare, but report separately from reproduction; do not adopt new baselines automatically.

When required, execute all cells of notebook 05, including both sweeps, with the verified repository Python kernel. Use the README procedure or nbclient with that kernel explicitly selected. Inspect exports first; write to a fresh validation/temporary directory, never original `results/` or notebook 05 in place. If blocked, report incomplete validation and the blocker.

Compare new raw CSVs (40 rows per method, 80 total), both summaries, and the comparison table against original artifacts. Verify keys, dimensions `[4,8,12,16,24,32,40,51]`, seeds `[42,123,456,789,2026]`, all scores and resource fields—not just averages. Read the new validation report, not an older successful run. Report maximum differences; distinguish figure comparison from numerical equivalence.

## Reference values

Use exact equality for dimensions/counts. For scores use `rtol=0` and the established `ATOL` in `tests/test_regression.py` (currently `1e-8`). Investigate tolerance changes; **never widen tolerances to make failures pass**.

| Quantity | Reference |
| --- | ---: |
| N | 45211 |
| D | 51 |
| Maximum row sparsity | 16 |
| nnz | 723376 |
| QOS LS-SVM resource estimate | 41 |
| Paper Ridge ordinary accuracy | 0.8995155858234207 |
| Paper Ridge balanced accuracy | 0.6300640934600773 |
| Balanced Ridge balanced accuracy | 0.8256193229214974 |
| CSR JL endpoint ordinary accuracy | 0.899537704823642 |
| CSR JL endpoint balanced-Ridge balanced accuracy | 0.8256067994399695 |

Preserve the limitations in `AGENTS.md`: sweep preprocessing leakage, row-only accounting, and representation-sensitive solvers. Keep empirical scores distinct from theoretical QOS estimates. Verify protected-file hashes after execution.

## Findings and final report

Classify findings with evidence; categories may overlap:

- **Code/test failure:** exceptions, broken interfaces/assertions, or stale source fingerprints.
- **Numerical drift:** counts change or scores exceed tolerance under unchanged methodology. Also report nonzero differences within tolerance.
- **Intentional methodology change:** explicitly documented/authorized protocol changes, distinct from bugs or baseline equivalence.
- **Dependency/environment drift:** interpreter, package, solver/backend, or environment mismatch; distinguish suspected causes from demonstrated ones.

Report:

1. **Scope/environment:** changed files, comparison basis, missing provenance, version differences.
2. **Checks:** commands and pass/fail counts; full rerun performed/skipped/blocked, rationale, artifact paths.
3. **Numerics:** expected, observed, delta, and status for every reference quantity; sweep coverage and maximum differences. Mark unmeasured quantities and historical evidence explicitly.
4. **Findings:** categories, methodological differences, artifact integrity, unresolved blockers.
5. **Scientific conclusions:** unchanged within tested scope, changed, or not established, supported by evidence. Passing regressions demonstrate reproducibility under existing assumptions, never quantum advantage. Never describe theoretical accounting as measured quantum hardware usage.
