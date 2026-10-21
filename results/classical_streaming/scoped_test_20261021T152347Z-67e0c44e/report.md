# Test-split confirmation of the scoped streaming benchmark (frozen selections)

Run `20261021T152347Z-67e0c44e`, executed from commit `8b9e832`. Source run: [`20261020T154320Z-d1fc3382`](../scoped_20261020T154320Z-d1fc3382/report.md). Wall time: 3.57 h.

**Label: measured, test split, frozen selections.** Each cell uses the configuration selected on validation in the source run, with no retuning. Test labels played no part in any selection. Results are classical and make no quantum claim.

## Protocol

- **Test data.** The CFPB primary test rows, February 15–28, 2026 (4,213 rows). Membership comes from the frozen split manifest, and the cleaned text is checked against the frozen curation hashes. Rows are in chronological order.
- **Ridge reference.** Frozen full Ridge (α = 1 for all three tasks, as selected on validation; unweighted; no intercept) is refitted on the same training rows. It must:
  - reproduce the source run's validation predictions **exactly**;
  - match the frozen step-1 audit's test accuracy and balanced accuracy to **1e-12**.

  Both checks passed.
- **Learner rebuild.** For each of the 108 cells and each evaluation seed (42/123/456), the selected configuration is rebuilt and retrained on the same chronological training cache through 5 passes. Before any test score is used, the rebuild must reproduce the source run **exactly**: the saved allocation, the validation accuracy and balanced accuracy at passes 1, 2 and 5, and the 5-pass validation predictions. **All 324 seeds reproduced exactly.** Learner state stays within budget.
- **Statistics.** Joint retention and k\* are defined as in the source run. The bootstrap is the same: 2,000 replicates with seed 42. Each replicate resamples test examples once and shares that resample across the reference and every budget, and draws seeds independently per budget.
- **Code.** [`experiments/cfpb_streaming_state/confirmation/scoped_test.py`](../../../experiments/cfpb_streaming_state/confirmation/scoped_test.py), with tests in `tests/test_scoped_test.py`. It sits outside the source-hash set of `scoped.py`, so the source run still validates. The source run directory is not modified.

## Ridge reference on test (primary rows)

| Task | N test | Accuracy | Balanced accuracy |
|---|--:|--:|--:|
| checking_vs_mortgage | 1,517 | 0.98154 | 0.96989 |
| credit_vs_checking | 2,203 | 0.92056 | 0.92056 |
| mortgage_vs_rest | 4,213 | 0.97911 | 0.91612 |

## k\* at 95% and 98% joint retention: test vs validation

Values are KiB, with the 95% bootstrap CI in brackets. "not reached" means the point estimate did not reach the threshold at any budget up to 512 KiB. NR is the fraction of bootstrap replicates that never reached it.

| Task | Track | Method | k\*₉₅ test | k\*₉₅ validation | k\*₉₈ test | k\*₉₈ validation |
|---|---|---|---|---|---|---|
| credit_vs_checking | A | Hashing + SGD | 16 [16, 16] | 16 [16, 32] | 32 [32, 64] | 64 [32, 128] |
| credit_vs_checking | A | Sparse JL + SGD | 16 [16, 16] | 16 [16, 32] | 32 [16, 64] | 64 [32, 128] |
| credit_vs_checking | A | AWM-Sketch | 16 [16, 16] | 16 [16, 16] | 16 [16, 32] | 32 [16, 128] |
| credit_vs_checking | B | Hashing + SGD | 16 [16, 16] | 16 [16, 32] | 32 [32, 64] | 64 [32, 512] (NR 1.1%) |
| credit_vs_checking | B | Sparse JL + SGD | 16 [16, 16] | 16 [16, 32] | 32 [16, 64] | 128 [32, 512] (NR 0.4%) |
| credit_vs_checking | B | AWM-Sketch | 16 [16, 16] | 16 [16, 32] | 32 [16, 64] | 64 [32, 512] (NR 0.3%) |
| checking_vs_mortgage | A | Hashing + SGD | 16 [16, 16] | 16 [16, 16] | 32 [16, 32] | 16 [16, 64] |
| checking_vs_mortgage | A | Sparse JL + SGD | 16 [16, 16] | 16 [16, 16] | 16 [16, 64] | 32 [16, 64] |
| checking_vs_mortgage | A | AWM-Sketch | 16 [16, 16] | 16 [4, 16] | 32 [16, 32] | 32 [16, 32] |
| checking_vs_mortgage | B | Hashing + SGD | 16 [4, 16] | 4 [4, 16] | 32 [16, 64] | 32 [16, 64] |
| checking_vs_mortgage | B | Sparse JL + SGD | 4 [4, 16] | 16 [4, 16] | 32 [16, 64] | 16 [16, 32] |
| checking_vs_mortgage | B | AWM-Sketch | 16 [16, 16] | 16 [4, 16] | 32 [32, 32] | 32 [16, 128] (NR 0.1%) |
| mortgage_vs_rest | A | Hashing + SGD | 32 [16, 32] | 16 [16, 32] | 64 [32, 64] | 32 [32, 128] |
| mortgage_vs_rest | A | Sparse JL + SGD | 16 [16, 32] | 16 [16, 32] | 64 [32, 64] | 32 [32, 128] |
| mortgage_vs_rest | A | AWM-Sketch | 16 [16, 16] | 16 [16, 16] | 32 [16, 128] | 32 [16, 64] |
| mortgage_vs_rest | B | Hashing + SGD | 32 [16, 32] | 16 [16, 32] | 128 [32, >512] (NR 37.5%) | 32 [16, >512] (NR 5.2%) |
| mortgage_vs_rest | B | Sparse JL + SGD | 32 [16, 32] | 16 [16, 32] | not reached [32, >512] (NR 39.2%) | 64 [32, 128] (NR 1.8%) |
| mortgage_vs_rest | B | AWM-Sketch | 32 [16, 32] | 16 [16, 32] | not reached [32, >512] (NR 37.9%) | 64 [32, >512] (NR 4.3%) |

**Results:**
- **95% is confirmed at small budgets.** On test, k\*₉₅ is 16 KiB in 13 of 18 groups and 4 KiB in one. The other four groups need 32 KiB: mortgage_vs_rest A with hashing, and all three mortgage_vs_rest B methods. At 32 KiB, all 18 groups are at or above 95% on test.
- **98% holds except on mortgage_vs_rest B.** 16 of 18 groups reach 98% by 128 KiB. On credit_vs_checking the test k\*₉₈ is lower than on validation.
- **mortgage_vs_rest B is the exception.** This is the imbalanced task (10% positive) on the causal track.
  - With sparse JL and AWM it **never reaches 98% on test** within 512 KiB; test retention plateaus at 97.5–97.9%, and NR is about 38–39%.
  - With hashing it reaches 98% at 128 KiB (98.07%) but falls back to 97.6% at 512 KiB, NR 37%.

  This 98% target is not supported on test for this group. A plausible but untested explanation is the representation gap between causal features and the TF-IDF Ridge reference, rather than memory.

**Test retention by budget across all 18 groups:**

| Budget | Retention range | Groups ≥ 95% | Groups ≥ 98% |
|---|---|--:|--:|
| 4 KiB | 82.6–95.2% | 1 | 0 |
| 16 KiB | 94.0–98.4% | 14 | 2 |
| 32 KiB | 96.8–99.5% | 18 | 13 |
| 64 KiB | 97.6–99.7% | 18 | 15 |
| 128 KiB | 97.5–100.3% | 18 | 16 |
| 512 KiB | 97.6–100.2% | 18 | 15 |

**After one pass** (descriptive), 84 of 108 cells reach 95% on test and 56 reach 98%.

## Caveats

- **Scope.** One test window (two weeks of February), three tasks, chronological order only, and three evaluation seeds.
- **Grid-limited.** k\* is resolved only to the budget grid. Retention is not monotone in budget everywhere (see mortgage_vs_rest B with hashing).
- **Track A's featurizer** is the frozen January TF-IDF (8.2 MB). Track B's causal featurizer is 47.9 KB. Learner state is reported separately, in the source run's `state.csv`.

## Files

| File | Contents |
|---|---|
| [kstar_test.csv](kstar_test.csv) | Test k\* with CIs and NR, next to the validation values |
| [retention_test.csv](retention_test.csv) | Test accuracy, balanced accuracy and joint retention with CIs, per budget |
| [test_reference.json](test_reference.json) | Ridge test reference: labels, predictions, complaint IDs, audit cross-check |
| `cells/*.json` | Per-seed test curves (passes 1/2/5), test predictions, reproduction flags |
| [manifest.json](manifest.json) | Source run and hashes, code hashes, environment, cell checksums |
