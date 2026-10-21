# CFPB scoped classical streaming-state benchmark

Run `20261020T154320Z-d1fc3382`, executed from commit `4c83221` (allocator `bookkeeping-v2`). The run started at 2026-10-20 15:43 UTC and took 24,841 s (6.90 h) of wall time.

**Label: measured, validation only.** The test split was never read (`test_access: false`). Test evaluation is a separate, unapproved step. The results are classical: they make no claim about quantum advantage or disadvantage.

## Verification (2026-10-21)

A read-only check was run before this report was written. Its script is not part of the repo.

- **Integrity.** The run's own `validate_resume` passed: completed-cell SHA-256 ledger, approved plan, source and environment fingerprints, preparation and spool hashes, and allocator `bookkeeping-v2` bounds for every tuning and evaluation allocation.
- **Cells.** All **108/108** cells match the approved plan, with status `completed` and seeds 42/123/456. Every seed has status `completed` and full checkpoint curves (passes 1, 2 and 5). There are no failed, diverged or truncated cells or seeds. Every prediction vector has the full validation length, with values ±1. Every learner state + 512 B fits its budget, and no tuning candidate diverged.
- **Recomputation.** All 36 `kstar.csv` rows (points, CIs and NR fractions) and all 108 `retention.csv` values were recomputed independently from the cell predictions, with **0 mismatches**.

On that basis the manifest status was changed from `cells_completed` to `completed`. The manifest's `verification` field records the change.

## Design

| Item | Setting |
|---|---|
| Cells | 108 = 3 tasks × 2 featurizer tracks × 3 methods × 6 learner budgets |
| Tasks | credit_vs_checking, checking_vs_mortgage, mortgage_vs_rest |
| Tracks | **A**: frozen January TF-IDF. **B**: causal document-local term counts |
| Methods | hashing + SGD, sparse JL + SGD, AWM-Sketch |
| Budgets | 4 / 16 / 32 / 64 / 128 / 512 KiB |
| Order | Chronological; passes after the first replay the stream |
| Tuning (seed 2026) | Successive halving over 8 candidates: η₀ ∈ {0.03, 0.1, 0.3, 1} × {inverse-time, inverse-sqrt with τ = N}. 8 candidates run 1 pass, the top 4 continue to 2, the top 2 to 5. The winner has the highest joint validation retention; ties go to the lower candidate index. That is 18 pass-equivalents per cell. |
| Evaluation | The selected configuration on seeds 42 / 123 / 456 for 5 passes, with checkpoints at passes 1, 2 and 5 |
| Learners | Unweighted, no intercept, half squared loss, λ = 1/N |

**Joint retention** compares the learner with frozen full Ridge (α = 1, unweighted, no intercept). It is min(accuracy ÷ Ridge accuracy, balanced accuracy ÷ Ridge balanced accuracy). Accuracy and balanced accuracy are each averaged over the three evaluation seeds before dividing.

**k\*** is the smallest budget in the grid at which the 5-pass joint retention is at or above the threshold. In every group, retention stays at or above the threshold at all budgets larger than k\*.

**Bootstrap.** There are 2,000 replicates with RNG seed 42, and each replicate does two things:
- **Examples are resampled once and shared.** One resample of validation examples (an index vector) is applied to the Ridge reference and to every budget, so the comparison between budgets is paired. All groups within a task use the same validation set and the same RNG draw sequence, so they get the same example resamples.
- **Seeds are resampled as whole blocks.** For each budget, the three evaluation seeds are drawn with replacement. Each drawn seed keeps its whole prediction vector, so per-seed predictions are never broken apart.

The seed draws are made **independently for each budget**: the same seed number is not tied together across budgets. That is deliberate, because a seed sets different hash or projection dimensions at different budgets, so seed 42 is not the same model at 4 KiB and at 64 KiB. **Method decision (2026-10-21):** keep per-budget seed draws. Seed identity is not shared across budgets, while validation examples stay paired across the reference and all budgets. The test-split confirmation uses the same bootstrap. Percentile 95% intervals use the lower/higher conventions. A replicate in which no budget reaches the threshold is counted as k\* = ∞. A blank CI upper bound means more than 2.5% of replicates never reached the threshold within 512 KiB.

**MISSION is excluded**, for the reason recorded in the manifest:
- it was the weakest method in every 128 KiB pilot cell;
- its small-budget effects of random vs chronological order (up to 14 accuracy points) need a separate multi-seed study;
- native MISSION has no L2 term, so it does not optimize the Ridge objective.

## Reference (frozen full Ridge, primary validation)

| Task | N train | N validation | Positive fraction | Accuracy | Balanced accuracy |
|---|--:|--:|--:|--:|--:|
| credit_vs_checking | 7,265 | 1,588 | 0.49 | 0.92128 | 0.92122 |
| checking_vs_mortgage | 4,987 | 1,107 | 0.73 | 0.97832 | 0.96638 |
| mortgage_vs_rest | 13,459 | 3,033 | 0.10 | 0.97923 | 0.91013 |

## k\* at 95% and 98% joint retention

Point estimates, with the 95% bootstrap CI in brackets. "NR" is the fraction of bootstrap replicates that never reached the threshold.

| Task | Track | Method | k\*₉₅ KiB [95% CI] | k\*₉₈ KiB [95% CI] |
|---|---|---|---|---|
| credit_vs_checking | A | Hashing + SGD | 16 [16, 32] | 64 [32, 128] |
| credit_vs_checking | A | Sparse JL + SGD | 16 [16, 32] | 64 [32, 128] |
| credit_vs_checking | A | AWM-Sketch | 16 [16, 16] | 32 [16, 128] |
| credit_vs_checking | B | Hashing + SGD | 16 [16, 32] | 64 [32, 512] (NR 1.1%) |
| credit_vs_checking | B | Sparse JL + SGD | 16 [16, 32] | 128 [32, 512] (NR 0.4%) |
| credit_vs_checking | B | AWM-Sketch | 16 [16, 32] | 64 [32, 512] (NR 0.3%) |
| checking_vs_mortgage | A | Hashing + SGD | 16 [16, 16] | 16 [16, 64] |
| checking_vs_mortgage | A | Sparse JL + SGD | 16 [16, 16] | 32 [16, 64] |
| checking_vs_mortgage | A | AWM-Sketch | 16 [4, 16] | 32 [16, 32] |
| checking_vs_mortgage | B | Hashing + SGD | 4 [4, 16] | 32 [16, 64] |
| checking_vs_mortgage | B | Sparse JL + SGD | 16 [4, 16] | 16 [16, 32] |
| checking_vs_mortgage | B | AWM-Sketch | 16 [4, 16] | 32 [16, 128] (NR 0.1%) |
| mortgage_vs_rest | A | Hashing + SGD | 16 [16, 32] | 32 [32, 128] |
| mortgage_vs_rest | A | Sparse JL + SGD | 16 [16, 32] | 32 [32, 128] |
| mortgage_vs_rest | A | AWM-Sketch | 16 [16, 16] | 32 [16, 64] |
| mortgage_vs_rest | B | Hashing + SGD | 16 [16, 32] | 32 [16, >512] (NR 5.2%) |
| mortgage_vs_rest | B | Sparse JL + SGD | 16 [16, 32] | 64 [32, 128] (NR 1.8%) |
| mortgage_vs_rest | B | AWM-Sketch | 16 [16, 32] | 64 [32, >512] (NR 4.3%) |

**Results:**
- **95% retention needs 16 KiB.** Every group reaches it at 16 KiB, with one exception: checking_vs_mortgage B with hashing reaches it at 4 KiB.
- **98% retention needs 16–128 KiB**, and most groups need 32–64 KiB.
- **Upper bounds can exceed the grid.** mortgage_vs_rest B with hashing and with AWM have upper CI bounds above 512 KiB (NR 5.2% and 4.3%). Their 98% k\* is not well determined within this grid.
- **No method dominates.** Differences between methods are mostly within one budget step and inside the CIs. This run supports no ranking of methods.

## Joint retention by budget (5 passes, seed mean)

Per-budget 95% CIs and the selected configurations are in [retention.csv](retention.csv).

| Task | Track | Method | 4 KiB | 16 KiB | 32 KiB | 64 KiB | 128 KiB | 512 KiB |
|---|---|---|--:|--:|--:|--:|--:|--:|
| credit_vs_checking | A | Hashing + SGD | 87.8% | 95.2% | 97.7% | 98.6% | 99.2% | 100.0% |
| credit_vs_checking | A | Sparse JL + SGD | 87.3% | 95.1% | 97.4% | 98.7% | 98.6% | 100.0% |
| credit_vs_checking | A | AWM-Sketch | 86.2% | 97.1% | 98.1% | 98.5% | 99.0% | 99.9% |
| credit_vs_checking | B | Hashing + SGD | 91.0% | 96.4% | 97.3% | 98.0% | 98.3% | 99.0% |
| credit_vs_checking | B | Sparse JL + SGD | 90.8% | 96.1% | 97.4% | 97.8% | 98.6% | 99.1% |
| credit_vs_checking | B | AWM-Sketch | 89.7% | 95.5% | 97.3% | 98.2% | 98.4% | 99.4% |
| checking_vs_mortgage | A | Hashing + SGD | 91.5% | 98.1% | 98.6% | 99.6% | 99.6% | 100.0% |
| checking_vs_mortgage | A | Sparse JL + SGD | 92.3% | 98.0% | 98.3% | 99.7% | 99.3% | 100.0% |
| checking_vs_mortgage | A | AWM-Sketch | 93.3% | 97.6% | 99.1% | 99.7% | 99.6% | 100.0% |
| checking_vs_mortgage | B | Hashing + SGD | 95.5% | 97.6% | 98.5% | 99.1% | 99.7% | 100.0% |
| checking_vs_mortgage | B | Sparse JL + SGD | 94.4% | 98.3% | 99.1% | 99.3% | 100.0% | 99.8% |
| checking_vs_mortgage | B | AWM-Sketch | 93.1% | 97.9% | 98.9% | 98.5% | 99.7% | 99.8% |
| mortgage_vs_rest | A | Hashing + SGD | 84.3% | 95.8% | 98.2% | 98.7% | 99.2% | 99.9% |
| mortgage_vs_rest | A | Sparse JL + SGD | 85.3% | 95.8% | 98.1% | 98.5% | 99.5% | 99.5% |
| mortgage_vs_rest | A | AWM-Sketch | 89.4% | 96.9% | 98.6% | 99.5% | 99.0% | 99.6% |
| mortgage_vs_rest | B | Hashing + SGD | 90.9% | 96.2% | 98.3% | 98.6% | 98.4% | 98.7% |
| mortgage_vs_rest | B | Sparse JL + SGD | 90.2% | 95.8% | 97.4% | 99.0% | 99.1% | 98.5% |
| mortgage_vs_rest | B | AWM-Sketch | 90.5% | 95.8% | 97.8% | 98.4% | 98.6% | 98.8% |

## Thresholds never reached

- **4 KiB:** 17 of 18 cells stay below 95% even after 5 passes. The exception is checking_vs_mortgage B with hashing (95.5%). The other 17 cells range from 84% to 94%.
- **98%:** 41 of 108 cells never reach it within 5 passes. All of them are below their group's k\*: every 4 KiB cell, most 16 KiB cells, and a few 32–64 KiB cells.
- **Every group-threshold reaches its point k\* within the grid.** No group's point estimate is "not reached."
- **Bootstrap non-reach.** Seven group-thresholds have a nonzero NR fraction, all at 98% and all on track B. The largest are mortgage_vs_rest B with hashing (5.2%) and with AWM (4.3%).

## Samples to target

**Label: descriptive.** These are seed-mean checkpoint crossings without CIs. For each group, the table gives the first evaluation checkpoint (passes 1, 2 or 5) at which the k\* cell's seed-mean joint retention reaches the threshold. Presentations = passes × N_train. On top of that, every cell spent 18 pass-equivalents on validation tuning and 15 on evaluation (3 seeds × 5 passes); neither is included in the presentation counts.

| Task | Track | Method | 95%: k\* KiB → first pass (presentations) | 98%: k\* KiB → first pass (presentations) |
|---|---|---|---|---|
| credit_vs_checking | A | Hashing + SGD | 16 → 1 (7,265) | 64 → 1 (7,265) |
| credit_vs_checking | A | Sparse JL + SGD | 16 → 1 (7,265) | 64 → 2 (14,530) |
| credit_vs_checking | A | AWM-Sketch | 16 → 1 (7,265) | 32 → 5 (36,325) |
| credit_vs_checking | B | Hashing + SGD | 16 → 1 (7,265) | 64 → 5 (36,325) |
| credit_vs_checking | B | Sparse JL + SGD | 16 → 1 (7,265) | 128 → 2 (14,530) |
| credit_vs_checking | B | AWM-Sketch | 16 → 1 (7,265) | 64 → 2 (14,530) |
| checking_vs_mortgage | A | Hashing + SGD | 16 → 1 (4,987) | 16 → 2 (9,974) |
| checking_vs_mortgage | A | Sparse JL + SGD | 16 → 1 (4,987) | 32 → 5 (24,935) |
| checking_vs_mortgage | A | AWM-Sketch | 16 → 1 (4,987) | 32 → 1 (4,987) |
| checking_vs_mortgage | B | Hashing + SGD | 4 → 1 (4,987) | 32 → 1 (4,987) |
| checking_vs_mortgage | B | Sparse JL + SGD | 16 → 1 (4,987) | 16 → 1 (4,987) |
| checking_vs_mortgage | B | AWM-Sketch | 16 → 1 (4,987) | 32 → 1 (4,987) |
| mortgage_vs_rest | A | Hashing + SGD | 16 → 1 (13,459) | 32 → 5 (67,295) |
| mortgage_vs_rest | A | Sparse JL + SGD | 16 → 1 (13,459) | 32 → 5 (67,295) |
| mortgage_vs_rest | A | AWM-Sketch | 16 → 1 (13,459) | 32 → 2 (26,918) |
| mortgage_vs_rest | B | Hashing + SGD | 16 → 1 (13,459) | 32 → 2 (26,918) |
| mortgage_vs_rest | B | Sparse JL + SGD | 16 → 1 (13,459) | 64 → 1 (13,459) |
| mortgage_vs_rest | B | AWM-Sketch | 16 → 1 (13,459) | 64 → 2 (26,918) |

**Over all 108 cells:**
- **95% after one pass:** 91 cells.
- **98% after one pass:** 44 cells.
- **Multiple passes help modestly.** At most budgets they add 0.5–1.5 points of retention. Replayed chronological passes are replay, not fresh data.

## Learner state vs featurizer state

**Learner state** is every allocated component of the learner: coefficients, sketch counters, keys, heap and index, projection and optimizer. It is measured after training. Every cell meets the final check that state + 512 bytes ≤ budget.

| Budget | Learner state, bytes (range over all cells) | 64-bit registers |
|---|--:|--:|
| 4 KiB | 3,525–3,574 | ≈ 441–447 |
| 16 KiB | 15,813–15,862 | ≈ 1,977–1,983 |
| 32 KiB | 32,205–32,246 | ≈ 4,026–4,031 |
| 64 KiB | 64,993–65,014 | ≈ 8,125–8,127 |
| 128 KiB | 130,537–130,550 | ≈ 16,318–16,319 |
| 512 KiB | 523,753–523,766 | ≈ 65,470–65,471 |

**Featurizer state** is the state needed to turn raw text into the stream the learner sees:

| Track | Featurizer state | Total persistent state at k\*₉₅ = 16 KiB | Featurizer share |
|---|--:|--:|--:|
| A: frozen TF-IDF (vocabulary + IDF) | 8,219,927 B (8.2 MB) | ≈ 8.24 MB | 99.8% |
| B: causal (tokenizer, no learned vocabulary) | 47,939 B (47 KB) | ≈ 63.8 KB | 75% |

**What this means:**
- **Track A's cost is its featurizer.** On track A, the frozen TF-IDF featurizer is about 500× the learner's state at the 95% target. It is also built from all January rows, so it is not a single-pass object.
- **Track B removes that cost.** It reaches the same retention targets against the same TF-IDF Ridge reference with a 47 KB featurizer, so its whole pipeline fits in about 64 KB at 95% and 64–179 KB at 98% (learner k\*₉₈ of 16–128 KiB).
- **The featurizer cost would apply to any learner fed this representation, a QOS data stream included.**
- **These sizes come from the Python implementation.** They are allocation inventories including Python/NumPy object overhead. They are not information-theoretic lower bounds.

## Caveats

- **The k\* values are optimistic.** Tuning, selection and k\* all use the same primary validation set. Confirmation on test needs a frozen-configuration evaluation, which is a separate step.
- **Limited scope.** Three tasks, chronological order only, one tuning seed and three evaluation seeds. Random-order cells were not run, and neither was MISSION.
- **The harness caches encoded rows.** Each (task, track) stream is encoded once into an in-memory cache. That cache is harness infrastructure and is not charged as learner state, just as the on-disk spools are not.
- **Track A sees all of January.** Its vocabulary and IDF were fitted on all January rows, so its chronological replay is conditional replay, not prospective preprocessing.
- **Track B changes the representation.** Its gap from the TF-IDF Ridge target mixes representation, optimization and sketching effects.
- **Earlier runs are kept for the record.**
  - The partial run `scoped_20261020T142519Z-60752b83` (15 cells, legacy allocator) stopped on the byte-ceiling check, and [ALLOCATION_FIX.md](../../../experiments/cfpb_streaming_state/ALLOCATION_FIX.md) explains the cause. Its cells are not used here.
  - The full-dimensional SGD convergence control [convergence_20261020T142038Z-c609c8bb](../convergence_20261020T142038Z-c609c8bb/checkpoints.csv) reached ≥ 99.3% joint retention by 5 passes in all 12 configurations. So the shortfalls at small budgets are not an optimizer failure.

## Files

| File | Contents |
|---|---|
| [kstar.csv](kstar.csv) | k\* point estimates, CIs and NR fractions |
| [retention.csv](retention.csv) | Joint retention with CIs and the selected configuration, per budget |
| [state.csv](state.csv) | Learner, featurizer and total state per cell |
| [samples_to_target.csv](samples_to_target.csv) | Descriptive checkpoint crossings for every cell and threshold |
| `cells/*.json` | Per-cell tuning rungs, allocations, evaluation curves and validation predictions |
| [manifest.json](manifest.json) | Source and environment fingerprints, cell checksums, verification record |
| [run_plan.json](run_plan.json), [preparation.json](preparation.json) | The approved plan and data preparation |

## Test-split confirmation

The frozen validation-selected configurations were later re-evaluated on the CFPB test split, without retuning: [scoped_test_20261021T152347Z-67e0c44e](../scoped_test_20261021T152347Z-67e0c44e/report.md), labelled "measured, test split, frozen selections". All 324 seeds reproduced this run's validation results exactly before being scored on test.
- **95%:** k\*₉₅ on test is 4–32 KiB, and every group is at or above 95% by 32 KiB.
- **98%:** reached by 128 KiB in 16 of 18 groups. mortgage_vs_rest B with sparse JL and with AWM do not reach it within 512 KiB.
