# Final summary: Quantum Oracle Sketching on bank-held data

Date: 2026-10-20. Paper: Zhao et al., arXiv:2604.07639v2 (Theorem F.13, Eqs. F48–F62); reference code `reference/quantum-oracle-sketching/` at `a3fe8d6`.

**Conclusion.** QOS is not useful for the CFPB complaint text or the Bank Marketing table studied here. The theorem's sparsity condition fails by orders of magnitude on both. Even with the measured, smaller sparsity exponent, the extrapolated cost of each data query is 10²³–10²⁷ samples. Streaming classical learners reach 95% of full Ridge on validation with 16 KiB of learner state, usually within one pass, and 98% with 16–128 KiB. None of this disproves Theorem F.13 or says anything about other datasets. Phase (b), an end-to-end QOS simulation, was cancelled ([decision](../docs/decisions/phase_b_cancelled.md)), so the κ, γ and δ factors keep their proof values.

**Labels used in this summary.** Every table carries one of these:

| Label | Meaning |
|---|---|
| **measured** | Computed from real data or real classical training runs |
| **simulated** | Classical numerical simulation of QOS primitives (expected-unitary model, reference convention); not hardware |
| **calculated** | Arithmetic on theorem formulas with measured inputs; constants and polylogs dropped, so orders of magnitude only |
| **estimated** | A back-of-envelope resource estimate or an extrapolation beyond the measured range |

Nothing here is measured quantum hardware usage.

## 1. Applicability audit (step 1)

**Label: measured inputs, calculated F61.** Sources: [docs/audit_step1_summary.md](../docs/audit_step1_summary.md), [results/audit/summary_step1.csv](audit/summary_step1.csv), [cfpb_step1/](audit/cfpb_step1/), [bank_step1/](audit/bank_step1/).

This audit asks whether the theorem's preconditions hold, and how large the F61 sample prefactor is. Conventions: unweighted ridge, no intercept, ‖X‖ = 1, labels ±1. Ridge α is selected on validation; ties go to the larger α.

| Dataset / family | Tasks | s_row_or_col | s / N | κ_reg (selected α) | γ p10 | log10 F61 @ selected α | log10 F61 at s = 1 (floor) |
|---|--:|--:|--:|--:|--:|--:|--:|
| CFPB pairwise | 6 | 2,916–5,992 | 0.58–0.71 | 5.4–16.5 | 0.008–0.11 | 36.7–40.7 | 11.1–14.7 |
| CFPB one-vs-rest | 4 | 8,776 | 0.65 | 18.7 | 0.007–0.13 | 40.9–42.9 | 13.3–15.3 |
| Bank (5 CV folds) | 5 | 36,168–36,169 | 1.00 | 40.8 | 0.13–0.15 | 44.8–44.9 | 12.2–12.3 |
| PaySim (analytic only) | — | 6,362,620 | 1.00 | — | — | s⁷ alone: 47.6 | — |

- **The sparsity condition fails.** The binding sparsity is column sparsity, and it scales with N. CFPB's heavy columns cover 58–71% of rows. Bank's and PaySim's dense numeric columns cover all rows.
- **Bank's σ_min = 0.** Bank has rank 43 < D = 51, so κ_reg grows like √N at a fixed α. The same holds for CFPB (D > N).
- **The prediction step and the trained model are far apart.** One prediction needs about 52 logical qubits under the reference's machine-size accounting (A9). The reusable-model prefactor F60 is 10¹²–10¹⁸. The two differ by more than 10 orders of magnitude.
- **Bank caveat.** The Bank audit uses its global encoding, which carries the known preprocessing-leakage caveat in AGENTS.md.

## 2. Tightness study: phase (a) and the N-dependence check

**Label: simulated (expected-unitary QOS, reference convention; ideal circulant sparsity).** Sources: [experiments/qos_tightness/N_DEPENDENCE.md](../experiments/qos_tightness/N_DEPENDENCE.md), [phase_a_20261019T145729Z-a1644ff3/](tightness/phase_a_20261019T145729Z-a1644ff3/) (N = 64), [phase_a_n128_20261019T153934Z-75edf052/](tightness/phase_a_n128_20261019T153934Z-75edf052/) (N = 128, including [comparison/](tightness/phase_a_n128_20261019T153934Z-75edf052/comparison/)), [experiments/qos_tightness/PLAN.md](../experiments/qos_tightness/PLAN.md).

M* is the number of samples a primitive needs to reach error ε*; we fit M* ∝ s^α. Settings: N = 64 used s = 2–32 with 5 instances (3 at s = 32); N = 128 used s = 2–16 with 3 instances; M runs from 10⁴ to 10¹¹. The CIs below reflect instance variation only.

| Primitive | ε* | Exponent of s (proof) | α at N = 64 (published) | α at N = 128 [95% CI] | Δ vs matched N = 64 |
|---|--:|--:|--:|--:|--:|
| Index oracle | 10⁻² | 3 | 2.200 | 2.225 [2.221, 2.231] | −0.007 |
| Index oracle | 10⁻³ | 3 | 2.191 | 2.207 [2.203, 2.213] | −0.016 |
| D.23 block encoding A/2, raw | 10⁻² | 5 | 3.147 | 3.273 [3.247, 3.298] | +0.062 |
| D.23 block encoding A/2, polylog-divided | 10⁻² | 5 | 2.671 | 2.787 [2.761, 2.813] | +0.071 |
| D.23 block encoding A/2, raw | 10⁻³ | 5 | 3.128 | 3.251 [3.226, 3.275] | +0.061 |
| D.23 block encoding A/2, polylog-divided | 10⁻³ | 5 | 2.755 | 2.872 [2.847, 2.896] | +0.066 |

- **The exponents are stable across N.** No change exceeds 0.3. Read the exponents to about ±0.2, the precision these finite grids support.
- **The error dependence matches.** β ≈ 1.01–1.02, consistent with the proof's ε⁻¹.
- **The proof's extra factor of s does not appear.** The proof gets it by splitting the precision budget; it is absent empirically.
- **The fitted range is small.** N ≤ 128, s ≤ 32, uniform sparsity. This does not show that the asymptotic bound is loose.
- **The element-oracle exponent is not estimable.** Its crossings fall below the sample grid.
- **Found along the way.** pyqsp 0.2.0's `PolySign(ensure_bounded=True)` returns a polynomial with |P| > 1 at s ≥ 16, and the reference then silently produces garbage angles. We replaced it with a verified-bounded sign polynomial ([repro](../experiments/qos_tightness/repro/)). It has not been reported upstream.

## 3. Empirical F61

**Label: estimated (extrapolation from small-size simulation; orders of magnitude only).** Sources: [docs/empirical_f61.md](../docs/empirical_f61.md), [results/audit/empirical_f61/](audit/empirical_f61/).

Only the block encoding's s⁵ is replaced, by the measured s^3.15 (N = 64 fit) to s^3.27 (N = 128 fit). The κ⁴, γ⁻⁴, δ⁻¹ and readout s² factors keep their proof values.

| Family | s | log10 F61, proof (s⁵) | log10 F61, empirical | Change (decades) | Samples per A/2 query, ε = 10⁻² |
|---|--:|--:|--:|--:|--:|
| CFPB pairwise (6) | 2,916–5,992 | 36.7–40.7 | 30.3–34.3 | −6.0 to −7.0 | 1.9×10²³ – 3.6×10²⁴ |
| CFPB one-vs-rest (4) | 8,776 | 40.9–42.9 | 33.6–36.1 | −6.8 to −7.3 | 6.2×10²⁴ – 1.3×10²⁵ |
| Bank (5 folds) | 36,168–36,169 | 44.8–44.9 | 36.4–37.1 | −7.9 to −8.4 | 5.3×10²⁶ – 1.3×10²⁷ |

- **The measured exponent helps but doesn't rescue QOS.** It lowers F61 by 6–8 decades, but the totals are still 10³⁰–10³⁷.
- **The per-query cost is large in its own right.** It is 10¹⁷–10²¹ times the dataset's nonzero count for every single block-encoding query.
- **The extrapolation is long.** The exponent was fitted at s ≤ 32 and is applied at s = 2,916–36,169, which is 2–3 decades beyond the fitted range.

## 4. QROM vs QOS per-query loading cost (CFPB one-vs-rest matrix)

**Label: estimated (back-of-envelope resource estimate; units differ, so compare orders of magnitude only).** Sources: [docs/qrom_vs_qos_estimate.md](../docs/qrom_vs_qos_estimate.md), [results/audit/qrom_vs_qos/](audit/qrom_vs_qos/) (`qrom_costs.csv`, `summary.json`), [tests/test_qrom_formulas.py](../tests/test_qrom_formulas.py).

The matrix has N = 13,459, D = 50,000, 1,563,584 nonzeros and s = 8,776. The QROM side assumes 1,024 dirty qubits and packed tables.

| | QROM (dense encoding + sequential packets) | QOS (empirical F61) |
|---|---|---|
| Cost per A/s query | 1.3×10⁵ Toffoli | not computed separately |
| Cost per A/2 query (same D.23 amplification for both) | 3.3×10¹⁰ Toffoli | 6×10²⁴ – 1×10²⁵ samples |
| Quantum memory | ≈ 1,080 qubits | ≈ 52 logical qubits (A9 accounting) |
| Is the data stored? | Yes: a 9–13 MB table, read once per query | Not in the machine, but the source must be re-streamed ≈ 4×10¹⁸ times per query, plus the 8.2 MB featurizer |

**What this means.** QOS saves the 9–13 MB table only inside the quantum machine. To do so it draws about 10²⁵ samples per query from a source that must itself be stored. Neither column includes the solver's query count. The packed QROM figures are lower bounds.

## 5. Classical streaming: k* and learner vs featurizer state

**Label: measured (validation only; the test split was not read).** Sources:
- [scoped run report](classical_streaming/scoped_20261020T154320Z-d1fc3382/report.md), with [kstar.csv](classical_streaming/scoped_20261020T154320Z-d1fc3382/kstar.csv), [retention.csv](classical_streaming/scoped_20261020T154320Z-d1fc3382/retention.csv), [state.csv](classical_streaming/scoped_20261020T154320Z-d1fc3382/state.csv) and [samples_to_target.csv](classical_streaming/scoped_20261020T154320Z-d1fc3382/samples_to_target.csv);
- [pilot report](classical_streaming/20261019T202457Z-4a8cdd1c/report.md);
- [protocol](../experiments/cfpb_streaming_state/PROTOCOL.md) and [allocation fix](../experiments/cfpb_streaming_state/ALLOCATION_FIX.md).

This asks how much state a streaming classical learner needs to match frozen full Ridge (α = 1, unweighted, no intercept) on CFPB.

**Setup:**
- **Metric.** Joint retention = min(accuracy, balanced accuracy) relative to Ridge.
- **Grid.** 3 tasks × 2 featurizer tracks × 3 methods × 6 learner budgets (4–512 KiB) = 108 cells, in chronological order.
- **Tuning.** Each cell is tuned by successive halving on seed 2026, then evaluated on seeds 42/123/456 through 5 passes.
- **k\*.** The smallest budget at which the 5-pass retention reaches the threshold. CIs come from a 2,000-replicate bootstrap that resamples validation examples (paired across budgets) and evaluation seeds.
- **Verification.** All 108 cells were checked, and the k\* and retention tables were recomputed independently with 0 mismatches.

**k\* by task, featurizer track and method.** Values are KiB of learner state, with the 95% CI in brackets.

| Task | Track | Hashing + SGD k\*₉₅ / k\*₉₈ | Sparse JL + SGD k\*₉₅ / k\*₉₈ | AWM-Sketch k\*₉₅ / k\*₉₈ |
|---|---|---|---|---|
| credit_vs_checking | A (frozen TF-IDF) | 16 [16, 32] / 64 [32, 128] | 16 [16, 32] / 64 [32, 128] | 16 [16, 16] / 32 [16, 128] |
| credit_vs_checking | B (causal) | 16 [16, 32] / 64 [32, 512] | 16 [16, 32] / 128 [32, 512] | 16 [16, 32] / 64 [32, 512] |
| checking_vs_mortgage | A | 16 [16, 16] / 16 [16, 64] | 16 [16, 16] / 32 [16, 64] | 16 [4, 16] / 32 [16, 32] |
| checking_vs_mortgage | B | 4 [4, 16] / 32 [16, 64] | 16 [4, 16] / 16 [16, 32] | 16 [4, 16] / 32 [16, 128] |
| mortgage_vs_rest | A | 16 [16, 32] / 32 [32, 128] | 16 [16, 32] / 32 [32, 128] | 16 [16, 16] / 32 [16, 64] |
| mortgage_vs_rest | B | 16 [16, 32] / 32 [16, >512] | 16 [16, 32] / 64 [32, 128] | 16 [16, 32] / 64 [32, >512] |

**Learner state vs featurizer state.**

| Quantity | Track A (frozen TF-IDF) | Track B (causal) |
|---|--:|--:|
| Learner state at k\*₉₅ = 16 KiB (allocated, measured) | 15.8–15.9 KB (≈ 1,980 64-bit registers) | 15.8–15.9 KB |
| Featurizer state (vocabulary/IDF or tokenizer) | 8,219,927 B (8.2 MB) | 47,939 B (47.9 KB) |
| Total persistent state at k\*₉₅ | ≈ 8.24 MB | ≈ 63.8 KB |
| Full Ridge coefficients, for comparison (pilot) | 400 KB payload | — |

**Results:**
- **95% needs 16 KiB.** It is reached at 16 KiB in 17 of 18 groups, and at 4 KiB in one.
- **98% needs 16–128 KiB.** Two track-B mortgage groups have CI upper bounds above 512 KiB.
- **Few passes are needed.** 91 of 108 cells reach 95% after a single pass, and 44 of 108 reach 98%.
- **4 KiB is too small.** At 4 KiB, 17 of 18 cells stay below 95% even after 5 passes.
- **The optimizer is not the limit.** A separate full-dimensional SGD control (12 configurations; local run `convergence_20261020T142038Z-c609c8bb`, not committed) reached ≥ 99.3% by 5 passes. So the small-budget shortfalls come from memory or representation, not optimization.
- **On track A, the featurizer dominates.** The frozen TF-IDF featurizer is about 500× the learner's state, and it was fitted on all January rows, so it is not single-pass. That cost would equally apply to any learner fed this representation, a QOS sample stream included.
- **Track B avoids it.** The causal featurizer reaches the same 95% target with about 64 KB in total.

**Caveats:**
- **Optimistic.** Selection, tuning and k\* all use the same validation set. Confirmation on test is a separate, unrun step.
- **Limited scope.** Chronological order only, three tasks, and MISSION excluded (reason recorded in the run manifest).
- **Python allocation inventories.** These are not lower bounds on the state needed.
- **No method ranking.** The data does not support one.

## What remains untested

- **End-to-end exponents.** κ_reg, γ and δ have not been measured end to end. The γ⁻³ vs γ⁻⁴ and log(1/δ) vs δ⁻¹ hypotheses are unresolved.
- **Reusable shadow readout.** It is untested, and so is its m-dependence.
- **Full solver composition.** It has only been calibrated at N = 4.
- **The CFPB toy and the element-oracle exponent.** Neither was run or estimated.
- **Beyond small sizes.** No exponents exist for N > 128, s > 32, or non-circulant sparsity.
- **Beyond the scoped classical grid.** Classical results are validation-only on 3 CFPB tasks. There is no test-set evaluation, no random-order or MISSION cells, and no Bank or PaySim streaming runs.

Details are in [docs/decisions/phase_b_cancelled.md](../docs/decisions/phase_b_cancelled.md).
