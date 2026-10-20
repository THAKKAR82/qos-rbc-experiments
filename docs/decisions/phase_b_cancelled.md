# Decision: QOS tightness study phase (b) cancelled

Date: 2026-10-20. Decided by: the project owner. The QOS investigation is closing.

## Decision

Phase (b) of the QOS tightness study is **cancelled**. This covers both candidate first increments that were awaiting approval:

- the reusable-shadow readout build; and
- the fixed-test end-to-end sweep.

No phase (b) sweep was run. The completed tiny calibration and optimization gates stay in the record as they are:

| Gate | Commits | Report |
|---|---|---|
| Tiny calibration | `a2f75a8` | [PHASE_B_CALIBRATION.md](../../experiments/qos_tightness/PHASE_B_CALIBRATION.md) |
| Optimization and fidelity audit | `58711ac` | [PHASE_B_OPTIMIZATION.md](../../experiments/qos_tightness/PHASE_B_OPTIMIZATION.md), [PHASE_B_FIDELITY_AUDIT.md](../../experiments/qos_tightness/PHASE_B_FIDELITY_AUDIT.md) |

## Reason

The project's conclusion is that QOS is not useful for the bank-held data studied here. That conclusion holds without phase (b). The audit estimated 2–4 engineering weeks to build a faithful end-to-end test, and the conclusion rests on four results that are already measured or calculated:

1. **Column sparsity.** On the frozen CFPB TF-IDF, s_row_or_col is 2,916–8,776, which is 58–71% of N. It is set by heavy columns, not rows. Bank's dense standardized columns give s = N. PaySim, analytically, gives s = N with any dense column. Sources: [audit summary](../audit_step1_summary.md), `results/audit/summary_step1.csv`.
2. **The measured sparsity exponent of the block encoding.** For Theorem D.23's A/2 block encoding, α_s ≈ 3.1–3.3 (raw) and 2.7–2.9 with polylogs divided out, at N = 64 and 128 with s ≤ 32. The proof's exponent is 5. Even the smaller measured exponent leaves the per-query sample cost astronomically large at CFPB's s.
   - Sources: [phase (a)](../../results/tightness/phase_a_20261019T145729Z-a1644ff3/) and [N-dependence check](../../experiments/qos_tightness/N_DEPENDENCE.md).
   - This is a simulation (expected-unitary QOS, reference convention), with the ideal circulant sparsity.
3. **Classical streaming results at kilobyte scale.**
   - At one pass, 85 of 96 untuned pilot configurations reach 95% of full Ridge on both accuracy and balanced accuracy, with 16–512 KiB of learner state.
   - At 16 KiB alone, 21 of 32 do.
   - Featurizer state is 8.2 MB for frozen TF-IDF, against 48 KB for causal features.
   - Source: [streaming pilot](../../results/classical_streaming/20261019T202457Z-4a8cdd1c/report.md).
4. **The QROM comparison.**
   - A QROM loader needs about 10⁵–10⁶ Toffoli gates per A/s block-encoding query for the 1.56 M-nonzero CFPB matrix in practical dirty-qubit regimes (dense encoding with sequential packets: 1.3×10⁵ with 1,024 dirty qubits, packed tables), while storing a 9–13 MB table.
   - QOS needs about 6×10²⁴–1×10²⁵ samples per A/2 query by extrapolation ([empirical F61](../empirical_f61.md)).
   - Source: [docs/qrom_vs_qos_estimate.md](../qrom_vs_qos_estimate.md), with numbers from [`results/audit/qrom_vs_qos/`](../../results/audit/qrom_vs_qos/). This is a back-of-envelope resource estimate.

None of this is a disproof of Theorem F.13 or a statement about other datasets. The conclusion is about applicability to these representations.

## What remains untested

- **κ_reg, γ and δ dependence end to end.** No measurement of the solver's query count against κ, the readout shots against γ, or the oracle-instantiation penalty against δ. The F61 factors κ⁴, γ⁻⁴ and δ⁻¹ remain proof values.
- **Reusable interferometric Clifford-shadow readout**, and with it the m-dependence of F61 (F83/F92). Only a fixed-test Hadamard measurement was calibrated, at N = 4.
- **Full solver composition** (inverse polynomial, fixed-point amplification, real-part LCU, selector) at any size beyond the N = 4 calibration, including the planned LCU query accounting.
- **The explicit-sampling (fresh-sample channel) model** against expected-unitary (refinement A). The γ⁻³ vs γ⁻⁴ and log(1/δ) vs 1/δ hypotheses are therefore **unresolved**.
- **The CFPB toy end-to-end runs** (refinement B: random-64 and top-64 column selections).
- **The element-oracle sparsity exponent.** Its crossings fall below the sample grid.
- **Exponents beyond small sizes** (N ≤ 128, s ≤ 32) and beyond uniform (circulant) sparsity. All exponent statements are finite-size, under the expected-unitary model.
