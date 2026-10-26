# Handoff

Date: 2026-10-26. Findings tag: `v1-qos-findings`. Binding policy: [AGENTS.md](../AGENTS.md).

## Goal

This project evaluates Quantum Oracle Sketching (QOS; Zhao et al., arXiv:2604.07639v2, Theorem F.13) on financial data. It asks whether the theorem's preconditions hold, what its resource formulas imply, and how QOS compares with classical streaming and sketching. It is a scientific evaluation, not an attempt to show quantum advantage. Negative results count.

## Conventions

- **Theorem conventions**, not sklearn defaults: unweighted ridge, no intercept, training matrix scaled to ‖X‖ = 1, labels ±1, λ = α/σ_max², α ∈ {0.1, 1, 10, 100}, δ = 0.1. α is selected on validation, and ties go to the larger α.
- **Sparsity:** s = max nonzeros per row **or** column (F.13), counted over nonzero rows and columns (p.143). Bank's legacy accounting is row-only (A9 machine size 41); the two are reported side by side.
- **Result labels:** measured / simulated / calculated / estimated, defined in [FINAL_SUMMARY.md](../results/FINAL_SUMMARY.md). Nothing here is measured quantum hardware usage.
- **Seeds:** 42 by default. CV is stratified. Temporal data uses chronological splits.
- **Records:** [results/experiment_log.csv](../results/experiment_log.csv) is an append-only ledger, written through `experiments/cfpb_streaming_state/analyze.append_exactly_once`. Run directories are immutable, and every run writes a `manifest.json` with code and data hashes.
- **Git workflow:** commit at the end of each step and never push. After changes to `src/`, run the `scientific-regression-check` skill (`.agents/skills/`). Ask before changing any metric, split, resource accounting or interpretation.

## Findings to date (details and sources in [FINAL_SUMMARY.md](../results/FINAL_SUMMARY.md))

1. **The sparsity condition fails on CFPB text and Bank Marketing.** The binding sparsity is column sparsity, and it grows with N: s/N = 0.58–0.71 on CFPB and 1.0 on Bank and PaySim. log10 F61 comes out at 37–45 (calculated).
2. **The measured block-encoding exponent is s^3.1–3.3**, against s⁵ in the proof. This was simulated at N ≤ 128 and s ≤ 32. Even so, the extrapolated cost is 10²³–10²⁷ samples per A/2 query (estimated).
3. **QROM vs QOS:** about 10⁵ Toffoli per A/s query with a stored 9–13 MB table, against about 10²⁵ samples per QOS query (estimated).
4. **Classical streaming learners reach 95% of full Ridge with 16–32 KiB of learner state** on CFPB (measured on validation and confirmed on the test split). The 8.2 MB TF-IDF featurizer dominates on track A.
5. **Phase (b), the end-to-end QOS simulation, was cancelled** ([decision](decisions/phase_b_cancelled.md)). κ, γ and δ keep their proof values, and the untested items are listed there.
6. **Public-data sparsity screen (2026-10-26): no passing case** ([README](../results/screen/README.md), [protocol](../experiments/sparsity_screen/PROTOCOL.md), [plan](superpowers/plans/2026-10-26-qos-sparsity-screen.md)). Calculated: s_col ≥ N/D, so any fixed-D encoding fails by construction. Measured: on AMLworld, hub source accounts make s grow with N; on PaySim, a coarse type × amount × hour cell sets s. Cold-start rates are 0.68 (AMLworld) and 0.75 (PaySim). Elliptic has no cross-step edges (cold start 1.0). The N ≤ 80k prefixes span only minutes to hours of data. Code: `src/sparsity_screen.py`, `src/screen_datasets.py`, `experiments/sparsity_screen/`.

## Where things live

| Path | Contents |
|---|---|
| `src/` | Reusable code: `qos_audit.py` (step-1 audit, F60/F61), `machine_size.py`, `sketches.py`, `evaluation.py`, the CFPB pipeline (`cfpb*.py`, `text_preprocessing.py`) |
| `experiments/qos_tightness/` | Tightness study: phase (a) simulation, N-dependence, the phase (b) gates |
| `experiments/cfpb_streaming_state/` | Classical streaming-state benchmark (`scoped.run_cell`, budgeted learners) |
| `notebooks/` | 05 is the Bank validated pipeline (regression reference); 06 and 07 are the audit notebooks; 08 is a demo |
| `results/` | `FINAL_SUMMARY.md`, `audit/`, `tightness/`, `classical_streaming/`, the ledger |
| `docs/` | Audit, F61, QROM and CFPB write-ups; `decisions/` holds the gate and cancellation records |
| `data/` | Registry (`manifest.csv`, `README.md`, `download/`). Raw data is local and gitignored. |
| `reference/quantum-oracle-sketching/` | Reference implementation at `a3fe8d6` (gitignored) |
| `tests/` | Unit tests; `test_regression.py` pins the Bank reference values |
