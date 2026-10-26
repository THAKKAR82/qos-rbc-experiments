# Public-data sparsity screen (2026-10-26)

**Question.** Does any public financial transaction dataset, under an encoding built for the purpose, satisfy Theorem F.13's sparsity condition? The condition is that s = max nonzeros per row or column stays bounded as N grows. This is a regime search, not an attempt to show an advantage. The selection criteria, thresholds and cases were fixed in [PROTOCOL.md](../../experiments/sparsity_screen/PROTOCOL.md) before any real data was read.

**Answer. No.** No public case passes. One fails only the cold-start gate and is inconclusive on sparsity; the others fail outright. k\* was not run, because no case passed both the sparsity and cold-start gates.

## Key result: fixed-D encodings fail by construction (calculated)

For N nonzero rows with mean row nonzero count r̄, the column counts sum to nnz = r̄N. So max column nnz ≥ r̄N/D ≥ N/D. A fixed feature set therefore gives s growing linearly in N, whatever the data. Bounded s needs D ≥ r̄N/s₀: the feature count must grow with N (entity or entity × time features). The bound is necessary, not sufficient. `paysim_naive` is the one illustration kept: s = N, against the bound r̄N/D = 48,000 at N = 80,000.

## Summary

Sources: [screen_20261026T142018Z/summary.csv](screen_20261026T142018Z/summary.csv) (with `scaling.csv` and `slopes.csv`), [controls_20261026T140837Z/](controls_20261026T140837Z/) and [elliptic_structure_20261026T141107Z/](elliptic_structure_20261026T141107Z/). Slopes are OLS fits of log s on log N, for N = 10k–80k with 5 replicate start offsets; the CIs are 95% cluster bootstraps over replicates. Cold start is the share of test rows with no feature in common with training, so their margin γ = 0. The cold-start gate is ≤ 0.10, as the mean over replicates at N = 80k.

| Case | Label | s at N = 80k | Slope [95% CI] | Sparsity | Cold start | Gate | Outcome |
|---|---|--:|---|---|--:|---|---|
| `paysim_naive` (fixed D = 10) | measured | 80,000 | 1.00 [1.00, 1.00] | FAIL | 0.00 | PASS | s = N, as the N/D bound requires |
| `paysim_time_local` (entity × window) | measured | 709–3,276 | 0.63 [0.40, 0.91] | INCONCLUSIVE | 0.754 | FAIL | not passing |
| `aml_acct_window` (account × day) | measured | 629–3,216 | 1.09 [0.98, 1.30] | FAIL | 0.680 | FAIL | not passing |
| `elliptic_structure` (graph only) | measured | — | not measured | skipped | 1.000 | FAIL | 0 of 234,355 edges cross time steps |
| `ctrl_bounded` | synthetic control | 16 | 0 [0, 0] | PASS | 0.00 | PASS | screener check passed |
| `ctrl_dense_col` | synthetic control | 80,000 | 1 [1, 1] | FAIL | 0.00 | PASS | screener check failed, as it should |
| `ctrl_bounded_margin` | synthetic best case | 16 | 0 [0, 0] | PASS | 0.00 | PASS | reference point only |

**Synthetic reference points** (audit at N = 80k, α selected on validation; costs calculated, constants and polylogs dropped):

| Control | α | κ_reg | γ_p10 | γ_p10 bound | Validation accuracy | log10 F61, proof | log10 F61, empirical |
|---|--:|--:|--:|--:|--:|--:|--:|
| `ctrl_bounded` | 10 | 3.45 | 0.0011 | 0.102 | 0.651 | 25.7 | 23.4–23.6 |
| `ctrl_bounded_margin` | 100 | 1.69 | 0.042 | 0.115 | 1.000 | 18.1 | 15.9–16.0 |

These are synthetic and say nothing about real data. Even the best case, with s = 16, near-perfect separability and κ ≈ 1.7, gives F61 ≈ 10¹⁶–10¹⁸ passes over the data. That comes from 16⁷ ≈ 10⁸·⁴ times γ⁻⁴ ≈ 10⁵·⁵.

## Findings

- **Hubs set s.** In both entity × window encodings, the typical column is nearly bounded: the col-nnz p99 slope is 0.10 [−0.20, 0.38] for PaySim and 0.19 [0.13, 0.28] for AML. The maximum column still grows roughly linearly with N. The theorem's s is that maximum, and it comes from a few hub entities (a merchant or account with thousands of transactions in one window). The same kind of heavy-column effect set s on CFPB.
- **Cold start is the binding problem for bounded-type features.** Entity × window columns rarely recur in later rows.
  - On AML, 61–78% of test rows have no training feature at N = 80k.
  - On PaySim, the rate is about 0 when the test slice stays inside the training slice's 24-step window, and about 1 when it crosses into a new window. The per-replicate values at 80k are 0.00, 0.80, 0.97, 1.00 and 1.00. So the cold-start rate mostly reflects window alignment.
  - Elliptic has no cross-step edges, so every later time step is cold start (measured 1.000 on the standard 34/15 split).
- **Margin ceiling (calculated).** With test entries |x| ≤ 1, Σᵢ γᵢ² ≤ ‖X′‖² ≤ s′_row·s′_col. Hence γ_p10 ≤ √(s′_row s′_col / m_served). For column-bounded test data, the served margin can only shrink like 1/√m, and γ⁻⁴ grows like m². The controls respect this (0.0011 ≤ 0.10 and 0.042 ≤ 0.12).

## Design limits and caveats

- **PaySim windows and the grid.** At N ≤ 80k, a PaySim prefix spans fewer steps than one 24-step window. So the screen cannot show whether s saturates once the stream exceeds the window. The inconclusive slope partly reflects this. The protocol was not changed after the results were seen; a longer grid or a shorter window would be a new, pre-registered run.
- **Coverage.** There is one encoding per real dataset, chronological prefixes only, and 5 start offsets. The CIs reflect variation between start offsets only.
- **Provenance.**
  - AMLworld comes from a Hugging Face mirror whose bytes are identical in three independent uploads. It has not been compared with the Kaggle original.
  - The Elliptic licence text was not verifiable.
  - The PaySim licence is unverified, as before.
- **What was not run.** k\* and the audit on real data (no case passed sparsity), capped encodings (they pass by construction), SEC XBRL (dropped by the user), and naive fixed-D encodings other than PaySim's.
