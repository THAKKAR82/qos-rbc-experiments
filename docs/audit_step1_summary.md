# QOS applicability audit, step 1: cross-dataset summary

Date: 2026-10-16. Paper: Zhao et al., arXiv:2604.07639v2 (1 Oct 2026), Theorem F.13 / Eqs. F48–F50, F60–F62.
Machine-readable table: [`results/audit/summary_step1.csv`](../results/audit/summary_step1.csv), built by [notebook 07](../notebooks/07_bank_qos_audit_step1.ipynb).

This is a **calculated / theoretical accounting** study. It is not a quantum simulation, not a hardware resource estimate, and not evidence for or against quantum advantage. F61 values are log10(M/(RN)) polynomial prefactors with the constants, the polylogs hidden in Õ, R (taken as 1) and b dropped. Read them as orders of magnitude only.

## Measured rows

Each range runs across the tasks in a family; the κ_reg and γ p10 columns also run across α ∈ {0.1, 1, 10, 100}.

**Headline: F61 at the α selected by validation accuracy.**
- CFPB selects one α per task on the validation split.
- Bank selects one α for all folds by mean CV-fold accuracy, on fold validation rows.
- Ties go to the larger α.
- The s = 1 floor keeps every other measured factor at the same α.

| Dataset / family | Tasks | s_row_only | s_row_or_col | s/N | κ_reg (all α) | γ p10 (all α) | Selected α | **log10 F61 @ p10, selected α** | s = 1 floor (log10), selected α |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| CFPB pairwise | 6 | 919–1,254 | 2,916–5,992 | 0.58–0.71 | 1.6–53 | 0.008–0.11 | 1 (5 tasks), 10 (1) | **36.7–40.7** | 11.1–14.7 |
| CFPB one-vs-rest | 4 | 1,254 | 8,776 | 0.65 | 2.1–59 | 0.007–0.13 | 1 | **40.9–42.9** | 13.3–15.3 |
| Bank (5 CV folds) | 5 | 16 | 36,168–36,169 | 1.00 | 41–1,289 | 0.13–0.15 | 100 | **44.8–44.9** | 12.2–12.3 |

**Secondary: the cost-minimizing α is a grid-edge value.** F61 decreases monotonically with α over this grid, so its minimum is α = 100 (the grid's upper edge) for every task. It is not an interior optimum, and a larger α would lower it further at some cost in accuracy.

| Dataset / family | log10 F61 @ p10, cost-min α = 100 (grid edge) | s = 1 floor (log10) |
|---|--:|--:|
| CFPB pairwise | 32.8–35.7 | 8.3–9.7 |
| CFPB one-vs-rest | 35.7–37.3 | 8.1–9.7 |
| Bank | 44.8–44.9 | 12.2–12.3 |

**Accuracy at the selected α** (unweighted ridge, no intercept, the model the theorem prices):
- CFPB pairwise: test accuracy 0.921–0.982, balanced 0.921–0.976.
- CFPB one-vs-rest: test accuracy 0.915–0.979, balanced 0.869–0.942.

**Bank's selection is effectively a tie.** Mean CV-fold accuracy is 0.89952–0.89963 across all four α, a spread of about 1×10⁻⁴. α = 100 wins by that margin and so coincides with the grid edge. Mean balanced accuracy is highest at α = 0.1 (0.631, against 0.625 at α = 100). There, κ_reg ≈ 1,289 and log10 F61 ≈ 51.0–51.1.

**Why κ_reg grows with N at a fixed sklearn α.**
- The theorem's λ is α/σ_max². With σ_min = 0, κ_reg = √((1+λ)/λ) ≈ σ_max/√α when λ ≪ 1.
- σ_max grows with N. Bank's σ_max is 407.5 for a fold's 36,169 training rows and 455.6 for all 45,211 rows. That ratio, 1.118, equals √(45,211/36,169), so σ_max grows like √N here and κ_reg does too at a fixed α.
- CFPB shows the same trend: σ_max is 11.9 at N = 4,904 and 18.7 at N = 13,459.

## Analytic row (not measured)

> ◇ **PaySim: analytic, not measured; representation not onboarded in this repo.** N = 6,362,620 data rows in `data/paysim.csv`. If the encoding contains any fully dense column (for example, the transaction amount), s_row_or_col = N, s/N = 1, and log10 s⁷ = 7·log10(6,362,620) ≈ **47.6**. No κ_reg, γ, α selection or accuracy exists for PaySim, and none is implied.

## Which condition fails, and by how much

Eq. F62 requires s, κ_reg, γ⁻¹ ≤ poly(log N, log D). The degree is unspecified, so (log2 N)² is used below only as an illustrative yardstick.

- **CFPB.** The **sparsity** condition fails. s_row_or_col is 2,916–8,776, which is 58–71% of N and 19–47× (log2 N)². κ_reg (5.4–19 at the selected α) and γ p10 (γ⁻¹ ≈ 8–150) are moderate, but even at s = 1 the remaining factors leave an F61 prefactor of about 10^11–10^15 at the selected α.
- **Bank.** **Sparsity** fails completely. The seven dense standardized numeric columns make s_row_or_col = N (36,169, which is 158× (log2 N)²), although rows hold only 16 nonzeros. **κ_reg** is also large (41–1,289), because rank 43 < D = 51 forces σ_min = 0 and λ = α/σ_max² is small. Margins are comfortable (γ p10 ≈ 0.13–0.15). The s = 1 floor is still about 10^12. These numbers come from the global encoding, which carries a known leakage caveat.
- **PaySim (analytic).** Any dense column makes **sparsity** fail at s = N = 6.4M, which is 12,456× (log2 N)². The s⁷ factor alone is about 10^47.6, before κ_reg or γ, which have not been measured.

## Provenance

| Item | Location |
|---|---|
| CFPB run (`20261016T161637Z-e70ae078`) | [`results/audit/cfpb_step1/`](../results/audit/cfpb_step1/) |
| Bank run (`20261016T175910Z-a543a93d`) | [`results/audit/bank_step1/`](../results/audit/bank_step1/) |
| CFPB modeling gate | [`docs/decisions/cfpb_modeling_gate.md`](decisions/cfpb_modeling_gate.md) (Recommendation B; Debt-collection caveats apply) |

Ledger rows cover both runs, the two F61 decomposition tables and this summary table. Bank's validated results and regression fixtures are unchanged by this audit.
