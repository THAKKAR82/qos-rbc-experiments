# Empirical F61: block-encoding sparsity exponent replaced by its measured value

Date: 2026-10-20. **Label: extrapolation from small-size simulation; orders of magnitude only.**
This is pure arithmetic on saved artifacts: nothing was re-solved. Generator: `scripts/empirical_f61.py`. Output: [`results/audit/empirical_f61/`](../results/audit/empirical_f61/), whose manifest records every input hash.

## What changes and what does not

F61's s⁷ factors as s⁵ × s²:
- **s⁵** is the per-query cost of Theorem D.23's block encoding. **This is the only factor replaced**, with the measured s^α.
- **s²** comes from the readout term Q² (F97/F104). It **keeps its proof value**.

The measured exponent comes from the D.23 A/2 block encoding (raw fit, ε* = 10⁻²):

| Fit | α |
|---|---|
| N = 64 | 3.147 (95% CI 3.127–3.167), the point value |
| N = 128 | 3.273 (95% CI 3.247–3.298), used to give a range |

**Not measured:** κ_reg⁴, γ⁻⁴, δ⁻¹, the min{s² log² D, log² m} term and the readout s². All keep proof values. Phase (b), which would have measured them, was cancelled ([decision](decisions/phase_b_cancelled.md)).

**Inputs** are the same as the step-1 summary: γ = p10, δ = 0.1.
- CFPB uses the validation-accuracy-selected Ridge α per task.
- Bank uses one α selected by mean CV-fold accuracy, an effective tie, from the global encoding (known leakage).

The "per-query samples" column is a different quantity from log10 F61 (which is log10(M/(RN)) with constants dropped). It is the measured M* of an A/2 block-encoding query at N = 64, s = 32, ε* = 10⁻² (median 1.28×10¹⁴), scaled by (s/32)^α × (padded dimension / 64). That dimension factor is the theorem's linear one, not a measurement. The ε* = 10⁻³ variant is 10.2× larger and is in the CSV.

## Per task (γ = p10, validation-selected α)

| Task | s | Ridge α | κ_reg | γ p10 | log10 F61, proof (s⁵) | log10 F61, empirical (s^3.15 … s^3.27) | Δ (decades) | per-query samples, ε=1e-2 (s^3.15 … s^3.27) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Debt vs Card | 5,714 | 10 | 5.4 | 0.056 | 37.4 | 30.4 … 30.9 | −7.0 … −6.5 | 1.6e+24 … 3.1e+24 |
| Debt vs Checking | 5,992 | 1 | 16.5 | 0.047 | 39.7 | 32.7 … 33.2 | −7.0 … −6.5 | 1.9e+24 … 3.6e+24 |
| Debt vs Mortgage | 3,601 | 1 | 15.9 | 0.055 | 37.8 | 31.2 … 31.7 | −6.6 … −6.1 | 3.8e+23 … 6.8e+23 |
| Card vs Checking | 5,175 | 1 | 14.6 | 0.018 | 40.7 | 33.8 … 34.3 | −6.9 … −6.4 | 1.2e+24 … 2.2e+24 |
| Card vs Mortgage | 2,916 | 1 | 11.9 | 0.053 | 36.7 | 30.3 … 30.7 | −6.4 … −6.0 | 1.9e+23 … 3.4e+23 |
| Checking vs Mortgage | 3,559 | 1 | 12.1 | 0.069 | 36.9 | 30.3 … 30.7 | −6.6 … −6.1 | 3.6e+23 … 6.5e+23 |
| Debt vs rest | 8,776 | 1 | 18.7 | 0.028 | 42.0 | 34.7 … 35.2 | −7.3 … −6.8 | 6.2e+24 … 1.3e+25 |
| Card vs rest | 8,776 | 1 | 18.7 | 0.017 | 42.9 | 35.6 … 36.1 | −7.3 … −6.8 | 6.2e+24 … 1.3e+25 |
| Checking vs rest | 8,776 | 1 | 18.7 | 0.026 | 42.2 | 34.9 … 35.4 | −7.3 … −6.8 | 6.2e+24 … 1.3e+25 |
| Mortgage vs rest | 8,776 | 1 | 18.7 | 0.054 | 40.9 | 33.6 … 34.1 | −7.3 … −6.8 | 6.2e+24 … 1.3e+25 |
| Bank fold 1 | 36,168 | 100 | 40.8 | 0.145 | 44.9 | 36.5 … 37.1 | −8.4 … −7.9 | 5.3e+26 … 1.3e+27 |
| Bank fold 2 | 36,169 | 100 | 40.8 | 0.148 | 44.9 | 36.4 … 37.0 | −8.4 … −7.9 | 5.3e+26 … 1.3e+27 |
| Bank fold 3 | 36,169 | 100 | 40.8 | 0.145 | 44.9 | 36.5 … 37.1 | −8.4 … −7.9 | 5.3e+26 … 1.3e+27 |
| Bank fold 4 | 36,169 | 100 | 40.8 | 0.155 | 44.8 | 36.4 … 36.9 | −8.4 … −7.9 | 5.3e+26 … 1.3e+27 |
| Bank fold 5 | 36,169 | 100 | 40.8 | 0.145 | 44.9 | 36.5 … 37.1 | −8.4 … −7.9 | 5.3e+26 … 1.3e+27 |

**How to read the table:**
- Replacing s⁵ with the measured s^~3.2 lowers F61 by 6–7 decades for CFPB and about 8 for Bank.
- It leaves totals of **10³⁰–10³⁶** passes for CFPB and **about 10³⁷** for Bank.
- The per-query cost is **10²³–10²⁵** samples (CFPB) and **10²⁶–10²⁷** (Bank). That is 10¹⁷–10¹⁹ and 10²¹ times the respective nonzero counts, per single block-encoding query.

## Caveats

- **Extrapolation range.** The exponent was fitted at N ≤ 128 and s ≤ 32, with uniform circulant sparsity, under the expected-unitary model. Here it is applied at s = 2,916–36,169, which is 2–3 decades beyond the fitted range.
- **Not measured.** κ, γ, δ and the readout s² keep their proof values.
- **Scope.** This is not a resource estimate, not a measurement of hardware usage, and not evidence about the asymptotic tightness of Theorem F.13.
