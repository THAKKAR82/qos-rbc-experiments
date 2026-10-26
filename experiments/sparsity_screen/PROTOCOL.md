# Sparsity screen: pre-registered protocol

Committed 2026-10-26, before any real data was read. Source: [the plan](../../docs/superpowers/plans/2026-10-26-qos-sparsity-screen.md), copied verbatim. Changes after this commit are dated amendments below; nothing above is edited.

## Key result: fixed-D encodings fail by construction (calculated)

Let X ∈ ℝ^{N×D} have N nonzero rows with mean row nonzero count r̄ ≥ 1, and let c_j be the nonzero count of column j. Then

  Σ_j c_j = nnz(X) = r̄ N  ⇒  max_j c_j ≥ r̄ N / D  ⇒  s ≥ s_col ≥ r̄ N / D ≥ N / D.

- **Fixed D means s grows at least linearly in N.** The log-log slope of s against N is 1, whatever the data. So the sparsity condition cannot hold for any fixed feature set, and re-measuring this per dataset adds nothing. This is why the screen keeps only one fixed-D illustration (`paysim_naive`).
- **Bounded s needs D ≥ r̄ N / s₀.** The feature count must grow linearly with N, which means entity-like or entity × time features.
- **The bound is necessary, not sufficient.**
  - On CFPB one-vs-rest (N = 13,459, D = 50,000, nnz = 1,563,584) it gives s_col ≥ 31, but the measured value is 8,776, driven by Zipf-heavy columns.
  - On Bank (N = 36,168 per fold, D = 51, r̄ = 16) it gives s_col ≥ 11,347. The measured value is 36,168, because some columns are dense.
- **Hypothesis this motivates, to be tested.** Entity × time features rarely recur in later rows. So chronological test rows tend to have no overlap with the training features (**cold start**), which gives γ = 0 for those rows, and the theorem does not apply to them.

## Pre-registered protocol

- **Selection criteria:** a candidate must be public; it must have its licence, terms or provenance recorded before use; it must be ≤ 2 GB on disk; it must have a binary label defensible without leakage; and it must have a time order. The cases are fixed to the list below. Nothing is added or dropped after results are seen. A case not run is reported with its reason.
- **N growth:** chronological prefix from a start offset. Train is the first N rows, validation the next ⌈0.15 N⌉ and test the next ⌈0.15 N⌉. The vocabulary and columns are fit on train only. Val/test columns outside the training support are dropped.
- **Grid:** N ∈ {10k, 20k, 40k, 80k}, truncated to what is available and recorded as `truncated`. R = 5 replicate start offsets, spaced evenly over the usable span (synthetic controls: 5 seeds).
- **Fit:** OLS of log s on log N over all (N, replicate) points. The 95% CI is a cluster bootstrap over replicates (2,000 resamples, seed 42). The t-interval is reported too. Fits are done for s_row_or_col (the verdict statistic), s_row, s_col and col-nnz p99.
- **Sparsity verdict (approved):** **PASS** if CI upper < 0.2. **FAIL** if CI lower > 0.5. Otherwise **INCONCLUSIVE**, which does not pass.
- **Cold-start rate:** the fraction of test rows with no nonzero in any column that has a nonzero in the training matrix. It is reported at every (N, replicate) for every case, and before κ/γ/F61. The gate is **cold-start ≤ 0.10, as the mean over replicates at the largest N**. *Proposed, and to be confirmed by the user at the step-1 report, before any real data is read.* Rationale: the audit's headline γ is the 10th percentile of test margins, so a cold-start rate above 0.10 forces γ_p10 = 0. The gate is necessary, not sufficient.
- **Downstream:** the audit (κ_reg, γ, F61) runs only for cases that pass sparsity, at the largest N, with α selected on validation (ties go to the larger α). k\* runs only for cases that pass sparsity **and** cold-start, at train N = 40k.

## Cases

| Case id | Source | Rows and binary task | Encoding | Role |
|---|---|---|---|---|
| `ctrl_bounded` | synthetic | planted y = sgn(Xw\* + noise) | n_total × 2 n_total, the sum of 16 seeded injective row→column maps | positive control: must PASS |
| `ctrl_dense_col` | synthetic | same | `ctrl_bounded` plus one all-ones column | negative control: must FAIL |
| `elliptic_structure` | Elliptic Bitcoin edge list and time steps (token-free mirror if one exists, else Kaggle `ellipticco/elliptic-data-set`) | — | structural check only: count edges whose endpoints are in different time steps | 0 edges: FAIL by the γ = 0 gate (under a step-aligned chronological split, every test row's neighbours are outside the training columns, so cold start = 1, calculated from the edge list), and the rest is skipped. > 0 edges: stop and report before building an Elliptic case. |
| `paysim_naive` | local `data/paysim.csv` (registered hash) | transactions, isFraud = +1 | fixed D = 10: type one-hot (5), log1p(amount), 4 balances, standardized on train | the one illustration of the N/D bound; expected FAIL |
| `paysim_time_local` | same | same | (nameOrig, 24-step window), (nameDest, 24-step window), (type, train-fitted amount decile, 1-step window); value 1, uncapped | screened |
| `aml_acct_window` | IBM AMLworld `HI-Small_Trans.csv` (token-free mirror if one exists, else Kaggle `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`, single file) | transactions, Is Laundering = +1 | (from_account, 1-day window), (to_account, 1-day window); value 1, uncapped | screened |

Removed from scope by the user: SEC XBRL, every other naive fixed-D encoding, and degree-capped variants (see the ledger ruling).

## Amendments

None.
