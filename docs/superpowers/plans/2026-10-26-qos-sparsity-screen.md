# QOS Sparsity-Condition Screen Implementation Plan (reduced scope, approved 2026-10-26)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Screen a few public financial transaction datasets, under bounded (entity × time-window) encodings, for Theorem F.13's sparsity condition. For each one, measure the cold-start rate. For cases that pass both gates only, measure κ_reg, γ, F61 (proof and empirical exponents) and classical streaming k\* on the same encoding.

**Architecture:** This is a regime search, not an advantage demonstration. Each dataset × encoding is a *case*. A case builds chronological train/val/test matrices at each grid N. The screener fits log s against log N across replicate start offsets, applies the pre-registered verdict rule, and reports the cold-start rate. Passing cases go to the existing `qos_audit.audit_binary`, which is unchanged, and to the existing streaming `scoped.run_cell` through a CSR-to-row adapter. Every case, failures included, gets a row in one summary table.

**Tech Stack:** Python 3.13 `.venv`, NumPy, SciPy sparse, pandas, scikit-learn. No new dependencies.

**Spec:** the user's task message and approval message of 2026-10-26 (session), plus `AGENTS.md`, `results/FINAL_SUMMARY.md` and `docs/decisions/phase_b_cancelled.md`.

## Global Constraints

- Conventions are unchanged: unweighted ridge, no intercept, ‖X‖ = 1, labels ±1, λ = α/σ_max², α ∈ {0.1, 1, 10, 100}, δ = 0.1, seed 42.
- s is `max_row_or_col_sparsity` on the **training** matrix, over nonzero rows and columns (p.143 convention, as in step 1). Fits use N = N_nonzero.
- **`src/qos_audit.py` is not modified.** If a case has D ≤ N and D > 2,000, the audit is reported as "audit not run" (decision 3a).
- The ledger is append-only, through `experiments/cfpb_streaming_state/analyze.append_exactly_once`. Never edit historical rows.
- Outputs go to `results/screen/<run_id>/`, which is immutable: refuse a non-empty directory.
- Raw data goes in `data/raw/screen/<dataset>/` (gitignored). Register each new source in `data/manifest.csv` and `data/download/README.md` with its SHA-256.
- **Kaggle credentials:** first look for a token-free public source. If a token is needed, the user places it in `~/.kaggle/kaggle.json`. Never read it, copy it, print it or commit it.
- Commit at the end of each task, on branch `qos-sparsity-screen`. **Never push.** Run the `scientific-regression-check` skill after any task that touches `src/`.
- Labels: the N/D bound is **calculated**. Sparsity slopes and cold-start rates are **measured**. F61 is **calculated**. Empirical F61 is **estimated (extrapolation)**. k\* is **measured, validation only**. Test splits feed cold-start and γ only.
- Never densify a matrix with D > 2,000.

## Key result: fixed-D encodings fail by construction (calculated)

Let X ∈ ℝ^{N×D} have N nonzero rows with mean row nonzero count r̄ ≥ 1, and let c_j be the nonzero count of column j. Then

  Σ_j c_j = nnz(X) = r̄ N  ⇒  max_j c_j ≥ r̄ N / D  ⇒  s ≥ s_col ≥ r̄ N / D ≥ N / D.

- **Fixed D means s grows at least linearly in N.** The log-log slope of s against N is 1, whatever the data. So the sparsity condition cannot hold for any fixed feature set, and re-measuring this per dataset adds nothing. This is why the screen keeps only one fixed-D illustration (`paysim_naive`).
- **Bounded s needs D ≥ r̄ N / s₀.** The feature count must grow linearly with N, which means entity-like or entity × time features.
- **The bound is necessary, not sufficient.**
  - On CFPB one-vs-rest (N = 13,459, D = 50,000, nnz = 1,563,584) it gives s_col ≥ 31, but the measured value is 8,776, driven by Zipf-heavy columns.
  - On Bank (N = 36,168 per fold, D = 51, r̄ = 16) it gives s_col ≥ 11,347. The measured value is 36,168, because some columns are dense.
- **Hypothesis this motivates, to be tested.** Entity × time features rarely recur in later rows. So chronological test rows tend to have no overlap with the training features (**cold start**), which gives γ = 0 for those rows, and the theorem does not apply to them.

## Pre-registered protocol (copied into `experiments/sparsity_screen/PROTOCOL.md` and committed before any real data is read)

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

## Review Focus

1. **Test rows empty on the training support:** the cold-start rate must count them, and the audit's γ = 0 must become the reason "theorem inapplicable", never a crash or NaN. (Task 1 tests `cold_start_rate`; Task 4 tests the reason.)
2. **Duplicate (row, column) pairs** (an account appearing twice in a window): nnz must count each pair once, with value 1. (Task 3.)
3. **A prefix shorter than the requested N:** the grid is truncated per replicate and recorded, never silently reused. (Task 1.)
4. **A degenerate slope fit** (identical s at every N): the CI is exactly (0, 0) and the verdict is PASS, without warnings. (Task 1.)
5. **A single-class training or validation prefix** (AML ≈ 0.1% positives): balanced accuracy is skipped with a recorded reason, and sparsity and cold-start are still measured. (Task 4.)

---

### Task 1: Screener core, synthetic controls, protocol (spec step 1; stop and report after it)

**Files:**
- Create: `src/sparsity_screen.py`
- Create: `experiments/sparsity_screen/PROTOCOL.md` (the key result and pre-registered protocol above, copied verbatim)
- Test: `tests/test_sparsity_screen.py`

**Interfaces:**
- Produces:
  - `@dataclass Encoded`: `X_train, y_train, X_val, y_val, X_test, y_test` (CSR float64, ±1 int labels) and `meta: dict` (at least `truncated: bool`).
  - `chronological_split(n_total: int, start: int, n_train: int, val_frac: float = .15, test_frac: float = .15) -> tuple[slice, slice, slice, bool]`. When `start + n_train·1.3 > n_total`, it shrinks n_train so that everything fits, and returns `truncated=True`.
  - `sparsity_point(X) -> dict` with `N_nonzero, D_nonzero, nnz, s_row, s_col, s_row_or_col, col_nnz_p99`.
  - `cold_start_rate(X_train, X_test) -> float`.
  - `fit_loglog_slope(N, s, replicate, *, n_boot: int = 2000, seed: int = 42) -> dict` with `slope, intercept, ci_boot, ci_t, n_points, n_replicates`.
  - `classify_slope(ci_lo, ci_hi, *, pass_below=.2, fail_above=.5) -> str`.
  - `bounded_control(n_total: int, s: int = 16, d_ratio: int = 2, seed: int = 42) -> tuple[csr_matrix, np.ndarray]`, `dense_column_control(...)` with the same signature, and `control_case(kind: str, s: int = 16, max_rows: int | None = None) -> Callable[[int, int], Encoded]`.
  - `screen_case(build, n_grid, replicates: int = 5) -> pd.DataFrame`, with one row per (N_requested, replicate): `N_requested, replicate, truncated, cold_start_rate` and the `sparsity_point` keys of X_train.

- [ ] **Step 1: Write the failing tests** for: control boundedness at each grid N; dense column s_col = N; slope recovery (1, and an exact 0 with CI (0, 0)); the inconclusive band; `cold_start_rate` on hand-built matrices (0, 0.5, 1); controls end to end (PASS and FAIL, cold start ≈ 0); truncation recorded.
- [ ] **Step 2:** `.venv/bin/python -m pytest -q tests/test_sparsity_screen.py`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement.** Each injective map is a seeded permutation of `range(d_ratio·n_total)` truncated to n_total. Duplicates are summed and then set to 1. `w*` is drawn from `default_rng([seed, 1])`, and labels are sgn(Xw\* + 0.1·noise), with ties going to +1. A replicate r uses seed `[seed, r]`. If every log s is identical, the slope is exactly 0 with CI (0, 0).
- [ ] **Step 4:** Rerun. Expected: PASS.
- [ ] **Step 5:** Run the full suite and `tests/test_regression.py`, then the `scientific-regression-check` skill. These are new modules, nothing imports them, and existing code is unchanged, so the Bank rerun should be skippable, with the evidence stated.
- [ ] **Step 6: Commit** `feat(screen): sparsity-scaling screener, synthetic controls, pre-registered protocol`. **Stop and report** the control results and ask the user to confirm the cold-start gate.

### Task 2: Elliptic structural check

**Files:** `experiments/sparsity_screen/elliptic_structure.py`, `experiments/sparsity_screen/acquire.md` (sources, licence and terms seen, commands, hashes), plus `data/manifest.csv` and `data/download/README.md` rows.

- [ ] Search for a token-free source with documented provenance. If there is none, ask the user to place the Kaggle token. Download only the edge list and the time-step column (read the features CSV's first two columns, without keeping the rest).
- [ ] Test (tiny fixture with 2 cross-step edges): `cross_step_edges(edges, step) == 2`.
- [ ] Run it and write `results/screen/elliptic_structure_<run_id>/structure.json` (edges total, cross-step count, steps, labeled counts). If the count is 0, add a FAIL row for the summary with the reason `"no cross-step edges: cold start = 1 under a step-aligned chronological split (calculated)"`. If it is > 0, stop and report. Commit.

### Task 3: PaySim and AMLworld loaders and encoders; AML acquisition

**Files:** `src/screen_datasets.py`; `tests/test_screen_datasets.py` with fixtures in `tests/fixtures/screen/` (≤ 50 invented rows each); updates to the manifest and README.

- Produces: `load_paysim(path) -> pd.DataFrame` and `load_amlworld(path) -> pd.DataFrame` (time-sorted, needed columns only, labels ±1); `keyed_csr(row_ids, keys, n_rows, vocabulary=None) -> (csr, vocab)` (duplicates go to 1; unseen keys are dropped when a vocabulary is given); and `CASES: dict[str, Callable[[int, int], Encoded]]` with exactly `paysim_naive`, `paysim_time_local` and `aml_acct_window`.
- Tests: the fixture loaders give the expected order and labels; duplicates count once (Review Focus 2); columns come from train only; `paysim_naive` has D = 10 and s_col = N_train; the time-local cases' columns are (entity, window) keys.
- Acquire AML as in Task 2, and verify the PaySim hash. Run the regression-check skill, then commit.

### Task 4: Screen run, cold start, gated audit, summary, ledger

**Files:** `experiments/sparsity_screen/run.py` (`scale`, `audit`, `summarize`), `experiments/sparsity_screen/tests/test_run.py`.

- Outputs in `results/screen/<run_id>/`:
  - `manifest.json`, `scaling.csv` and `slopes.csv`;
  - `audit/<case>.json`, for PASS cases only;
  - `summary.csv` and `summary.md`, with the columns `case, dataset, encoding, N_max, D_train, s_row_or_col@N_max, NoverD_bound@N_max, slope, ci_lo, ci_hi, sparsity_verdict, cold_start@N_max, cold_start_gate, reason, kappa_reg_sel, gamma_p10, log10_F61_proof, log10_F61_emp_lo, log10_F61_emp_hi, ridge_val_acc, ridge_val_bal_acc, full_model_bytes, kstar95_kib, kstar98_kib`. The cold-start columns come before κ/γ/F61. The k\* columns stay blank until Task 5. The Elliptic row comes from Task 2.
  - `ledger_rows.csv`.
- Tests on the controls: the summary verdicts are PASS and FAIL. A synthetic case that is entirely cold-start gets `reason == "theorem inapplicable: gamma_p10 = 0"` and no NaN (Review Focus 1). A single-class prefix gets a reason containing "single class" (Review Focus 5). Rerunning into an existing directory raises `FileExistsError`.
- Real run, then inspect `summary.md`, then append to the ledger once, then commit.

### Task 5: Streaming k\* (only for cases passing sparsity and cold start)

**Files:** `experiments/sparsity_screen/kstar.py`, `experiments/sparsity_screen/tests/test_kstar.py`.

- `csr_to_rows(X, y) -> list[(uint64[n, 2] keys, float64 values, int y)]`, with the column index in `keys[:, 0]` as in CFPB track A. Tests: a CSR round trip, and one 16 KiB `hash_sgd` cell on `ctrl_bounded` at N = 2,000 that stays within budget.
- First pilot one cell and re-estimate. Then run 3 methods × 6 budgets at train N = 40k. Write `kstar.csv` and a new `summary_with_kstar.csv`, append ledger rows and commit. If no case passes both gates, record "k\* not run: no case passed both gates" in the summary.

### Task 6: Write-up

- `results/screen/README.md`: the key-result derivation, the summary table with labels, negatives included.
- Append a short section to `results/FINAL_SUMMARY.md` with the N/D derivation and the screen's verdicts. The v1 content stays unchanged above it; it is a dated addendum.
- Update `docs/HANDOFF.md`. Commit, then run the whole-branch review.

## Estimates (reduced scope)

| Task | Agent time | Compute | Disk |
|---|---|---|---|
| 1 controls | 45–60 min | < 2 min | — |
| 2 Elliptic check | 20–30 min, plus a possible token wait | < 1 min, plus a ~5–10 min download | ≤ 0.7 GB (less if a mirror carries only the edge list and classes) |
| 3 loaders, AML acquisition | 1–1.5 h | ~5 min download | ≈ 0.5 GB (PaySim is local) |
| 4 screen, audit, summary | 1–1.5 h | ≈ 15–20 min (3 real cases × 4 N × 5 replicates, plus ≤ 3 audits) | < 0.2 GB |
| 5 k\* | 30–45 min if anything passes | 1.5–3 h per case passing both gates (likely 0–1); pilot first | < 0.1 GB |
| 6 write-up and review | 1–1.5 h | — | — |

**Total:** about 4.5–6 h of agent time. Compute is about 30 min, plus 0–6 h for k\*. Disk is about 1.2 GB.
