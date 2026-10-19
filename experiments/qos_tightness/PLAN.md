# QOS tightness study, step 2: approved plan

**Goal.** Measure how QOS's sample requirement scales with sparsity s, regularized condition number κ, and test margin γ at small simulable sizes. Compare the measurements with the proof's dependence: Theorem F.13 / Eq. F61 and Eqs. F87–F104 of arXiv:2604.07639v2.

**Status of results.** Every result is a *simulation (expected-unitary QOS, reference convention)*. None is a hardware resource estimate, and none is evidence for or against quantum advantage.

## Conventions and conditions

- Code lives in `experiments/qos_tightness/` and runs only under `.venv-qos-ref`, pinned in `requirements-lock.txt` against reference commit `a3fe8d6`. Nothing JAX-dependent goes in `src/` or the main env.
- Outputs go to `results/tightness/<phase>_<run_id>/` with a manifest. Ledger rows are append-only and labelled as above.
- Synthetic circulant support is the **ideal-sparsity case** (uniform row and column s). Real data's s is set by heavy columns.
- **Block encoding (primary path):** Lemma D.22 sketched A/s, then Theorem D.23 odd-polynomial amplification to A/2.
  - Comparison path, labelled "optimistic: reference convention": multiply by s classically.
- **Sign QSVT (primary path):** a verified-bounded odd Chebyshev erf polynomial, with degree scaled with s to reach the D.23 oracle precision ε/(s ln(1/ε)). Angles are verified against the polynomial to 1e-10.
  - a2′ (i): the reference pyqsp `PolySign` at degree 101, with its unbounded-polynomial / non-converged-angle failure recorded as found (see `repro/`).
  - a2′ (ii): our verified-bounded polynomial at fixed degree 101, which isolates the true fixed-degree floor.
- **QSVT evaluation:** the singular-value shortcut under Halmos completion, cross-checked against the reference's dense `apply_qsvt` at N ≤ 32.

## Phases and stop points

1. **Environment, calibration, D.23 validation and dense cross-check.** Stop and report.
   - The first validation failed as found (commit `eb6329a`). The sign path was fixed (`f4cc24c`) and re-validated.
2. **(a) Primitives** at N = 64:
   - element oracle (predicted s¹ at fixed N; ε¹);
   - index oracle (Lemma D.21: s³, ε¹);
   - block encoding A/2 (Theorem D.23: s⁵, ε¹);
   - the a2′ (i)/(ii) comparisons.
   - Run over 5 instance seeds (42, 123, 456, 789, 2026), using M at target ε* = 1e-2 **and** 1e-3 (to check β).
   - Report global fits with 95% bootstrap CIs, **plus local slopes between consecutive s**.
   - Stop and report before (b).
3. **(b) End to end:**
   - **Synthetic data:** sweep s, κ (via λ) and γ (constructed test points); expected-unitary.
   - **Hypothesis check (refinement A):** at least 3 points per γ and δ sweep re-run with **explicit sampling** (`qos_sampling.py`, fresh samples per shot) at the smallest feasible size. γ⁻³ vs γ⁻⁴ and log(1/δ) vs 1/δ count as supported only if both models agree.
   - **CFPB toy (refinement B):** two labelled column selections, (i) a seeded random 64 columns and (ii) the top 64 by training document frequency, each with its resulting s reported. This toy is a derived dataset object; register it in `data/manifest.csv` per the provenance convention.
   - **Query accounting (refinement D):** must include the **real-part extraction (LCU) step**. The reference implements each QSVT polynomial as the real part of the block, and the dense cross-check shows the imaginary part is nonzero (0.3–0.5). So every QSVT stage (sign, amplification, pseudo-inverse) needs the LCU combination of the circuit with its conjugate. This costs one ancilla, and its query cost must be counted explicitly (per the reference's claim, same samples, but the circuit is still applied twice). Not implemented yet.

## Approved N-dependence check before (b)

The 2026-10-19 handoff requests a primary-path repeat at N = 128,
s = {2, 4, 8, 16}, seeds = {42, 123, 456}, with the unchanged M grid
10^4 through 10^11 and targets ε* = 10^-2 and 10^-3. Reuse the phase-(a)
measurement and fit functions. Checkpoint each point and monitor process RSS
against the 4 GiB gate; wrap the run in `caffeinate -i`.

Report raw and polylog-divided α_s alongside the frozen N = 64 results,
flagging absolute changes greater than 0.3. Also recompute N = 64 fits from
the saved points restricted to the same s and seeds, to separate N dependence
from differences in the fitted sparsity range. Preserve the original bootstrap
convention and report local slopes. Existing results remain immutable.

Command (from `experiments/qos_tightness/`):
`caffeinate -i ../../.venv-qos-ref/bin/python -m qostight.phase_a run --n-check --workers 2 --append-ledger`.
Resume by adding `--resume results/tightness/<run_directory>`.
Commit the results, report, and **stop before phase (b)**.

Completed 2026-10-19: [N-dependence report](N_DEPENDENCE.md). All 96 points
completed; no estimable global exponent changes by more than 0.3 against
either the published or matched N = 64 baseline. Phase (b) has not started.

## Proof-derived exponents (comparison targets)

Phase-(b) plan/calibration gate completed 2026-10-19: see
[detailed plan](PHASE_B_PLAN.md) and [calibration/approval report](PHASE_B_CALIBRATION.md).
The tiny fixed-test pipeline passes; full sweeps remain unapproved and unready.
Conservative phase finding exceeds the three-hour budget in conditional forecasts;
verified explicit sampling, reusable Clifford shadows and nested accounting remain gates.

| Component | Source | Dependence |
|---|---|---|
| Sparse index oracle | Lemma D.21 | Õ(R N s³/ε) |
| Block encoding of A/2 | Theorem D.23 | Õ(R N s⁵/ε): s³ per oracle × s amplification degree × s precision split |
| Solver queries | F87 | Q₀ = O(κ² log²(κ/ε)) |
| Readout precision / queries | F94–F96, F90/F92 | ε = γ/(3√s); O(s·(min{s log D, log m} + log 1/δ)/γ²) |
| Total | F97–F104 | M = Õ(RN s⁷ κ⁴ γ⁻⁴ δ⁻¹ min{s² log² D, log² m}) |
