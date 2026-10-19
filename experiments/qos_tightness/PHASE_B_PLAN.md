# Phase (b): plan and calibration gate

2026-10-19. Base commit `aa252fa`; reference `a3fe8d6fe777750749f5e90131180e016088bdaf`.
This phase is authorized for inspection, planning and a tiny calibration prototype only.
Full sweeps require approval. No changes to `src/`, the main environment, Bank,
prior results or reference code are proposed.

## Verified theorem definitions

Source: [Zhao et al., version 2 PDF](https://arxiv.org/pdf/2604.07639v2),
printed pages 142–151, Theorem F.13 and Lemmas F.14–F.16.
The arXiv HTML uses global equation numbering; equation labels below refer to the PDF.

- F46–F50: unweighted ridge, no intercept, ||X||₂=1;
  w=(XᵀX+λI)⁻¹Xᵀy; γ(x)=|xᵀw|/||w||₂ (NOT normalized overlap);
  κ_reg=√((1+λ)/(σ_min²+λ)); r_reg=(||Xw−y||²+λ||w||²)/||y||²≤0.99.
- s is maximum row OR column nonzero count. Test vectors have at most s
  nonzeros and ||x||∞≤1. N,D count nonzero rows/features. R measures sample
  repetition; the planned independent sampler has R=1. b denotes data precision,
  not a sample exponent; m is the number of test vectors.
- F64–F73: Aλ=[X;√λ I]/√(1+λ), b=[y;0]/||y||. Apply the odd inverse
  polynomial to Aλ†, not Aλ. P≈1/(2κ_reg x), η=ε_solver/(80κ_reg).
- D.23 encodes X/2. The prototype constructs Cλ=[C_X;√λ I/2]/√(1+λ),
  so Cλ≈Aλ/2. Its inverse interval starts at 1/(2κ_reg), and the target
  is P(x)=1/(4κ_reg x). No free classical doubling of the encoded matrix.
- F75–F79: ideal success amplitude ≥1/(20κ_reg), approximate amplitude
  ≥1/(40κ_reg). Fixed-point amplification costs O(κ_reg log(1/ε_solver));
  combined solver Q₀=O(κ_reg² log²(κ_reg/ε_solver)).
- F83/F92: reusable interferometric Clifford shadows need
  O((min{s log(D/ε),log m}+log(1/δ))/ε²) controlled solver calls.
  A fixed-test Hadamard measurement is a separate, useful diagnostic.
- F94–F97: normalized overlap ≥γ/√s; choose ε=γ/(3√s).
  Q=Õ(s κ_reg² [min{s log D,log m}+log(1/δ)]/γ²).
- F99–F104: instantiate EACH X and y oracle to ε₁=δ/(6Q), including
  fresh sample blocks for repeated uses. Then M_X=Õ(R N s⁵ Q²/δ), with
  the label loading cost added. This gives F61:
  M=Õ(R N s⁷ κ_reg⁴ γ⁻⁴ δ⁻¹ min{s² log²D,log²m}).
  The displayed F61 hides δ logarithms; use the preceding Q expression
  for finite-m calculations (substituting m=1 into log²m would spuriously give zero).
  The m-dependent factor also contains s powers: isolate its chosen regime
  before calling a fitted slope an estimate of the displayed s⁷ prefactor.

## Missing components and correctness gates

1. Sparse primitives and bounded sign polynomials are validated. Verify archived
   hashes, pinned versions and reference status again; retain prior data unchanged.
2. Add a separate instance generator with exactly s entries per row/column and
   a known zero singular value: signed circulant paired differences. This permits
   λ=1/(κ_reg²−1) without destroying sparsity. Signs are correlated through row
   and column sign matrices, not iid entrywise. This is a new Phase-(b) family,
   not a change to Phase (a). Report measured full singular spectra and κ.
3. Build and densely verify the selector construction for Aλ/2, an odd bounded
   inverse polynomial, the inverse action on [y;0], flat-label QOS loading,
   fixed-point amplification, and readout. Compare to independent normal-equation
   ridge and augmented least squares. Keep leakage; never renormalize a success
   branch for free before calculating readout probabilities.
4. Dense QSVT at N≤32: verify polynomial response to 1e-10, dilation unitarity,
   real-part LCU, stage errors and clean ancillas. Dirty ancillas are a negative
   control, not an allowed input. Stagewise canonical Halmos completion is the
   reference simulation convention; it does not validate random changing garbage
   blocks in a nested sampled circuit.
5. The prototype inverse uses a bounded smooth piecewise extension of 1/x.
   Validate its interval error and actual degree, but do not call it a
   proof-optimal polynomial. Calibrate degree versus κ and ε before exponent
   sweeps; excessive precision scaling requires a separate optimized construction.
6. Use theorem-certified amplification threshold 1/(40κ_reg), rather than selecting
   degree using the observed success amplitude. Report actual amplitude as a
   diagnostic. Preserve all boundedness and phase-verification failure gates.
7. A tiny fixed-test Hadamard calibration can pass while the full F61 experiment
   remains unready. The reusable Clifford-shadow implementation, statistical
   sampler and literal nested circuit accounting remain explicit approval gates.

## Proposed experiment grid (not launched)

Primary N=D=64. Five independent instance seeds 42,123,456,789,2026.
Baseline s=4, κ_reg=4, γ=0.1, δ=0.1. One-factor sweeps:

| Factor | Values | Held fixed |
|---|---|---|
| s | 2,4,8,16 | κ=4, γ=0.1, δ=0.1 |
| κ_reg | 2,4,8,16 | s=4, γ=0.1, δ=0.1 |
| γ | 0.3,0.1,0.03,0.01 | s=4, κ=4, δ=0.1 |
| δ | 0.2,0.1,0.05,0.02 | s=4, κ=4, γ=0.1 |

For the proposed reusable-shadow primary experiment, fix m=16 test vectors
per instance, with both signs and verified minimum margin; measure simultaneous
correctness across that set. The finite-m logarithmic factor is then explicit.
Use a genuinely uniform Clifford sampler with a verified second-moment/shadow
identity, not an uncalibrated short random Clifford circuit or a Haar substitute.
Keep the one-test Hadamard diagnostic alongside it. These are proposed choices
for approval, not completed implementations.

Use ε*=0.01 and 0.001, with actual solver error target
min{ε*,γ/(6√s)} reported explicitly. This prevents an ε* choice exceeding
its margin error budget. The primitive ε_BE, inverse η and amplification
leakage tolerance are distinct columns. Keep a proof-budget calculation
ε₁=δ/(6Q) alongside an empirically calibrated sample-budget search; do not
impose the theorem's powers in the empirical search and then fit them back.

The union has 13 parameter configurations × 5 seeds × 2 precision settings =
130 configurations, before adaptive sample brackets, spot checks and toys.
Use the phase-(a) eight-point elementary M grid initially (1,040 evaluations),
with explicit left/right censoring. Refinement of brackets must be recorded.
Smallest M with success ≥1−δ is a discrete-grid estimand unless bracketed and
refined; retain failures and nonmonotonic points. Use independent validation
repetitions after selecting a budget, to avoid optimistic selection bias.

Construct s-supported test vectors of fixed norm (initially 1) by changing
orientation relative to w, never merely shrinking x: shrinking changes F48
margin but leaves normalized-state readout difficulty unchanged. Verify actual
γ, overlap, norm, support and infinity norm for every point. Reject/report
infeasible constructions without silently replacing seeds or targets. The
s/γ² readout prediction is a worst-case bound; unit-norm test vectors need
not saturate it. Add a separately labelled near-√s-norm family only after
its feasibility is checked. The primary family is idealized uniform sparsity.

N=128: propose baseline and one high-s point per seed after N64 timings;
no automatic reduction of instances or accuracy requirements.

## Statistics and model separation

- Expected-unitary: deterministic canonical-completion circuit for each instance
  and M. Calculate exact binomial success for the fixed-test diagnostic, then
  validate with 200 independent shot repetitions. Report two-sided 95% intervals
  and one-sided lower bounds for threshold decisions. Zero failures in 200 trials
  only certifies failure probability below about 1.5% at one-sided 95% confidence;
  δ=0.01 cannot be certified that way. The proposed δ floor is 0.02.
- Bootstrap independently generated matrix instances (paired across factor values),
  seed 42, 2,000 replicates. Report global fits, adjacent local slopes, raw and
  explicitly defined log-factor-divided fits, and leave-one-instance-out sensitivity.
  Q₀ versus κ, shots versus actual normalized overlap, and total samples are
  separate outputs. Quantized degrees and short ranges limit interpretation.
- Explicit sampling: mandatory ≥3 γ and ≥3 δ values (shared baseline allowed),
  at the smallest validated size. Call `qos_sampling.py` primitives with fresh
  RNG substreams per shot AND per elementary oracle invocation. Its index
  routine hard-codes degree 51 and unverified PolySign, so first port the streamed
  routine into this experiment with injected verified angles; do not patch upstream
  or silently use its degree-51 path. Compare its low-degree behavior with upstream
  as a labelled diagnostic, then validate the corrected primary path.
- Fresh draws between shots alone are insufficient if repeated inverse/QSVT queries
  within a shot reuse one random oracle. Real-part LCU conjugate branches may share
  a sample block under the reference convention, but subsequent logical queries
  require independent blocks. Test changing-garbage completions before using an
  SVD shortcut for this stochastic circuit. Average probabilities/channels, not
  amplitudes, across samples. Add finite-shot uncertainty separately.
- Clifford shadows are required to test the reusable-model/m dependence in F61.
  Hadamard-only results must remain fixed-test results. δ-dependent shots alone
  cannot test the theorem's oracle-instantiation δ penalty.
- γ⁻³ or log(1/δ) improvements remain unresolved unless both models agree with
  adequate statistical resolution; even agreement is finite-size evidence only.

## CFPB toy (planned, not generated at this gate)

Use 64 seed-selected training rows from one registered binary task, with its
frozen January TF-IDF vocabulary and IDF. `data/manifest.csv` says the validated
representation is reconstructed, not persisted: reproduce using the pinned main
preprocessing environment in a SEPARATE preflight process, compare recorded
statistics/vocabulary/IDF provenance, export once, then load into the reference
environment. Do not silently refit vocabulary on 64 rows or mix environments.
Choose (A) 64 seeded random columns from the full training vocabulary and
(B) top 64 by FULL training document frequency, deterministic tie-break by feature
index. Use the same rows for both; record zero rows/columns, both sparsities,
actual s, rank, norms and normalization. Random columns may yield an almost-empty
matrix: retain that negative result rather than resampling until usable. Register
both derived objects and parent hashes in the data manifest when approved.
Toy accuracy and balanced accuracy are both required; neither is the CFPB benchmark.
Nonuniform sparsity requires the padded Ns-slot sampler of F99 rather than uniform
nnz sampling without correction. The old `d23_amplify` rounds √(s_row s_col),
so a separately validated max-s padded construction is required for the toys.

## Accounting and outputs

Per row record N,D, row/column s, actual κ_reg, γ, overlap, δ, all ε values,
polynomial degrees, label loading, logical queries, controlled branch applications,
LCU overhead, readout shots, elementary M, total X/y samples, success probability
and confidence intervals, measured runtime/RSS, seed, versions and hashes.

Each real-part LCU adds one clean ancilla and uses the circuit and its adjoint.
Hermitianization likewise has two controlled branches where used. Record both
branch applications and unique fresh sample blocks: the reference permits shared
samples across paired conjugate branches, not across repeated logical calls.
Any prototype recurrence is calculated circuit accounting, not measured gate
execution; certify the final recurrence against a literal circuit trace before
sweeping. Report a no-sharing alternative separately if relevant.

Equivalent row-data passes = total independently drawn training-row samples / N;
equivalent nonzero-entry passes = X entry draws / nnz(X). They are different
quantities. One row draw followed by one padded slot gives one matrix sample;
do not count a row as s free element samples. Simulation array evaluations do
not consume the analytically accounted trillions of samples.

Outputs: immutable run directories with manifests, checkpoints, stage-error
CSV, degree/runtime/RSS plots, prediction-success tables, exponent/local-slope
plots and model-comparison plots. Ledger is append-only with exact label
`simulation (expected-unitary QOS, reference convention)`; model-specific fields
must additionally distinguish explicit sampling and theorem arithmetic.

## Approval gate

Run only a tiny correctness/calibration prototype now, with per-process 4 GiB RSS
monitoring, ≥4 GiB available before stages and ≥2 GiB while running, under
`caffeinate -i`. Preserve old hashes and run isolated tests plus the existing
57-test tightness suite. No shared source change is needed, so no Bank rerun.
Write measured calibration and conditional runtime estimates in a separate report.
If full estimated runtime exceeds three hours, or an essential construction is
unvalidated, present alternatives and STOP. Do not launch any complete sweeps.
