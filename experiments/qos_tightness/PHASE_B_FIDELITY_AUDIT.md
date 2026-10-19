# Phase (b) theorem-to-implementation audit and bounded next steps

Audit date: 2026-10-19. Baseline: `6d25652`. Reference:
`a3fe8d6fe777750749f5e90131180e016088bdaf`. This audit accompanies one approved
optimization/correctness gate. No full parameter sweeps are authorized or run.

The implementation supports a **partial, deterministic fixed-test simulation**.
It does not yet simulate Theorem F.13 end to end. Faster phase synthesis changes
its cost, not this classification. See [measured gate report](PHASE_B_OPTIMIZATION.md).

## Source and exact requirements

Rechecked against [arXiv:2604.07639v2](https://arxiv.org/pdf/2604.07639v2),
printed pages 146–151, especially F.13, F64–F79 and F83–F104. Equation numbers
refer to the PDF, not the HTML's global numbering. The source date is
1 October 2026. The implementation and prior plan/calibration were read before
changes; the working tree started clean at `6d25652`.

The theorem concerns no-intercept, unweighted ridge, spectral norm ||X||=1,
maximum **row-or-column** sparsity s, regularized residual r_reg<=0.99,
and s-sparse tests with infinity norm <=1. The margin is |xᵀw|/||w||,
not the normalized overlap. A finite precision b, repetition parameter R,
and the number m of test vectors also enter its resource statements.
The synthetic calibration checks its actual singular spectrum, regularized
condition number, residual, support, margin and test norm. It only covers
uniform sparsity, binary flat labels, R=1 as a modelling assumption, and one
unit-norm test per instance. It does not implement finite-b arithmetic or the
general hierarchical data process.

| Equations / component | Implementation at this gate | Fidelity and consequence |
|---|---|---|
| F64–F69: augmented ridge and nontrivial overlap | `instance`, `augmented_selector`, independent normal equations and augmented least squares | Faithful algebra for these instances. D.23 encodes X/2, so the selector physically encodes A_lambda/2 with the identity branch also divided by 2. |
| F70–F74: inverse of the augmented adjoint | Odd Chebyshev approximation to 1/(4 kappa x), interval [1/(2 kappa),1], eta=eps_state/(80 kappa) | Faithful target and factor-two convention. C2 splice/interpolation is a numerical construction; its degree is not established to achieve the proof's asymptotic degree. |
| F75–F79: fixed-point amplification | Odd erf approximation with threshold 1/(40 kappa); amplitude error <=eps_state²/128 on a grid; leakage included | Faithful threshold and full-state diagnostic. A canonical dilation of the success vector is transformed and checked densely. The circuit does **not** literally amplify the previously built solver unitary using its reflections and adjoints. That compositional gate is still missing. |
| F83–F86: reusable interferometric Clifford shadow | Fixed-test Hadamard interference with x in the reference arm; exact binomial model and simulated shots | Missing reusable model. The required reference arm is fixed at zero; test dependence belongs in the classical observables. Current shots cannot be reused for different x. |
| F87: controlled solver and inverse queries | Stagewise dense QSVT and SVD agreement; canonical completions | Partial. Dense checks validate each stage under the completion convention, not one composed cV with all ancillary registers or freshly changing random oracles. |
| F88–F92: sparse covering-net or finite-m readout | No net, rounding, multi-test estimator or uniform Clifford sampler | Missing. The covering net is an analytical tool; a future finite-m pilot can use the finite-m branch without enumerating a net, but cannot claim the uniform sparse-test branch. |
| F93–F96: overlap/margin/error allocation | Constructed unit-norm x, checked margin gamma; fixed eps_state=.01 | Valid for tested instances. Unit test norm does not saturate the sqrt(s) norm bound. The general solver/shadow split must be explicit before varying gamma. |
| F97: total X/y query count Q | `query_accounting` degree-product branch recipe | Unvalidated analytical recipe. It is not a traced nested circuit and cannot certify the F97 Q, particularly after adding shadows. |
| F98: retained classical model space | No shadow records/tableaux | Missing. Python dense-array RSS measures classical simulator memory, not theorem machine size. |
| F99: approximate X block encoding, then random channel | eps_BE=1e-4; expected-unitary sparse primitives; elementary M=1e8 | Partial primitive simulation only. Neither error is allocated delta/(6Q), and expected-unitary response is not random-channel distance. |
| F100–F101: y preparation and repeated sampled queries | Expected-unitary flat-label loading; y error measured | Restricted diagnostic. General D.25 shared preprocessing record, conditioned fresh blocks, inverse/controlled channel semantics and preprocessing cost are missing. |
| F102: accumulated failure budget | delta affects fixed-test shot count only | Missing. Existing binomial success does not certify the theorem's oracle-instantiation error budget. |
| F103–F104: complete memory/sample accounting | Labelled recipe counts, measured simulator runtime/RSS | Partial analytical accounting. No measured consumption of the reported large sample totals, no complete theorem-resource certification. |

In particular, grid-bounded polynomials and a 2,001-point circuit-response check
are numerical validation, not a rigorous uniform approximation proof. The
calibration's dirty-ancilla checks reject inappropriate inputs; they do not
prove the ancilla lifecycle of an unimplemented composed circuit. Both remain
limitations after optimization.

## Readout and accuracy ledger to implement separately

The exact interferometric state is
`(|0>|0> + |1>|w>)/sqrt(2)`. With
`|x±>=(|0>|0> ± |1>|x>)/sqrt(2)`, the observable
`O_x=|x+><x+|−|x−><x−|` has trace zero, tr(O_x²)=2, and expectation
Re<x|w>. This retains sign, unlike measuring |x><x| on |w> alone.
Its circuit uses one controlled preparation per independently measured copy.

For a finite-m model, use the same Clifford/outcome records for all test vectors,
including tests selected after collection but independently of those records.
Do not silently select tests by looking for small observed shadow error.
F88–F92 instead provide a uniform sparse-test route through a covering net and
on-demand rounding; this is a distinct later validation target.

For the accuracy ledger, retain the following quantities separately:

1. `epsilon = gamma_test/(3 sqrt(s))` (F96), solver state error at most
   epsilon/2, and a separate shadow estimation/rounding allocation. For a finite
   test set there is no net approximation; a concrete conservative concentration
   bound or independently calibrated estimator must determine its shot budget.
2. Ideal query algorithm success at least 1−delta/2. Let L be its number of
   independently prepared shadow copies and Q0 its traced X/y query cost,
   including inverse/controlled uses. Q=L Q0. Neither unknown big-O constants
   nor the present binomial shot count determine a certified L.
3. `epsilon1=delta/(6Q)`. Keep all three errors: X approximation E1,
   random X channel E2, and y channel E3, each at most Q epsilon1=delta/6.
   Their sum is delta/2. Preserve the paper's distance conventions when converting
   channel bounds to total-variation or success guarantees; do not double-count
   or drop factors by switching conventions.
4. Fresh blocks per logical randomized query, including inverse and controlled
   versions. Paired coherent conjugate branches require an explicitly justified
   sharing rule; they do not authorize reusing a fixed random oracle through an
   entire inverse/FP sequence. Cache **classical phases**, never sampled oracle
   realizations. The y preprocessing record is generated once and shared as in
   D.25; subsequent blocks remain fresh conditioned on it.
5. Sample X via a uniform row and uniform padded slot, retaining the slot label.
   This gives Ns possible slots, including dummy slots for nonuniform sparsity.
   Count X and y streams and label preprocessing separately. One row/slot draw
   is one matrix sample, not s free samples. Report row passes /N and entry
   passes /nnz(X) separately.

With n=ceil(log2(max(N,D))), the F99 and F101 bound expressions before
suppressing logarithms are respectively proportional to
`R 2^n n² s^5 Q² log^4(nsQ/delta)/delta` and
`R 2^n n² Q² log^4(Q/delta)/delta` (unknown constants).
These are analytical expressions, not exact finite sample budgets. F104 follows
by substituting the F97 query bound. Keep the additive log(1/delta) in F92/F97:
putting m=1 into the abbreviated log²(m) term must not produce zero samples.
Likewise the min factor can itself contain s dependence. Fitting theorem-assigned
budgets would merely recover imposed powers, not measure tightness.

## Safe optimization boundary

Only opt-in phase synthesis and storage change. `phase_cache.py` batches the
same real 3-vector recurrences used by pyqsp 0.2.0, takes the equivalent real FFT,
then uses the same Newton initialization, update order, 1-norm criterion 1e-12,
100-iteration limit and QSVT conversion. The original backend is still the default.
No upstream package or reference file is modified. Both parities are compared
against upstream Jacobians and Newton phases; finite differences independently
check the Jacobian. Existing scientific tests remain intact.

Disk entries are keyed by exact float64 coefficient bytes, implementation and
upstream code hashes, reference commit, dependency versions, convention and
tolerances. A hit checks identity, coefficient equality, checksum, shape and
finiteness, then reruns the original reference response check to 1e-10.
Corruption fails closed. Misses are checked before atomic storage. In-process
LRU reuse remains available. Reconstructing polynomials and checking phases on a
disk hit are deliberately retained, so warm cost is not zero.

The original 4,095 degree cap remains the default. The controlled kappa=4 job
explicitly permits 8,191, with the same polynomial target, degree sequence,
threshold, tolerances and seed. This is a disclosed resource-limit change,
not a different inverse or amplification algorithm. The 4 GiB RSS limit and
available-memory gates remain active. A conservative workspace estimate also
rejects phase synthesis before allocating a predicted excessive Jacobian.

## Three possible outcomes

| Outcome | Evidence needed | Status after this gate |
|---|---|---|
| Faithful end-to-end simulation of the theorem | Composed controlled solver/FPAA, uniform Clifford readout reused across tests, fresh sampled channels with shared-record semantics, delta accuracy ledger, traced queries/samples and independent end-to-end statistical validation | **Not achieved.** Optimizing phase finding cannot supply missing algorithmic components. |
| Partial simulation with analytical accounting | Explicitly identified simulated stages, numerical gates, separate analytical costs for absent components and stated constants/assumptions | **Current outcome.** Fixed-test expected-unitary chain plus a provisional branch recipe. Accounting is not yet a certified theorem budget. |
| Primitive-level tightness results only | Independently validated oracle/block-encoding error versus elementary samples, under an explicit simulation model | **Existing Phase (a) outcome.** Its finite-size findings remain valid within that scope; they do not become F61 exponent measurements by adding solver timing. |

No outcome here establishes quantum advantage or disproves asymptotic powers.
A numerical phase failure would be an implementation limitation; failure of an
instance to meet theorem assumptions would invalidate that instance's test;
degree-cap failure only says the current search did not find an acceptable
candidate within its permitted sequence. None by itself refutes the theorem.

## Estimated implementation work before further approval

These are engineering estimates, not measurements or promised completion times.
The components below are **not implemented by this gate**.

| Increment | Construction and acceptance gate | Estimated work / complexity |
|---|---|---|
| Tiny reusable-shadow readout | Exact uniform Clifford or equivalent uniform stabilizer-basis sampler at three qubits; Born outcomes; inverse measurement channel; retained records; independent mean/second-moment checks; median-of-means or justified finite-sample estimator; both-sign held-out tests | 2–4 focused engineering days. A verified sampler and its distribution tests dominate. Dense state dimension 8 for ideal N=4 readout; O(L d² + L m d) arithmetic with dense gates/overlaps, O(L q²) retained tableau records when implemented. |
| Full deterministic solver composition | Preserve all success flags, selector, real-part LCU, inverse calls, label preparation and reflection-controlled FPAA; compare its output to the stagewise shortcut without resetting garbage | 3–6 days. Circuit state/density dimension depends on actual ancilla liveness, so establish that count before allocating dense unitaries. State-vector application should replace full-unitary storage where equivalent and independently checked. |
| Query/accuracy planner | Trace real calls at X/y and elementary levels, including adjoints/controls and branch sharing; allocate epsilon1; separate one-time y preparation; retain symbolic constants; unit-test counted low-degree circuits | 2–4 days, partly overlapping composition. Analytical budgets alone cannot certify stochastic error. |
| Fresh-sample channel validation | Port the reference streaming primitive with injected verified phases; preserve RNG order/substreams; validate paired branches and D.25 conditioning; compare averaged channels/probabilities and composed circuits on tiny cases | 5–10 days after the circuit/trace is stable. Runtime could dominate; no credible full-run estimate before a per-query sampled-channel pilot. |

Total scope is approximately 2–4 engineering weeks with overlap and revision,
not an additional small optimization. No new dependencies are installed now.
For the tiny three-qubit readout, exact enumeration of the finite Clifford group
is a possible test oracle, but storing millions of dense Cliffords would be
wasteful. An exact symplectic/tableau sampler or proven equivalent uniform
stabilizer-state measurement construction needs its own fidelity proof and tests;
a short random circuit or Haar sampler is not a silent substitute.
The accompanying report proposes a more concrete small-instance route:
enumerate stabilizer-state records and sample their exact Born-weighted marginal
distribution, proving its equivalence to uniform Clifford measurement first.
The runtime proxy benchmarks only lookup/draw arithmetic, not that construction.

Recommended next approval is **only the tiny reusable-shadow readout gate**:
N=D=4, s=2, kappa=2, gamma=.3, delta=.1, seed 42, m=4 fixed tests with both
signs; compare ideal ridge-state and finite-M expected-unitary/leakage inputs
using the same records across tests. Leakage requires retaining the clean-flag
subspace (larger dimension if included), not free postselection. Validate the
ideal eight-dimensional readout first; do not hide a leakage omission in an
"end-to-end" label. Choose L from an explicit concentration/error target, then
independently test simultaneous correctness. No exponent fit and no delta sweep.
The measured solver cost and a bounded runtime forecast are in the accompanying
optimization report. A 10,000-record throughput preflight should precede the
statistical run, with a ten-minute execution budget and checkpoint/stop on
exceeding it. This next gate remains partial simulation with analytical oracle
accounting; full F61 tightness remains unresolved.
