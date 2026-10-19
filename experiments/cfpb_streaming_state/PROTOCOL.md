# Approved validation-only pilot

Approval: user instruction following published checkpoint `13aae4f`. This
execution protocol fixes pilot choices before seeing any new model scores.
It implements only the correctness/calibration/one-pass pilot portions of
`docs/cfpb_streaming_state_plan.md`; it does not execute its tuning matrix.

## Fixed design and access

- Tasks: credit-versus-checking (near-balanced training counts, 7,265 rows)
  and mortgage-versus-rest (minority positive class, 13,459 rows). These are
  the plan's prespecified contrasting tasks, chosen without test outcomes.
- Four methods, representations A/B, chronological/random orders, algorithm
  ceilings 16/128/512 KiB, seed 42: exactly 96 planned trajectories.
- One training pass; checkpoints at ceil(N/8), ceil(N/4), ceil(N/2), N.
  Calibration is the first grid trajectory, credit/checking, A, hashing,
  chronological, 16 KiB; it is not an extra training run.
- Half squared loss, no intercept, no class weights, no averaging, float64
  model state. eta0=.1 and eta(t)=eta0/(1+eta0*lambda*t). Hashing/JL/AWM use
  lambda=1/N (Ridge alpha=1); native MISSION uses lambda=0. These are diagnostic
  settings, not validation-tuned optima. JL and adaptive sketches use depth 3.
  Adaptive exact-state allocation targets 45% of available buffer bytes;
  heap capacity rounds down to a power of two. Remaining state goes to counters.
  Every byte ceiling includes Python object/NumPy allocation overhead and
  algorithm metadata, with 512 bytes reserved for scalar object de-aliasing.
- No trial retries with changed parameters, no extra seeds, no balanced-model
  fits, and no empirical minimum-memory or method-superiority claim.
- Primary and frozen natural-stream **validation** only, including the frozen
  strata. The loader filters to January training and February 1–14 validation
  before rehydrating accepted texts. It never builds a test matrix or evaluates
  a test model. Reference checks read only validation columns of the audit CSV.
- The full Ridge reference selects alpha from {.1,1,10,100} by frozen validation
  accuracy, ties to larger alpha. Source values are frozen float32 TF-IDF cast
  to float64; LSQR tolerance 1e-10 and limit 20,000. Its batch fitting resources
  are reported separately. No real-data full-dimensional SGD trajectories are
  added beyond the 96 ceiling; dense-gradient/sklearn comparisons use toy unit
  fixtures. Thus this pilot cannot diagnose full-data optimization convergence.

Track A reads frozen common-January vocabulary/IDF and is replay conditional on
that preprocessing. Track B uses only each document's normalized term counts,
without vocabulary/DF/IDF fitting. Its 128-bit BLAKE2b token IDs are finite hash
identities, not a proof of collision impossibility. No cross-document token
dictionary is kept. Both tracks retain the frozen offline-curation caveat.

The offline input spool holds accepted training/validation text on disk under
the ignored `data_cache/` subdirectory. It is research access infrastructure,
not free model state or an online curator. Random replay explicitly owns an
N-element int32 permutation and N-element int64 seek-offset array. Both are
counted in stream-access state. Chronological reading uses one bounded 8 KiB
file buffer and cursor; it owns no N-sized ordering array. Report external
spool/index bytes, setup/transient permutation allocation, and preprocessing
passes separately. Raw source CSV scanning is an offline integrity/selection
operation; it does not expose future rows to the learner.

## Algorithm conformance and sources

Independent implementations use the following primary references:

- [AWM paper, Algorithm 2](https://arxiv.org/pdf/1711.02305), author repository
  [wmsketch](https://github.com/stanford-futuredata/wmsketch), inspected revision
  `4907851642f520a752c3591612c168da40b03989`. No license file was present; no
  upstream source is copied or vendored. The implementation is written from
  the mathematical algorithm and checked with independent dense toy updates.
  The paper's projected-tail predictor is used: with unscaled Count-Sketch
  counters S, tail prediction uses the mean signed counter across rows, while
  admission/eviction queries use the median. This differs from the released
  logistic code's median-tail predictor and sorted snapshot candidates. The
  pilot uses half squared loss, canonical feature-order sequential candidates,
  and negative loss-gradient updates, including the rejected-candidate branch.
  It does not reproduce the ambiguous positive sign printed in that branch.
- [MISSION paper, Algorithm 1](https://arxiv.org/pdf/1806.04310), author repository
  [MISSION](https://github.com/rdspring1/MISSION), inspected revision
  `03b5bb2bb95a7f7f5ecee3ff4777042d00924b20` (Apache-2.0). The accumulator retains
  every signed update; only bounded active candidates contribute to prediction.
  Touched coordinates are queried/admitted sequentially, as in the released
  implementation. The pilot maintains exact heap priorities, without its 1.05
  priority slack. Untouched estimates can become stale under collisions; no
  exhaustive global top-k recovery is claimed. Half-loss eta=.1 corresponds
  to eta=.05 in the paper's unhalved squared-loss gradient convention.

All methods use seeded 64-bit mixing on the two halves of the feature ID, with
separate bucket/sign namespaces. JL has three disjoint row blocks and scaling
1/sqrt(3); hashing is its depth-one case. This is an empirical seeded sparse
projection, not the repository's occupancy-normalized legacy JL, and not a
certified finite-independence JL distribution. No existing sketch is modified.
The separate legacy adapter preserves the existing permutation/sign RNG order,
occupancy normalization and sparse multiplication. A toy row-wise equivalence
test checks exact equality and charges its stored CSR mapping. It adds no
real-data training trajectory to the pilot.

The active index uses fixed NumPy key/value/heap/position arrays and a bounded
open-address table with backshift deletion. No growing feature dictionary,
tombstone store, gradient history, or dormant dense model is retained. AWM
uses lazy global L2 scaling; MISSION has no hidden L2 shrinkage.

## Measurement, correctness, and stopping

Persistent state is a deduplicated inventory of reachable Python objects and
owned NumPy buffers including unused capacity. Model coefficients, sketch
counters, active indices, projection parameters, optimizer, preprocessing, and
stream access are separate categories. Native array registers and allocated-byte
64-bit equivalents are both reported. Object sizes are not serialized sizes;
interpreter/library code, allocator fragmentation and other native overhead
appear in absolute process RSS. Explicit per-document workspace is a measured
inventory lower bound; peak RSS also captures temporaries inside libraries.

The encoder's stop-word and vocabulary state is charged even when library
objects are shared. Preprocessing state is measured after fixed-vocabulary
initialization. Model state is audited throughout updates and at checkpoints.
One worker/learner runs at a time, with single-thread numerical libraries.

Independent tests cover projected SGD against dense gradients and sklearn,
AWM/MISSION against direct tiny sketch updates, heap/index eviction and ties,
lazy regularization, deterministic snapshots/restart, fixed allocation,
read-only prediction, and causal/temporal guards. Full reference preparation
checks frozen text hashes, IDs/order, vocabulary/IDF, row-wise feature equality,
and validation Ridge predictions/metrics against the audit's conventions.

The one-hour benchmark wall budget includes preparation, Ridge reproduction,
calibration, process startup, training, checkpoint evaluation, and output I/O.
Development and unit/regression tests are separate. Stop before launching more
work at the cap, on failed correctness/memory gates, or at the 5 GiB output cap.
Preserve the repository's 4 GiB process-RSS / 4 GiB start-available / 2 GiB
running-available / rising-swap safeguards. Completed runs never get overwritten.
Orderly interruption/deadline checkpoints preserve model, cursor and evaluation
history. A hard crash with uncertain post-checkpoint access is failed rather
than automatically rerun. Final ledger publication uses unique trajectory IDs
and preserves every pre-existing ledger byte.

Only validation pilot summaries may inform a separately proposed reduced final
grid. One seed gives no across-seed uncertainty estimate. The full benchmark,
additional pilot fits, test evaluation, and GitHub push remain unauthorized.
