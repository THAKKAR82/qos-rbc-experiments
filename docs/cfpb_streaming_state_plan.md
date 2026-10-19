# CFPB classical streaming-state benchmark: implementation plan

Status: **proposal awaiting approval; no implementation or experiments authorized by this document**. Prepared 2026-10-19 against repository commit `a480ebc`. The QOS Phase (b) partial simulation remains frozen. This proposal does not extend it.

The question is: **how much persistent classical state, and how many training-example presentations, suffice to retain 95%, 98%, or 99% of the full Ridge classifier's predictive performance?** The answer will be an empirical frontier over specified algorithms, populations, representations, access budgets, and metrics. It will not be an information-theoretic minimum or a claim about quantum advantage.

## 1. Repository findings and immutable inputs

Reviewed `AGENTS.md`; the CFPB specification, curation validity review and modeling decision; baseline/compression protocols and reports; and `src/cfpb.py`, `cfpb_prepare.py`, `text_preprocessing.py`, `cfpb_baselines.py`, `holdout_evaluation.py`, `compression.py`, `cfpb_compression.py`, `sketches.py`, and `qos_audit.py`. These imply several distinctions that the implementation must preserve:

- The previous compression study primarily compared balanced, intercept-bearing LinearSVC models. Its hashing control used logistic SGD, five passes, and training-frequency weights. Its feature counts and saved batch-model measurements are **not streaming-state measurements** for this study.
- The applicability audit uses ten binary, **unweighted, no-intercept Ridge** tasks. Its validation criterion is ordinary accuracy, with ties favoring larger Ridge alpha. This is the primary reference here.
- The existing loader materializes a dataframe of accepted texts. It is useful for offline integrity checks, but cannot be called a bounded-memory learner or hidden inside its memory measurement.
- The existing sparse JL is a balanced, signed, occupancy-normalized, one-entry-per-column construction with a stored mapping and a specified RNG order. It must not silently become a different projection.

Frozen inputs:

| Item | Accepted location / identity |
| --- | --- |
| Raw source | `data/CCDB_Export_15_January_2026_through_February_2026.csv` |
| Source SHA-256 | `664f1cae8d3a19dfd6894e2ca760f5b3f0bb4b71e466f38235473e1ea5b29f6d` |
| Curation | `validation/cfpb/20261014T192947Z-7b6668f9/` |
| Baselines and frozen vocabulary/IDF provenance | `validation/cfpb/20261014T204807Z-2270924a/` |
| Earlier compression, historical context only | `validation/cfpb/20261016T125619Z-30526eac/` |
| Binary task definitions / Ridge results | `results/audit/cfpb_step1/` |
| Curation authorization and limitations | [modeling decision](decisions/cfpb_modeling_gate.md), [validity review](cfpb_curation_validity_review.md) |

Read-only preparation for this plan verified the current source hash and 87 recorded output-artifact hashes across those four accepted result directories. No models were fitted. Acquisition date remains unknown; the source filename does not establish it.

The inspected main environment is Python 3.13.9, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, pandas 3.0.6, psutil 7.2.2, and threadpoolctl 3.7.0. Record the actual versions again at execution; do not upgrade them for this study without a separate dependency/regression assessment.

Preserve primary January 1–31 training (13,459 rows), February 1–14 validation (3,033), and February 15–28 test (4,213). Preserve the secondary natural-stream validation (3,634) and test (5,214), evaluated with the **same primary-trained models**, including frozen seen-in-training / seen-in-validation / new-in-test strata. Never add secondary rows to training or tune on secondary test.

| Binary task | Train | Validation | Test | Positive training rows |
| --- | ---: | ---: | ---: | ---: |
| debt_vs_credit | 8,472 | 1,926 | 2,696 | 4,881 |
| debt_vs_checking | 8,555 | 1,950 | 2,701 | 4,881 |
| debt_vs_mortgage | 6,194 | 1,445 | 2,010 | 4,881 |
| credit_vs_checking | 7,265 | 1,588 | 2,203 | 3,591 |
| credit_vs_mortgage | 4,904 | 1,083 | 1,512 | 3,591 |
| checking_vs_mortgage | 4,987 | 1,107 | 1,517 | 3,674 |
| debt_vs_rest | 13,459 | 3,033 | 4,213 | 4,881 |
| credit_vs_rest | 13,459 | 3,033 | 4,213 | 3,591 |
| checking_vs_rest | 13,459 | 3,033 | 4,213 | 3,674 |
| mortgage_vs_rest | 13,459 | 3,033 | 4,213 | 1,313 |

Use the audit's class order, task masks, labels ±1, and score-zero prediction of +1. Preserve all text cleaning, grouping, exclusions, and temporal boundaries. Frozen grouping uses information across the source period: the population is an offline-curated population of earliest eligible text-group representatives, **not a prospective online deduplication process or representative incoming complaint population**. Debt collection's 35.14% removal and template-frequency suppression remain material limitations.

## 2. Two separately reported representation tracks

These tracks answer different questions. Approval of the plan would authorize a new comparison, not a change to any established benchmark.

**A — frozen-feature replay (direct audit comparison).** Use the common four-class January vocabulary of 50,000 word unigrams/bigrams and its frozen IDF, including for pairwise tasks. Preserve English stop words, tokenization, `min_df=2`, smooth IDF, ordinary term frequency, L2 row normalization, and float32 CSR feature values. Cast those already-rounded values to float64 as the audit does; do not substitute a newly fitted float64 vectorizer. Stream rows of this representation to each learner. Vocabulary/IDF construction consumes all 13,459 January observations even when a binary task is smaller; disclose that supporting-data access.

Frozen vocabulary/IDF see later January observations. Consequently, chronological **model updates** in A are a controlled replay, not prospective within-January preprocessing. A prefix learning curve measures additional learner access conditional on the full-January preprocessing artifact. It cannot establish that only that prefix was needed end to end. No February text or label enters preprocessing or updates.

**B — causal feature construction on the frozen population.** Use the same rows, cleaning, tokenization, stop words, and unigram/bigram rules, but stateless normalized term frequencies, without learned vocabulary, `min_df` filtering, or IDF. Normalize the unprojected per-document counts before applying any sketch; do not make method-specific normalization an unnoticed confound. Hash token identities into fixed 128-bit keys; retain no corpus-sized dictionary. Key/hash collision behavior must be specified, tested, and disclosed. Per-document token-count workspace is transient and measured.

This proposed new representation uses no later document to encode the current one. It is compared to the same frozen TF-IDF Ridge target, so a gap can reflect representation as well as learning/compression. A supplementary full batch Ridge on the same term-frequency representation, if feasible, would separate those effects; it is not a replacement target and must report its own construction state. Frozen population selection still prevents an unconditional real-world prospective claim. Label availability is assumed immediate after an example; received date is not proof of actual narrative-publication/label-availability time.

For both tracks:

- Chronological order is `(date_received, integer complaint_id)`. Only the first pass is a forward chronological stream; later passes are **chronological replay** and require retained/external training data. Label them accordingly.
- Random order is a uniform permutation without replacement of January training examples, independently reshuffled per pass. This is randomized replay, not an IID-with-replacement sample stream. Use matched orders across methods and budgets.
- Checkpoints occur after `ceil(N/8)`, `ceil(N/4)`, `ceil(N/2)`, `N`, `2N`, `5N`, and `10N` presentations. Count first-pass unique examples separately from repeated presentations.
- Validation is February-only and cannot update weights, features, optimizer clocks, or RNG state. Selecting hyperparameters using February validation is retrospective model development; January prefix scores are not online decisions made at that January date.
- No online updates during February evaluation. Prequential error during January can be diagnostic, but is not the requested held-out Ridge-retention target.

## 3. Methods and fidelity requirements

| Method | Planned role and state |
| --- | --- |
| Full Ridge | Non-streaming accuracy reference, all 50,000 frozen features; alpha selected from `{0.1,1,10,100}` by primary validation accuracy, ties to larger alpha. Measure fitting resources separately from final coefficient state. |
| Feature hashing + SGD | Signed feature hashing of the track's input coordinates; online squared-error SGD, no intercept or class weights. Count weights, hash seeds, optimizer state, and any averaging buffers. |
| Sparse JL + SGD | A **new, explicitly named seed-generated sparse projection** with multiple signed entries per column, each scaled by `1/sqrt(q)`, followed by the same SGD family. Use disjoint row blocks to avoid within-column duplicates; tune `q` in `{3,5}`. No dense projection or full feature map. |
| AWM-Sketch | Active exact weights plus a Weight-Median sketch of the tail; squared-loss adaptation, no intercept or weights. Account for sketch, bounded active set, keys, position index, and update metadata. |
| MISSION | Native squared-loss gradient accumulation in Count-Sketch with bounded active-feature recovery and sparse prediction; no intercept or class weights. Retain the accumulator across evictions. Native unregularized MISSION remains a candidate in its own right. |
| Full-dimensional online SGD | Optimization control on A: can the same optimizer reach Ridge-like prediction with ample state and the same access budget? Failure limits conclusions about small-state methods. |
| Optional batch feature selection | Training-only frequency or supervised feature ranking followed by Ridge. Reference for representational compression, never a streaming-state competitor; report ranking arrays and data passes. |

The new JL construction is motivated by [sparse JL transforms](https://arxiv.org/abs/1012.1577). Fixed small `q` at empirical budgets does not itself certify a requested distortion bound. Test norms/inner products diagnostically without selecting projection seeds by test performance. A `q=1` construction can serve as a hashing equivalence test, not an independent algorithm win.

Separately, retain a **legacy-JL compatibility control** in A: a row-wise adapter must match `src/sketches.py`'s exact assignment, signs, division by square-root occupancy, and RNG consumption. Its mapping counts toward state. Do not change that function, reuse its old batch measurements, or call the new projection the validated legacy variant. Where the legacy mapping exceeds budget, report `infeasible_state`, not poor accuracy. This control is outside the four-method main matrix.

The [AWM paper, Algorithm 2](https://arxiv.org/pdf/1711.02305) and [author implementation](https://github.com/stanford-futuredata/wmsketch) are the fidelity references. Prediction combines exact active contributions with a projected sketch-tail contribution; a median-decoded full-vector predictor is a different procedure. Admissions, evictions, and sketch corrections must follow the algorithm. Label squared loss as an adaptation of the published experimental setup. Check gradient signs against the loss derivative and author code, rather than reproducing a possibly ambiguous typeset update blindly.

The [MISSION paper, Algorithm 1](https://arxiv.org/pdf/1806.04310) and [author implementation](https://github.com/rdspring1/MISSION) are its fidelity references. The sketch accumulates signed gradient updates; predictions use the recovered active model. Recovery must be a bounded implementation, not a hidden scan/dictionary over every observed feature. Pin and inspect exact upstream revisions and licenses before porting; revisions are not yet pinned by this plan. Document any discrepancy between paper abstraction and the author's candidate-heap implementation. Native MISSION is not a Ridge solver. Adding L2 later would require an explicit regularized variant: shrinking the entire accumulator is not automatically equivalent to regularizing only its active predictor. That extension is outside the initial implementation proposal.

For the objective-aligned SGD methods, the audit objective is

`||X w - y||² + alpha_Ridge ||w||²`.

Using per-example half squared error and `lambda ||w||² / 2` gives `lambda = alpha_Ridge / N`. Treating Ridge alpha as an SGD penalty directly is incorrect. This convention is for the full training objective; do not secretly rescale it at each checkpoint. Use unscaled frozen `X`: audit scaling by its spectral norm and adjusting lambda is algebraically equivalent for predictions. Verify against the saved audit results before relying on that equivalence. Retain audit LSQR tolerance `1e-10`, iteration limit 20,000, and convergence/residual checks for reference reproduction; never use Ridge's representation-sensitive automatic solver as an unnoticed replacement.

Start numerical fidelity work with float64 state for all methods. Float32 is a separately labeled memory-precision sensitivity after equivalence and stability checks, with bytes credited honestly. Never clip gradients or alter loss/normalization merely to make a difficult candidate pass; divergent candidates are reported.

## 4. Operational memory and data-access definitions

The independent memory budget is **peak persistent algorithm state during learning**, not feature count, heap size, nonzero coefficient count, or serialized file size. Let `B_alg` include all maintained model/sketch/projection/optimizer state, including capacities reserved but not populated. Configuration must fit the budget at every update, including active-set replacement and averaging transitions.

Report the following disjoint categories and their sums:

| Category | Included items |
| --- | --- |
| `model` | Coefficients and exact active predictor values |
| `sketch` | All sketch counters, accumulators, collision-resolution state |
| `projection` | Hash seeds/functions, legacy feature mapping, occupancy/sign arrays if retained |
| `optimizer` | Momentum/averaging buffers, lazy scaling, clocks, RNG state, running statistics |
| `active_index` | Feature keys, heap capacity, positions/hash table, tombstones and allocator capacity |
| `preprocessing` | Vocabulary strings and lookup structure, IDF, stop words, tokenizer/configuration state |
| `stream_access` | Resident permutation/index, cursor and offsets, replay buffers or resident raw documents |
| `transient` | Current document/batch, sparse gradient/projection workspace; excluded from persistent state but measured separately |

`B_alg = model + sketch + projection + optimizer + active_index`;
`B_total_persistent = B_alg + preprocessing + stream_access`.

Also report inference-only retained state, maximum resident mutable workspace, total process peak RSS, and process baseline RSS. Include Python/container/object overhead when those objects are used; a packed-array idealization is an additional calculated column, not the measured footprint. Deduplicate aliased buffers. Audit allocation capacities and resident objects; sampled RSS alone cannot resolve kilobyte-sized state. Save native register counts by dtype/key width and **64-bit-equivalent registers = ceil(bytes/8)** for each aggregate. These are classical storage units, not QOS machine-size registers.

Use one serial learner per process for measurement. The harness may own fixed datasets and labels for evaluation, but expose its memory separately and include it in process RSS. It must not give the learner free ranking statistics, a corpus dictionary, random-access feature map, or cached future data. Report external source/replay/index disk bytes and reads separately. A permutation stored on disk still consumes external state and access; a resident int32 permutation costs `4N` bytes (up to 53,836). Caches used by the actual pipeline count once in total state; benchmark-only cross-trial caches must be identified and both cold and amortized timing reported.

For every trajectory record training presentations, unique training IDs, fractional/completed passes, feature-coordinate visits, preprocessing presentations and passes, validation/test reads, and HPO aggregate presentations. Pairwise A preprocessing uses rows outside that task; these are supporting observations, not zero-cost sample access. Building the frozen vocabulary/IDF requires a corpus-statistics pass and transformation requires reading documents again; reusing an artifact saves execution now but not its historical data requirement. Preserve row-group privacy by saving IDs/hashes and scores, not narrative text.

## 5. Benchmark matrix and hyperparameter selection

The complete matrix is a **costed research proposal, not permission to execute it**.

| Axis | Proposed main matrix |
| --- | --- |
| Tasks | All ten frozen binary tasks |
| Representations | A and B, reported separately |
| Methods | Hashing-SGD, sparse-JL-SGD, AWM-Sketch, MISSION |
| Algorithm-state ceilings | 4, 8, 16, 32, 64, 128, 256, 512 KiB |
| Orders | Random permutations; chronological first pass then labeled replay |
| Evaluation seeds | Existing list `[42,123,456,789,2026]` |
| Presentation checkpoints | N/8, N/4, N/2, N, 2N, 5N, 10N, rounded up |
| Precision | Float64 primary; float32 outside the main matrix |
| Primary selection | Validation ordinary accuracy; then balanced accuracy; then fixed configuration-ID order |
| Evaluation | Accuracy, balanced accuracy, macro-F1, class recalls/supports, confusion matrices; primary and natural-stream populations |

A ceiling includes overhead: a nominal 4 KiB budget is not permission for 512 float64 coefficients plus free metadata. Derive the largest feasible dimensions/heap capacities from a measured allocation inventory before fitting. Use actual measured bytes on plots. Report infeasible cells. Budget endpoints can be extended only by a recorded, validation-only amendment with a new cost estimate and approval; no automatic full sweep.

Use separate deterministic RNG namespaces for order, projection, optimizer, and tuning, rooted at seed 42. Preserve the evaluation seed list. Tuning streams must be disjoint from evaluation streams even if a displayed root seed is shared. All competitors see the same example sequence at each evaluation seed. Five chronological repetitions quantify sketch/optimizer randomness, not five different chronological datasets. For deterministic controls record one run, not five fictitious replicates.

For each task/method/track/order/budget, propose **24 predeclared, stratified joint configurations**, each run through five passes. Cover step sizes `{0.01,0.1,1,10}` on normalized inputs; constant, inverse-square-root, and inverse-time schedules with explicit time offsets; and, where applicable, penalties `{0,0.1/N,1/N,10/N,100/N}`, averaging on/off, sketch depths `{1,3,5}`, and exact-active-state fractions `{0.25,0.5,0.75}`. JL uses `q={3,5}`. The exact balanced configuration table, time-offset values, and tie rules must be frozen before the first fit; this is a joint design, not an exhaustive Cartesian search. MISSION's native candidate space uses step/schedule/sketch/heap choices, without claiming its zero penalty matches Ridge regularization.

Extend the best two validation configurations at five passes and two prespecified slow-schedule safeguards to ten passes; evaluate these four on a second independent tuning stream through ten passes. This costs `24*5 + 4*5 + 4*10 = 180` training-pass equivalents per cell, before evaluation. All methods receive the same candidate-example budget. Report time as well: equal presentations are not equal computational work. Select one configuration per checkpoint using only candidates actually evaluated there; report candidate counts and the long-horizon screening limitation. The chosen configuration must fit the same state ceiling, including averaging.

Do not launch the full search if the pilot cannot establish sensible numerical step ranges. An optimum at a search boundary, divergence of most candidates, or an unconverged full-dimensional SGD control triggers a validation-only tuning review. It does not justify a memory lower-bound claim. Expansion needs a costed amendment; no selective extra tuning for a favored method. Save all candidate outcomes, including failures, and the aggregate cost of adaptive allocation.

Balanced models form a **separate secondary profile** in A: inverse-training-frequency weights, a separately selected no-intercept weighted Ridge reference, and separately tuned weighted streaming models at up to three primary-validation-selected frontier budgets. Select this profile by validation balanced accuracy, with ordinary accuracy also reported. Its full-January class-count pass is charged. Its predictions and k-star estimates cannot be pooled with unweighted results. Do not preload full-January class weights into causal first-pass B; any future past-only weighting experiment would be a different objective. The balanced profile is staged after the unweighted study and separately costed below.

## 6. Defining k-star and statistical uncertainty

Let `F_t,m` be full Ridge's score on task `t` for metric `m`. For retention `r` in `{0.95,0.98,0.99}`, the target is literally `r * F_t,m`. For each method, track, order, and sample budget `M`, report the smallest **tested** state ceiling whose mean evaluation-seed score meets the validation target. Name it `k_star_grid`, with actual byte range and 64-bit-equivalent registers. Also report the corresponding total persistent state and unique/presented samples. This is a discrete empirical upper bound on achievable state within the tested methods, not a universal minimum.

Conversely, estimate `M_star_grid(B,r)` as the first tested presentation checkpoint meeting the target at budget B, with unique examples and preprocessing access alongside it. Report the nondominated `(B,M)` frontier: there need not be one pair minimizing both. A's end-to-end unique-input requirement already includes all January preprocessing rows. A per-checkpoint validation-selected envelope can use different configurations at different M; label it as separately selected models and also show fixed-configuration trajectories. It is not one adaptive learner choosing its own best stopping time for free.

Report separate accuracy, balanced-accuracy, and macro-F1 retention curves. The headline safeguard is the smallest budget satisfying **both ordinary and balanced accuracy** at the requested retention, still using the primary accuracy-selected model family. Do not silently redefine retention as macro-F1 or rebalance training. Include majority-class predictions and a separately labeled excess-over-baseline retention diagnostic: raw 95% accuracy retention can be trivial on an imbalanced task. An excess-score ratio is undefined when the reference does not beat its baseline.

Use paired prediction-level uncertainty: the compressed and full predictions must be evaluated on the same resampled observations. Save each seed's score, mean, SD, and interval. Plan 2,000 reproducible bootstrap replicates (seed 42) that resample evaluation seeds and complaint groups; keep each natural-stream group intact. Show calendar-day/block resampling as a temporal-dependence sensitivity, with the limitation that each holdout spans only 14 days. The primary curated split has distinct groups, but this does not establish independent underlying incidents. These intervals are conditional on frozen training data and do not quantify new training populations. Five seeds provide limited tail precision.

For each fixed task/method/order/track, form 95% simultaneous paired confidence bands over tested budgets, presentation checkpoints, and the two headline metrics, using a bootstrap maximum-deviation construction. This addresses the per-frontier search and all three retention cutoffs; it is not family-wide control across every task/method, and bootstrap coverage is approximate. Any overall best-method claim needs a separate declared multiplicity family. Report descriptive cross-task summaries without treating overlapping binary tasks as independent replications.

Validation chooses configurations and the proposed threshold budgets. Freeze and hash that selection before any new-method test evaluation. Test confirms those fixed candidates (and predeclared neighboring budgets for descriptive context); **do not reselect the minimum on test**. A candidate is confirmed only when the lower confidence bound on `S - r*F` meets zero for both headline metrics. Distinguish point-estimate crossing from confidence-supported retention. If confirmation fails, report `not_confirmed`, rather than selecting a larger test-favored budget. Historical test results already exist, so this is not a pristine unseen test dataset; registration now only prevents additional method-specific test tuning.

Report grid intervals rather than interpolation as fact: below-grid, above-grid/not-attained, indeterminate, and state-infeasible are different statuses. A last-failed/first-passed grid bracket is descriptive; without monotonicity it does not rule out success at an untested smaller budget. Nonmonotone curves stay visible. Isotonic smoothing may be a labeled visualization only. Near 99% retention, uncertainty may prevent resolving adjacent budgets; that is a valid result, not a reason to change thresholds or add seeds after looking at test. A larger seed count requires a prospective amendment. A small-state failure within ten passes is an access-budget failure, not proof that more memory is necessary at unlimited time.

## 7. Proposed output schema and artifacts

New implementation would live under `experiments/cfpb_streaming_state/`; new results under `validation/cfpb_streaming_state/<UTC>-<commit>/`. No edits to existing `src`, CFPB fixtures, notebooks, audit results, QOS code, or `.venv-qos-ref` are proposed. Use the existing main `.venv`. Keep report figures and interpretation in this new experiment directory or a new notebook, with no extraction into shared `src`.

| Output | Required fields / purpose |
| --- | --- |
| `manifest.json` | schema version, UTC, commit/dirty diff hash, source and frozen-artifact hashes, Python/dependency/OS/CPU versions, thread limits, code/config hashes, upstream revisions/licenses, unknown provenance explicitly null |
| `protocol.json` | task definitions, representations, population/ordering contracts, budgets, candidate tables, seeds/namespaces, metrics, stopping and uncertainty rules |
| `state_inventory.csv` | trajectory/checkpoint, category, object/buffer name, dtype, shape, capacity, native registers, payload bytes, overhead bytes, allocation/alias ID, measurement method; persistent/transient classification |
| `trials.csv` | trial/config ID, task/track/order/seed/role, method variant and loss, alpha/step/schedule, dimensions/depth/heap, nominal budget, actual peak state, presentations/unique IDs/passes, validation scores, status/failure, wall/CPU times |
| `checkpoints.csv` | trajectory/checkpoint, metric/partition/population/stratum, score/support, algorithm/preprocess/access/total bytes, registers, inference bytes, peak RSS/workspace, data and coordinate visits, update/transform/I/O/evaluation times |
| `predictions` shards | frozen ID/group/date references, trajectory/checkpoint, partition/population, signed score, prediction, label; compressed, no text; join primary/natural overlaps by ID |
| `selection.json` | chosen reference alpha, per-budget/checkpoint configs, candidate counts/costs, frontier choices, selection timestamp/hash before test, balanced-profile selections separately |
| `frontiers.csv` | task/method/track/order/M/metric/r, reference score, target, candidate budget and actual bytes/registers, k-star/M-star, unique/preprocessing access, mean/SD/interval, grid bounds, validation crossing/test confirmation, censoring status |
| `cost_ledger.csv` | preprocessing, tuning, final fitting, inference, bootstrap and artifact I/O costs; unique/supporting/presented observations; external bytes; cold/amortized distinction |
| `report.md` and figures | state–performance and state–sample Pareto frontiers, paired uncertainty, failures, no-access-budget substitutions, scientific interpretation and limitations |

Store scalar checkpoint metrics for all trials; retain detailed validation predictions for finalists and final trajectories, and test predictions only after the lock. Do not retain every large model/checkpoint or bootstrap replicate in RAM. Missing/failed metrics are null with explicit status, never zero. Report timing scopes so training-only speed is not presented as end-to-end speed.

## 8. Resource forecasts before implementation

The following historical measurements are read from accepted artifacts, **not new benchmarks**:

| Historical accepted run | Sum of timed stages | Peak process RSS |
| --- | ---: | ---: |
| CFPB baseline | 18.90 s | 581,779,456 bytes |
| CFPB batch compression | 114.10 s | 777,338,880 bytes |

Earlier hashing jobs took 7.30–7.71 s per feature budget for five passes over 13,459 rows, including three candidate models and validation. Text rehydration took 5.36 s and TF-IDF fitting approximately 2.9 s. These establish that data preparation is modest on the recorded machine. They do **not** establish throughput for binary squared-loss SGD, AWM, MISSION, strict per-trial accounting, or new code. The previous fits shared work and used a different objective.

Calculated storage anchors, not measured learner footprints:

- Full Ridge coefficients: `50,000 * 8 = 400,000` bytes, before object/configuration overhead.
- Primary train CSR at frozen float32: 1,563,584 nonzeros and 12,562,512 payload bytes. Casting values to float64 adds 6,254,336 bytes. No dense train matrix is needed.
- Vocabulary token UTF-8 payload 674,283 bytes, 50,001 int32 offsets 200,004 bytes, and float32 IDF 200,000 bytes total **1,074,287 bytes** before lookup structures, metadata, and overhead. The Python representation will be larger. Thus tiny A model budgets do not imply tiny total memory.
- A packed legacy-JL int32 mapping plus int8 signs alone is 250,000 bytes; normalization and model state add more. The existing NumPy representation can be larger. Packing must preserve exact projection behavior before any savings are credited.
- AWM/MISSION sketch counters cost `8*d*w` bytes in float64, plus active values/keys/index storage. In B, a key alone costs 16 bytes; a heap entry is not one scalar register. Exact byte capacity must be measured before assigning `h` and `w`.

The full main matrix has `10*2*4*8*2*5 = 6,400` final trajectories; checkpoints reuse trajectories rather than create new fits. Across all ten tasks, one pass presents 94,213 rows and visits 10,945,088 frozen nonzeros. At ten passes, final fitting costs **301,481,600 presentations per track**. The proposed tuning protocol costs **1,085,333,760 per track**, giving **2,773,630,720 presentations across A and B**, excluding references, diagnostics, preprocessing, evaluation and bootstrap. Infeasible cells lower this total; extra approved tuning raises it. Balanced A at three budgets adds at most **520,055,760** under the same tuning/final protocol. Report this development cost separately from the deployed learner's samples.

For the two-track main matrix, illustrative aggregate throughput scenarios of 50,000 / 5,000 / 500 examples per second imply **15.4 hours / 6.4 days / 64.2 days of training alone**. These are arithmetic scenarios, not measured or promised runtimes. At about 116 nonzeros per frozen row, 500 examples/s already entails roughly 58,000 input-coordinate visits/s before sketch depth, heap work, and regularization. B's raw features can have higher nnz. Python active-set implementations may be slower still. Evaluation at multiple checkpoints and extensive HPO add material overhead. A complete matrix is therefore not justified by the old two-minute compression run.

Forecast process memory for the serial prototype: **0.25–1.0 GiB** with bounded batches, one learner, and streamed prediction output, informed only by historical process peaks. This is not a guarantee. Planned algorithm budgets are 4–512 KiB; runtime libraries, harness data and transient storage explain the difference. Cached predictions for all final trajectories at all checkpoints could reach multiple GB on disk; keep shards on disk, do not concatenate them all in memory. Record actual disk usage and set a provisional 5 GiB new-output cap. No full `N x D` dense array or corpus-sized per-method hidden state is permitted.

Before expansion, replace these scenarios with pilot-measured per-method/track rates at multiple state budgets, measured peak state/RSS, and prediction-output sizes. Forecast each cell from its own update/transform/evaluation timings, with a conservative overhead allowance; do not extrapolate only the fastest method.

## 9. Staged implementation, validation, and stopping

**Stage 0 — implementation and correctness, only after approval.** Develop the isolated reader/state ledger, then hashing/SGD and sparse JL, then the two adaptive sketches. Pin author-code references and write an algorithm-conformance note. Approximate engineering effort is 4–8 focused working days for a defensible implementation and accounting, not a measured estimate: the bounded active-set implementations and allocation audit dominate. No new dependency installation is assumed; if author code needs a compiler/backend or dependency change, document the choice and main-environment impact before adopting it.

Required gates before performance experiments:

1. Reproduce frozen IDs, task labels and float32 feature values; verify source/fixture hashes and no held-out access during fitting. Reproduce each full-Ridge alpha's predictions, convergence and validation selection. Record score discrepancies and maximum numerical differences; investigate before proceeding.
2. Test row-wise updates against small dense reference calculations, including collisions, cancellation, empty documents, admission/eviction, equal-weight ties, duplicate coordinates, and bounded-capacity indexing. Verify AWM and MISSION's distinct prediction/update semantics. No dense oracle state in measured runs.
3. Verify chunk-size invariance where the advertised algorithm is per-example SGD; batching must not substitute a different gradient method. Check restart/resume equality including RNG, optimizer clocks, active indices, and lazy scaling.
4. Audit allocation at empty, full and replacement states. Averaging, tombstones and feature keys must fit budgets. Test that diagnostic/evaluation calls neither mutate learners nor advance training RNG. Test chronological first-pass access with a reader that rejects future rows.
5. Run existing appropriate regression checks in the main environment and new meaningful correctness tests; hash protected fixtures before/after. No Bank rerun is needed absent a change to shared code/environment, but any such proposed change requires a separate regression assessment. Never rewrite fixtures to pass.

**Stage 1 — smallest useful feasibility pilot, recommended next approval.** Two tasks: `credit_vs_checking` (near-balanced pair) and `mortgage_vs_rest` (imbalanced OVR). Four methods, A/B, random/chronological, seed 42, state ceilings 16/128/512 KiB, one pass with prefix checkpoints: at most **96 trajectories and 994,752 training presentations**. Use prespecified stable diagnostic configurations, not claim tuned best-method rankings or k-star. Add full Ridge reproduction, full-dimensional SGD and legacy-JL compatibility checks separately; evaluate validation only, keeping new-method test locked. The pilot's purpose is fidelity, memory feasibility, numerical stability, measured throughput, and an honest full-study cost forecast.

For 994,752 presentations, the same illustrative 50,000/5,000/500 samples/s scenarios give 20 seconds / 3.3 minutes / 33.2 minutes of updates, before setup/evaluation. Budget **up to 60 minutes wall time for the performance pilot**, with a checkpoint after each trajectory and a report even if incomplete. The cap is a stopping rule, not a prediction that all runs will finish. Correctness tests and development time are separate. No algorithm may be replaced by an easier approximation to meet this cap.

**Stage 2 — pilot review and a tractable primary study.** Choose a costed scope using the measured rates. Prefer completing all four methods on the two pilot tasks with real validation tuning, five evaluation seeds and the memory/sample frontier before expanding to ten tasks. This requires another approval. Preserve both first-pass and replay results; do not quietly drop a slow method or the causal track and then claim a complete comparison. A useful negative finding is that preprocessing dominates total state, or that an adaptive method is too costly at the allowed access budget.

**Stage 3 — full matrix / balanced profile / optional references.** Only if Stage 2 provides resolvable frontiers and an approved resource budget. No automatic expansion, no new population, and no reused batch compression point labeled as streaming state.

At every stage, preserve the existing safeguards: serial execution, single-thread numerical libraries, resource checks before stages, at least 4 GiB available to start and 2 GiB while running, process RSS at most 4 GiB, and stop on rising swap I/O for three successive at-least-one-second samples. Checkpoint completed outputs atomically and never overwrite accepted artifacts. Stop on source/hash mismatch, future access, state-budget breach, invalid numerical/reference behavior, or corrupt checkpoint; record the failure. Individual divergent hyperparameter trials fail explicitly, without silent clipping. Stop expansion when projected runtime exceeds its approved cap, output exceeds its storage cap, or tuning/search uncertainty prevents a meaningful conclusion.

**Current stopping point:** planning only. No experiment, model fit, dependency change, QOS work, or push has been performed for this proposal. Approval is requested for the plan and, preferably, only Stage 0 plus the bounded Stage 1 pilot; later stages remain separate decisions.
