# CFPB sparse-text QOS benchmark specification

**Version:** 1.0 — frozen design, 2026-10-14. **Status:** specification only; no benchmark results exist.

This document defines the experiment to implement after separate authorization. Changes to these rules require a new specification version and a new run directory. Do not alter the validated Bank methodology, PaySim analysis, source CSV, or historical results.

## 1. Scientific question and scope

Measure how classical feature compression affects financial complaint classification, predictive quality, measured classical resources, and conditional theoretical QOS accounting. This is a financial analogue of sparse text classification, not fraud detection, verification of allegations, or a demonstration of quantum advantage.

The input is **only `Consumer complaint narrative`**. The target is the consumer-selected `Product`. No structured field may enter the feature vector, feature selection, weighting, or model. Dates and IDs serve curation and splitting; labels serve conflict auditing and supervised learning. Company and other metadata may support descriptive audits only. Text cleaning must never depend on the example's product label or other metadata value.

### Source identity

- File: `data/CCDB_Export_15_January_2026_through_February_2026.csv`.
- SHA-256: `664f1cae8d3a19dfd6894e2ca760f5b3f0bb4b71e466f38235473e1ea5b29f6d`.
- Onboarding counted 1,038,383 complaints and 34,351 non-empty narratives (3.3081242663%), received January 1–February 28, 2026.
- Record actual acquisition provenance if available; otherwise mark it unknown. Do not infer acquisition or snapshot date from the filename.
- A hash mismatch blocks this specification's run until the input change is investigated. Never replace this source silently.

### Primary classes and fixed label order

| Label | Exact source product | Onboarding narrative rows |
| --- | --- | ---: |
| 0 | Debt collection | 11,752 |
| 1 | Credit card | 5,771 |
| 2 | Checking or savings account | 5,771 |
| 3 | Mortgage | 2,108 |

The raw primary population is 25,402 rows. **This is not the final curated N.** Do not merge prepaid cards into credit cards. Consumer reporting is excluded from v1's modeled tasks because of exceptional narrative-selection bias and duplication; a five-class extension requires a separate specification amendment.

## 2. Chronological partitions

Use `Date received`, parsed as calendar dates, with inclusive bounds:

| Partition | Dates | Raw primary rows before curation |
| --- | --- | ---: |
| Train | 2026-01-01 through 2026-01-31 | 16,327 |
| Validation | 2026-02-01 through 2026-02-14 | 3,735 |
| Test | 2026-02-15 through 2026-02-28 | 5,340 |

Reject missing/invalid dates or IDs into an audit table. Unexpected dates outside this interval, duplicate complaint IDs, or contradictory records for one ID block release pending investigation. IDs must parse as integers. Sort deterministically by `(Date received, integer Complaint ID)`; do not use CSV order.

No random CV is the primary evaluation. Fit vocabulary, document-frequency thresholds, IDF, training class weights, and estimators on **curated January training data only**. Validation chooses eligible hyperparameters and operating points. Do not refit on January plus validation in v1: keeping the January vocabulary and model makes the reported resource and population comparisons consistent.

Freeze the run manifest, curation policy, hyperparameters, and validation selections before computing test predictions. Test is used once per frozen run to evaluate the prespecified comparisons, not to choose classes, dimensions, seeds, or methods. Failed or negative test outcomes do not authorize retuning on test.

## 3. Text normalization and artifact policy

Retain source text unchanged. Apply the following to private working strings; log exclusions by complaint ID and reason, without publishing raw narratives in result tables.

1. Treat null/empty/Unicode-whitespace-only narratives as unavailable; do not turn nulls into the literal word `nan`.
2. For the **raw normalized duplicate key**, apply Python Unicode `str.lower()`, replace each maximal Unicode whitespace run (`re` pattern `\s+`) with one ASCII space, and strip leading/trailing whitespace. Do not perform Unicode compatibility normalization, accent stripping, punctuation removal, stemming, number masking, or spelling correction.
3. Flag structured-heading narratives using this case-insensitive pattern against the raw normalized text:

   ```text
   (?<!\w)(?:product|sub[ -]?product|issue|sub[ -]?issue|company|company response(?: to consumer)?|complaint id|date received|date sent to company|submitted via)\s*:
   ```

   Quarantine the **entire flagged narrative** from primary and secondary scored populations. This conservative rule can exclude legitimate prose; report its class/time counts. It avoids guessing where a copied field block ends, and does not compare a heading's value with the true label. Do not describe every flag as proven leakage. No manual label-informed rewriting or exceptions in v1.
4. For the **model text**, replace standalone runs of two or more `x` characters (`\bx{2,}\b` after lowercasing) with a space, then collapse whitespace again. This removes redaction tokens without creating a replacement-token feature. Keep numbers, punctuation, ordinary financial terms, and company names embedded in otherwise eligible prose. Do not strip words such as `mortgage`, `credit card`, or `collection`.
5. Quarantine text that is empty after cleaning or has no tokens under the vectorizer token pattern `(?u)\b\w\w+\b`. Do not impose a length cutoff, truncate long narratives, translate, or silently discard a detected language. Record character/token lengths and language-review flags; v1 has no automated language filter.

Numeric values, dates, and redactions can still encode artifacts; their effects remain limitations to report. Any additional cleaning is a new version, not an unrecorded fix. The onboarding measurements retained redaction tokens and structured-heading rows, so they must not be reused as post-cleaning regression targets.

## 4. Exact duplicates, conflicts, and near-template groups

### Audit universe and exact groups

Build the raw normalized duplicate audit across **all 11 product categories with narratives**, before primary-class selection. A normalized group with more than one distinct non-empty source `Product` is conflicting: quarantine every member from both scored populations. Do not majority-vote, relabel, or keep only the desired class. Missing product labels are ineligible and must be counted.

For the remaining artifact-eligible rows, group identical model text as well. Removal of redaction tokens can make previously different strings identical. Quarantine conflicting-label model-text groups using the same rule. Record exact-original-text duplicates, raw-normalized duplicates, and model-text duplicates separately.

### Near-template definition

- Tokenize model text into ordered Unicode word tokens with `(?u)\b\w+\b`; do not remove stop words for similarity measurement.
- Represent each text by its **set of consecutive word trigrams**.
- Compare distinct model-text groups only when both have at least 20 distinct trigrams. Shorter texts retain exact duplicate handling; record this near-template coverage limitation.
- Define an undirected edge when exact Jaccard similarity is at least 0.85. Compare integers (`20 * intersection_size >= 17 * union_size`) to avoid boundary rounding.
- Near-template groups are connected components, including singleton model-text groups. Transitive chains count as one component even if two endpoints have similarity below 0.85; report component sizes and chaining implications.
- Production curation must find all qualifying edges under these rules, using a verified lossless candidate procedure or exhaustive comparison with safe bounds. Approximate LSH may accelerate exploration but cannot silently define the frozen groups. The onboarding capped LSH results are lower bounds, not the benchmark grouping manifest. If complete grouping is not locally feasible, stop and propose a versioned change rather than silently capping candidates.
- Construct this graph across eligible narratives from all source products. If a component contains different product labels, quarantine the **whole component**, including desired-class rows. Never split a component by its labels.

### Primary earliest-occurrence policy

For each non-conflicting component, keep exactly its earliest eligible complaint by `(Date received, integer Complaint ID)`. Include it only if its product is one of the four primary labels. Assign it to its actual received-date partition. All later members are excluded from the primary scored population and logged, even if they occur in another partition. Never move later examples backward into training. Primary group IDs must be disjoint across train, validation, and test.

Store stable group IDs from SHA-256 of sorted member complaint IDs, plus a curation-version identifier. Record original and retained dates, group size, conflict status, exclusion reason, and representative ID. Do not use group IDs as predictors.

This global conflict/group audit uses future text and labels for **offline benchmark curation**. It is not an online deployment algorithm and must not be described as fully prospective data preparation. It does not authorize fitting vocabulary or IDF on future data. Test labels are used only for the prespecified audit and final scoring, not policy adjustment.

### Secondary natural-stream evaluation

Keep all artifact-eligible, non-conflicting rows of the four classes in the February validation/test windows, including later members of non-conflicting duplicate/template groups. Evaluate **the same primary-trained, frozen models**; do not train a second model or tune against these secondary scores.

Report overall natural-stream scores and separate strata: groups seen in January training, groups seen earlier only in validation, and groups first seen in test (using mutually exclusive assignment in that order). Preserve chronological dates and natural class frequencies. This measures performance on repeated as well as novel eligible complaints; it is not an evaluation of all CFPB complaints or quarantined text. Primary claims remain based on the group-disjoint population.

## 5. Representation contracts

### Primary TF-IDF

Use the following explicit `TfidfVectorizer` settings on the model text:

```text
analyzer="word", ngram_range=(1, 2), max_features=50000,
min_df=2, max_df=1.0, stop_words="english",
lowercase=True, strip_accents=None,
token_pattern=r"(?u)\b\w\w+\b",
binary=False, sublinear_tf=False, use_idf=True, smooth_idf=True,
norm="l2", dtype=float32
```

Use the pinned environment's English stop-word list and record its digest. Vocabulary selection, term-frequency ranking under the feature cap, and IDF all use primary training rows only. Store the ordered vocabulary, IDF, parameters, and their hashes. Transform validation, test, and secondary rows without refitting. Report actual D, never assume that the cap is reached.

Keep canonical CSR: sum duplicate coordinates, eliminate explicit zeros, sort indices, and count mathematical nonzeros. Log zero-feature rows by partition. Do not silently drop those rows after splitting; score them and report their prevalence, subject to the stopping gates below.

### Unigram reference-style control

Use the same curation, partitions, and TF-IDF defaults except `ngram_range=(1, 1)`, `min_df=1`, and `max_features=None`. Retain English stop words, L2, and float32. This follows the local IMDb reference's full-feature representation most closely. Run unweighted Ridge with alpha 10 (IMDb-style) and alpha 1 (Newsgroups-style), explicitly labeled as separate fixed controls.

This is representation/classifier alignment, not exact paper reproduction: CFPB is multiclass, curated, chronological, and train-only vectorized. The reference scripts combine source train/test data, vectorize globally, and use `cv=5`; do not reproduce that leakage in this primary benchmark. Local reference commit: `a3fe8d6fe777750749f5e90131180e016088bdaf`.

### Vocabulary reduction

Fit separate training-only primary-style vectorizers with caps `[20000, 50000, 100000]`, all with `min_df=2`. The 50k run is the primary baseline, not a duplicate run. Add one fixed rare-feature control using cap 50k and `min_df=5`. Record actual vocabularies and D: a capped vocabulary can change membership as `min_df` changes, so nnz need not be monotonic.

## 6. Dtype and numerical policy

- Primary TF-IDF, model-input matrices, and sketch outputs are **float32 CSR**. Never densify the full document-feature matrix. Index dtype is recorded; use int32 when safe and int64 when required, without overflow.
- Existing sketch functions may construct float64 projections and upcast sparse inputs. For CFPB, an explicit caller-side cast back to float32 is required after sketching, followed by canonicalization. Include casting time, transient memory, and post-cast nnz in measurements. This is a documented CFPB policy, not a change to Bank helpers.
- Identity endpoints retain identical CSR values and dtype. Use the same full input and classifier settings as the corresponding uncompressed baseline. Do not introduce solver differences through dense/CSR conversion.
- Ridge uses `solver="auto"` under the pinned environment. Record the actual solver where exposed; otherwise mark it unresolved rather than infer it. Solver/format/dtype changes require an explicit new comparison.
- Required numerical sensitivity control: recompute January TF-IDF weights in float64 using the **frozen primary vocabulary**, then transform and fit the full unsketched unweighted and balanced Ridge configurations in float64. Keep this control separate; do not select the reporting dtype from test scores. Record prediction disagreement and metric differences versus float32. Float64 doubles value payload, not necessarily total CSR memory.
- No CFPB score regression targets exist yet. Once established, preserve fixtures and apply repository score tolerance `rtol=0, atol=1e-8` to reproducibility checks within the same frozen configuration. Float32-versus-float64 controls are sensitivity comparisons, not required to agree within that tolerance. Investigate numerical drift; never widen tolerance to conceal a failure.

## 7. Models, compression controls, and seeds

All models include an intercept. No resampling, synthetic examples, target encoding, or class-dependent text filtering. Balanced weights use **primary training label frequencies only**. No test-time threshold tuning; predict the class with the largest decision score, breaking exact ties by the fixed label order.

| Method | Frozen configuration / role |
| --- | --- |
| Full sparse Ridge | `alpha=10`, `solver="auto"`, `class_weight=None`, `tol=1e-4`, `random_state=42` |
| Balanced sparse Ridge | Same, `class_weight="balanced"` |
| Linear SVM | `LinearSVC`, `penalty="l2"`, `loss="squared_hinge"`, `dual="auto"`, `class_weight="balanced"`, `tol=1e-4`, `max_iter=20000`, `random_state=42`; select C from `[0.1, 1, 10]` by validation macro-F1 |
| Vocabulary controls | Both fixed Ridge configurations on the representations in section 5 |
| Balanced bucketing | Existing balanced assignment and occupancy normalization, sparse path; both fixed Ridge configurations |
| Sparse JL | Existing balanced signed projection, occupancy normalization, **sparsity=1 only**; both fixed Ridge configurations |
| Streaming hashing | `HashingVectorizer`, same model text/tokenization/English stop words and uni+bigrams, `alternate_sign=True`, `binary=False`, `norm="l2"`, float32; no fitted IDF |
| Incremental classifier | `SGDClassifier(loss="log_loss", penalty="l2", learning_rate="optimal", fit_intercept=True, average=False, random_state=42, shuffle=False)` via `partial_fit`; alpha in `[1e-5, 1e-4, 1e-3]` selected on validation |

Streaming hashing uses dimensions `[16384, 32768, 65536, 131072]` and chronological training order, batches of 1,024, exactly five passes. Apply explicit sample weights from January class counts; do not use an unsupported automatic balanced-weight option in `partial_fit`. Supply all four classes on the first batch. Record that training-class counting and repeated passes require replay; this is not a strict one-pass, unknown-class-frequency claim. Do not perform early stopping on test. Compare candidate `(hash dimension, alpha)` pairs on validation.

For bucketing and JL, use sorted unique dimensions from `[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]` that are strictly below actual primary D, then append D as the identity endpoint. Use sketch seeds **`[42, 123, 456, 789, 2026]`**. Apply the same projection to every partition. Run the identity once per classifier and reuse its recorded result; do not count deterministic copies as five independent observations.

Use seed **42** for any other randomized diagnostic or sampling operation. Primary curation/splits contain no random sampling. Run one experiment worker at a time; pin BLAS/OpenMP threads to one for the reproducibility run and record those settings. Hashing's mapping is fixed by the pinned implementation; do not invent a hash seed API.

SVM and streaming selection ties: largest validation macro-F1, then balanced accuracy, then smaller C for SVM; smaller hash dimension then larger alpha for streaming. Use full-precision recorded metrics. No other hyperparameter search is authorized by v1.

For each Ridge/sketch family, retain the entire prespecified curve. Select one validation operating point as the smallest k with mean seed-level macro-F1 within **0.01 absolute** of that family's full primary-TF-IDF Ridge baseline; if none qualifies, select the uncompressed endpoint. Test all prespecified curve points for descriptive comparison, but never select a new winner from test. Do not select a favorable sketch seed.

These are classical sketch experiments with theoretical QOS accounting. They do not execute a quantum classifier. A binary CFPB companion is not part of v1; it requires its own task/specification amendment.

## 8. Metrics and aggregation

Compute the following for every scored configuration and eligible evaluation population:

- Macro-F1, with all four fixed labels and `zero_division=0`.
- Balanced accuracy (unadjusted mean class recall).
- Ordinary accuracy.
- Per-class precision, recall, F1, and support.
- Confusion matrix in fixed label order, with raw counts and row-normalized proportions.
- Majority-class baseline, predicting the most common **training** class (ties by label order), scored on each population.

Every primary partition must retain all classes; missing classes block release. Secondary strata with absent classes must flag which metrics are undefined or not comparable: report per-class support and leave four-class balanced accuracy undefined rather than silently average over a different class set.

For sketch curves, report the arithmetic mean and **sample standard deviation (`ddof=1`) across the five seed-level holdout metrics**. Deterministic singleton controls have SD recorded as null, not zero. Do not flatten scores across partitions, pool seeds into one classifier, call seed SD a confidence interval, or adopt the reference's flattened-score SEM convention. Statistical confidence intervals are outside v1; report group counts and dependence limitations instead.

Record fit time, vocabulary/IDF fit time, transform time, sketch/cast time, prediction time, total passes, peak process RSS, CSR payload, model parameter payload, and any measured temporary storage. Unknown measurements remain null with an explanation. RSS is not equivalent to matrix payload. Measure test scoring without fitting anything.

## 9. QOS accounting contract

For each actual training representation, report:

```text
N_train, D_actual, nnz_train, s_row, s_column,
s_reference = max(s_row, s_column)
Q_row = 2*ceil(log2(N_train + 2*D_actual)) + ceil(log2(s_row + 1)) + 4
Q_reference = 2*ceil(log2(N_train + 2*D_actual)) + ceil(log2(s_reference + 1)) + 4
```

Use `src/machine_size.py` where compatible, without changing its Bank defaults. Count mathematical nonzeros after canonicalization and the final dtype cast. Feature dimension is actual output D, not requested k when an endpoint differs. Do not use average row nnz as maximum sparsity. Record each sketch seed separately because cancellation can change nnz and sparsity.

Also report representation statistics for validation/test independently, but **never add holdout rows to N_train** to describe a trained model's resource estimate. Any all-row descriptive accounting must be separately labeled.

Label `Q_row` as the **Bank row-only convention** and `Q_reference` as the **reference row-or-column convention**. Both are conditional theoretical LS-SVM resource accounting. The +4 represents three logical qubits plus one classical scalar register under the reference's convention. They are not measured hardware qubits, bytes, or directly comparable units to classical CSR memory.

The scalar LS-SVM formula describes a binary/one-vs-rest subproblem. For four classes, report it as a **per-subproblem proxy**, not total multiclass quantum cost. Do not silently multiply by four or claim total time/memory without an execution/access model. QOS sample access, text parsing, vocabulary/IDF construction, random access/replay, and output handling remain explicit assumptions/costs. Applying the formula to a hashing matrix is descriptive only; it is not the resource usage of SGD.

Onboarding's uncurated 50k representation had training N=16,327, nnz=1,972,014, s_row=1,325, s_column=12,876, and estimates 49/52. **These are historical feasibility observations, not targets for this curated and artifact-cleaned benchmark.**

## 10. Output and provenance contract

Future runs write only to a fresh directory:

```text
validation/cfpb/<UTC-YYYYMMDDTHHMMSSZ>-<unique-run-id>/
```

Never overwrite a run or write CFPB outputs into validated Bank artifacts. No run directory or result files are created by adopting this document.

Required outputs:

| Filename | Required content / row unit |
| --- | --- |
| `manifest.json` | Spec version/hash, source path/hash, actual run UTC time, root/reference commits or null, environment/package versions, hardware, seeds, threads, exact model/vectorizer settings, execution status |
| `population_counts.csv` | Stage × product × partition counts, availability, exclusions, retained groups, and zero-feature rows |
| `daily_class_counts.csv` | Received date × product × raw/curated population counts |
| `curation_audit.csv` | Complaint ID, dates, product, raw/model-text hashes, group/component ID, representative ID, conflict/artifact flags, primary eligibility, exclusion reasons; no raw prose |
| `duplicate_groups.csv` | Exact/normalized/template group ID, member count, date span, partition span, distinct-label count, disposition |
| `near_template_edges.csv` | Model-text group pair, intersection/union counts, exact Jaccard, component ID; candidate-search method/completeness in manifest |
| `split_manifest.csv` | Eligible complaint/group ID, label, partition, primary/natural-stream membership, seen-group stratum; no feature metadata |
| `vectorizer_manifest.json` | Ordered vocabulary/IDF artifact names and hashes, fitted row-ID hash, parameters, D, stop-word digest, dtype |
| `vocabulary.csv` / `idf.npy` | Primary feature index/token and matching IDF; controls use uniquely named equivalents. Treat vocabulary as derived text, not privacy-cleared publication content |
| `representation_stats.csv` | Configuration × partition × seed: N, actual D, nnz, density, mean/max row nnz, max column nnz, zero rows, CSR/index dtype, CSR bytes, dense-equivalent bytes |
| `metrics_per_run.csv` | Configuration × evaluation population × seed: all aggregate metrics, model, representation, requested k, actual D, split, convergence/status |
| `per_class_metrics.csv` | Configuration × population × seed × fixed class: precision, recall, F1, support |
| `confusion_matrices.csv` | Configuration × population × seed × true/predicted label: count and row proportion |
| `sweep_summary.csv` | Method × model × k × population: seed count, metric means/sample SD, explicitly identified resource summaries |
| `resource_accounting.csv` | Training configuration × seed: both sparsity conventions and Q estimates, per-subproblem designation, accounting assumptions |
| `runtime_memory.csv` | Configuration × stage × seed: elapsed seconds, peak RSS, payloads, passes, thread settings, measurement method |
| `validation_selection.json` | Frozen validation-selected SVM/streaming settings and sketch operating points; timestamp/hash before test scoring |
| `dtype_sensitivity.csv` | Fixed float32/float64 controls, metric deltas, prediction disagreement, solver/dtype and resource differences |
| `experiment_log.csv` | One row per reported metric/accounting quantity using ledger schema below |
| `benchmark_report.md` | Coverage/curation impacts, empirical outcomes, both QOS conventions, limitations, stopping-gate outcomes, explicit absence of quantum-advantage claims |

Use a unique `configuration_id` throughout. Include `run_id`, `spec_version`, `population` (`primary`/`natural_stream`/named stratum), `partition`, and `seed` where applicable. Missing/undefined fields are null, not invented values. Serialize numeric precision sufficient for future regression comparisons.

### Experiment ledger

Preserve the established columns:

```text
experiment_id,dataset,date,git_commit,preprocessing,n_samples,n_features,
max_sparsity,sketch_method,sketch_dimension,seed,model,metric,result,notes
```

- `dataset`: `CFPB Jan-Feb 2026 primary4`, with population identified in notes.
- `date`: actual execution date, not complaint date or onboarding date; `git_commit` is blank if unavailable.
- `n_samples`: training N; notes include evaluation N and partition. `n_features`: pre-sketch vocabulary D; `sketch_dimension`: actual compressed D, blank for uncompressed models.
- `max_sparsity`: row maximum of the actual training representation, retaining the ledger's existing convention. Put column maximum and reference convention in notes and `resource_accounting.csv`.
- `seed`: actual randomized sketch/model seed; blank for deterministic representations with no applicable seed, not a fictitious replicated run.
- `metric`: explicit names such as `macro_f1`, `balanced_accuracy`, `accuracy`, `qos_size_row_only`, `qos_size_row_or_column`. Resource rows must state that they are theoretical per-subproblem accounting.
- `notes`: run/configuration ID, spec/source hashes, split, population, vocabulary hash, dtype, solver, training/evaluation sizes, seeds' role, source output filename, and limitations.

Write the run-local ledger first. Do not append to `results/experiment_log.csv` during specification work; any later publication of ledger entries must preserve existing rows and reference a validated run.

## 11. Release gates and stopping criteria

### Block implementation results from being accepted

- Source identity mismatch, invalid partition membership, future-fitted vocabulary/IDF, duplicate-group overlap in primary partitions, or incomplete near-template grouping.
- Unresolved conflicting labels, missing classes, or fewer than **1,000 primary training examples total**, **200 training examples per class**, or **100 validation/test examples per class** after curation. These are minimum feasibility gates, not claims of adequate statistical power.
- More than **1% zero-feature rows in any primary partition**, undocumented manual curation, or unexplained changes to counts/parameters.
- Nonfinite features/scores, failed convergence, silently changed solver/dtype, or irreproducible seeded transforms. A failed configuration is not assigned a successful score; any remedy is documented before final test evaluation.
- Before a heavy stage, available RAM below 4 GiB: defer the stage. Stop a pilot if process RSS exceeds 4 GiB, available RAM falls below 2 GiB, or sustained memory pressure/swap activity is observed. Record monitoring limitations. Do not solve memory pressure by densifying, omitting costs, or relying on swap.

If curation removes more than 30% of any primary class, or any template component contains more than 10% of that class, pause for a documented validity review before models are fitted. Do not weaken the duplicate threshold simply to retain examples.

### Deprioritize the scientific benchmark or compression claim

- If the best validation-selected full classifier improves macro-F1 by less than **0.05 absolute** over the training-majority baseline, investigate label/text adequacy and deprioritize expensive sweeps if no independent validity explanation is found.
- If audited artifact/template exclusions are essential to apparently strong results, report the dependence and do not promote the raw-population result as robust evidence.
- If no compressed setting with `k <= D/4` stays within **0.01 absolute validation macro-F1** of its corresponding full Ridge baseline, retain the negative result and deprioritize further compression sweeps. Do not enlarge tolerances on test.
- If compressed runs do not improve measured memory or runtime despite reduced D, report no demonstrated classical resource benefit. Do not substitute dense-equivalent memory or theoretical QOS scaling for measured evidence.
- If high D is mainly rare artifacts or the effective curated corpus becomes too small/repetitive to test a stable tradeoff, seek another corpus rather than inflating dimensions with identifiers.
- Failure to outperform an inexpensive hashing/streaming baseline is a valid outcome. A passing regression or favorable QOS accounting estimate never establishes practical quantum advantage.

## 12. Required implementation review, not implementation authorization

Before creating the experiment, review the curation counts and exact near-template grouping cost under this policy, then confirm the output contracts and stopping gates can be met. Existing Bank evaluation helpers default to shuffled CV and two metrics; those defaults do not implement this CFPB protocol. Any necessary future extensions must be explicit and preserve Bank regressions.

The onboarding identified a promising high-dimensional sparse regime, but all post-curation N/D/nnz/sparsity values, model quality, and training resource measurements remain unestablished. This specification authorizes none of those experiments by itself.
