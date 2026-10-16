# CFPB curation and sparse text preparation

This implements the curation/text stage of [benchmark specification v1.0](cfpb_benchmark_spec.md). It does not train CFPB classifiers, run sketches, change Bank preprocessing, or create a notebook.

From the repository root:

```sh
.venv/bin/python -m src.cfpb_prepare
```

The runner verifies the frozen source hash and creates a fresh directory under `validation/cfpb/`. It scans 100,000 source rows at a time, retains narrative-bearing rows, audits **all products before class selection**, and fits both TF-IDF vocabularies/IDFs only on curated January training rows. It does not persist dense or sparse feature matrices; rebuild them from the immutable source, split manifest, frozen configuration, and saved vocabulary/IDF metadata.

## Modules

- `src/cfpb.py`: frozen normalization/artifacts, source validation, chronological splits, complete template grouping, conflict quarantine, primary and natural-stream membership, and release gates.
- `src/text_preprocessing.py`: generic explicit TF-IDF factories, training-only fit/heldout transform, canonical float32 CSR, and sparse representation/accounting statistics.
- `src/cfpb_prepare.py`: streaming input, source/provenance checks, resource monitoring, fresh output directories, audits, summaries, vocabulary/IDF exports, and text-only reports.

Importing these modules does not set global thread environment variables. The command-line runner explicitly constrains numerical threads to one. Resource monitoring checks available RAM and process RSS; when the OS exposes swap counters, three successive increases sampled at least one second apart stop the run. Unavailable swap monitoring is disclosed in the manifest.

## Completeness of near-template grouping

For each distinct eligible model text, form the exact set of Unicode word trigrams. Sets smaller than 20 have exact-duplicate handling only, as specified. All other sets share one deterministic token ordering: document frequency, then lexical trigram order.

For a set of size `n`, index its first `n - ceil(17*n/20) + 1` tokens. A pair with Jaccard at least 17/20 must share a prefix token: its intersection has at least `ceil(17*n/20)` tokens in each set, so the globally earliest shared token cannot occur after either prefix. Process sets by increasing size and reject previous sets of size `m` only when `20*m < 17*n`, a necessary size bound. Verify every remaining candidate using exact set intersections and `20*intersection >= 17*union`.

There are no LSH approximations, collision-based equality decisions, candidate caps, or random seeds in grouping. Connected components include transitive chains. Tests compare the complete edge set with exhaustive enumeration, including threshold boundaries, short sets, transitive groups, and cross-class conflicts.

## Validated text-stage output

The accepted run is:

`validation/cfpb/20261014T192947Z-7b6668f9/`

It retained **20,705** primary rows: **13,459 train**, **3,033 validation**, **4,213 test**. Primary training TF-IDF has D=50,000 and nnz=1,563,584; the unigram control has D=21,514 and nnz=950,946. Both are float32 CSR with no zero-feature primary rows. The run took approximately 48 seconds and peaked around 1.48 GiB RSS. These are text-preparation measurements, not classifier resource requirements.

Two corrected full runs produced byte-identical audit/group/split/count/vocabulary/IDF artifacts; representation statistics match after excluding the run identifier. An earlier run (`20261014T192547Z-c143a6c6`) is marked `invalid_reporting_bug` because its class-count reports accidentally accessed the pandas `product` method. That reporting bug was fixed and covered by an end-to-end test; do not use the invalid reports.

### Modeling remains gated

The specification requires review before modeling if more than 30% of a primary class is removed or a template component exceeds 10% of that class. Both gates triggered for Debt collection:

- 11,752 raw narratives → 7,622 retained: **35.14% removed**.
- Largest component: 1,576 rows, **13.41%** of the raw class.

Grouping and sparse text validation completed; these flags are not algorithm failures or permission to weaken curation. `release_gates.json` records `modeling_permitted=false`. Resolve the documented validity review before any classifier fitting, which was outside this implementation task in any event.

## Audit and accounting interpretation

`curation_audit.csv` maps every original source row to a reason or primary membership. Flags for downstream checks on empty/invalid rows may be blank because those checks were not performed. Narrative-bearing rows have exact, normalized, model-text, and template-group identifiers where applicable. `split_manifest.csv` includes the secondary natural-stream membership and seen-group strata, without using metadata as predictors.

Exclusion reasons are mutually exclusive by stage precedence; flag counts and duplicate-stage counts are not generally additive. Full-population conflict audits include non-primary products. Global future text/labels inform offline curation, never vocabulary/IDF fitting. This is not a prospective online curation claim.

`representation_stats.csv` contains actual N/D/nnz, row/column sparsity, density, dtype, and payload for each partition/population. `resource_accounting.csv` and the run-local ledger report training-only theoretical binary-subproblem QOS quantities: primary 49 row-only / 52 row-or-column, unigram 46 / 50. These are neither measured hardware memory nor complete multiclass costs nor evidence of quantum advantage.

Vocabulary files contain derived text and are not privacy-cleared publication artifacts. No raw complaint prose is written to the audit tables. Existing `results/experiment_log.csv` and all validated Bank files remain unchanged.

## Tests

```sh
.venv/bin/python -m pytest -q tests --ignore=tests/test_regression.py
.venv/bin/python -m pytest -q tests/test_regression.py
.venv/bin/python -m pip check
```

The end-to-end synthetic test verifies complete-source membership mapping, class counts, training-only transformations, immutable input, and byte-identical deterministic artifacts across fresh runs. The small fixture explicitly bypasses real-population minimum-support gates; real runs never bypass those gates.
