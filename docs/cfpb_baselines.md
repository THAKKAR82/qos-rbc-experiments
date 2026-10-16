# CFPB uncompressed baselines

The accepted first modeling run is [`20261014T204807Z-2270924a`](../validation/cfpb/20261014T204807Z-2270924a/benchmark_report.md). It uses unchanged [specification v1.0](cfpb_benchmark_spec.md), the accepted January-only preparation, and [Recommendation B](cfpb_curation_validity_review.md). No compression or new curation is implemented.

From the repository root:

```sh
.venv/bin/python -m src.cfpb_baselines
```

This command creates a fresh run, verifies source/preparation/code/environment fingerprints, reconstructs January vocabulary/IDF exactly, fits uncompressed models, freezes primary-validation choices, then evaluates test and natural-stream populations. It never overwrites preparation artifacts or publishes ledger entries automatically. A run ends `completed_pending_validation`; accept and publish only after regression, artifact-integrity and result-consistency checks. A fresh rerun is a reproducibility run, not permission to retune after seeing test scores.

`src/holdout_evaluation.py` provides explicit Ridge/SAGA/LinearSVC factories, fixed-label metrics, sparse-input checks, coefficient-array accounting and validation-only selection. `src/cfpb_baselines.py` owns CFPB artifact loading, row-order verification, representation reconstruction, resource monitoring and the staged execution. Existing Bank evaluation and preprocessing modules are untouched.

The only search is the frozen balanced-SVM C grid `[0.1, 1, 10]`. User-authorized additions are fixed-C=1 unweighted LinearSVC and unweighted/balanced SAGA logistic regression; no logistic search. Fixed controls include unigram Ridge alpha 10 and alpha 1, a January-majority classifier, and separately recomputed float64 primary-vocabulary Ridge. Weighting uses January labels only. Both date boundaries and curation exclusions remain frozen.

Key accepted outputs:

- [`baseline_results.csv`](../validation/cfpb/20261014T204807Z-2270924a/baseline_results.csv): tidy population/partition metrics and resource measurements. `validation_N` and `test_N` give primary holdout counts; `evaluation_N` gives the actual row's population/stratum support. `nnz` and `input_csr_bytes` describe training input. Times are seconds, memory fields are bytes.
- [`baseline_summary.csv`](../validation/cfpb/20261014T204807Z-2270924a/baseline_summary.csv): primary validation/test comparisons; nonselected SVM candidates have validation-only scores.
- [`validation_selection.json`](../validation/cfpb/20261014T204807Z-2270924a/validation_selection.json): timestamped choices hashed before test predictions.
- [`per_class_metrics.csv`](../validation/cfpb/20261014T204807Z-2270924a/per_class_metrics.csv) and [`confusion_matrices.csv`](../validation/cfpb/20261014T204807Z-2270924a/confusion_matrices.csv): fixed label order, supports, counts and normalized confusion values.
- [`natural_stream_comparison.csv`](../validation/cfpb/20261014T204807Z-2270924a/natural_stream_comparison.csv): sensitivity deltas for the same January-trained models. Stratum metrics are in the main table; absent-class balanced accuracy is undefined and flagged.
- [`dtype_sensitivity.csv`](../validation/cfpb/20261014T204807Z-2270924a/dtype_sensitivity.csv), [`runtime_memory.csv`](../validation/cfpb/20261014T204807Z-2270924a/runtime_memory.csv), and [`validation_checks.json`](../validation/cfpb/20261014T204807Z-2270924a/validation_checks.json).

Balanced SVM C=1 won validation (macro-F1 0.900900); its test macro-F1 is 0.903611. Balanced logistic had the highest observed test macro-F1, 0.908258, without replacing the validation winner. Sparse primary fits took approximately 0.23–1.02 seconds. The full report discusses class weighting, the difficult Credit card class, unigram controls, repeated-template strata and the limited practical resource savings available when full models are already inexpensive.

The initial run `20261014T204643Z-9fd78bfc` is retained as a reproducibility comparison, not the accepted timing source: its timing interval included monitor-thread startup/shutdown. The corrected run reproduced all predictions, metric/confusion outputs and exported coefficient arrays exactly. No hyperparameters changed. Absolute RSS includes resident representations/models and is not incremental estimator memory; sampling can miss brief spikes, and OS swap counters were unavailable.

Validation: 133 fast tests (including the 34 CFPB/text/holdout tests), nine Bank regression tests, and `pip check` passed. No existing scientific file or fixture changed. A full Bank rerun was unnecessary because these modules are isolated from Bank and all relevant source/input/environment hashes matched. The root experiment ledger appends accepted aggregate primary/natural-stream metrics while retaining its existing entries; the run-local ledger additionally records defined stratum metrics. Passing these checks establishes reproducibility within this scope, not broader deployment validity.
