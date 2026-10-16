# CFPB compression results

Accepted run: [`20261016T125619Z-30526eac`](../validation/cfpb/20261016T125619Z-30526eac/benchmark_report.md). Read the [predeclared stage protocol](cfpb_compression_protocol.md) for the user-authorized grid/classifier extensions and unchanged curation.

```sh
# Fresh scientific run; ends pending validation, never auto-publishes a ledger.
.venv/bin/python -m src.cfpb_compression
# Presentation-only tables/plots for a run:
.venv/bin/python -m src.compression_reporting validation/cfpb/<run-id>
```

Do not rerun reporting into an accepted immutable run; use a copy when changing presentation. New runs require the accepted preparation, source, environment and full-baseline artifacts. Hashing trains before TF-IDF reconstruction; all primary validation choices are frozen before test predictions. There is no automatic parameter retuning or curation change.

- `src/compression.py`: deterministic January-frequency ranking, explicit float32 sparse truncation/sketch transforms, bounded-batch hashing/SGD replay, retention and seed summaries.
- `src/cfpb_compression.py`: CFPB orchestration, full-endpoint reproduction, validation-only selection, restricted natural-stream sensitivity, timing/resource accounting and separate output files.
- `src/compression_reporting.py`: tables and four figures, keeping theoretical accounting visually separate from classical measurements.
- `tests/test_compression.py`: six tests covering ranking/heldout immutability, all method endpoints, sparse/dtype preservation, seeded behavior, source non-mutation, batching/no fitted hashing vocabulary, retention/schema/sample-SD and accounting integration.

Validation favors truncation at 500–10,000 features, bucketing slightly at 20,000, and the shared full model at 50,000. Minimum tested k for 99%/98%/95% validation macro-F1 retention: truncation 5,000/1,000/500; bucketing and JL 10,000/5,000/2,000; hashing not reached/10,000/5,000. These are validation decisions, not heldout guarantees: truncation at 5,000 retains 99.41% validation but 98.59% test macro-F1.

The full report includes test-based descriptive minima, per-class/confusion results, seed SD, and representative natural-stream sensitivity. Full-model predictions/parameters match the accepted balanced LinearSVC C=1 baseline. Truncation is competitive and substantially reduces model arrays; the projections save little sparse matrix payload and generally fit more slowly here. Hashing is a useful vocabulary-free comparator but needs five replay passes and measured tokenization costs. Matrix payload, parameter arrays, replay/batch memory and whole-process RSS are different quantities.

CFPB remains a useful secondary sparse-text compression/QOS-accounting benchmark. Its inexpensive classical baselines weaken motivation based on this scale alone; theoretical accounting does not establish hardware requirements or practical benefit. No circuit or next-dataset work was performed.

Validation: 139 fast tests (40 CFPB/text/holdout/compression), nine Bank scientific regressions, and `pip check` passed. All 208 score rows and QOS/retention calculations were independently checked. Existing Bank and accepted CFPB artifacts remain unchanged; root ledger publication only appends accepted entries and preserves previous bytes. Full Bank rerun was unnecessary because shared code, input artifacts and numerical environment did not change.
