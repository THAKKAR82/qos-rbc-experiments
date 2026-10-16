# Dataset registry

[`manifest.csv`](manifest.csv) identifies scientifically distinct source, population,
and representation objects. Empty cells mean unknown/not applicable, not zero.
Counts are records (CSV parsing), not physical lines. `n_columns` counts raw columns
including labels/legacy fields; `encoded_dimension_D` counts model features.
`size_bytes`/`sha256` describe actual files only. Reconstructed matrices have neither;
CSR payloads and configuration hashes are recorded separately.

Raw data are normally local: size, evolving exports, acquisition terms, and complaint
text make blind Git publication unsuitable. **No new raw bytes are proposed.** The
already tracked `bank_marketing_qos_encoded.csv` remains untouched and tracked.
Ignoring a file does not remove it from history.

## Layout and existing paths

- `raw/`: ignored destination for future acquisitions.
- `processed/`: ignored destination for future large generated representations.
- `metadata/`: lightweight checksums, configuration, and artifact references.
- `download/`: acquisition instructions, not downloaded payloads.

Existing data remain at their legacy `data/*.csv` paths because experiment code uses
those paths. Empty ignored raw/processed folders need not survive a clone; create
with `mkdir -p data/raw data/processed`. Do not move fixtures during a provenance task.
Existing CFPB outputs remain in ignored `validation/cfpb/`; no copies of complaint
IDs, exclusions, vocabulary tokens, or narratives are added here.

## Obtain and verify

Follow [acquisition instructions](download/README.md), then from the repository root:

```sh
python scripts/dataset_checksum.py data/bank_marketing_qos_encoded.csv data/paysim.csv
python scripts/dataset_checksum.py data/CCDB_Export_15_January_2026_through_February_2026.csv
```

The standard-library helper streams 1 MiB chunks and outputs CSV path, size, SHA-256.
Compare exactly with `manifest.csv` or [local checksums](metadata/local_file_checksums.csv).
A different checksum is a new source version, not permission to update a fixture.
Original acquisition timestamps are unknown; filenames and filesystem modification
times do not establish acquisition provenance. Inspection date is 2026-10-16.

## Objects and experiment mapping

- `bank_marketing_raw`: the **local augmented source**, not pristine UCI data despite
  the short ID. Notebook 02 adds legacy columns; notebook 05 selects the original
  features. `bank_marketing_encoded_v1` is the separate 51-feature sweep matrix,
  reconstructed by notebook 05. Known global-preprocessing leakage remains.
- `paysim_raw`: local transactions, analytic applicability discussion only. No accepted
  conservative matrix exists, so no encoded PaySim row is claimed.
- `cfpb_raw_2026_01_02`: frozen source for `src/cfpb_prepare.py`;
  `cfpb_curated_primary_v1`: primary membership from the split manifest;
  `cfpb_natural_stream_v1`: secondary validation/test membership;
  `cfpb_tfidf_50k_v1` and `cfpb_tfidf_unigram_v1`: January training matrices,
  reconstructed rather than saved. Validation/test statistics are in the metadata
  bundle. Baseline/compression runners consume the same accepted preparation.
- `cfpb_raw_august_export`: present, but not the accepted experiment input.
- `pbmc68k_reference`: external reference loader, no local measured matrix.
  Reference-only IMDb, 20 Newsgroups and Dorothea code is not evidence of local use;
  these are not instantiated project dataset objects in this registry.

`results/audit/bank_step1`, `results/audit/cfpb_step1`, and
`results/audit/summary_step1.csv` hold task/fold-specific audit quantities. They must
not overwrite generic dataset statistics. QOS accounting is theoretical, not
measured hardware usage or evidence of quantum advantage.

## CFPB reconstruction and limitations

[Artifact index](metadata/cfpb_artifact_index.csv) records exact paths, hashes, and
sizes for accepted preparation outputs and relevant tracked code/policy.
[Reproducibility metadata](metadata/cfpb_reproducibility.json) preserves parameters,
training-ID/vocabulary/IDF hashes, source identity, versions, and partition statistics.

1. Obtain the exact source and verify its checksum.
2. Use the pinned environment and source/configuration fingerprints in the bundle.
3. Run `.venv/bin/python -m src.cfpb_prepare` only when preparation is intended;
   it creates a fresh directory. This provenance task did not run it.
4. Compare deterministic `split_manifest.csv` (IDs, group membership, primary and
   natural-stream flags), `curation_audit.csv` (exclusions), duplicate groups/edges,
   vocabulary/IDF and configuration hashes against the index. Run IDs/timings and
   manifests containing them need not be byte-identical across runs.

The original artifacts are ignored, not available just by cloning. Exact source
availability is therefore a reproducibility dependency: checksums do not recover
missing data. The historical acquisition URL/time is unrecorded. No vocabulary
publication is implied: tokens are derived complaint text and are not privacy-cleared.
The preparation's original modeling gate was resolved by the tracked validity review
and `docs/decisions/cfpb_modeling_gate.md`; preserve Recommendation B caveats.

## Tracking decisions

| Object | Full-file decision |
| --- | --- |
| Bank upstream data | CC BY 4.0 permits redistribution with attribution; local augmented fixture already tracked, unchanged |
| PaySim | Redistribution unclear from verified source; metadata/acquisition instructions only |
| CFPB sources/populations/text representations | Metadata only by project policy; public availability is not an instruction to mirror narratives |
| PBMC68k | Reference metadata only; exact dataset terms/version and local bytes unverified |

No inspected source established a categorical redistribution prohibition; uncertainty
is explicitly distinct from prohibition. Source/terms evidence is linked in the
acquisition instructions. No new full dataset file is proposed for commit.

## Add a dataset

Assign a unique stable ID and register each scientific object separately with a valid
parent. Record verified source/version/terms, local path and streaming checksum;
leave unknowns blank and explain them. State the population/partition of every
statistic. Keep raw/generated bytes ignored, preserve accepted fixtures, and review
any proposed publication explicitly. Add lightweight metadata, not narrative dumps.
Run `python -m pytest -q tests/test_dataset_tracking.py`; do not run experiments just
to populate a registry. Use `git status` and explicit file lists when staging.
