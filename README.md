# Quantum Oracle Sketching experiments

This project rigorously evaluates Quantum Oracle Sketching (QOS) for large-scale machine learning relevant to financial services. It asks when feature compression preserves useful ML performance and whether theoretical QOS resource requirements offer meaningful benefits against strong classical baselines. Negative results are valid outcomes.

**Empirical classification performance and theoretical resource accounting are separate.** Ridge scores are measured classical cross-validation results. Sketches are classical feature transformations. QOS machine-size estimates are calculated under specified assumptions; they are not measured quantum hardware usage or evidence that a quantum classifier was executed.

## Repository

| Path | Purpose |
| --- | --- |
| `AGENTS.md` | Stable scientific and development policy |
| `data/` | [Dataset registry and acquisition instructions](data/README.md); local raw inputs, checksums, and reconstruction metadata |
| `notebooks/00–04` | Earlier exploratory notebooks; unchanged by the refactor |
| `notebooks/05_qos_paper_style_bank_svm.ipynb` | Bank configuration, experiment execution, figures, exports, and validation |
| `src/` | Preprocessing, sketches, resource accounting, and Ridge/CV/sweep evaluation |
| `tests/` | Unit tests and nine scientific regression tests; frozen original definitions in `tests/fixtures/` |
| `results/` | Immutable validated CSVs/figures and the experiment ledger |
| `validation/bank_marketing/run-*/` | Separate rerun outputs and numerical comparison reports |
| `reference/quantum-oracle-sketching/` | Local QOS reference implementation |

## Environment and tests

For dataset versions, redistribution decisions, and checksum verification, see
[`data/README.md`](data/README.md) and [`data/manifest.csv`](data/manifest.csv).
Raw source files, curated membership, reconstructed feature matrices, and task-specific
audit outputs are distinct objects. Most raw inputs and CFPB preparation artifacts
are local and ignored; cloning alone does not provide them. The existing tracked
Bank fixture remains unchanged. No additional raw data publication is proposed.

The verified environment uses Python 3.13.9. From the repository root, create or activate a virtual environment and install the pinned dependencies:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

Use the **root** `requirements.txt`; the empty legacy `results/requirements.txt` is not the environment specification. The tests require the local dataset and reference checkout.

```sh
# Fast tests, including selected Bank scores; does not rerun full CV sweeps.
python -m pytest -q tests/test_preprocessing.py tests/test_machine_size.py tests/test_sketches.py tests/test_evaluation.py

# Nine scientific regression tests.
python -m pytest -q tests/test_regression.py
```

Numerical regression tolerance is `rtol=0, atol=1e-8`; dimensions and counts are checked exactly. Do not replace fixtures or widen tolerances to conceal discrepancies.

## Run notebook 05

Open `notebooks/05_qos_paper_style_bank_svm.ipynb` in a notebook-capable editor, select `.venv/bin/python`, restart the kernel, and run all cells in order. The working directory must be inside this repository; paths are resolved from its root. No dataset download is needed when `data/bank_marketing_qos_encoded.csv` is present.

Alternatively, execute headlessly after activating the environment:

```sh
python -m ipykernel install --user --name qos-rbc --display-name "QOS RBC (.venv)"
mkdir -p validation/notebook_execution
python -m nbconvert --to notebook --execute notebooks/05_qos_paper_style_bank_svm.ipynb \
  --ExecutePreprocessor.kernel_name=qos-rbc --ExecutePreprocessor.timeout=600 \
  --output=bank_marketing_executed.ipynb --output-dir=validation/notebook_execution
```

Each run creates a fresh `validation/bank_marketing/run-*/` directory for CSVs, figures, and `validation_report.json`. The notebook compares new CSVs and baselines against the original results without overwriting them. The headless command also saves an executed notebook separately.

## Validated Bank Marketing benchmark

Notebook 05 explicitly selects nine categorical and seven numerical features, one-hot encodes categories with unknown values ignored, and standardizes numerical features. Both Ridge configurations use `alpha=10`, `solver="auto"`; paper-style Ridge uses no class weighting and balanced Ridge uses `class_weight="balanced"`. CV uses five shuffled stratified folds with seed 42.

| Quantity | Validated value |
| --- | ---: |
| Samples, N | 45,211 |
| Encoded features, D | 51 |
| Maximum row sparsity | 16 |
| Nonzero values | 723,376 |
| QOS machine-size estimate | 41 |
| Paper-style Ridge ordinary accuracy | 0.8995155858 |
| Paper-style Ridge balanced accuracy | 0.6300640935 |
| Balanced Ridge balanced accuracy | 0.8256193229 |

The row-only LS-SVM estimate is `2 ceil(log2(N + 2D)) + ceil(log2(s + 1)) + 4`. The reference interprets `+4` as three logical qubits plus one classical scalar register. **41 is an accounting estimate conditional on the sparsity convention, not 41 measured hardware qubits.** Sparse nonzero counts and streaming feature dimensions are storage proxies, not directly comparable measured byte footprints.

Both balanced bucketing and signed sparse JL (`sparsity=1`) were evaluated at dimensions `[4, 8, 12, 16, 24, 32, 40, 51]`, with sketch seeds `[42, 123, 456, 789, 2026]`. Summaries use means and sample standard deviations across seed-level CV means, rather than the reference's flattened-score SEM.

The [completed refactor validation](validation/bank_marketing/run-29qfoy4_/validation_report.json) reproduced all 80 sweep rows, both summaries, the comparison table, and requested baselines with zero parsed numerical difference. All five regenerated figures were pixel-identical to the originals.

## Limitations and next direction

- Baseline pipelines fit preprocessing within CV folds. Sketch sweeps fit it globally before CV: **known preprocessing leakage is preserved**, limiting interpretation and direct baseline/sweep comparisons.
- Bank uses maximum **row sparsity**; the QOS reference uses maximum row-or-column sparsity. The estimate of 41 is conditional on the Bank convention.
- Dense bucketing and CSR JL select different `RidgeClassifier(solver="auto")` solvers. Full-dimension CSR JL scores are approximately `0.8995377048` ordinary accuracy and `0.8256067994` balanced-Ridge balanced accuracy; this small difference is retained.
- Only JL `sparsity=1` is validated. Higher sparsity is rejected; the reference includes collision repair not validated by this Bank workflow.
- Ordinary accuracy alone is inadequate for this imbalanced dataset. Balanced accuracy must also be considered.

**Bank Marketing is a pipeline-validation benchmark, not evidence of practical quantum advantage.** With only 51 encoded features, classical streaming is already competitive. The next research direction is larger and/or higher-dimensional financial datasets, with separately reviewed leakage-free or time-aware evaluation and resource assumptions. No next-dataset experiment has started.

## Experiment ledger

[`results/experiment_log.csv`](results/experiment_log.csv) records one row per metric. An experiment ID can repeat for metrics from the same run. Entries describe the verified October 14, 2026 rerun, dated from its saved notebook execution timestamps in `America/Toronto`; they do not assign dates to the unknown original executions. Git commits are blank because the inspected workspace has no root Git metadata.

`n_features` is the encoded input dimension; `sketch_dimension` is the output dimension and is blank for unsketched baselines. `max_sparsity` describes the evaluated matrix under the row-only convention. `seed` is the sketch seed (blank when no sketch was used); CV seed 42 and source artifacts are recorded in notes. The QOS accounting row is explicitly theoretical. Preserve historical entries and add future runs with their actual provenance; do not infer missing dates or commits.
