---
name: dataset-onboarding
description: Review a new dataset before implementing QOS experiments in this repository; assess data quality, prediction validity, evaluation splits, classical scaling, and QOS suitability without modifying data or scientific code.
---

# Dataset onboarding

## Establish the task

Read repository-root `AGENTS.md` first, then `README.md` and relevant dataset documentation. Paths below are repository-relative. Preserve the established Bank benchmark; its historical leakage and row-only convention are not defaults to impose on a new experiment.

Identify source/version, files, schema, observation unit, intended target, prediction time, outcome horizon, and label availability. Separate benchmark reproduction from a banking-relevant prediction task. Inspect available evidence before asking questions; ask only for consequential unknowns such as target meaning or feature availability. Do not infer availability from column names alone.

This is a pre-experiment review: do not modify the dataset, scientific code, notebooks, or validated artifacts unless explicitly requested. Do not start training sweeps or implement the recommended experiment. Use bounded read-only profiling and temporary/in-memory representations where practical.

## Select profiling mode

Enable `large_dataset_mode` before loading data when any of these apply: several million rows, files of several hundred MB or larger, an expected in-memory representation consuming a substantial fraction of available RAM, or an explicit request for memory-safe profiling. If size is uncertain, inspect file metadata/header first and use this mode conservatively.

When active, read [references/large_dataset_mode.md](references/large_dataset_mode.md) before profiling. It defines bounded sampling, chunked scans, machine-aware estimates, staged experiment recommendations, and six additional report sections. Otherwise follow the normal bounded profiling workflow below.

## Profile and trace validity

Start with schema, metadata, and bounded samples; use chunked scans for exact counts when feasible. Record sample selection, coverage, and seed (default 42 for random profiling). Distinguish full-data measurements, sample measurements, estimates, bounds, and unknowns; do not label a prefix sample representative of temporal data.

Report:

- N, total raw columns, and candidate predictor count excluding target/metadata; categorical versus numerical columns based on meaning as well as dtype.
- Missingness counts/rates, exact duplicate rows where feasible, repeated entities/events, conflicting labels, and possible cross-split overlap. Do not silently deduplicate legitimate repeated transactions.
- Target counts/prevalence and imbalance; missing/delayed labels, sampling or case-control selection, and implications for evaluation.
- Identifiers/quasi-identifiers, categorical cardinalities, temporal columns, candidate leakage and obvious post-outcome variables. Explain what would make each available or unavailable at prediction time; label unresolved risks.
- D under an explicit candidate encoding. Estimate before materializing high-cardinality one-hot features; disclose unseen-category and missing-value handling. Avoid inflating D with identifiers merely to make QOS look promising.

## Choose evaluation before preprocessing

Tie the split to the deployment question:

- Future transactions/events: chronological holdout or rolling evaluation; account for label delays and overlapping outcome windows with appropriate gaps. Do not default to random CV.
- Unseen customers/accounts/entities: entity/group-disjoint splits; combine with time constraints when both matter. Predicting future events for known entities is a different task—state which is intended.
- Independent static classification: stratified splitting/CV where class counts permit; preserve official benchmark splits for reproduction, documenting any separate operational evaluation.

Fit learned preprocessing, feature selection, target encoding, resampling, and tuning within training partitions only. Flag target-derived aggregates, future information, duplicate contamination, and post-outcome features. Recommend exclusions, not silent dataset edits.

For imbalanced classification, report ordinary accuracy for comparability plus balanced accuracy and suitable recall, precision, PR-AUC/average precision (state which), or cost/threshold metrics. Define the positive class, select thresholds on training/validation data, and keep the final holdout untouched. Use task-appropriate metrics if the target is not classification.

## Estimate representation and QOS relevance

Inspect `src/preprocessing.py` before reusing it; its centering and density threshold may be unsuitable for large sparse data. Do not change it during review. For a feasible training-only pilot, report encoded D, dtype/format, maximum row sparsity s, mathematical nnz, density, and sparse-storage suitability without densifying large matrices. Reuse `src/machine_size.py` where compatible. Sample maxima are not established full-data maxima; mark projections and bounds explicitly.

Compare like-for-like candidate representations:

- Dense payload: N × D × dtype bytes; separate raw table/file size and peak working-memory estimates.
- Sparse payload: values plus indices and row/column pointers; distinguish stored entries from mathematical nnz and include format/dtype assumptions.
- Classical streaming/sketching: state sketch dimension, passes, model/buffer memory and access costs. D alone is only a storage proxy, not total training memory.

Assess N, D, sparsity, storage pressure, and whether low dimensionality already makes classical streaming competitive. Describe arrival order, stationarity, replay/random access, dependence between samples, and whether a file can merely be iterated versus genuinely supporting the assumed sampling model. Check the relevant local QOS reference's access assumptions; identify unsupported assumptions and data-loading/oracle-construction costs.

Use `qos_svm_machine_size(N, D, s)` only for a compatible LS-SVM scenario with well-defined N, D, and s. If inputs are conditional estimates, label the entire result conditional; otherwise withhold it. State that Bank uses maximum row sparsity while the reference uses maximum row-or-column sparsity, and name the convention used. Do not silently substitute one or mix values from different representations. Label the result **theoretical resource accounting**, not measured quantum hardware usage; mixed accounting units are not directly comparable to classical bytes. Machine-size scaling alone never demonstrates quantum advantage.

## Final onboarding report

Use these eight sections:

1. **Dataset summary:** source/version, coverage, N, raw features, types, missingness, duplicates, and estimate status.
2. **Prediction target:** observation unit, positive class/outcome, timing, imbalance, benchmark versus operational objective.
3. **Leakage and validity risks:** suspect variables/entities, evidence, recommended exclusions, unresolved questions.
4. **Recommended split/evaluation strategy:** concrete time/group/entity/stratified design, preprocessing boundaries, metrics, tuning and holdout rules.
5. **Classical representation and scaling:** encoding, D, s, nnz, memory units/assumptions, dense/sparse/streaming comparisons.
6. **QOS relevance:** access feasibility, conditional resource estimate or reason withheld, scientific value and limitations.
7. **Recommended first experiment:** smallest informative comparison, strong classical controls, proposed sketch settings, feasibility checks, and success/stop criteria; no implementation.
8. **Reasons not to use this dataset, if applicable:** blockers and what evidence could resolve them; explicitly say if none identified.

When `large_dataset_mode` is active, append its six required report sections before the classification.

Conclude with exactly one classification, confidence, and reasons:

- **strong QOS benchmark candidate:** meaningful scale/sparsity contrasts, credible access assumptions, and a valid evaluation path; not evidence of advantage.
- **useful secondary benchmark:** informative complementary regimes with stated limitations.
- **pipeline-validation only:** useful correctness checks but insufficient scale or dimensionality to probe a meaningful resource tradeoff.
- **poor fit for QOS:** fundamental validity, access, or representation problems undermine the intended experiment.

If evidence is incomplete, make the classification provisional and identify the blocking evidence; do not default to a strong recommendation.
