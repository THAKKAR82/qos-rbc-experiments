# AGENTS.md

## Project purpose

This repository investigates Quantum Oracle Sketching (QOS) for
large-scale machine-learning problems relevant to financial services.

The goal is scientific evaluation, not to force a quantum advantage.

We reproduce and extend the methodology from:

https://github.com/haimengzhao/quantum-oracle-sketching

Primary questions:

1. Can QOS-style data access reduce machine-size requirements for
   large classical datasets?
2. At what dataset sizes, feature dimensions, and sparsities does
   QOS become interesting relative to classical streaming/sketching?
3. How much downstream ML performance is retained under compression?
4. Which conclusions are measured, calculated, simulated,
   theoretical, or speculative?

## Scientific rules

Do not claim quantum advantage unless the experiment actually
supports it.

Always distinguish:

- measured empirical results
- calculated metrics
- theoretical resource estimates
- simulations
- assumptions
- predictions or hypotheses

A theoretical QOS machine-size estimate is NOT equivalent to having
executed that algorithm on quantum hardware.

Do not describe theoretical qubit/resource accounting as an
experimental quantum speedup.

Preserve classical baselines even when they outperform QOS.

Negative results are valid results.

## QOS reference implementation

The primary external reference is:

haimengzhao/quantum-oracle-sketching

When reproducing algorithms or resource formulas, stay as close as
practical to the authors' implementation.

If our implementation differs materially from the reference repo,
document the difference.

## Project structure

- `src/` contains reusable implementation.
- `notebooks/` contains experiments and analysis.
- `data/` contains local datasets.
- `results/` contains generated tables and figures.
- `tests/` contains regression and correctness tests.

Do not put reusable algorithm implementations directly into notebooks.

Notebooks should primarily:
- load data
- configure experiments
- call functions from `src`
- visualize results
- document scientific interpretation

## Python conventions

Python >= 3.13 in the current local development environment.

Prefer:
- NumPy
- SciPy sparse matrices
- pandas
- scikit-learn
- PennyLane when actual quantum-circuit simulation is needed

Use deterministic random seeds for experiments.

Default seed: 42.

Avoid converting large sparse matrices to dense arrays unless
explicitly justified.

Use type hints for new reusable functions where practical.

Write short docstrings explaining scientific meaning, not merely syntax.

## ML evaluation

For imbalanced financial classification datasets:

- report ordinary accuracy when reproducing the QOS paper methodology
- also report balanced accuracy
- do not use ordinary accuracy alone to assess model quality

Use stratified cross-validation unless there is a strong reason not to.

Avoid leakage between preprocessing, training, and evaluation.

For temporal transaction datasets, investigate temporal splits before
using random cross-validation.

## Resource accounting

Current LS-SVM QOS machine-size formula:

2 * ceil(log2(N + 2D))
+ ceil(log2(s + 1))
+ 4

where:
- N = number of samples
- D = feature dimension
- s = maximum row sparsity

Treat this as theoretical resource accounting derived from the
reference QOS implementation.

Do not call these numbers measured RAM usage or hardware qubit counts
without qualification.

## Current validated Bank Marketing results

Full encoded representation:

- N = 45,211
- D = 51
- maximum row sparsity = 16
- QOS machine-size estimate = 41
- paper-style Ridge accuracy ≈ 0.8995
- balanced Ridge balanced accuracy ≈ 0.8256

Balanced bucketing and sparse JL sweeps have been validated for:

k = [4, 8, 12, 16, 24, 32, 40, 51]

Bank Marketing is considered a pipeline-validation dataset, not a
strong demonstration of QOS advantage because D is small.

## Testing requirements

Before considering a code change complete:

1. Run relevant tests.
2. Run formatting/linting if configured.
3. Confirm existing Bank Marketing regression values have not changed
   unexpectedly.
4. State what changed and why.
5. State any scientific assumptions introduced.

## Working style

For nontrivial changes:

1. Inspect relevant files first.
2. Explain the implementation plan briefly.
3. Make the smallest coherent change.
4. Run tests or validation.
5. Summarize changed files and results.

Do not refactor unrelated code during a focused task.

Do not silently change experimental methodology.

Ask before changing definitions of metrics, train/test splitting,
resource accounting, or scientific interpretation.