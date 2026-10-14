"""Ridge/CV evaluation preserving the validated Bank methodology.

No preprocessing or matrix-format conversion occurs here. Baseline callers can
pass a preprocessing Pipeline to cross_validation_scores for fold-local fitting.
Sketch sweeps receive already encoded features: Bank historically fits those
features globally before CV, a known leakage risk deliberately not fixed here.

Bank uses shuffled stratified CV, balanced Ridge as an additional model, and
sample standard deviation across seed-level CV means. The inspected QOS SVM
reference uses unshuffled cv=5 and flattened-score SEM. Default accounting also
retains Bank's row-only sparsity, not the reference row-or-column maximum.
"""

import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .machine_size import matrix_nnz, max_row_sparsity, qos_svm_machine_size


BANK_SKETCH_DIMENSIONS = (4, 8, 12, 16, 24, 32, 40, 51)
BANK_SKETCH_SEEDS = (42, 123, 456, 789, 2026)
RUN_COLUMNS = (
    "k", "seed", "paper_accuracy", "balanced_accuracy", "streaming_size",
    "sparse_size", "qos_size", "max_sparsity",
)


def make_ridge_classifier(*, class_weight=None, alpha=10.0, solver="auto"):
    """Build paper Ridge by default; class_weight='balanced' builds balanced Ridge.

    All other sklearn defaults remain unchanged, including random_state=None.
    'auto' intentionally selects different solvers for dense and sparse data.
    """
    return RidgeClassifier(alpha=alpha, solver=solver, class_weight=class_weight)


def make_stratified_cv(*, n_splits=5, seed=42):
    """Return the Bank shuffled stratified CV splitter (five folds, seed 42)."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def cross_validation_scores(estimator, X, y, *, scoring="accuracy", cv=None):
    """Return per-fold ordinary or balanced accuracy without changing X or y.

    Scoring is independent of class weighting: paper Ridge can also be scored
    with balanced_accuracy, as in the full-representation Bank baseline.
    Custom cv must be a reusable splitter or a re-iterable collection of splits.
    sklearn clones the estimator (including any caller-supplied Pipeline).
    """
    if scoring not in ("accuracy", "balanced_accuracy"):
        raise ValueError("scoring must be 'accuracy' or 'balanced_accuracy'")
    if cv is None:
        cv = make_stratified_cv()
    return cross_val_score(estimator, X, y, cv=cv, scoring=scoring)


def evaluate_sketch(
    X, y, sketch, *, k, seed=42, cv=None,
    sparsity_fn=max_row_sparsity, nnz_fn=matrix_nnz,
    qos_size_fn=qos_svm_machine_size,
):
    """Sketch and evaluate one (method, k, seed); return one notebook-style row.

    sketch(X, n_buckets=k, seed=seed) must return a matrix, not reference metadata.
    It and resource callbacks must not mutate their inputs. The matrix is passed
    unchanged to both models and accounting callbacks; dense bucketing remains
    dense and JL remains CSR. k records the requested dimension, streaming_size
    the actual output dimension (which may be smaller at an identity endpoint).
    """
    if cv is None:
        cv = make_stratified_cv()
    X_sketch = sketch(X, n_buckets=k, seed=seed)
    paper = cross_validation_scores(
        make_ridge_classifier(), X_sketch, y, cv=cv, scoring="accuracy",
    )
    balanced = cross_validation_scores(
        make_ridge_classifier(class_weight="balanced"), X_sketch, y,
        cv=cv, scoring="balanced_accuracy",
    )
    N, D = X_sketch.shape
    sparsity = sparsity_fn(X_sketch)
    return {
        "k": k,
        "seed": seed,
        "paper_accuracy": float(paper.mean()),
        "balanced_accuracy": float(balanced.mean()),
        "streaming_size": D,
        "sparse_size": nnz_fn(X_sketch),
        "qos_size": qos_size_fn(N, D, sparsity),
        "max_sparsity": sparsity,
    }


def run_sketch_sweep(
    X, y, sketch, *, dimensions=BANK_SKETCH_DIMENSIONS, seeds=BANK_SKETCH_SEEDS,
    cv=None, sparsity_fn=max_row_sparsity, nnz_fn=matrix_nnz,
    qos_size_fn=qos_svm_machine_size,
):
    """Return one method's tidy results in dimension-then-seed order.

    Defaults are explicitly Bank-specific, not inferred or randomly sampled.
    Other experiments must supply their own dimensions/seeds. No files are
    written. Resource callbacks use the same signatures as machine_size helpers.
    """
    if cv is None:
        cv = make_stratified_cv()
    seeds = tuple(seeds)  # Reuse even if a caller supplies a seed iterator.
    rows = [
        evaluate_sketch(
            X, y, sketch, k=k, seed=seed, cv=cv,
            sparsity_fn=sparsity_fn, nnz_fn=nnz_fn, qos_size_fn=qos_size_fn,
        )
        for k in dimensions for seed in seeds
    ]
    return pd.DataFrame(rows, columns=RUN_COLUMNS)


def summarize_sweep(results):
    """Aggregate a single method's rows into the notebook's summary schema.

    Means and sample std (ddof=1) are over seed-level CV means, never flattened
    fold scores or SEM. One seed yields NaN std, as in pandas/the notebook.
    Call separately for each sketch method; grouping is by k only.
    """
    return results.groupby("k", sort=True).agg(
        paper_accuracy_mean=("paper_accuracy", "mean"),
        paper_accuracy_std=("paper_accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        streaming_size=("streaming_size", "mean"),
        sparse_size=("sparse_size", "mean"),
        qos_size=("qos_size", "mean"),
        max_sparsity=("max_sparsity", "mean"),
    ).reset_index()
