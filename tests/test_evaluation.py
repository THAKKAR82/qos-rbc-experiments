"""Evaluation contracts and selected Bank regressions; no artifact writes."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

import src.evaluation as evaluation
from src.preprocessing import make_preprocessor
from src.sketches import balanced_feature_bucket, sparse_jl_features

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("weight", [None, "balanced"])
def test_exact_ridge_configuration(weight):
    model = evaluation.make_ridge_classifier(class_weight=weight)
    expected = RidgeClassifier(alpha=10., solver="auto", class_weight=weight)
    assert model.get_params() == expected.get_params()
    assert model.random_state is None


def test_exact_cv_configuration():
    cv = evaluation.make_stratified_cv()
    assert isinstance(cv, StratifiedKFold)
    assert cv.n_splits == 5
    assert cv.shuffle is True
    assert cv.random_state == 42
    X, y = np.zeros((30, 2)), np.array([0] * 20 + [1] * 10)
    expected = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for actual, original in zip(cv.split(X, y), expected.split(X, y)):
        for a, b in zip(actual, original):
            np.testing.assert_array_equal(a, b)


@pytest.fixture
def small_data():
    X = np.random.default_rng(42).normal(size=(50, 8))
    y = np.array([0] * 35 + [1] * 15)
    return X, y


@pytest.mark.parametrize("scoring", ["accuracy", "balanced_accuracy"])
def test_scoring_and_determinism(small_data, scoring):
    X, y = small_data
    model = evaluation.make_ridge_classifier()
    before_X, before_y = X.copy(), y.copy()
    actual = evaluation.cross_validation_scores(model, X, y, scoring=scoring)
    repeat = evaluation.cross_validation_scores(model, X, y, scoring=scoring)
    expected = cross_val_score(model, X, y, scoring=scoring, cv=evaluation.make_stratified_cv())
    assert actual.shape == (5,)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, repeat)
    np.testing.assert_array_equal(X, before_X)
    np.testing.assert_array_equal(y, before_y)
    assert not hasattr(model, "coef_")  # CV fits clones, not the caller's model.


def test_scoring_is_not_inferred_from_class_weight(small_data):
    X, y = small_data
    model = evaluation.make_ridge_classifier()
    ordinary = evaluation.cross_validation_scores(model, X, y, scoring="accuracy")
    balanced = evaluation.cross_validation_scores(model, X, y, scoring="balanced_accuracy")
    assert not np.array_equal(ordinary, balanced)
    with pytest.raises(ValueError, match="scoring"):
        evaluation.cross_validation_scores(model, X, y, scoring="roc_auc")


@pytest.mark.parametrize("sketch,sparse_output", [(balanced_feature_bucket, False), (sparse_jl_features, True)])
def test_evaluation_passes_representation_scoring_and_callbacks_unchanged(monkeypatch, small_data, sketch, sparse_output):
    X, y = small_data
    projected = sketch(X, 4)
    cv = evaluation.make_stratified_cv()
    seen = []
    def fixed_sketch(matrix, *, n_buckets, seed):
        assert matrix is X and n_buckets == 4 and seed == 123
        return projected
    def spy(estimator, matrix, labels, *, cv, scoring):
        assert matrix is projected and labels is y
        assert sp.issparse(matrix) is sparse_output
        seen.append((estimator.class_weight, scoring, cv))
        return np.array([0.7] * 5) if scoring == "accuracy" else np.array([0.6] * 5)
    def sparsity(matrix):
        assert matrix is projected
        return 3
    def nnz(matrix):
        assert matrix is projected
        return 101
    def qos(N, D, s):
        assert (N, D, s) == (50, 4, 3)
        return 29
    monkeypatch.setattr(evaluation, "cross_val_score", spy)
    row = evaluation.evaluate_sketch(X, y, fixed_sketch, k=4, seed=123, cv=cv,
                                     sparsity_fn=sparsity, nnz_fn=nnz, qos_size_fn=qos)
    assert seen == [(None, "accuracy", cv), ("balanced", "balanced_accuracy", cv)]
    assert list(row) == list(evaluation.RUN_COLUMNS)
    assert row == pytest.approx(dict(k=4, seed=123, paper_accuracy=.7,
        balanced_accuracy=.6, streaming_size=4, sparse_size=101, qos_size=29, max_sparsity=3))


@pytest.mark.parametrize("constructor", [np.array, sp.csr_matrix])
def test_real_sweep_determinism_and_non_mutation(small_data, constructor):
    dense, y = small_data
    X = constructor(dense)
    original_y = y.copy()
    original_data = X.data.copy() if sp.issparse(X) else X.copy()
    if sp.issparse(X):
        original_indices, original_indptr = X.indices.copy(), X.indptr.copy()
    kwargs = dict(dimensions=[4, 8], seeds=[42, 123])
    actual = evaluation.run_sketch_sweep(X, y, balanced_feature_bucket, **kwargs)
    repeat = evaluation.run_sketch_sweep(X, y, balanced_feature_bucket, **kwargs)
    pd.testing.assert_frame_equal(actual, repeat)
    assert list(zip(actual.k, actual.seed)) == [(4, 42), (4, 123), (8, 42), (8, 123)]
    np.testing.assert_array_equal(X.data if sp.issparse(X) else X, original_data)
    if sp.issparse(X):
        np.testing.assert_array_equal(X.indices, original_indices)
        np.testing.assert_array_equal(X.indptr, original_indptr)
    np.testing.assert_array_equal(y, original_y)


def test_sweep_defaults_order_and_callback_forwarding(monkeypatch, small_data):
    X, y = small_data
    calls = []
    custom_nnz = lambda matrix: 999
    def fake(matrix, labels, sketch, **kwargs):
        assert matrix is X and labels is y
        assert sketch is balanced_feature_bucket
        assert kwargs["nnz_fn"] is custom_nnz
        calls.append((kwargs["k"], kwargs["seed"]))
        return {"k": kwargs["k"], "seed": kwargs["seed"]}
    monkeypatch.setattr(evaluation, "evaluate_sketch", fake)
    result = evaluation.run_sketch_sweep(X, y, balanced_feature_bucket, nnz_fn=custom_nnz)
    assert calls == [(k, seed) for k in [4, 8, 12, 16, 24, 32, 40, 51]
                     for seed in [42, 123, 456, 789, 2026]]
    assert result.shape == (40, 8)
    calls.clear()
    evaluation.run_sketch_sweep(X, y, balanced_feature_bucket, dimensions=[2, 4],
                                seeds=iter([7, 8]), nnz_fn=custom_nnz)
    assert calls == [(2, 7), (2, 8), (4, 7), (4, 8)]


def test_aggregation_sample_std_single_seed_and_non_mutation():
    rows = pd.DataFrame([
        [8, 42, .5, .6, 8, 12, 20, 3],
        [4, 42, .2, .4, 4, 10, 18, 2],
        [4, 123, .6, .8, 4, 14, 20, 4],
    ], columns=evaluation.RUN_COLUMNS)
    before = rows.copy(deep=True)
    result = evaluation.summarize_sweep(rows)
    expected = pd.DataFrame({
        "k": [4, 8], "paper_accuracy_mean": [.4, .5],
        "paper_accuracy_std": [np.sqrt(.08), np.nan],
        "balanced_accuracy_mean": [.6, .6],
        "balanced_accuracy_std": [np.sqrt(.08), np.nan],
        "streaming_size": [4., 8.], "sparse_size": [12., 12.],
        "qos_size": [19., 20.], "max_sparsity": [3., 3.],
    })
    pd.testing.assert_frame_equal(result, expected, rtol=0, atol=1e-15)
    pd.testing.assert_frame_equal(rows, before)


@pytest.mark.parametrize("method", ["bucket", "sparse_jl"])
def test_aggregation_matches_saved_bank_summary(method):
    raw = pd.read_csv(ROOT / f"results/bank_{method}_sweep_raw.csv")
    expected = pd.read_csv(ROOT / f"results/bank_{method}_sweep_summary.csv")
    pd.testing.assert_frame_equal(evaluation.summarize_sweep(raw), expected, rtol=0, atol=1e-12)


def test_pipeline_remains_fold_local():
    frame = pd.DataFrame({"category": ["a", "b"] * 20, "number": np.arange(40.)})
    y = np.array([0, 1] * 20)
    pipeline = Pipeline([("preprocess", make_preprocessor(["category"], ["number"])),
                         ("classifier", evaluation.make_ridge_classifier())])
    scores = evaluation.cross_validation_scores(pipeline, frame, y)
    assert scores.shape == (5,)
    assert not hasattr(pipeline.named_steps["preprocess"], "transformers_")
    expected = cross_val_score(pipeline, frame, y, cv=evaluation.make_stratified_cv(), scoring="accuracy")
    np.testing.assert_array_equal(scores, expected)


@pytest.fixture(scope="module")
def bank_data():
    frame = pd.read_csv(ROOT / "data/bank_marketing_qos_encoded.csv")
    categorical = ["job", "marital", "education", "default", "housing", "loan",
                   "contact", "month", "poutcome"]
    numeric = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    # Preserve the sweep's globally fitted preprocessing, including known leakage.
    return (make_preprocessor(categorical, numeric).fit_transform(frame),
            frame.y.map({"no": 0, "yes": 1}).to_numpy())


@pytest.mark.parametrize("sketch,method", [(balanced_feature_bucket, "bucket"),
                                          (sparse_jl_features, "sparse_jl")])
def test_bank_known_k16_seed42_run(bank_data, sketch, method):
    X, y = bank_data
    result = evaluation.run_sketch_sweep(X, y, sketch, dimensions=[16], seeds=[42])
    saved = pd.read_csv(ROOT / f"results/bank_{method}_sweep_raw.csv")
    expected = saved[(saved.k == 16) & (saved.seed == 42)].reset_index(drop=True)
    # Existing CSV qos_size is float; the extracted integer formula is exact.
    pd.testing.assert_frame_equal(result, expected, check_dtype=False, rtol=0, atol=1e-8)
