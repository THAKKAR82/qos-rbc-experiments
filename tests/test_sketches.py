"""Sketch contracts, direct notebook/reference parity, and selected Bank scores.

No notebook exports or reference imports with filesystem side effects are run.
Bank transform comparisons retain the historical globally fitted preprocessing;
this is regression coverage, not a correction of the known leakage issue.
"""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.machine_size import matrix_nnz, max_row_sparsity, qos_svm_machine_size
from src.preprocessing import make_preprocessor
from src.sketches import balanced_feature_bucket, sparse_jl_features

ROOT = Path(__file__).resolve().parents[1]
METHODS = [balanced_feature_bucket, sparse_jl_features]


@pytest.fixture(scope="module")
def originals():
    fixture = json.loads((ROOT / "tests/fixtures/bank_original_cells.json").read_text())
    namespace = {"np": np, "sp": sp}
    for index, digest in [
        (12, "f3cf73523435ea6f84157f8322796e55a9f9726768bcd0e2d66a5c2171cfa4ef"),
        (23, "3adedd32f721a73034db8cff028f1657c189e751e630d7e08c1225497682c8b2"),
    ]:
        source = fixture["cells"][str(index)]
        assert hashlib.sha256(source.encode()).hexdigest() == digest
        tree = ast.parse(source)
        assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
        exec(compile(tree, f"bank-cell-{index}", "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def reference():
    path = ROOT / "reference/quantum-oracle-sketching/real_datasets/sketch_utils.py"
    # Execute source without importing it: do not create __pycache__ in reference.
    namespace = {"__name__": "bank_reference_comparison"}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("sketch", METHODS)
def test_determinism_seeds_shape_and_normalization(sketch):
    X = np.eye(23)
    actual = sketch(X, 5)
    repeated = sketch(X, 5, seed=42)
    changed = sketch(X, 5, seed=123)
    dense = actual.toarray() if sp.issparse(actual) else actual
    other = changed.toarray() if sp.issparse(changed) else changed
    np.testing.assert_array_equal(dense, repeated.toarray() if sp.issparse(repeated) else repeated)
    assert not np.array_equal(dense, other)
    assert actual.shape == (23, 5)
    assert (np.count_nonzero(dense, axis=1) == 1).all()
    occupancy = np.count_nonzero(dense, axis=0)
    assert occupancy.max() - occupancy.min() == 1
    np.testing.assert_allclose(dense.T @ dense, np.eye(5), rtol=0, atol=1e-15)
    rows, cols = np.nonzero(dense)
    np.testing.assert_array_equal(np.abs(dense[rows, cols]), 1 / np.sqrt(occupancy[cols]))
    if sketch is sparse_jl_features:
        assert sp.isspmatrix_csr(actual)
        assert (dense < 0).any() and (dense > 0).any()
    else:
        assert isinstance(actual, np.ndarray)
        assert (dense >= 0).all()


@pytest.mark.parametrize("sketch", METHODS)
@pytest.mark.parametrize("constructor", [np.array, sp.csr_matrix, sp.csc_matrix, sp.csr_array])
@pytest.mark.parametrize("k", [4, 7])
def test_identity_is_independent_copy(sketch, constructor, k):
    X = constructor([[1., 0., -2., 3.], [0., 5., 0., 4.]])
    result = sketch(X, k)
    expected = X.toarray() if sp.issparse(X) else X.copy()
    np.testing.assert_array_equal(result.toarray() if sp.issparse(result) else result, expected)
    assert result.shape == X.shape
    assert result is not X
    if sketch is sparse_jl_features:
        assert sp.isspmatrix_csr(result)
    elif sp.issparse(X):
        assert result.format == X.format
    else:
        assert isinstance(result, np.ndarray)
    if sp.issparse(result):
        result.data[:] = 99
    else:
        result[:] = 99
    np.testing.assert_array_equal(X.toarray() if sp.issparse(X) else X, expected)


@pytest.mark.parametrize("sketch", METHODS)
def test_sparse_noncanonical_input_no_densification_or_mutation(sketch, monkeypatch):
    X = sp.csr_matrix((np.array([2., -2., 0., 5., 3.]),
                       np.array([0, 0, 1, 2, 4]), np.array([0, 3, 5])), shape=(2, 8))
    buffers = [X.data.copy(), X.indices.copy(), X.indptr.copy()]
    canonical = X.has_canonical_format
    dense = X.toarray()
    expected = sketch(dense, 3)
    expected = expected.toarray() if sp.issparse(expected) else expected
    with monkeypatch.context() as patch:
        def forbidden(*args, **kwargs):
            raise AssertionError("Sparse input must not be densified")
        patch.setattr(sp.csr_matrix, "toarray", forbidden)
        patch.setattr(sp.csr_matrix, "todense", forbidden)
        result = sketch(X, 3)
    assert sp.isspmatrix_csr(result)
    np.testing.assert_allclose(result.toarray(), expected, rtol=0, atol=1e-15)
    for actual, before in zip([X.data, X.indices, X.indptr], buffers):
        np.testing.assert_array_equal(actual, before)
    assert X.has_canonical_format == canonical


@pytest.mark.parametrize("sketch", METHODS)
def test_exact_cancellation_and_dense_non_mutation(sketch):
    projection = sketch(np.eye(20), 4)
    projection = projection.toarray() if sp.issparse(projection) else projection
    indices = np.flatnonzero(projection[:, 0])
    j, k = indices[:2]
    X = np.zeros((1, 20))
    X[0, j] = 1
    X[0, k] = -np.sign(projection[j, 0]) * np.sign(projection[k, 0])
    before = X.copy()
    for source in [X, sp.csr_matrix(X)]:
        result = sketch(source, 4)
        assert matrix_nnz(result) == 0
        if sp.issparse(result):
            assert result.nnz == 0  # Cancellation zeros are removed.
    np.testing.assert_array_equal(X, before)


@pytest.mark.parametrize("k", [2, 5, 13, 20])
@pytest.mark.parametrize("seed", [42, 123])
def test_notebook_and_reference_parity(originals, reference, k, seed):
    X = np.random.default_rng(2026).normal(size=(11, 13))
    X[np.abs(X) < 0.5] = 0
    before = X.copy()
    for sketch, ref_name in [(balanced_feature_bucket, "bucket_features"),
                             (sparse_jl_features, "sparse_jl_features")]:
        actual = sketch(X, k, seed)
        original = originals[sketch.__name__](X, k, seed)
        np.testing.assert_array_equal(actual.toarray() if sp.issparse(actual) else actual,
                                      original.toarray() if sp.issparse(original) else original)
        expected, _ = reference[ref_name](X, k, seed)
        # Dense Bank division/accumulation differs from reference CSR arithmetic.
        np.testing.assert_allclose(actual.toarray() if sp.issparse(actual) else actual,
                                   expected.toarray(), rtol=1e-15, atol=1e-15)
        if sketch is balanced_feature_bucket:
            sparse_actual = sketch(sp.csr_matrix(X), k, seed)
            np.testing.assert_array_equal(sparse_actual.toarray(), expected.toarray())
        else:
            np.testing.assert_array_equal(actual.toarray(), expected.toarray())
    np.testing.assert_array_equal(X, before)


@pytest.mark.parametrize("sparsity", [0, 2, 4])
@pytest.mark.parametrize("k", [4, 8])
def test_unvalidated_sparsity_is_rejected_even_at_identity(sparsity, k):
    with pytest.raises(ValueError):
        sparse_jl_features(np.eye(8), k, sparsity=sparsity)


@pytest.mark.parametrize("sketch", METHODS)
@pytest.mark.parametrize("k,error", [(0, ValueError), (-1, ValueError), (2.5, TypeError), (True, TypeError)])
def test_invalid_dimensions(sketch, k, error):
    with pytest.raises(error):
        sketch(np.eye(8), k)


def test_single_bucket_contract():
    assert balanced_feature_bucket(np.eye(4), 1).shape == (4, 1)
    with pytest.raises(ValueError, match="n_buckets >= 2"):
        sparse_jl_features(np.eye(4), 1)
    np.testing.assert_array_equal(sparse_jl_features(np.ones((2, 1)), 1).toarray(), np.ones((2, 1)))


@pytest.fixture(scope="module")
def bank_data():
    frame = pd.read_csv(ROOT / "data/bank_marketing_qos_encoded.csv")
    categorical = ["job", "marital", "education", "default", "housing", "loan",
                   "contact", "month", "poutcome"]
    numeric = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    X = make_preprocessor(categorical, numeric).fit_transform(frame)
    return X, frame.y.map({"no": 0, "yes": 1}).to_numpy()


@pytest.mark.parametrize("sketch,method", [(balanced_feature_bucket, "bucket"),
                                          (sparse_jl_features, "sparse_jl")])
def test_bank_sweep_transforms_and_saved_counts(bank_data, originals, sketch, method):
    X, _ = bank_data
    saved = pd.read_csv(ROOT / f"results/bank_{method}_sweep_raw.csv")
    for row in saved.itertuples(index=False):
        actual = sketch(X, row.k, row.seed)
        expected = originals[sketch.__name__](X, row.k, row.seed)
        if sp.issparse(actual):
            assert (actual != expected).nnz == 0
        else:
            assert isinstance(actual, np.ndarray)
            np.testing.assert_array_equal(actual, expected)
        assert actual.shape == (45211, row.k)
        assert matrix_nnz(actual) == row.sparse_size
        assert max_row_sparsity(actual) == row.max_sparsity
        assert qos_svm_machine_size(*actual.shape, max_row_sparsity(actual)) == row.qos_size


@pytest.mark.parametrize("sketch,method", [(balanced_feature_bucket, "bucket"),
                                          (sparse_jl_features, "sparse_jl")])
def test_bank_k16_seed42_scores(bank_data, sketch, method):
    X, y = bank_data
    projected = sketch(X, 16, seed=42)
    saved = pd.read_csv(ROOT / f"results/bank_{method}_sweep_raw.csv")
    row = saved[(saved.k == 16) & (saved.seed == 42)].iloc[0]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for weight, scoring, field in [(None, "accuracy", "paper_accuracy"),
                                    ("balanced", "balanced_accuracy", "balanced_accuracy")]:
        scores = cross_val_score(RidgeClassifier(alpha=10., solver="auto", class_weight=weight),
                                 projected, y, cv=cv, scoring=scoring)
        np.testing.assert_allclose(scores.mean(), row[field], rtol=0, atol=1e-8)
