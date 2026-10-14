"""Bank row-only accounting; do not substitute reference row-or-column maxima."""
import numpy as np
import pytest
from scipy import sparse

from src.machine_size import matrix_nnz, max_row_sparsity, qos_svm_machine_size


@pytest.mark.parametrize("constructor", [np.array, sparse.csr_matrix, sparse.csc_matrix,
                                         sparse.coo_matrix, sparse.csr_array])
def test_dense_sparse_counts_and_non_mutation(constructor):
    X = constructor([[1., 0., 2.], [3., 0., 0.], [4., 0., 0.], [5., 0., 0.]])
    before = X.copy()
    assert matrix_nnz(X) == 5
    assert max_row_sparsity(X) == 2  # Column maximum is 4; deliberately not used.
    if sparse.issparse(X):
        np.testing.assert_array_equal(X.data, before.data)
        if hasattr(X, 'indices'):
            np.testing.assert_array_equal(X.indices, before.indices)
            np.testing.assert_array_equal(X.indptr, before.indptr)
        else:
            np.testing.assert_array_equal(X.row, before.row)
            np.testing.assert_array_equal(X.col, before.col)
    else:
        np.testing.assert_array_equal(X, before)


@pytest.mark.parametrize("constructor", [sparse.csr_matrix, sparse.csr_array])
def test_explicit_zeros_duplicates_and_no_densification(constructor, monkeypatch):
    # Duplicate entries cancel in row 0; row 1 contains an explicit zero.
    X = constructor((np.array([2., -2., 0., 5.]), np.array([0, 0, 1, 2]),
                     np.array([0, 2, 4])), shape=(2, 3))
    before = (X.data.copy(), X.indices.copy(), X.indptr.copy(), X.has_canonical_format)
    def forbidden(*args, **kwargs):
        raise AssertionError("Sparse accounting must not densify")
    monkeypatch.setattr(constructor, 'toarray', forbidden)
    monkeypatch.setattr(constructor, 'todense', forbidden)
    assert matrix_nnz(X) == 1
    assert max_row_sparsity(X) == 1
    for actual, expected in zip((X.data, X.indices, X.indptr), before[:3]):
        np.testing.assert_array_equal(actual, expected)
    assert X.has_canonical_format == before[3]


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 0), (3, 4)])
def test_empty_and_zero_matrices(shape):
    for X in [np.zeros(shape), sparse.csr_matrix(shape)]:
        assert matrix_nnz(X) == max_row_sparsity(X) == 0


@pytest.mark.parametrize("N,D,s,expected", [
    (45211, 51, 16, 41),
    (5, 1, 0, 10), (6, 1, 0, 10), (7, 1, 0, 12),  # N+2D = 7,8,9
    (1, 4, 2, 14), (1, 4, 3, 14), (1, 4, 4, 15),  # s+1 = 3,4,5
    (2**54 - 3, 1, 0, 112), (2**54 - 2, 1, 0, 112),
    (2**54 - 1, 1, 0, 114),  # Exact integer boundaries beyond float precision.
])
def test_formula_boundaries(N, D, s, expected):
    assert qos_svm_machine_size(N, D, s) == expected
    assert isinstance(qos_svm_machine_size(N, D, s), int)


def test_numpy_integer_arguments():
    assert qos_svm_machine_size(np.int64(45211), np.int64(51), np.int64(16)) == 41


@pytest.mark.parametrize("args,error", [
    ((0, 1, 0), ValueError), ((1, 0, 0), ValueError), ((1, 1, -1), ValueError),
    ((1.5, 1, 0), TypeError), ((True, 1, 0), TypeError),
])
def test_invalid_formula_arguments(args, error):
    with pytest.raises(error):
        qos_svm_machine_size(*args)


@pytest.mark.parametrize("X", [np.array(1), np.zeros(3), np.zeros((1, 2, 3))])
def test_non_matrix_inputs(X):
    for function in [matrix_nnz, max_row_sparsity]:
        with pytest.raises(ValueError, match="two-dimensional"):
            function(X)
