"""Phase-1 building blocks: guards, matrices, polynomials, QSVT shortcut, accounting."""
import math

import numpy as np
import pytest


# ---------------------------------------------------------------- environment / reference
def test_env_guard_accepts_qos_ref_venv_only():
    from qostight import env
    env.require_qos_ref_venv()  # Running under .venv-qos-ref.
    with pytest.raises(RuntimeError):
        env.require_qos_ref_venv(prefix="/Users/x/code/qos-rbc-experiments/.venv")


def test_reference_loader_pins_commit():
    from qostight import env
    ref = env.load_reference()
    assert ref.commit == env.REFERENCE_COMMIT == "a3fe8d6fe777750749f5e90131180e016088bdaf"
    for name in ("qos", "qsvt", "utils", "primitives"):
        assert hasattr(ref, name)
    with pytest.raises(RuntimeError):
        env.load_reference(expected_commit="0" * 40)


# ---------------------------------------------------------------- matrices
@pytest.mark.parametrize("N,D,s", [(16, 16, 1), (16, 16, 4), (32, 32, 8), (32, 16, 4)])
def test_circulant_matrix_uniform_sparsity_and_norm(N, D, s):
    from qostight.matrices import circulant_sparse
    X = circulant_sparse(N, D, s, seed=42)
    rows, cols = (X != 0).sum(1), (X != 0).sum(0)
    assert rows.max() == rows.min() == s
    assert cols.max() == cols.min() == s * N // D
    assert np.linalg.norm(X, 2) == pytest.approx(1.0)
    assert np.abs(X).max() <= 1.0
    np.testing.assert_array_equal(X, circulant_sparse(N, D, s, seed=42))
    assert not np.array_equal(X, circulant_sparse(N, D, s, seed=43))


# ---------------------------------------------------------------- sign degree
def test_sign_degree_scales_with_s_and_is_odd():
    from qostight.polynomials import sign_degree
    degrees = [sign_degree(s, eps=1e-4) for s in (2, 4, 8, 16, 32)]
    assert all(d % 2 == 1 for d in degrees)
    assert all(b > a for a, b in zip(degrees, degrees[1:]))
    # d = O(log(1/eps) / lambda*), lambda* = sin(pi / (4s + 2)) (Lemma D.21).
    for s, d in zip((2, 4, 8, 16, 32), degrees):
        assert d >= math.log(1e4) / math.sin(math.pi / (4 * s + 2))


# ---------------------------------------------------------------- amplification polynomial (D.23)
@pytest.mark.parametrize("s", [2, 4, 8, 16])
def test_amplification_polynomial_is_valid_qsvt_polynomial(s):
    from qostight.polynomials import amplification_polynomial
    eps = 1e-4
    P = amplification_polynomial(s, eps=eps)
    x = np.linspace(-1, 1, 20001)
    assert np.max(np.abs(P(x))) <= 1.0  # bounded on [-1, 1]
    np.testing.assert_allclose(P(-x), -P(x), atol=1e-12)  # odd
    inner = x[np.abs(x) <= 1 / s]
    assert np.max(np.abs(P(inner) - s / 2 * inner)) <= eps
    assert P.degree() % 2 == 1


def test_amplification_degree_grows_linearly_in_s():
    from qostight.polynomials import amplification_polynomial
    d = {s: amplification_polynomial(s, eps=1e-4).degree() for s in (4, 8, 16)}
    assert 1.5 < d[8] / d[4] < 2.6 and 1.5 < d[16] / d[8] < 2.6


# ---------------------------------------------------------------- singular-value transform
def test_singular_value_transform_matches_definition():
    from qostight.qsvt_tools import singular_value_transform
    rng = np.random.default_rng(42)
    B = rng.standard_normal((6, 4)); B /= 2 * np.linalg.norm(B, 2)
    P = np.polynomial.Polynomial([0, 3, 0, -1])  # odd
    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    np.testing.assert_allclose(singular_value_transform(B, P), U @ np.diag(P(S)) @ Vt, atol=1e-12)
    # Odd polynomial of the Hermitian embedding equals the SV transform off-diagonal.
    H = np.block([[np.zeros((6, 6)), B], [B.T, np.zeros((4, 4))]])
    w, V = np.linalg.eigh(H)
    PH = V @ np.diag(P(w)) @ V.T
    np.testing.assert_allclose(PH[:6, 6:], singular_value_transform(B, P), atol=1e-12)


def test_dense_qsvt_cross_check_agrees_with_shortcut_small():
    from qostight.qsvt_tools import dense_qsvt_vs_shortcut
    rng = np.random.default_rng(42)
    B = rng.standard_normal((4, 4)); B /= 1.25 * np.linalg.norm(B, 2)
    P = np.polynomial.Chebyshev([0, 0.5, 0, -0.2])  # odd, bounded by 0.7 on [-1, 1]
    result = dense_qsvt_vs_shortcut(B, P)
    assert result["max_abs_diff"] < 1e-6


# ---------------------------------------------------------------- accounting
def test_samples_per_block_encoding_query():
    from qostight.accounting import samples_per_be_query, amplified_be_samples
    per = samples_per_be_query(n_row_index=10, n_col_index=20, n_element=5)
    assert per == 10 + 20 + 2 * 5  # O_row, O_col, O_ele and O_ele^dagger (Lemma D.22)
    assert amplified_be_samples(per, amplification_degree=7) == 7 * per
