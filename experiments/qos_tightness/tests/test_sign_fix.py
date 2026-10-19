"""Fix for the sign-QSVT path: verified-bounded sign polynomial, D.23 precision rule, verified angles."""
import math

import numpy as np
import pytest

from qostight.matrices import circulant_sparse


def lam_star(s):
    return math.sin(math.pi / (4 * s + 2))


@pytest.mark.parametrize("s", [2, 8, 16, 32])
def test_sign_polynomial_bounded_odd_and_accurate(s):
    from qostight.polynomials import sign_polynomial
    eps = 1e-6
    P, info = sign_polynomial(s, eps=eps)
    x = np.linspace(-1, 1, 200001)
    assert np.max(np.abs(P(x))) <= 1.0
    np.testing.assert_allclose(P(-x), -P(x), atol=1e-13)
    outer = x[np.abs(x) >= lam_star(s)]
    assert np.max(np.abs(P(outer) - (1 - eps) * np.sign(outer))) <= eps
    assert info["achieved_error"] <= eps and info["bounded"] and P.degree() % 2 == 1


def test_sign_polynomial_fixed_degree_reports_floor_but_stays_bounded():
    from qostight.polynomials import sign_polynomial
    P, info = sign_polynomial(16, eps=1e-6, degree=101)
    assert P.degree() == 101 and info["bounded"]
    assert np.max(np.abs(P(np.linspace(-1, 1, 200001)))) <= 1.0
    assert info["achieved_error"] > 1e-6  # degree 101 cannot reach eps at s = 16: the true fixed-degree floor


def test_d23_oracle_precision_rule():
    from qostight.polynomials import d23_oracle_eps
    assert d23_oracle_eps(1e-4, 8) == pytest.approx(1e-4 / (8 * math.log(1e4)))


def test_verified_angles_accepts_valid_and_rejects_unbounded():
    from qostight.qsvt_tools import verified_qsvt_angles, AngleVerificationError
    from qostight.polynomials import sign_polynomial
    P, _ = sign_polynomial(8, eps=1e-6)
    angles, check = verified_qsvt_angles(P)
    assert check["max_dev"] <= 1e-10
    with pytest.raises(AngleVerificationError):
        verified_qsvt_angles(P * 1.002)  # |P| > 1: no valid phases exist


def test_reference_pyqsp_path_records_failure_at_s16_instead_of_raising():
    from qostight.blockenc import sign_angles
    out = sign_angles(16, method="pyqsp", degree=101)
    assert out["method"] == "pyqsp PolySign (reference path)" and out["verified"] is False
    assert out["max_dev"] > 1e-3


def test_regression_s8_floor_passes_after_fix():
    from qostight.blockenc import sketched_block_encoding, d23_amplify
    A = circulant_sparse(16, 16, 8, seed=42)
    be = sketched_block_encoding(A, 10**10)
    assert be["sign_method"] == "verified Chebyshev erf (primary)"
    assert np.linalg.norm(d23_amplify(be)["block"] - A / 2, 2) <= 1e-3


def test_regression_s16_converges_after_fix():
    from qostight.blockenc import sketched_block_encoding, d23_amplify
    A = circulant_sparse(32, 32, 16, seed=42)
    be = sketched_block_encoding(A, 10**10)
    assert np.linalg.norm(d23_amplify(be)["block"] - A / 2, 2) <= 1e-3
