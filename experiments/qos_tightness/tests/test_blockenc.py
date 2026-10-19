"""Sketched sparse block encoding (Lemmas D.21/D.22) and D.23 amplification, expected-unitary model."""
import numpy as np
import pytest

from qostight.matrices import circulant_sparse


@pytest.fixture(scope="module")
def A():
    return circulant_sparse(8, 8, 2, seed=42)


def test_sketched_block_encoding_converges_to_A_over_s(A):
    from qostight.blockenc import sketched_block_encoding
    out = sketched_block_encoding(A, unit_num_samples=10**10)
    assert out["s_row"] == out["s_col"] == 2
    assert np.linalg.norm(out["B"] - A / 2, 2) <= 1e-3  # A / sqrt(s_r s_c) = A / s


def test_sketched_block_encoding_error_decreases_with_samples(A):
    from qostight.blockenc import sketched_block_encoding
    errs = [np.linalg.norm(sketched_block_encoding(A, unit_num_samples=m)["B"] - A / 2, 2)
            for m in (10**4, 10**5, 10**6)]
    assert errs[0] > errs[1] > errs[2]


def test_sample_counts_follow_reference_and_lemma_d22(A):
    from qostight.blockenc import sketched_block_encoding
    out = sketched_block_encoding(A, unit_num_samples=1000)
    n = out["samples"]
    assert n["element"] == 1000
    # Reference: unit samples * (angle count - 1) * index bit length, per index oracle.
    assert n["row_index"] == 1000 * out["sign_degree"] * 3 == n["col_index"]
    assert n["per_be_query"] == n["row_index"] + n["col_index"] + 2 * n["element"]


def test_sign_degree_override_is_used(A):
    from qostight.blockenc import sketched_block_encoding
    assert sketched_block_encoding(A, unit_num_samples=1000, sign_degree=101)["sign_degree"] == 101


def test_d23_amplification_recovers_A_over_2(A):
    from qostight.blockenc import sketched_block_encoding, d23_amplify, classical_rescaling
    be = sketched_block_encoding(A, unit_num_samples=10**10)
    amp = d23_amplify(be, eps=1e-4)
    assert np.linalg.norm(amp["block"] - A / 2, 2) <= 2e-3
    assert amp["samples_total"] == amp["degree"] * be["samples"]["per_be_query"]
    opt = classical_rescaling(be)
    assert opt["label"] == "optimistic: reference convention"
    assert np.linalg.norm(opt["block"] - A / 2, 2) <= 2e-3
    assert opt["samples_total"] == be["samples"]["per_be_query"]
