"""Independent fidelity checks against upstream recurrence, phases and circuits."""
import contextlib
import io
import json

import numpy as np
import pytest
from numpy.polynomial import Chebyshev
from pyqsp.sym_qsp_opt import SymmetricQSPProtocol, newton_solver

from qostight.phase_cache import BatchedSymmetricQSP, batched_newton, cached_verified_angles, _array_hash
from qostight.qsvt_tools import AngleVerificationError


@pytest.mark.parametrize('parity', [0, 1])
@pytest.mark.parametrize('n', [1, 2, 9, 33])
def test_batched_value_and_jacobian_match_upstream(parity, n):
    phases = np.random.default_rng(42).uniform(-.3, .3, n)
    old = SymmetricQSPProtocol(phases, parity).gen_jacobian()
    new = BatchedSymmetricQSP(phases, parity).gen_jacobian()
    for a, b in zip(old, new):
        np.testing.assert_allclose(a, b, atol=2e-14, rtol=2e-13)


@pytest.mark.parametrize('parity', [0, 1])
def test_newton_matches_upstream_and_finite_difference(parity):
    coef = np.array([.3, .12, -.03, .002])
    seq, err, iterations = batched_newton(coef, parity)
    with contextlib.redirect_stdout(io.StringIO()):
        _, olderr, olditerations, oldseq = newton_solver(coef, parity)
    assert iterations == olditerations and err < 1e-12 and olderr < 1e-12
    np.testing.assert_allclose(seq.full_phases, oldseq.full_phases, atol=2e-14)
    _, jac = seq.gen_jacobian()
    for j in range(len(coef)):
        shift = np.eye(len(coef))[j]*1e-6
        plus = BatchedSymmetricQSP(seq.reduced_phases+shift, parity).gen_jacobian()[0]
        minus = BatchedSymmetricQSP(seq.reduced_phases-shift, parity).gen_jacobian()[0]
        np.testing.assert_allclose((plus-minus)/2e-6, jac[:, j], atol=3e-10)


def test_cache_hit_rechecks_response_identity_and_corruption(tmp_path, monkeypatch):
    p = Chebyshev([0, .4, 0, .1])
    angles, miss = cached_verified_angles(p, tmp_path)
    assert not miss['cache_hit'] and miss['max_dev'] < 1e-10
    def no_synthesis(*args, **kwargs):
        raise AssertionError('cache hit synthesized phases')
    monkeypatch.setattr('qostight.phase_cache.batched_newton', no_synthesis)
    cached, hit = cached_verified_angles(p, tmp_path)
    assert hit['cache_hit'] and hit['verification_seconds'] > 0
    np.testing.assert_array_equal(angles, cached)
    path = next(tmp_path.glob('*.npz'))
    with np.load(path, allow_pickle=False) as data:
        saved = {k:data[k] for k in data.files}
    saved['angles'] = saved['angles']+.1
    np.savez(path, **saved)
    with pytest.raises(AngleVerificationError, match='integrity'):
        cached_verified_angles(p, tmp_path)
    # Even internally consistent altered phases must fail the independent circuit check.
    metadata = json.loads(str(saved['metadata']))
    metadata['array_sha256'] = _array_hash(saved['coefficients'], saved['angles'])
    saved['metadata'] = json.dumps(metadata)
    np.savez(path, **saved)
    with pytest.raises(AngleVerificationError, match='response deviates'):
        cached_verified_angles(p, tmp_path)


def test_cache_key_changes_with_coefficients_and_tolerance(tmp_path):
    _, first = cached_verified_angles(Chebyshev([0,.4,0,.1]), tmp_path)
    _, different = cached_verified_angles(Chebyshev([0,.41,0,.1]), tmp_path)
    _, stricter = cached_verified_angles(Chebyshev([0,.4,0,.1]), tmp_path, tol=1e-11)
    assert len({first['cache_key'], different['cache_key'], stricter['cache_key']}) == 3
    assert not different['cache_hit'] and not stricter['cache_hit']


def test_newton_nonconvergence_is_rejected():
    with pytest.raises(AngleVerificationError, match='did not converge'):
        batched_newton(np.array([.9, .01]), 1, maxiter=1)
