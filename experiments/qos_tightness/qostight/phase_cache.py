"""Opt-in, equivalent symmetric-QSP Newton synthesis and verified disk reuse.

The 3-vector recurrence and FFT are those of pyqsp 0.2.0 sym_qsp_opt;
only evaluation across Chebyshev nodes is batched. No polynomial is changed.
Upstream and the default Phase-(a)/(b) backend remain untouched.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from pyqsp.sym_qsp_opt import SymmetricQSPProtocol

from .qsvt_tools import AngleVerificationError, check_angles


class BatchedSymmetricQSP(SymmetricQSPProtocol):
    def gen_jacobian(self):
        n = len(self.reduced_phases)
        theta = np.arange(n + 1) * (np.pi / (2*n))
        # Match upstream arccos(cos(theta)), including its endpoint rounding.
        t = np.arccos(np.cos(theta))
        ct, st = np.cos(2*t), np.sin(2*t)
        cp, sp = np.cos(2*self.reduced_phases), np.sin(2*self.reduced_phases)
        left = np.zeros((n, n+1, 3))
        left[-1, :, 1] = 1
        for k in range(n-2, -1, -1):
            l = left[k+1]
            a = cp[k+1]*l[:, 0] + sp[k+1]*l[:, 1]
            b = -sp[k+1]*l[:, 0] + cp[k+1]*l[:, 1]
            left[k, :, 0] = ct*a + st*l[:, 2]
            left[k, :, 1] = b
            left[k, :, 2] = -st*a + ct*l[:, 2]
        right = np.zeros((n+1, 3))
        if self.parity == 0:
            right[:, 0] = 1
        else:
            right[:, 0], right[:, 2] = np.cos(t), np.sin(t)
        mat = np.zeros((4*n, n+1))
        for k in range(n):
            mat[:n+1, k] = 2*(left[k, :, 0]*(-sp[k]*right[:, 0]-cp[k]*right[:, 1])
                                      + left[k, :, 1]*(cp[k]*right[:, 0]-sp[k]*right[:, 1]))
            if k == n-1:
                mat[:n+1, n] = sp[k]*right[:, 0]+cp[k]*right[:, 1]
            else:
                a = cp[k]*right[:, 0]-sp[k]*right[:, 1]
                b = sp[k]*right[:, 0]+cp[k]*right[:, 1]
                z = right[:, 2].copy()
                right[:, 0], right[:, 1], right[:, 2] = ct*a-st*z, b, st*a+ct*z
        mat[n+1:2*n+1] = (-1)**self.parity * mat[n-1::-1]
        mat[2*n+1:] = mat[2*n-1:0:-1]
        # Real FFT gives exactly the same nonnegative-frequency coefficients.
        transformed = np.fft.rfft(mat, axis=0).real
        transformed[1:-1] *= 2
        transformed /= 4*n
        return (transformed[self.parity:2*n:2, -1].copy(),
                transformed[self.parity:2*n:2, :-1].copy())


def batched_newton(coef, parity, crit=1e-12, maxiter=100):
    """Same start, update order, 1-norm criterion and iteration limit as upstream."""
    seq = BatchedSymmetricQSP(reduced_phases=np.asarray(coef)/2, parity=parity)
    for iteration in range(1, maxiter+1):
        value, jacobian = seq.gen_jacobian()
        residual = value-coef
        error = float(np.linalg.norm(residual, ord=1))
        seq.update_reduced_phases(seq.reduced_phases-np.linalg.solve(jacobian, residual))
        if error < crit:
            return seq, error, iteration
    raise AngleVerificationError(f'Newton did not converge in {maxiter} iterations: {error}')


def _array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.asarray(a, dtype='<f8').tobytes())
    return h.hexdigest()


def cached_verified_angles(poly, cache_dir=None, tol=1e-10):
    """Every hit and miss passes the original 2,001-node reference circuit check.

    Cache identity includes exact coefficients, conventions, code, pinned
    reference and dependency versions. Corrupt entries fail closed. No pickle.
    """
    from .env import REFERENCE_COMMIT, load_reference
    from .resources import GIB, gate_ok
    import psutil
    import pyqsp.sym_qsp_opt
    started = time.perf_counter()
    ref = load_reference()
    coef = np.asarray(poly.coef, dtype='<f8')
    even, odd = np.any(np.abs(coef[::2]) > 1e-8), np.any(np.abs(coef[1::2]) > 1e-8)
    if even == odd:
        raise AngleVerificationError('polynomial must have definite nonzero parity')
    parity = 0 if even else 1
    identity = dict(schema=1, backend='batched-pyqsp-symmetric-newton',
                    coefficient_sha256=_array_hash(coef), coefficient_count=len(coef),
                    parity=parity, newton_tol=1e-12, maxiter=100, response_tol=tol,
                    response_nodes=2001, conversion='pi*[1/4,1/2,...,1/4]',
                    reference_commit=REFERENCE_COMMIT,
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    upstream_sha256=hashlib.sha256(Path(pyqsp.sym_qsp_opt.__file__).read_bytes()).hexdigest(),
                    versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','pyqsp','jax')})
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir)/f'{key}.npz' if cache_dir else None
    hit = path is not None and path.exists()
    if hit:
        with np.load(path, allow_pickle=False) as stored:
            metadata = json.loads(str(stored['metadata']))
            angles = stored['angles'].copy()
            if (metadata['identity'] != identity or not np.array_equal(stored['coefficients'], coef)
                    or metadata['array_sha256'] != _array_hash(coef, angles)):
                raise AngleVerificationError(f'cache integrity failed: {path}')
        synthesis = metadata['synthesis']
    else:
        n = len(coef[parity::2])
        rss, available = psutil.Process().memory_info().rss, psutil.virtual_memory().available
        # Conservative workspace preflight; runtime RSS/available monitor remains required.
        if not gate_ok(available, rss, before_stage=True) or rss+160*n*n > 4*GIB:
            raise MemoryError(f'phase-synthesis memory preflight failed: n={n}, rss={rss}')
        t = time.perf_counter()
        seq, err, iterations = batched_newton(coef[parity::2], parity)
        phi = seq.full_phases
        conversion = np.full(len(phi), np.pi/2)
        conversion[0] = conversion[-1] = np.pi/4
        if len(phi) == 1:
            raise AngleVerificationError('degree-zero conversion is outside this prototype')
        angles = phi+conversion
        synthesis = dict(seconds=time.perf_counter()-t, residual_l1=err, iterations=iterations)
    if angles.shape != coef.shape or not np.all(np.isfinite(angles)):
        raise AngleVerificationError('invalid phase shape or nonfinite phases')
    t = time.perf_counter()
    deviation = check_angles(poly, angles, ref)
    verification_seconds = time.perf_counter()-t
    if not deviation <= tol:
        raise AngleVerificationError(f'QSVT response deviates from P by {deviation:.3e} > {tol:g}')
    if path is not None and not hit:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(identity=identity, synthesis=synthesis, array_sha256=_array_hash(coef, angles))
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as f:
            temp = Path(f.name)
            np.savez(f, coefficients=coef, angles=angles, metadata=json.dumps(metadata, sort_keys=True))
        temp.replace(path)
    return angles, dict(max_dev=deviation, degree=int(poly.degree()), cache_hit=hit, cache_key=key,
                        synthesis=synthesis, verification_seconds=verification_seconds,
                        elapsed_seconds=time.perf_counter()-started)
