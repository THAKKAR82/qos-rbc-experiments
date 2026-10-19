"""Phase 1: calibration of feasible sizes, Theorem D.23 validation, dense-QSVT cross-check.

Simulation (expected-unitary QOS, reference convention). Not a hardware estimate.
"""
import numpy as np

from .blockenc import classical_rescaling, d23_amplify, sketched_block_encoding
from .matrices import circulant_sparse
from .polynomials import amplification_polynomial
from .qsvt_tools import dense_qsvt_vs_shortcut, singular_value_transform
from .resources import measured


def spec(a):
    return float(np.linalg.norm(a, 2))


def calibrate_one(N, s, unit_num_samples, poly_eps=1e-4, seed=42):
    A = circulant_sparse(N, N, s, seed=seed)
    with measured() as m:
        be = sketched_block_encoding(A, unit_num_samples)
        amp = d23_amplify(be, eps=poly_eps)
    return {"status": "ok" if m["gate_violation"] is None else "memory_gate", "N": N, "D": N, "s": s,
            "unit_num_samples": int(unit_num_samples), "index_oracle_dim": N * s * N,
            "sign_degree": be["sign_degree"], "amplification_degree": amp["degree"],
            "seconds": m["seconds"], "peak_rss_bytes": int(m["peak_rss_bytes"]),
            "be_error_spec": spec(amp["block"] - A / 2), "gate_violation": m["gate_violation"]}


def d23_validation(N, s, M_grid, poly_eps=1e-4, seed=42, large_M_tol=1e-3):
    """Exact-input check (must be <= poly_eps) and sketched convergence to A/2 (largest M <= large_M_tol)."""
    A = circulant_sparse(N, N, s, seed=seed)
    P = amplification_polynomial(s, eps=poly_eps)
    exact_err = spec(singular_value_transform(A / s, P) - A / 2)
    rows = [{"N": N, "s": s, "input": "exact A/s", "path": "primary: Theorem D.23 amplification",
             "unit_num_samples": None, "error_spec": exact_err, "criterion": f"<= poly_eps {poly_eps:g}",
             "pass": exact_err <= poly_eps, "amplification_degree": int(P.degree()), "samples_total": None}]
    M_max = max(M_grid)
    for M in M_grid:
        be = sketched_block_encoding(A, M)
        rows.append({"N": N, "s": s, "input": "sketched (diagnostic)", "path": "subnormalized A/s vs A/s (no amplification)",
                     "unit_num_samples": int(M), "error_spec": spec(be["B"] - A / s), "criterion": "diagnostic",
                     "pass": None, "amplification_degree": None,
                     "samples_total": be["samples"]["per_be_query"]})
        for out in (d23_amplify(be, eps=poly_eps), classical_rescaling(be)):
            err = spec(out["block"] - A / 2)
            is_max = M == M_max
            rows.append({"N": N, "s": s, "input": "sketched", "path": out["label"], "unit_num_samples": int(M),
                         "error_spec": err,
                         "criterion": f"<= {large_M_tol:g} at largest M" if is_max else "diagnostic",
                         "pass": (err <= large_M_tol) if is_max else None,
                         "amplification_degree": out.get("degree"), "samples_total": out["samples_total"]})
    return rows


def _pseudo_inverse_polynomial(kappa, eps=1e-3):
    import pyqsp
    from numpy.polynomial import Chebyshev
    from .env import load_reference
    ref = load_reference()
    with ref.utils.suppress_stdout_stderr():
        coefs, scale = pyqsp.poly.PolyOneOverX().generate(kappa=kappa, epsilon=eps, return_coef=True,
                                                          ensure_bounded=True, return_scale=True,
                                                          chebyshev_basis=True)
    return Chebyshev(np.asarray(coefs, dtype=np.float64)), float(np.asarray(scale).ravel()[0])


def dense_crosscheck(N, s, seed=42, unit_num_samples=10**6, lam=0.3):
    """Dense apply_qsvt (reference) vs singular-value shortcut, on a sketched A/s and on an augmented A_lambda."""
    A = circulant_sparse(N, N, s, seed=seed)
    be = sketched_block_encoding(A, unit_num_samples)
    P_amp = amplification_polynomial(s, eps=1e-4)
    r1 = dense_qsvt_vs_shortcut(be["B"], P_amp)
    A_lam = 0.999 * np.vstack([A, np.sqrt(lam) * np.eye(N)]) / np.sqrt(1 + lam)
    sv = np.linalg.svd(A_lam, compute_uv=False)
    kappa = float(np.ceil(sv.max() / sv.min()))
    P_inv, scale = _pseudo_inverse_polynomial(kappa)
    r2 = dense_qsvt_vs_shortcut(A_lam, P_inv)
    return [dict(N=N, s=s, polynomial="D.23 amplification", degree=int(P_amp.degree()),
                 input="sketched A/s (M=%g)" % unit_num_samples, **r1),
            dict(N=N, s=s, polynomial="1/x pseudo-inverse", degree=int(P_inv.degree()), kappa=kappa,
                 inverse_scale=scale, input=f"0.999 * A_lambda (lambda={lam})", **r2)]
