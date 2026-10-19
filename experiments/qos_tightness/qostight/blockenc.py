"""Sparse block encoding from expected-unitary QOS oracles (reference qos.py), plus amplification.

Primary path (Theorem D.23): sketched A/s block encoding (Lemma D.22 via the reference's
sparse index and element oracles) followed by odd-polynomial singular value amplification
to A/2. Comparison path, labelled "optimistic: reference convention": multiply by s
classically, as the reference tests do (not a physical operation; no extra queries).
"""
import math
from functools import lru_cache

import numpy as np

from .env import load_reference
from .polynomials import amplification_polynomial
from .qsvt_tools import singular_value_transform

SIGN_SCALE = 0.9999  # reference default target magnitude of the pyqsp sign polynomial
PRIMARY_SIGN = "verified Chebyshev erf (primary)"
REFERENCE_SIGN = "pyqsp PolySign (reference path)"


@lru_cache(maxsize=None)
def sign_angles(s, method="verified", degree=None, eps=None):
    """Sign-QSVT angles for the sparse index oracle of sparsity s.

    method="verified": our bounded erf polynomial; with degree=None the degree is chosen to reach eps
      (primary), with degree fixed it is the a2'(ii) comparison. Angles must reproduce P (else raise).
    method="pyqsp": the reference's qsvt.get_qsvt_angles_sign (threshold 0.8*pi/(4s+2), rescale 0.9999),
      the a2'(i) comparison. Its verification result is RECORDED, not enforced.
    """
    from .polynomials import sign_polynomial
    from .qsvt_tools import check_angles, verified_qsvt_angles
    ref = load_reference()
    if method == "verified":
        P, info = sign_polynomial(s, eps=eps if eps is not None else 1e-6, degree=degree)
        angles, chk = verified_qsvt_angles(P, ref)
        return dict(angles=np.asarray(angles),
                    method=PRIMARY_SIGN if degree is None else "verified Chebyshev erf, fixed degree",
                    comparison=None if degree is None else "a2' (ii): verified-bounded, fixed degree",
                    degree=info["degree"], verified=True,
                    max_dev=chk["max_dev"], achieved_error=info["achieved_error"], eps=info["eps"])
    if method == "pyqsp":
        from numpy.polynomial import Chebyshev
        import pyqsp
        d = int(degree if degree is not None else 101)
        threshold = float(math.pi / (4 * s + 2) * 0.8)
        with ref.utils.suppress_stdout_stderr():
            angles, _ = ref.qsvt.get_qsvt_angles_sign(degree=d, threshold=threshold, rescale=SIGN_SCALE)
            coefs, _ = pyqsp.poly.PolySign().generate(degree=d, delta=2.0 / threshold, ensure_bounded=True,
                                                      return_scale=True, chebyshev_basis=True,
                                                      max_scale=SIGN_SCALE)
        P = Chebyshev(np.asarray(coefs, dtype=np.float64))
        dev = check_angles(P, np.asarray(angles), ref)
        grid = np.linspace(-1, 1, 200001)
        lam = math.sin(math.pi / (4 * s + 2))
        outer = np.abs(grid) >= lam
        return dict(angles=np.asarray(angles), method=REFERENCE_SIGN,
                    comparison="a2' (i): reference pyqsp path, failure recorded as found", degree=d,
                    verified=bool(dev <= 1e-10), max_dev=dev, max_abs=float(np.max(np.abs(P(grid)))),
                    achieved_error=float(np.max(np.abs(P(grid[outer]) - SIGN_SCALE * np.sign(grid[outer])))),
                    eps=None)
    raise ValueError(method)


def sketched_block_encoding(A, unit_num_samples, sign_degree=None, sign_method="verified", eps_target=1e-4,
                            return_oracles=False):
    """Expected-unitary QOS block encoding of A / sqrt(s_r s_c) (= A/s for uniform sparsity).

    Primary: verified sign polynomial at the Theorem D.23 oracle precision eps / (s ln(1/eps)).
    """
    from .polynomials import d23_oracle_eps
    ref = load_reference()
    import jax.numpy as jnp
    A = np.asarray(A, dtype=np.float64)
    s_row, s_col = int((A != 0).sum(1).max()), int((A != 0).sum(0).max())
    signs, oracles = {}, {}
    for axis, s_ax, name in ((0, s_row, "row_index"), (1, s_col, "col_index")):
        signs[name] = sign_angles(s_ax, method=sign_method, degree=sign_degree,
                                  eps=d23_oracle_eps(eps_target, s_ax) if sign_method == "verified" else None)
    with ref.utils.suppress_stdout_stderr():
        element, n_ele = ref.qos.q_oracle_sketch_matrix_element(jnp.asarray(A), int(unit_num_samples))
        for axis, name in ((0, "row_index"), (1, "col_index")):
            oracles[name] = ref.qos.q_oracle_sketch_matrix_index(
                jnp.asarray(A), int(unit_num_samples), axis=axis, degree=signs[name]["degree"],
                scale=SIGN_SCALE, angle_set=jnp.asarray(signs[name]["angles"]))
    B = ref.utils.block_encoding_from_sparse_oracles(oracles["row_index"][0], oracles["col_index"][0], element)
    samples = {"element": int(n_ele), "row_index": int(oracles["row_index"][1]),
               "col_index": int(oracles["col_index"][1])}
    samples["per_be_query"] = samples["row_index"] + samples["col_index"] + 2 * samples["element"]
    info = {k: {kk: vv for kk, vv in v.items() if kk != "angles"} for k, v in signs.items()}
    extra = ({"oracles": {"element": np.asarray(element), "row_index": np.asarray(oracles["row_index"][0]),
                          "col_index": np.asarray(oracles["col_index"][0])}} if return_oracles else {})
    return {**extra, "B": np.asarray(B).real, "B_imag_max": float(np.max(np.abs(np.asarray(B).imag))),
            "s_row": s_row, "s_col": s_col, "subnormalization": math.sqrt(s_row * s_col),
            "sign_degree": signs["row_index"]["degree"], "sign_method": signs["row_index"]["method"],
            "sign_info": info, "eps_target": eps_target,
            "unit_num_samples": int(unit_num_samples), "samples": samples}


def d23_amplify(be, eps=1e-4):
    """Theorem D.23: odd polynomial P ~ (s/2) x on |x| <= 1/s applied to the A/s block encoding."""
    s = be["subnormalization"]
    P = amplification_polynomial(int(round(s)), eps=eps)
    return {"block": singular_value_transform(be["B"], P), "degree": int(P.degree()),
            "poly_eps": eps, "label": "primary: Theorem D.23 amplification",
            "samples_total": int(P.degree()) * be["samples"]["per_be_query"]}


def classical_rescaling(be):
    return {"block": be["B"] * be["subnormalization"] / 2, "label": "optimistic: reference convention",
            "samples_total": be["samples"]["per_be_query"]}
