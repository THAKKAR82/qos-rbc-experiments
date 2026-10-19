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
from .polynomials import amplification_polynomial, sign_degree as default_sign_degree
from .qsvt_tools import singular_value_transform

SIGN_SCALE = 0.9999  # reference default target magnitude of the sign polynomial


@lru_cache(maxsize=None)
def _sign_angles(degree, s):
    """Reference convention: threshold = 0.8 * pi / (4s + 2), rescale 0.9999 (qos.q_oracle_sketch_matrix_index)."""
    ref = load_reference()
    with ref.utils.suppress_stdout_stderr():
        angles, _ = ref.qsvt.get_qsvt_angles_sign(degree=degree, threshold=float(math.pi / (4 * s + 2) * 0.8),
                                                  rescale=SIGN_SCALE)
    return angles.astype(ref.utils.real_dtype)


def sketched_block_encoding(A, unit_num_samples, sign_degree=None, sign_eps=1e-4):
    """Expected-unitary QOS block encoding of A / sqrt(s_r s_c) (= A/s for uniform sparsity)."""
    ref = load_reference()
    import jax.numpy as jnp
    A = np.asarray(A, dtype=np.float64)
    s_row, s_col = int((A != 0).sum(1).max()), int((A != 0).sum(0).max())
    degrees = {}
    oracles = {}
    with ref.utils.suppress_stdout_stderr():
        element, n_ele = ref.qos.q_oracle_sketch_matrix_element(jnp.asarray(A), int(unit_num_samples))
        for axis, s_ax, name in ((0, s_row, "row_index"), (1, s_col, "col_index")):
            d = int(sign_degree) if sign_degree is not None else default_sign_degree(s_ax, sign_eps)
            degrees[name] = d
            oracles[name] = ref.qos.q_oracle_sketch_matrix_index(
                jnp.asarray(A), int(unit_num_samples), axis=axis, degree=d,
                scale=SIGN_SCALE, angle_set=_sign_angles(d, s_ax))
    B = ref.utils.block_encoding_from_sparse_oracles(oracles["row_index"][0], oracles["col_index"][0], element)
    samples = {"element": int(n_ele), "row_index": int(oracles["row_index"][1]),
               "col_index": int(oracles["col_index"][1])}
    samples["per_be_query"] = samples["row_index"] + samples["col_index"] + 2 * samples["element"]
    return {"B": np.asarray(B).real, "B_imag_max": float(np.max(np.abs(np.asarray(B).imag))),
            "s_row": s_row, "s_col": s_col, "subnormalization": math.sqrt(s_row * s_col),
            "sign_degree": degrees["row_index"], "sign_degrees": degrees,
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
