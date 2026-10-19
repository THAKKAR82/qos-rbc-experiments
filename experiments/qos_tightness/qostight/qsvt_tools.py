"""Singular-value QSVT shortcut and its dense-circuit cross-check.

Under the reference convention (fixed, data-independent garbage blocks; Halmos completion),
QSVT with an odd polynomial P on a block encoding of B yields P^(SV)(B) = U P(S) V^T.
dense_qsvt_vs_shortcut checks this against the reference's explicit apply_qsvt circuit.
"""
import numpy as np


def singular_value_transform(B, P):
    U, S, Vt = np.linalg.svd(np.asarray(B), full_matrices=False)
    return U @ np.diag(P(S)) @ Vt


def qsvt_angles(P, ref):
    """Reference convention: sym_qsp phases in the Chebyshev basis, converted as in qsvt.get_qsvt_angles."""
    import jax.numpy as jnp
    from pyqsp import angle_sequence
    phi, _, _ = angle_sequence.QuantumSignalProcessingPhases(
        np.asarray(P.coef, dtype=np.float64), method="sym_qsp", chebyshev_basis=True)
    phi_to_angle = jnp.array([1 / 4] + [1 / 2] * (phi.shape[0] - 2) + [1 / 4]) * jnp.pi
    return phi + phi_to_angle


def dense_qsvt_vs_shortcut(B, P, ref=None):
    """Run the reference's dense apply_qsvt on the Halmos dilation of B's Hermitian embedding."""
    from .env import load_reference
    ref = ref or load_reference()
    import jax.numpy as jnp
    m, n = B.shape
    H = np.block([[np.zeros((m, m)), B], [B.T, np.zeros((n, n))]])
    U = ref.utils.halmos_dilation(jnp.asarray(H, dtype=jnp.complex128))
    V = ref.utils.hermitian_block_encoding(U)
    Vq = ref.qsvt.apply_qsvt(V, num_ancilla=2, angle_set=qsvt_angles(P, ref))
    block = np.asarray(ref.utils.get_block_encoded(Vq, num_ancilla=2))
    shortcut = singular_value_transform(B, P)
    return {"max_abs_diff": float(np.max(np.abs(block.real[:m, m:] - shortcut))),
            "max_abs_imag": float(np.max(np.abs(block.imag))),
            "diag_block_max": float(max(np.max(np.abs(block.real[:m, :m])),
                                        np.max(np.abs(block.real[m:, m:]))))}


class AngleVerificationError(RuntimeError):
    pass


def check_angles(P, angles, ref=None, n=2001):
    """Max |QSVT response - P| on [-1, 1], using the reference's apply_qsvt_diag on 2x2 blocks."""
    from .env import load_reference
    import jax.numpy as jnp
    ref = ref or load_reference()
    x = np.linspace(-1, 1, n)
    c = np.sqrt(1 - x**2)
    U = jnp.stack([jnp.asarray(x), jnp.asarray(c), jnp.asarray(c), -jnp.asarray(x)]).reshape(2, 2, -1)
    out = np.asarray(ref.qsvt.apply_qsvt_diag(U, num_ancilla=1, angle_set=jnp.asarray(angles))[0, 0]).real
    return float(np.max(np.abs(out - P(x))))


def verified_qsvt_angles(P, ref=None, tol=1e-10):
    """sym_qsp phases for P (reference conversion), rejected unless the circuit reproduces P within tol."""
    import contextlib, io
    from .env import load_reference
    ref = ref or load_reference()
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            angles = qsvt_angles(P, ref)
        except Exception as exc:  # pyqsp may raise for invalid polynomials
            raise AngleVerificationError(f"phase finding failed: {exc}") from exc
    dev = check_angles(P, angles, ref)
    if not dev <= tol:
        raise AngleVerificationError(f"QSVT response deviates from P by {dev:.3e} > {tol:g}")
    return angles, {"max_dev": dev, "degree": int(P.degree())}
