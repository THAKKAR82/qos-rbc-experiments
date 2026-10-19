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
