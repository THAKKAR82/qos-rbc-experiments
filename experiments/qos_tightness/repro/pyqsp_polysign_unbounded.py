"""Minimal reproduction: pyqsp PolySign(ensure_bounded=True) can return max|P| > 1 on [-1, 1].

Standalone: depends only on pyqsp (and its numpy/scipy dependencies); no repository code.
Parameters are the sign-QSVT settings of a sparse index oracle with sparsity s
(threshold = 0.8*pi/(4s+2), delta = 2/threshold, max_scale = 0.9999), as used by the
quantum-oracle-sketching reference (qsvt.get_qsvt_angles_sign).

For each case it reports:
  1. max|P| on a dense grid of [-1, 1] and by multi-start bounded optimization;
  2. whether sym_qsp phase finding converges (its own printed final error);
  3. the max deviation between the QSP response of the returned phases and P,
     using an independent numpy evaluator validated on the bounded control case.
Run:  python pyqsp_polysign_unbounded.py
"""
import contextlib
import io
import math
import platform
import re
import sys
from importlib.metadata import version

import numpy as np
import scipy
from scipy.optimize import minimize_scalar
from numpy.polynomial import Chebyshev
import pyqsp
from pyqsp import angle_sequence

CASES = [("control s=8", 8, 101), ("s=16", 16, 195), ("s=32", 32, 383)]


def max_abs_dense(P, n=200001):
    x = np.linspace(-1.0, 1.0, n)
    v = np.abs(P(x))
    return float(v.max()), float(x[v.argmax()])


def max_abs_multistart(P, starts=400):
    best = (0.0, 0.0)
    edges = np.linspace(-1.0, 1.0, starts + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        r = minimize_scalar(lambda t: -abs(P(t)), bounds=(a, b), method="bounded",
                            options={"xatol": 1e-13})
        if -r.fun > best[0]:
            best = (float(-r.fun), float(r.x))
    return best


def qsp_response(phi, x):
    """Im <0| e^{i phi_0 Z} prod_k W(x) e^{i phi_k Z} |0>, W(x) = [[x, i s], [i s, x]], s = sqrt(1-x^2).

    Symmetric-QSP (Wx) convention of pyqsp's sym_qsp in the Chebyshev basis: the target
    polynomial is the IMAGINARY part. Validated by the bounded control case (dev ~1e-14).
    """
    out = np.empty_like(x)
    for j, xv in enumerate(x):
        sv = math.sqrt(max(0.0, 1.0 - xv * xv))
        W = np.array([[xv, 1j * sv], [1j * sv, xv]])
        U = np.diag([np.exp(1j * phi[0]), np.exp(-1j * phi[0])])
        for p in phi[1:]:
            U = U @ W @ np.diag([np.exp(1j * p), np.exp(-1j * p)])
        out[j] = U[0, 0].imag
    return out


def main():
    print("python", platform.python_version(), "| numpy", np.__version__, "| scipy", scipy.__version__,
          "| pyqsp", version("pyqsp"))
    for label, s, degree in CASES:
        threshold = 0.8 * math.pi / (4 * s + 2)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            coefs, scale = pyqsp.poly.PolySign().generate(
                degree=degree, delta=2.0 / threshold, ensure_bounded=True, return_scale=True,
                chebyshev_basis=True, max_scale=0.9999)
        P = Chebyshev(np.asarray(coefs, dtype=float))
        dense, x_dense = max_abs_dense(P)
        multi, x_multi = max_abs_multistart(P)
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            phi, _, _ = angle_sequence.QuantumSignalProcessingPhases(
                np.asarray(coefs, dtype=float), method="sym_qsp", chebyshev_basis=True)
        errs = [float(m) for m in re.findall(r"err:\s*([0-9.eE+-]+)", buf2.getvalue())]
        converged = "Stop criteria satisfied" in buf2.getvalue()
        x = np.linspace(-1, 1, 801)
        dev = float(np.max(np.abs(qsp_response(np.asarray(phi), x) - P(x))))
        print(f"{label}: degree={degree} threshold={threshold:.5f} delta={2 / threshold:.2f}")
        print(f"  max|P| dense grid = {dense:.6f} at x={x_dense:+.5f}; multi-start = {multi:.6f} at x={x_multi:+.5f}"
              f"  -> {'UNBOUNDED (>1)' if max(dense, multi) > 1 else 'bounded'}")
        print(f"  sym_qsp: iterations={len(errs)} final err={errs[-1] if errs else float('nan'):.3e} "
              f"converged={converged}")
        print(f"  max |QSP response - P| on [-1,1] = {dev:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
