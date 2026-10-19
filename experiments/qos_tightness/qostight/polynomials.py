"""QSVT polynomials used by the theorem's constructions (Lemma D.21, Theorem D.23)."""
import math

import numpy as np
from numpy.polynomial import Chebyshev
from scipy.special import erf, erfcinv


def sign_degree(s, eps=1e-4):
    """Odd degree d >= ln(1/eps)/lambda*, lambda* = sin(pi/(4s+2)) (Lemma D.21: d = O(log(1/eps)/lambda*))."""
    d = math.ceil(math.log(1 / eps) / math.sin(math.pi / (4 * s + 2)))
    return d if d % 2 else d + 1


def amplification_polynomial(s, eps=1e-4, max_degree=20001):
    """Odd P, |P| <= 1 on [-1, 1], |P(x) - (s/2) x| <= eps for |x| <= 1/s (Theorem D.23 amplification).

    Target (s/2) x * w(x) with an erf window w ~ 1 on |x| <= 1/s and ~ 0 beyond 2/s, so the
    target peaks below 1; Chebyshev interpolation degree then grows linearly in s.
    """
    k = 2 * erfcinv(eps / 4) * s  # window edge sharpness, set so the window error is << eps
    c = 2.0 / s
    def target(x):
        return (s / 2) * x * 0.5 * (erf(k * (x + c)) - erf(k * (x - c)))
    x = np.linspace(-1, 1, 20001)
    inner = x[np.abs(x) <= 1 / s]
    deg = 2 * math.ceil(k * 2) + 1
    while deg <= max_degree:
        P = Chebyshev.interpolate(target, deg)
        coef = P.coef.copy(); coef[0::2] = 0.0  # enforce exact oddness
        P = Chebyshev(coef)
        if (np.max(np.abs(P(inner) - s / 2 * inner)) <= eps and np.max(np.abs(P(x))) <= 1.0):
            return P
        deg = 2 * math.ceil(deg * 0.6) + 1
    raise RuntimeError(f"no valid amplification polynomial up to degree {max_degree} for s={s}")
