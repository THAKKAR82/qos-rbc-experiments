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


def d23_oracle_eps(eps, s):
    """Theorem D.23 instantiates each sparse oracle at eps3 = Theta(eps / (s log(1/eps)))."""
    return eps / (s * math.log(1 / eps))


def sign_polynomial(s, eps=1e-6, degree=None, max_degree=8001):
    """Odd, verified-bounded P ~ (1 - eps) sign(x) on |x| >= lambda* = sin(pi/(4s+2)) (Lemma D.21).

    Chebyshev interpolation of (1 - eps) erf(k x), k = erfcinv(eps/2) / lambda*. Boundedness and
    accuracy are checked on a dense grid. With `degree` fixed (the a2' comparison), the polynomial is
    rescaled if needed to stay bounded and its achieved error is reported, not guaranteed.
    """
    lam = math.sin(math.pi / (4 * s + 2))
    k = erfcinv(eps / 2) / lam
    x = np.linspace(-1, 1, 200001)
    outer = np.abs(x) >= lam
    def build(deg):
        P = Chebyshev.interpolate(lambda t: (1 - eps) * erf(k * t), deg)
        coef = P.coef.copy(); coef[0::2] = 0.0
        return Chebyshev(coef)
    def check(P):
        vals = P(x)
        return float(np.max(np.abs(vals))), float(np.max(np.abs(vals[outer] - (1 - eps) * np.sign(x[outer]))))
    if degree is not None:
        P = build(degree)
        peak, err = check(P)
        if peak > 1.0:
            P = P * ((1 - eps) / peak); peak, err = check(P)
        return P, {"degree": int(degree), "achieved_error": err, "max_abs": peak, "bounded": peak <= 1.0,
                   "lambda_star": lam, "eps": eps}
    deg = 2 * math.ceil(k) + 1
    while deg <= max_degree:
        P = build(deg)
        peak, err = check(P)
        if peak <= 1.0 and err <= eps:
            return P, {"degree": int(P.degree()), "achieved_error": err, "max_abs": peak, "bounded": True,
                       "lambda_star": lam, "eps": eps}
        deg = 2 * math.ceil(deg * 0.625) + 1
    raise RuntimeError(f"no bounded sign polynomial up to degree {max_degree} for s={s}, eps={eps}")
