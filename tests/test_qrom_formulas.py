"""QROM Toffoli formulas transcribed from arXiv:2605.20334 and arXiv:2610.02321 (no scientific computation)."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qrom_vs_qos", ROOT / "scripts/qrom_vs_qos.py")
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def test_exact_formulas_at_a_point():
    L, b, lam = 2**20, 32, 8
    assert q.toffoli("unary", L, b, lam) == L
    assert q.toffoli("selectswap_dirty", L, b, lam) == 2 * L / lam + 4 * b * (lam - 1)
    assert q.toffoli("selectcopy", L, b, lam) == 2 * L / lam + 2 * b * (lam - 1) + 2 * lam - 6
    assert q.toffoli("sequential_packets", L, b, lam) == (1 + 1 / b) * L / lam + 2 * (b + 1) * (b * lam - 2)
    assert q.toffoli("dense", L, b, lam) == L / lam + (4 * b + 2) * lam - 6
    assert q.toffoli("dense_sequential", L, b, lam) == 0.5 * (1 + 1 / b) * L / lam + 3 * b * (b + 1) * lam - 3 * (b + 1)
    assert q.dirty_qubits("selectswap_dirty", b, lam) == b * (lam - 1)
    assert q.dirty_qubits("sequential_packets", b, lam) == b * lam - 1
    assert q.dirty_qubits("dense_sequential", b, lam) == b * lam


def test_leading_term_ratios_match_papers():
    L, b, lam = 2**40, 32, 8  # huge L: leading terms dominate
    ss = q.toffoli("selectswap_dirty", L, b, lam)
    assert ss / q.toffoli("dense", L, b, lam) == pytest.approx(2.0, rel=1e-6)
    assert ss / q.toffoli("dense_sequential", L, b, lam) == pytest.approx(4 * b / (b + 1), rel=1e-6)  # ~3.9 at b=32
    assert ss / q.toffoli("sequential_packets", L, b, lam) == pytest.approx(2 * b / (b + 1), rel=1e-6)


def test_best_lambda_respects_dirty_budget_and_powers_of_two():
    r = q.best_lambda("dense_sequential", 10**6, 16, dirty_budget=256)
    assert r["lambda"] & (r["lambda"] - 1) == 0 and r["dirty_qubits"] <= 256
    free = q.best_lambda("selectswap_dirty", 10**6, 16, dirty_budget=None)
    assert free["toffoli"] <= q.toffoli("selectswap_dirty", 10**6, 16, 2)


def test_general_bounds_reduce_to_named_constructions_at_mu_equals_b():
    L, b, lam = 2**20, 16, 8
    # arXiv:2605.20334 Eq. A1 at mu=b (alpha=1) is SelectCopy exactly
    assert q.toffoli_general("sequential_packets", L, b, lam, mu=b) == q.toffoli("selectcopy", L, b, lam)
    # arXiv:2610.02321 Eq. 9 at mu=b (one packet) is dense encoding Eq. 6 exactly (2L rows-block divides N)
    assert q.toffoli_general("dense_sequential", L, b, lam, mu=b) == q.toffoli("dense", L, b, lam)
    assert q.dirty_general("sequential_packets", b, lam, mu=3) == 3 * (lam - 1)
    assert q.dirty_general("dense_sequential", b, lam, mu=3) == 3 * lam


def test_optimized_packet_width_never_worse_than_endpoints():
    for method, endpoint in (("sequential_packets", "selectcopy"), ("dense_sequential", "dense")):
        r = q.best_general(method, 10**6, 32, dirty_budget=None)
        # General bounds are integer (ceil) bounds; the named endpoints assume exact divisibility.
        # At mu=b the ceilings add at most (ceil(b/mu)+1) = 2 Toffolis.
        assert r["toffoli"] <= q.best_lambda(endpoint, 10**6, 32, dirty_budget=None)["toffoli"] + 2
        assert 1 <= r["mu"] <= 32
