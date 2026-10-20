"""QROM vs QOS per-query loading cost for the CFPB training matrix. Back-of-envelope resource estimate.

QROM Toffoli formulas are transcribed exactly from:
  - Motlagh & Pocrnic, "Halving the cost of QROM", arXiv:2605.20334 (SelectSwap baseline as quoted from
    Berry et al. 2019; SelectCopy; sequential bit packets, Eq. 7);
  - Motlagh, "Halving the cost of QROM again via dense encoding", arXiv:2610.02321 (Eqs. 6 and 8).
L = table entries, b = bits per entry, lambda = power-of-two block size. Leading-order formulas assume
power-of-two sizes; tables are padded accordingly (the papers' general bounds differ by rounding).
QOS per-query samples are read from results/audit/empirical_f61 (extrapolated; its padded-dimension
factor is the theorem's linear one, NOT measured).
Run: .venv/bin/python scripts/qrom_vs_qos.py
"""
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/audit/qrom_vs_qos"
EMPIRICAL = ROOT / "results/audit/empirical_f61/empirical_f61.csv"
LABEL = "back-of-envelope resource estimate"

# CFPB frozen training matrix (audit step 1; all January rows, the one-vs-rest matrix)
N_ROWS, D_COLS, NNZ, S_ROW, S_COL = 13_459, 50_000, 1_563_584, 1_254, 8_776
B_VALUE = 32                                   # float32 TF-IDF value
B_COL_INDEX = math.ceil(math.log2(D_COLS))     # row-index oracle outputs a column index: 16 bits
B_ROW_INDEX = math.ceil(math.log2(N_ROWS))     # column-index oracle outputs a row index: 14 bits
AMPLIFICATION_DEGREE_PER_S = 919 / 32          # measured degree of our D.23 polynomial at s=32 (phase (a))

METHODS = ("unary", "selectswap_dirty", "selectcopy", "sequential_packets", "dense", "dense_sequential")
# Table methods: alpha=b special cases (Eq. 7 / Eq. 8) are kept in toffoli() for the leading-term checks;
# the cost table uses the packet-width-optimized general bounds for the two packet methods.
GENERAL = {"sequential_packets": "arXiv:2605.20334 Theorem 1 (Eq. A1), optimal packet width mu",
           "dense_sequential": "arXiv:2610.02321 Eq. 9, optimal packet width mu"}
SOURCES = {"unary": "Babbush et al. 2018 (as quoted in arXiv:2605.20334): ~L",
           "selectswap_dirty": "Berry et al. 2019 (as quoted in arXiv:2605.20334): 2L/lam + 4b(lam-1)",
           "selectcopy": "arXiv:2605.20334 Sec. II.B: 2L/lam + 2b(lam-1) + 2lam - 6",
           "sequential_packets": "arXiv:2605.20334 Eq. 7 (alpha=b): (1+1/b)L/lam + 2(b+1)(b lam - 2)",
           "dense": "arXiv:2610.02321 Eq. 6: L/lam + (4b+2)lam - 6",
           "dense_sequential": "arXiv:2610.02321 Eq. 8: (1/2)(1+1/b)L/lam + 3b(b+1)lam - 3(b+1)"}


def toffoli(method, L, b, lam):
    if method == "unary":
        return L
    if method == "selectswap_dirty":
        return 2 * L / lam + 4 * b * (lam - 1)
    if method == "selectcopy":
        return 2 * L / lam + 2 * b * (lam - 1) + 2 * lam - 6
    if method == "sequential_packets":
        return (1 + 1 / b) * L / lam + 2 * (b + 1) * (b * lam - 2)
    if method == "dense":
        return L / lam + (4 * b + 2) * lam - 6
    if method == "dense_sequential":
        return 0.5 * (1 + 1 / b) * L / lam + 3 * b * (b + 1) * lam - 3 * (b + 1)
    raise ValueError(method)


def dirty_qubits(method, b, lam):
    return {"unary": 0, "selectswap_dirty": b * (lam - 1), "selectcopy": b * (lam - 1),
            "sequential_packets": b * lam - 1, "dense": b * lam, "dense_sequential": b * lam}[method]


def best_lambda(method, L, b, dirty_budget=None):
    """Minimize Toffoli over power-of-two lambda with 1 < lambda < L (and 2 lambda < L for dense)."""
    if method == "unary":
        return {"lambda": 1, "toffoli": float(L), "dirty_qubits": 0}
    best = None
    lam = 2
    while 2 * lam < L:
        dq = dirty_qubits(method, b, lam)
        if dirty_budget is None or dq <= dirty_budget:
            t = toffoli(method, L, b, lam)
            if best is None or t < best["toffoli"]:
                best = {"lambda": lam, "toffoli": float(t), "dirty_qubits": dq}
        lam *= 2
    return best


def toffoli_general(method, L, b, lam, mu):
    """Packet-width-general bounds.

    sequential_packets: arXiv:2605.20334 Theorem 1 / Eq. A1,
        (ceil(b/mu)+1)(ceil(L/lam)+lam-3) + (lam-1)(mu(floor(b/mu)+1) + b mod mu), dirty mu(lam-1).
    dense_sequential: arXiv:2610.02321 Eq. 9 with block rows lam,
        (m+1)(Q+lam-3) + 2 lam (b+mu), m = ceil(b/mu), Q = ceil(L/(2 lam)), dirty mu lam.
    """
    if method == "sequential_packets":
        return (math.ceil(b / mu) + 1) * (math.ceil(L / lam) + lam - 3) + (lam - 1) * (mu * (b // mu + 1) + b % mu)
    if method == "dense_sequential":
        m, Q = math.ceil(b / mu), math.ceil(L / (2 * lam))
        return (m + 1) * (Q + lam - 3) + 2 * lam * (b + mu)
    raise ValueError(method)


def dirty_general(method, b, lam, mu):
    return mu * (lam - 1) if method == "sequential_packets" else mu * lam


def best_general(method, L, b, dirty_budget=None):
    """Minimize the general bound jointly over packet width mu in 1..b and power-of-two lam, 1 < lam < L/2."""
    best = None
    for mu in range(1, b + 1):
        lam = 2
        while 2 * lam < L:
            dq = dirty_general(method, b, lam, mu)
            if dirty_budget is None or dq <= dirty_budget:
                t = toffoli_general(method, L, b, lam, mu)
                if best is None or t < best["toffoli"]:
                    best = {"lambda": lam, "mu": mu, "toffoli": float(t), "dirty_qubits": dq}
            lam *= 2
    return best


def tables(layout):
    """Lemma D.22 lookups per A/s query: row index, column index, element and element^dagger."""
    if layout == "packed":
        L_row = L_col = L_ele = NNZ   # lower bound: ignores CSR/CSC offset arithmetic
    else:
        L_row = L_ele = N_ROWS * S_ROW
        L_col = D_COLS * S_COL
    return [("row_index", L_row, B_COL_INDEX, 1), ("col_index", L_col, B_ROW_INDEX, 1), ("element", L_ele, B_VALUE, 2)]


def storage_bytes():
    csr = 4 * NNZ + 2 * NNZ + 4 * (N_ROWS + 1)
    csc_extra = 2 * NNZ + 4 * (D_COLS + 1)
    padded = (N_ROWS * S_ROW * (B_COL_INDEX + B_VALUE) + D_COLS * S_COL * B_ROW_INDEX) / 8
    return dict(csr_bytes=csr, csr_plus_csc_bytes=csr + csc_extra, padded_tables_bytes=padded)


def qos_per_query():
    rows = [r for r in csv.DictReader(open(EMPIRICAL)) if r["task_id"].endswith("__vs__rest")]
    s = {int(r["s"]) for r in rows}
    assert s == {S_COL}, s  # the one-vs-rest tasks use the full January training matrix
    r = rows[0]
    return {k: float(r[k]) for k in r if k.startswith("per_query_samples")}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out = []
    amp_degree = AMPLIFICATION_DEGREE_PER_S * S_COL
    for layout in ("packed", "padded"):
        for budget in (256, 1024, None):
            for method in METHODS:
                per_as, dirty, lam_info = 0.0, 0, []
                for name, L, b, uses in tables(layout):
                    r = best_general(method, L, b, budget) if method in GENERAL else best_lambda(method, L, b, budget)
                    per_as += uses * r["toffoli"]
                    dirty = max(dirty, r["dirty_qubits"])
                    lam_info.append(f"{name}:L={L},b={b},lambda={r['lambda']}" + (f",mu={r['mu']}" if "mu" in r else ""))
                address = max(math.ceil(math.log2(L)) for _, L, _, _ in tables(layout))
                out.append(dict(layout=layout, dirty_budget=budget if budget is not None else "unconstrained",
                                method=method, formula=GENERAL.get(method, SOURCES[method]),
                                toffoli_per_As_query=per_as, toffoli_per_A2_query=per_as * amp_degree,
                                amplification_degree_used=amp_degree,
                                max_dirty_qubits=dirty, address_qubits=address,
                                output_qubits=max(b for _, _, b, _ in tables(layout)),
                                lambdas="; ".join(lam_info), label=LABEL))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "qrom_costs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    qos = qos_per_query()
    summary = dict(label=LABEL, date_utc=datetime.now(timezone.utc).isoformat(),
                   matrix=dict(N=N_ROWS, D=D_COLS, nnz=NNZ, s_row=S_ROW, s_col=S_COL, s_row_or_col=S_COL,
                               bits=dict(value=B_VALUE, col_index=B_COL_INDEX, row_index=B_ROW_INDEX)),
                   storage=storage_bytes(), qos_per_A2_query_samples=qos,
                   qos_note="extrapolated from N<=128, s<=32 simulation with measured s^alpha; padded-dimension "
                            "(linear in size) factor is the theorem's, NOT measured",
                   amplification_note=f"QROM per-A/2 counts multiply the Lemma D.22 A/s query by our measured D.23 "
                                      f"polynomial degree ~{AMPLIFICATION_DEGREE_PER_S:.1f}*s = {amp_degree:.3g} at s={S_COL}; "
                                      "a QROM-based solver need not use D.23 amplification at all",
                   sources=SOURCES, inputs_sha256={str(EMPIRICAL.relative_to(ROOT)): sha(EMPIRICAL)},
                   script_sha256=sha(Path(__file__)))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    summary["outputs_sha256"] = {"qrom_costs.csv": sha(OUT / "qrom_costs.csv")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    for r in out:
        print(f"{r['layout']:6s} {str(r['dirty_budget']):13s} {r['method']:18s} A/s {r['toffoli_per_As_query']:.3e}  "
              f"A/2 {r['toffoli_per_A2_query']:.3e}  dirty {r['max_dirty_qubits']}")
    print(json.dumps(summary["storage"]), json.dumps(qos))


if __name__ == "__main__":
    main()
