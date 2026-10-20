"""Empirical F61: step-1 per-task numbers with the block-encoding s^5 replaced by measured s^alpha.

Pure arithmetic on saved artifacts (no re-solves). Label: extrapolation from small-size simulation;
orders of magnitude only. Run from the repository root with the main .venv:
    .venv/bin/python scripts/empirical_f61.py
"""
import csv
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import qos_audit as qa  # noqa: E402

PHASE_A = ROOT / "results/tightness/phase_a_20261019T145729Z-a1644ff3"
PHASE_A_N128 = ROOT / "results/tightness/phase_a_n128_20261019T153934Z-75edf052"
CFPB = ROOT / "results/audit/cfpb_step1"
BANK = ROOT / "results/audit/bank_step1"
OUT = ROOT / "results/audit/empirical_f61"
FIT_EPS = "0.01"


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def d23_alpha(run_dir):
    row = next(r for r in read(run_dir / "alpha_fits.csv")
               if r["quantity"] == "d23" and r["eps"] == FIT_EPS and r["polylog"] == "raw")
    return float(row["alpha"]), row["ci95"]


def anchor(eps):
    v = [float(r["M_star"]) for r in read(PHASE_A / "mstar.csv")
         if r["quantity"] == "d23" and r["s"] == "32" and r["eps"] == eps]
    return statistics.median(v)


def selected(summary, decomposition, pooled_metric=None):
    """CFPB: per task max val_accuracy, ties -> larger alpha. Bank: one alpha by mean CV-fold accuracy."""
    if pooled_metric:
        means = {}
        for r in summary:
            means.setdefault(float(r["alpha"]), []).append(float(r[pooled_metric]))
        alpha = max(means, key=lambda a: (statistics.mean(means[a]), a))
        picks = {r["task_id"]: alpha for r in summary}
    else:
        picks = {}
        for r in summary:
            key = (float(r["val_accuracy"]), float(r["alpha"]))
            if r["task_id"] not in picks or key > picks[r["task_id"]][0]:
                picks[r["task_id"]] = (key, float(r["alpha"]))
        picks = {t: v[1] for t, v in picks.items()}
    by = {(r["task_id"], float(r["alpha"])): r for r in decomposition}
    sums = {(r["task_id"], float(r["alpha"])): r for r in summary}
    return [(t, a, by[(t, a)], sums[(t, a)]) for t, a in picks.items()]


def main():
    a64, ci64 = d23_alpha(PHASE_A)
    a128, ci128 = d23_alpha(PHASE_A_N128)
    anchors = {e: anchor(e) for e in ("0.01", "0.001")}
    rows = []
    for dataset, d, pooled in (("CFPB Jan-Feb 2026 primary4", CFPB, None),
                               ("Bank Marketing (bank-full), global encoding (known leakage)", BANK, "test_accuracy")):
        summary = read(d / "summary.csv")
        decomposition = read(d / "derived" / "f61_decomposition_gamma_p10.csv")
        for task, alpha_ridge, dec, summ in selected(summary, decomposition, pooled):
            e64, e128 = qa.empirical_f61(dec, a64), qa.empirical_f61(dec, a128)
            dim = qa.padded_dimension(int(summ["N_nonzero"]), int(summ["D_nonzero"]))
            row = dict(dataset=dataset, task_id=task, ridge_alpha=alpha_ridge, s=int(dec["s"]),
                       N=int(summ["N_nonzero"]), D=int(summ["D_nonzero"]), padded_dim=dim,
                       kappa_reg=float(dec["kappa_reg"]), gamma_p10=float(dec["gamma"]), m=int(dec["m"]),
                       delta=float(dec["delta"]),
                       log10_F61_proof=e64["log10_total_proof"],
                       log10_F61_empirical_N64=e64["log10_total_empirical"],
                       log10_F61_empirical_N128=e128["log10_total_empirical"],
                       log10_s_BE_proof=e64["log10_s_block_encoding_proof"],
                       log10_s_BE_measured_N64=e64["log10_s_block_encoding_measured"],
                       log10_s_BE_measured_N128=e128["log10_s_block_encoding_measured"],
                       log10_s_readout_proof=e64["log10_s_readout_proof"],
                       log10_kappa4_proof=float(dec["log10_kappa^4"]), log10_gamma_m4_proof=float(dec["log10_gamma^-4"]),
                       log10_delta_m1_proof=float(dec["log10_delta^-1"]), log10_min_term_proof=float(dec["log10_min_term"]),
                       kappa_gamma_delta_status=e64["kappa_gamma_delta_status"])
            for e in ("0.01", "0.001"):
                tag = "1e-2" if e == "0.01" else "1e-3"
                for lab, al in (("N64", a64), ("N128", a128), ("proof_s5", qa.PROOF_BE_S_EXPONENT)):
                    row[f"per_query_samples_eps{tag}_{lab}"] = qa.anchored_per_query_samples(
                        dec["s"], dim, al, anchors[e])
            row["label"] = qa.EMPIRICAL_LABEL
            rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "empirical_f61.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    inputs = [PHASE_A / "alpha_fits.csv", PHASE_A / "mstar.csv", PHASE_A_N128 / "alpha_fits.csv",
              CFPB / "summary.csv", CFPB / "derived/f61_decomposition_gamma_p10.csv",
              BANK / "summary.csv", BANK / "derived/f61_decomposition_gamma_p10.csv"]
    manifest = dict(date_utc=datetime.now(timezone.utc).isoformat(), label=qa.EMPIRICAL_LABEL,
                    method="F61 log10 passes with the Theorem D.23 block-encoding s^5 replaced by measured s^alpha; "
                           "readout s^2, kappa^4, gamma^-4, delta^-1 and min term kept at proof values (not measured)",
                    alpha_N64=dict(value=a64, ci95=ci64, source="phase (a) d23 raw, eps*=1e-2"),
                    alpha_N128=dict(value=a128, ci95=ci128, source="N=128 check d23 raw, eps*=1e-2"),
                    per_query_anchor=dict(samples_eps1e_2=anchors["0.01"], samples_eps1e_3=anchors["0.001"],
                                          at="N=64, s=32, median over 3 instances (phase (a) mstar)",
                                          scaling="(s/32)^alpha x (padded_dim/64); dimension factor is the theorem's, not measured"),
                    selection="CFPB: per task max validation accuracy, ties to larger alpha; Bank: one alpha by mean CV-fold accuracy",
                    inputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in inputs},
                    script_sha256=sha(Path(__file__)), qos_audit_sha256=sha(ROOT / "src/qos_audit.py"))
    manifest["outputs_sha256"] = {"empirical_f61.csv": sha(OUT / "empirical_f61.csv")}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for r in rows:
        print(f"{r['task_id'][:44]:44s} s={r['s']:>6d} proof={r['log10_F61_proof']:.1f} "
              f"emp={r['log10_F61_empirical_N64']:.1f}..{r['log10_F61_empirical_N128']:.1f} "
              f"perq(1e-2)={r['per_query_samples_eps1e-2_N64']:.1e}..{r['per_query_samples_eps1e-2_N128']:.1e}")
    print("alpha", a64, a128, "anchors", anchors)


if __name__ == "__main__":
    main()
