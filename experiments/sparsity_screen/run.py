"""Sparsity screen on public data (steps 3-4): scaling fits, cold start, gated audit, summary.

Gates in protocol order (PROTOCOL.md): (1) sparsity verdict on the log s vs log N
slope; (2) cold start <= 0.10 at the largest N; the audit (kappa_reg, gamma, F61)
runs for sparsity PASS only, at the largest N, replicate 0. Every case gets a
summary row with fixed-string reasons, failures included. The controls and the
Elliptic structural check are read from their own committed run directories.

    .venv/bin/python -m experiments.sparsity_screen.run run --controls <dir> --elliptic <dir>
    .venv/bin/python -m experiments.sparsity_screen.run publish results/screen/screen_<run_id>
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import uuid

import numpy as np
import pandas as pd

from src import qos_audit as qa
from src import screen_datasets as sd
from src import sparsity_screen as ss
from src.cfpb_prepare import ROOT, git_commit, write_json

from .controls import EMPIRICAL_ALPHA_S, LEDGER, PROTOCOL, _selected, ledger_prefix, publish

GRID = (10_000, 20_000, 40_000, 80_000)
COLD_START_GATE = 0.10  # Approved 2026-10-26 (PROTOCOL.md amendment A1).
DESCRIPTIONS = {
    "paysim_naive": dict(dataset="PaySim", encoding="fixed D = 10: type one-hot, log1p money standardized",
                         label="public data (N/D illustration)"),
    "paysim_time_local": dict(dataset="PaySim", encoding="(orig, 24-step), (dest, 24-step), (type, amount decile, step)",
                              label="public data"),
    "aml_acct_window": dict(dataset="IBM AMLworld HI-Small", encoding="(from account, day), (to account, day)",
                            label="public data")}
DATA_FILES = ("data/paysim.csv", "data/raw/screen/amlworld/HI-Small_Trans.csv")
CODE = ("src/sparsity_screen.py", "src/screen_datasets.py", "src/qos_audit.py", "src/machine_size.py",
        "experiments/sparsity_screen/run.py", "experiments/sparsity_screen/controls.py")
COLUMNS = ["case", "dataset", "encoding", "label", "N_max", "D_train", "s_row_or_col_at_N_max", "NoverD_bound_at_N_max",
           "slope", "ci_lo", "ci_hi", "sparsity_verdict", "cold_start_at_N_max", "cold_start_gate", "reason",
           "audit_N", "alpha", "kappa_reg", "gamma_p10", "gamma_p10_bound", "val_accuracy", "val_balanced_accuracy",
           "log10_F61_proof", "log10_F61_emp_lo", "log10_F61_emp_hi", "full_model_bytes", "kstar95_kib", "kstar98_kib"]
INTERPRETATION = {"sparsity": "measured (chronological prefixes of public data)",
                  "audit": "calculated / theoretical accounting with measured inputs",
                  "empirical": qa.EMPIRICAL_LABEL,
                  "bound": "calculated inequality (Schur bound on the test matrix)"}


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_case(enc: ss.Encoded) -> tuple[dict | None, dict, str | None]:
    """Audit one Encoded split unless a recorded reason forbids it (decision 3a, single class)."""
    p = ss.sparsity_point(enc.X_train)
    if len(np.unique(enc.y_train)) < 2 or len(np.unique(enc.y_val)) < 2:
        return None, {}, "audit not run: single class in training or validation prefix"
    if p["D_nonzero"] <= p["N_nonzero"] and p["D_nonzero"] > qa.DENSE_SVD_MAX_D:
        return None, {}, "audit not run: D <= N with D > 2000 (decision 3a)"
    audit = qa.audit_binary(enc.X_train, enc.y_train, enc.X_val, enc.y_val, enc.X_test, enc.y_test)
    row = _selected(audit["alpha_sweep"])
    inputs = row["costs_gamma_p10"]["inputs"]
    fields = dict(audit_N=audit["N_nonzero"], alpha=row["alpha"], kappa_reg=row["kappa_reg"],
                  gamma_p10=inputs["gamma_test"],
                  gamma_p10_bound=ss.gamma_upper_bound(enc.X_train, enc.X_test, inputs["m"]),
                  val_accuracy=row["validation"]["accuracy"],
                  val_balanced_accuracy=row["validation"]["balanced_accuracy"])
    if row["costs_gamma_p10"]["log10_F61_passes"] is None:
        return audit, fields, "theorem inapplicable: gamma_p10 = 0"
    decomposition = dict(qa.f61_decomposition(inputs), s=inputs["s"])
    empirical = [qa.empirical_f61(decomposition, a)["log10_total_empirical"] for a in EMPIRICAL_ALPHA_S]
    fields.update(log10_F61_proof=decomposition["log10_total"], log10_F61_emp_lo=min(empirical),
                  log10_F61_emp_hi=max(empirical))
    return audit, fields, None


def screen_one(case, build, description, *, grid, replicates, n_boot):
    """Scaling table, slope fits and summary row (audit fields only after a sparsity PASS)."""
    df = ss.screen_case(build, grid, replicates).assign(case=case)
    fits = {stat: ss.fit_loglog_slope(df.N_nonzero, df[stat], df.replicate, n_boot=n_boot)
            for stat in ("s_row_or_col", "s_row", "s_col", "col_nnz_p99")}
    slopes = [dict(case=case, statistic=stat, slope=f["slope"], ci_boot_lo=f["ci_boot"][0], ci_boot_hi=f["ci_boot"][1],
                   ci_t_lo=f["ci_t"][0], ci_t_hi=f["ci_t"][1], n_points=f["n_points"], n_replicates=f["n_replicates"],
                   verdict=ss.classify_slope(*f["ci_boot"])) for stat, f in fits.items()]
    fit = fits["s_row_or_col"]
    top = df[df.N_requested == df.N_requested.max()]
    verdict = ss.classify_slope(*fit["ci_boot"])
    cold = float(top.cold_start_rate.mean())
    row = dict(case=case, **description, N_max=int(top.N_nonzero.max()), D_train=int(top.D_nonzero.max()),
               s_row_or_col_at_N_max=int(top.s_row_or_col.max()),
               NoverD_bound_at_N_max=float((top.nnz / top.D_nonzero).mean()),
               slope=fit["slope"], ci_lo=fit["ci_boot"][0], ci_hi=fit["ci_boot"][1], sparsity_verdict=verdict,
               cold_start_at_N_max=cold, cold_start_gate="PASS" if cold <= COLD_START_GATE else "FAIL",
               full_model_bytes=8 * int(top.D_nonzero.max()))
    reasons = []
    if verdict == "FAIL":
        reasons.append(f"s grows with N (slope {fit['slope']:.2f}, CI [{fit['ci_boot'][0]:.2f}, {fit['ci_boot'][1]:.2f}])")
    elif verdict == "INCONCLUSIVE":
        reasons.append(f"inconclusive slope CI [{fit['ci_boot'][0]:.2f}, {fit['ci_boot'][1]:.2f}]")
    if row["cold_start_gate"] == "FAIL":
        reasons.append(f"cold start {cold:.3f} > {COLD_START_GATE:.2f}")
    audit = enc = None
    if verdict == "PASS":
        enc = build(int(top.N_requested.iloc[0]), 0)
        audit, fields, why = audit_case(enc)
        row.update(fields)
        if why:
            reasons.append(why)
    row["reason"] = "; ".join(reasons) or "passes sparsity and cold start; see costs"
    return df, slopes, row, audit, enc


def _ledger(row, audit, *, run_id, date, commit, rel):
    base = dict(dataset=f"sparsity screen {row['dataset']}:{row['case']}", date=date, git_commit=commit or "",
                preprocessing=f"{row['encoding']}; chronological prefix, train-only vocabulary",
                n_samples=row["N_max"], n_features=row["D_train"], max_sparsity=row["s_row_or_col_at_N_max"],
                sketch_method="none", sketch_dimension="", seed=qa.SEED,
                model="none (sparsity screen)" if audit is None else
                      "unweighted ridge, no intercept, ||X||=1, lam=alpha/sigma_max^2 (LSQR)")
    def one(metric, value, kind, **extra):
        notes = dict(interpretation=INTERPRETATION[kind], run_id=run_id, case=row["case"], label=row["label"],
                     reason=row["reason"], artifact=f"{rel}/summary.csv", **extra)
        return dict(base, experiment_id=f"screen-{run_id}-{row['case']}-{metric}", metric=metric, result=value,
                    notes=json.dumps(qa._clean(notes), sort_keys=True))
    out = [one("s_row_or_col_loglog_slope", row["slope"], "sparsity", ci_boot=[row["ci_lo"], row["ci_hi"]],
               verdict=row["sparsity_verdict"]),
           one("cold_start_rate_mean_at_N_max", row["cold_start_at_N_max"], "sparsity", gate=row["cold_start_gate"])]
    if audit is not None:
        audit_rows = qa.ledger_rows(dict(task_id=row["case"]), audit, run_id=run_id, date=date, commit=commit,
                                    output_dir=f"{rel}/audit", notes_common=dict(label=row["label"]),
                                    dataset=f"sparsity screen {row['dataset']}", preprocessing=base["preprocessing"],
                                    experiment_prefix="screen")
        out += [dict(r, experiment_id=f"{r['experiment_id']}-{r['metric']}") for r in audit_rows]
        out.append(one("gamma_p10_upper_bound", row["gamma_p10_bound"], "bound", alpha=row["alpha"]))
        if not math.isnan(row.get("log10_F61_emp_lo", math.nan)):
            out += [one("log10_F61_empirical_lo", row["log10_F61_emp_lo"], "empirical"),
                    one("log10_F61_empirical_hi", row["log10_F61_emp_hi"], "empirical")]
    return out


def _external_rows(controls_dir, elliptic_dir) -> list[dict]:
    rows = []
    if controls_dir is not None:
        c = pd.read_csv(Path(controls_dir) / "controls_summary.csv")
        for r in c.to_dict("records"):
            cold = r["cold_start_mean_at_N_max"]
            rows.append(dict({k: r.get(k) for k in COLUMNS if k in r}, dataset="synthetic", encoding=r["case"],
                             cold_start_at_N_max=cold, cold_start_gate="PASS" if cold <= COLD_START_GATE else "FAIL",
                             full_model_bytes=8 * int(r["D_train"]),
                             reason=f"control run {Path(controls_dir).name}"))
    if elliptic_dir is not None:
        e = json.loads((Path(elliptic_dir) / "structure.json").read_text())
        rate = e["cold_start_step_aligned"]["rate"]
        rows.append(dict(case="elliptic_structure", dataset="Elliptic Bitcoin", label="public data",
                         encoding="graph only: in/out neighbour ids", N_max=e["labeled"]["illicit"] + e["labeled"]["licit"],
                         sparsity_verdict="not measured (skipped)", cold_start_at_N_max=rate,
                         cold_start_gate="PASS" if rate <= COLD_START_GATE else "FAIL",
                         reason=f"{e['reason']}; cross-step edges {e['cross_step_edges']} of {e['n_edges']}; "
                                f"cold start {rate:.3f} on the 34/15 step split; run {Path(elliptic_dir).name}"))
    return rows


def _markdown(frame: pd.DataFrame) -> str:
    def fmt(v):
        if isinstance(v, float):
            return "" if math.isnan(v) else f"{v:.4g}"
        return str(v)
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in frame.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def run_screen(output_dir, *, builders=None, descriptions=DESCRIPTIONS, grid=GRID, replicates=sd.REPLICATES,
               n_boot=ss.N_BOOT, controls_dir=None, elliptic_dir=None, ledger=LEDGER):
    """Screen every case into a NEW directory; never touches the ledger (see publish)."""
    out = Path(output_dir)
    out = out if out.is_absolute() else ROOT / out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; screen outputs are immutable")
    (out / "audit").mkdir(parents=True, exist_ok=True)
    real = builders is None
    builders = builders or {c: sd.case(c) for c in sd.CASE_IDS}
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    manifest = dict(run_id=run_id, date_utc=now.isoformat(), git_commit=git_commit(ROOT),
                    code_sha256={p: _sha(ROOT / p) for p in CODE}, protocol_sha256=_sha(PROTOCOL),
                    data_sha256={p: _sha(ROOT / p) for p in DATA_FILES} if real else "synthetic builders (test)",
                    inputs=dict(controls=str(controls_dir) if controls_dir else None,
                                elliptic=str(elliptic_dir) if elliptic_dir else None),
                    config=dict(cases=list(builders), grid=list(grid), replicates=replicates, n_boot=n_boot,
                                audit="largest N, replicate 0; alpha by validation accuracy, ties -> larger",
                                cold_start_gate=COLD_START_GATE, alphas=list(qa.ALPHAS), seed=qa.SEED,
                                thresholds=dict(pass_below=ss.PASS_BELOW, fail_above=ss.FAIL_ABOVE),
                                empirical_alpha_s=list(EMPIRICAL_ALPHA_S)),
                    interpretation=INTERPRETATION, ledger=ledger_prefix(ledger),
                    python=sys.version, platform=platform.platform(),
                    packages={n: importlib.metadata.version(n) for n in ["numpy", "scipy", "scikit-learn", "pandas"]},
                    status="running")
    write_json(out / "manifest.json", qa._clean(manifest))
    rel = str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out)
    scaling, slopes, rows, ledger = [], [], [], []
    for case, build in builders.items():
        df, sl, row, audit, enc = screen_one(case, build, descriptions[case], grid=grid, replicates=replicates,
                                             n_boot=n_boot)
        scaling.append(df)
        slopes += sl
        rows.append(row)
        if audit is not None:
            write_json(out / "audit" / f"{case}.json", qa._clean(dict(case=case, run_id=run_id, meta=enc.meta, **audit)))
        ledger += _ledger(row, audit, run_id=run_id, date=now.date().isoformat(), commit=manifest["git_commit"], rel=rel)
        print("screened", case, row["sparsity_verdict"], row["cold_start_gate"], row["reason"], flush=True)
    rows += _external_rows(controls_dir, elliptic_dir)
    summary = pd.DataFrame(rows).reindex(columns=COLUMNS)
    pd.concat(scaling).to_csv(out / "scaling.csv", index=False)
    pd.DataFrame(slopes).to_csv(out / "slopes.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(_markdown(summary))
    pd.DataFrame(ledger).to_csv(out / "ledger_rows.csv", index=False)
    manifest.update(status="completed", artifacts_sha256={str(p.relative_to(out)): _sha(p)
                    for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"})
    write_json(out / "manifest.json", qa._clean(manifest))
    return out, run_id


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--controls", required=True)
    r.add_argument("--elliptic", required=True)
    p = sub.add_parser("publish")
    p.add_argument("output")
    args = parser.parse_args()
    if args.command == "run":
        name = "screen_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out, run_id = run_screen(ROOT / "results/screen" / name, controls_dir=ROOT / args.controls,
                                 elliptic_dir=ROOT / args.elliptic)
        print(out, run_id)
    else:
        print("appended", publish(args.output, LEDGER))


if __name__ == "__main__":
    main()
