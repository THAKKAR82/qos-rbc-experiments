"""Synthetic controls for the sparsity screen (step 1): scaling fits, cold start, audit, F61.

ctrl_bounded (s <= 16, weak planted labels) and ctrl_dense_col (plus one all-ones
column) validate the screener. ctrl_bounded_margin keeps the bounded support but
plants a class-block linear model with a margin: a "synthetic best case" reference
for kappa_reg, gamma and F61, not evidence about real data.

    .venv/bin/python -m experiments.sparsity_screen.controls run
    .venv/bin/python -m experiments.sparsity_screen.controls publish results/screen/controls_<run_id>
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import uuid

import pandas as pd

from src import qos_audit as qa
from src import sparsity_screen as ss
from src.cfpb_prepare import ROOT, git_commit, write_json

from ..cfpb_streaming_state.analyze import append_exactly_once

GRID = (10_000, 20_000, 40_000, 80_000)
CASES = {"ctrl_bounded": ("bounded", "synthetic control"),
         "ctrl_dense_col": ("dense_col", "synthetic control"),
         "ctrl_bounded_margin": ("bounded_margin", "synthetic best case")}
EMPIRICAL_ALPHA_S = (3.147, 3.273)  # Block-encoding s-exponent fits at N = 64 and N = 128 (docs/empirical_f61.md).
LEDGER = ROOT / "results/experiment_log.csv"
PROTOCOL = Path(__file__).with_name("PROTOCOL.md")
CODE = ("src/sparsity_screen.py", "src/qos_audit.py", "src/machine_size.py",
        "experiments/sparsity_screen/controls.py")
INTERPRETATION = {"sparsity": "measured on synthetic data (screener control)",
                  "audit": "calculated / theoretical accounting on synthetic data",
                  "empirical": qa.EMPIRICAL_LABEL + "; synthetic data",
                  "bound": "calculated inequality (Schur bound on the test matrix)"}


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _selected(sweep):
    """Highest validation accuracy; ties -> larger alpha (step-1 convention)."""
    return sorted(sweep, key=lambda r: (r["validation"]["accuracy"], r["alpha"]), reverse=True)[0]


def summary_row(case, label, fit, scaling, enc, audit) -> dict:
    top = scaling[scaling.N_requested == scaling.N_requested.max()]
    row = _selected(audit["alpha_sweep"])
    inputs = row["costs_gamma_p10"]["inputs"]
    decomposition = dict(qa.f61_decomposition(inputs), s=inputs["s"])
    empirical = [qa.empirical_f61(decomposition, a)["log10_total_empirical"] for a in EMPIRICAL_ALPHA_S]
    return dict(case=case, label=label, N_max=int(top.N_nonzero.max()), D_train=audit["D_nonzero"],
                s_row_or_col_at_N_max=int(top.s_row_or_col.max()), slope=fit["slope"],
                ci_lo=fit["ci_boot"][0], ci_hi=fit["ci_boot"][1],
                sparsity_verdict=ss.classify_slope(*fit["ci_boot"]),
                cold_start_mean_at_N_max=float(top.cold_start_rate.mean()),
                audit_N=audit["N_nonzero"], alpha=row["alpha"], kappa_reg=row["kappa_reg"],
                gamma_p10=inputs["gamma_test"], gamma_min=row["margins_inf_rescaled"]["q0"],
                gamma_p10_bound=ss.gamma_upper_bound(enc.X_train, enc.X_test, inputs["m"]),
                val_accuracy=row["validation"]["accuracy"],
                val_balanced_accuracy=row["validation"]["balanced_accuracy"],
                log10_F61_proof=decomposition["log10_total"],
                log10_F61_emp_lo=min(empirical), log10_F61_emp_hi=max(empirical),
                **{f"log10_{k}": decomposition[f"log10_{k}"] for k in qa.F61_FACTORS})


def _extra_ledger_rows(summary: dict, *, run_id, date, commit, output_dir) -> list[dict]:
    base = dict(dataset=f"sparsity screen {summary['label']}:{summary['case']}", date=date, git_commit=commit or "",
                preprocessing="synthetic; chronological 70/15/15-style split (train N, +15% val, +15% test)",
                n_samples=summary["audit_N"], n_features=summary["D_train"],
                max_sparsity=summary["s_row_or_col_at_N_max"], sketch_method="none", sketch_dimension="",
                seed=qa.SEED, model="unweighted ridge, no intercept, ||X||=1, lam=alpha/sigma_max^2 (LSQR)")
    def row(metric, value, kind, **extra):
        notes = dict(interpretation=INTERPRETATION[kind], run_id=run_id, case=summary["case"],
                     label=summary["label"], artifact=f"{output_dir}/controls_summary.csv", **extra)
        return dict(base, experiment_id=f"screen-controls-{run_id}-{summary['case']}-{metric}",
                    metric=metric, result=value, notes=json.dumps(qa._clean(notes), sort_keys=True))
    return [row("s_row_or_col_loglog_slope", summary["slope"], "sparsity", ci_boot=[summary["ci_lo"], summary["ci_hi"]],
                verdict=summary["sparsity_verdict"]),
            row("cold_start_rate_mean_at_N_max", summary["cold_start_mean_at_N_max"], "sparsity"),
            row("gamma_p10_upper_bound", summary["gamma_p10_bound"], "bound", alpha=summary["alpha"]),
            row("log10_F61_empirical_lo", summary["log10_F61_emp_lo"], "empirical", alpha_s=max(EMPIRICAL_ALPHA_S)),
            row("log10_F61_empirical_hi", summary["log10_F61_emp_hi"], "empirical", alpha_s=min(EMPIRICAL_ALPHA_S))]


def run_controls(output_dir, *, grid=GRID, replicates=5, audit_n=GRID[-1], n_boot=ss.N_BOOT):
    """Screen and audit the three controls into a NEW directory; never touches the ledger."""
    out = Path(output_dir)
    out = out if out.is_absolute() else ROOT / out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; screen outputs are immutable")
    (out / "audit").mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    ledger_bytes = LEDGER.read_bytes()
    manifest = dict(run_id=run_id, date_utc=now.isoformat(), git_commit=git_commit(ROOT),
                    code_sha256={p: _sha(ROOT / p) for p in CODE}, protocol_sha256=_sha(PROTOCOL),
                    config=dict(grid=list(grid), replicates=replicates, audit_n=audit_n, audit_replicate=0,
                                n_boot=n_boot, alphas=list(qa.ALPHAS), delta=qa.DELTA, seed=qa.SEED,
                                empirical_alpha_s=list(EMPIRICAL_ALPHA_S),
                                thresholds=dict(pass_below=ss.PASS_BELOW, fail_above=ss.FAIL_ABOVE)),
                    cases={c: dict(kind=k, label=lab) for c, (k, lab) in CASES.items()},
                    interpretation=INTERPRETATION,
                    ledger=dict(path=str(LEDGER.relative_to(ROOT)), original_length=len(ledger_bytes),
                                original_sha256=hashlib.sha256(ledger_bytes).hexdigest()),
                    python=sys.version, platform=platform.platform(),
                    packages={n: importlib.metadata.version(n) for n in ["numpy", "scipy", "scikit-learn", "pandas"]},
                    status="running")
    write_json(out / "manifest.json", qa._clean(manifest))
    scaling, slopes, summary, ledger = [], [], [], []
    rel = str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out)
    for case, (kind, label) in CASES.items():
        build = ss.control_case(kind)
        df = ss.screen_case(build, grid, replicates).assign(case=case)
        scaling.append(df)
        fits = {stat: ss.fit_loglog_slope(df.N_nonzero, df[stat], df.replicate, n_boot=n_boot)
                for stat in ("s_row_or_col", "s_row", "s_col", "col_nnz_p99")}
        slopes += [dict(case=case, statistic=stat, slope=f["slope"], ci_boot_lo=f["ci_boot"][0],
                        ci_boot_hi=f["ci_boot"][1], ci_t_lo=f["ci_t"][0], ci_t_hi=f["ci_t"][1],
                        n_points=f["n_points"], n_replicates=f["n_replicates"],
                        verdict=ss.classify_slope(*f["ci_boot"])) for stat, f in fits.items()]
        enc = build(audit_n, 0)
        audit = qa.audit_binary(enc.X_train, enc.y_train, enc.X_val, enc.y_val, enc.X_test, enc.y_test)
        write_json(out / "audit" / f"{kind}.json",
                   qa._clean(dict(case=case, label=label, run_id=run_id, meta=enc.meta, **audit)))
        row = summary_row(case, label, fits["s_row_or_col"], df, enc, audit)
        summary.append(row)
        audit_rows = qa.ledger_rows(dict(task_id=case), audit, run_id=run_id, date=now.date().isoformat(),
                                    commit=manifest["git_commit"], output_dir=f"{rel}/audit",
                                    notes_common=dict(label=label, protocol_sha256=manifest["protocol_sha256"]),
                                    dataset=f"sparsity screen {label}",
                                    preprocessing="synthetic; chronological split (train N, +15% val, +15% test)",
                                    experiment_prefix="screen-controls")
        # qa.ledger_rows shares one ID across metrics; append_exactly_once needs unique IDs.
        ledger += [dict(r, experiment_id=f"{r['experiment_id']}-{r['metric']}") for r in audit_rows]
        ledger += _extra_ledger_rows(row, run_id=run_id, date=now.date().isoformat(),
                                     commit=manifest["git_commit"], output_dir=rel)
        print("control", case, row["sparsity_verdict"], flush=True)
    pd.concat(scaling).to_csv(out / "scaling.csv", index=False)
    pd.DataFrame(slopes).to_csv(out / "slopes.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "controls_summary.csv", index=False)
    pd.DataFrame(ledger).to_csv(out / "ledger_rows.csv", index=False)
    manifest.update(status="completed", artifacts_sha256={str(p.relative_to(out)): _sha(p)
                    for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"})
    write_json(out / "manifest.json", qa._clean(manifest))
    return out, run_id


def publish(output_dir, ledger=LEDGER) -> int:
    """Append this run's ledger rows exactly once (rows already present are skipped)."""
    ledger = Path(ledger)
    before = ledger.read_bytes()
    rows = pd.read_csv(Path(output_dir) / "ledger_rows.csv", dtype=str, keep_default_na=False)
    return append_exactly_once(ledger, rows.to_dict("records"), len(before), hashlib.sha256(before).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run")
    p = sub.add_parser("publish")
    p.add_argument("output")
    args = parser.parse_args()
    if args.command == "run":
        out, run_id = run_controls(ROOT / "results/screen" / ("controls_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
        print(out, run_id)
    else:
        print("appended", publish(args.output))


if __name__ == "__main__":
    main()
