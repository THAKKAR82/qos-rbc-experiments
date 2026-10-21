"""Test-split confirmation of a completed scoped run, with FROZEN selections (no retuning).

For every cell, rebuild the validation-selected configuration on evaluation seeds 42/123/456, retrain on
the same chronological training cache through 5 passes, and require exact reproduction of the saved
allocation, validation checkpoint scores and 5-pass validation predictions before scoring the CFPB test
split (primary rows, 2026-02-15..28). Test labels never influence selection. k* uses the run's paired
bootstrap (shared example resample; per-budget seed draws). The source run directory is never modified.

    .venv/bin/python -m experiments.cfpb_streaming_state.confirmation.scoped_test run <scoped-run-dir> [--resume <out>]
"""
import argparse
import csv
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import scoped as sc
from ..data import ROOT, PREPARATION, atomic_json, sha256, task_mask
from ..methods import allocation_metadata, budgeted

INTERPRETATION = "measured, test split, frozen selections"
SOURCE = ROOT / "data/CCDB_Export_15_January_2026_through_February_2026.csv"


class ReproductionError(RuntimeError):
    pass


# ---------------------------------------------------------------- one cell
def evaluate_frozen_cell(saved, train_rows, val_rows, test_rows, check=None):
    n = len(train_rows)
    spec = saved["selected_config"]
    yv = np.array([r[2] for r in val_rows], dtype=int)
    yt = np.array([r[2] for r in test_rows], dtype=int)
    evaluation = []
    for s in saved["evaluation"]:
        seed = s["seed"]
        m = budgeted(saved["method"], saved["budget_bytes"], n, seed=seed, eta=spec["eta"],
                     schedule=spec["schedule"], tau=n if spec["schedule"] == "inverse_sqrt" else None)
        if allocation_metadata(m) != saved["allocations"]["evaluation"][str(seed)]:
            raise ReproductionError(f"{saved['id']} seed {seed}: allocation differs from the saved run")
        done, curve = 0, []
        for i, p in enumerate(sc.EVAL_CHECKPOINT_PASSES):
            sc._train(m, train_rows, done, p, check)
            done = p
            pv = sc._predict(m, val_rows)
            acc, ba = sc._scores(yv, pv)
            if (acc, ba) != (s["curve"][i]["accuracy"], s["curve"][i]["balanced_accuracy"]):
                raise ReproductionError(f"{saved['id']} seed {seed}: validation scores at {p} passes not reproduced")
            pt = sc._predict(m, test_rows)
            tacc, tba = sc._scores(yt, pt)
            curve.append(dict(passes=p, accuracy=tacc, balanced_accuracy=tba))
        if pv.tolist() != s["predictions"]:
            raise ReproductionError(f"{saved['id']} seed {seed}: 5-pass validation predictions not reproduced")
        if m.state_bytes() + sc.STATE_MARGIN_BYTES > saved["budget_bytes"]:
            raise ReproductionError(f"{saved['id']} seed {seed}: learner state exceeds budget")
        evaluation.append(dict(seed=seed, validation_reproduced=True, test_curve=curve,
                               test_primary=dict(accuracy=curve[-1]["accuracy"],
                                                 balanced_accuracy=curve[-1]["balanced_accuracy"]),
                               test_predictions=pt.tolist(), learner_state_bytes=int(m.state_bytes())))
    return dict(id=saved["id"], task=saved["task"], representation=saved["representation"], method=saved["method"],
                budget_bytes=saved["budget_bytes"], selected_candidate=saved["selected_candidate"],
                selected_config=spec, evaluation=evaluation)


# ---------------------------------------------------------------- CFPB test rows and Ridge reference
def load_test_rows():
    """Primary test rows from the frozen split manifest, cleaned and hash-checked, chronological order."""
    import pandas as pd
    from src.cfpb import CLASSES, SOURCE_SHA256, clean_narrative
    from src.cfpb_prepare import check_resources
    if sha256(SOURCE) != SOURCE_SHA256:
        raise ValueError("Source SHA mismatch")
    rows = pd.read_csv(PREPARATION / "split_manifest.csv", dtype=str, keep_default_na=False)
    rows = rows[(rows.split == "test") & (rows.primary == "True")].copy()
    wanted = set(rows.complaint_id)
    parts = []
    for chunk in pd.read_csv(SOURCE, usecols=["Complaint ID", "Consumer complaint narrative", "Product", "Date received"],
                             dtype=str, keep_default_na=False, chunksize=100000):
        check_resources()
        parts.append(chunk[chunk["Complaint ID"].isin(wanted)].copy())
    raw = pd.concat(parts, ignore_index=True)
    rows = rows.merge(raw, left_on="complaint_id", right_on="Complaint ID", validate="one_to_one")
    if (set(rows.complaint_id) != wanted or not rows.Product.eq(rows["product"]).all()
            or not rows["Date received"].eq(rows.date_received).all()):
        raise ValueError("Frozen test membership/metadata mismatch")
    rows["text"] = rows["Consumer complaint narrative"].map(clean_narrative)
    accepted = {}
    for chunk in pd.read_csv(PREPARATION / "curation_audit.csv", usecols=["complaint_id", "model_text_hash"],
                             dtype=str, keep_default_na=False, chunksize=100000):
        sub = chunk[chunk.complaint_id.isin(wanted)]
        accepted.update(zip(sub.complaint_id, sub.model_text_hash))
    if not all(hashlib.sha256(r.text.encode()).hexdigest() == accepted[r.complaint_id] for r in rows.itertuples()):
        raise ValueError("Frozen cleaned-text hash mismatch")
    rows["numeric_id"] = rows.complaint_id.astype(np.int64)
    rows = rows.sort_values(["date_received", "numeric_id"]).reset_index(drop=True)
    rows["label"] = rows.label.astype(int)
    if not rows["product"].eq(rows.label.map(dict(enumerate(CLASSES)))).all():
        raise ValueError("Label mapping drift")
    if not rows.date_received.between("2026-02-15", "2026-02-28").all():
        raise ValueError("Test rows outside the frozen test window")
    return rows[["complaint_id", "date_received", "label", "text"]]


def _read_spool(path):
    with open(path, "rb") as f:
        return [json.loads(line) for line in f]


def ridge_test_reference(run, prep, task, test_frame):
    """Refit frozen full Ridge (validation-selected alpha) and predict test. Must reproduce the run's
    validation predictions exactly and the frozen step-1 audit's test metrics to 1e-12."""
    import pandas as pd
    from scipy.sparse.linalg import lsqr
    from src.holdout_evaluation import classification_metrics
    from ..data import Encoder
    spec, ref = sc.SCOPED_TASKS[task], prep["references"][task]
    alpha = ref["selected_alpha"]
    vec = Encoder("A_frozen").vectorizer
    train = _read_spool(run / prep["spools"][task]["train"]["path"])
    val = [r for r in _read_spool(run / prep["spools"][task]["validation"]["path"]) if r["primary"]]
    X = vec.transform([r["text"] for r in train]).astype(np.float64)
    y = np.array([r["y"] for r in train], dtype=np.float64)
    w = lsqr(X, y, damp=np.sqrt(alpha), atol=1e-10, btol=1e-10, iter_lim=20000)
    if w[1] not in (1, 2, 4, 5):
        raise ValueError("Ridge LSQR did not converge")
    pv = np.where(vec.transform([r["text"] for r in val]).astype(np.float64) @ w[0] >= 0, 1, -1)
    if pv.tolist() != ref["selected_primary_predictions"]:
        raise ReproductionError(f"{task}: refitted Ridge does not reproduce saved validation predictions")
    frame = test_frame[task_mask(test_frame.label.to_numpy(), task, sc.SCOPED_TASKS)]
    yt = np.where(frame.label.to_numpy() == spec["positive"], 1, -1)
    pt = np.where(vec.transform(frame.text.tolist()).astype(np.float64) @ w[0] >= 0, 1, -1)
    metrics = classification_metrics(yt, pt, labels=(-1, 1))[0]
    audit = pd.read_csv(ROOT / "results/audit/cfpb_step1/summary.csv",
                        usecols=["task_id", "alpha", "test_accuracy", "test_balanced_accuracy"])
    frozen = audit[(audit.task_id == spec["audit_id"]) & (audit.alpha == alpha)].iloc[0]
    np.testing.assert_allclose([metrics["accuracy"], metrics["balanced_accuracy"]],
                               [frozen.test_accuracy, frozen.test_balanced_accuracy], atol=1e-12, rtol=0)
    return dict(alpha=alpha, N_test=len(yt), accuracy=float(metrics["accuracy"]),
                balanced_accuracy=float(metrics["balanced_accuracy"]), audit_match_atol=1e-12,
                validation_predictions_reproduced=True, labels=yt.tolist(), predictions=pt.tolist(),
                complaint_ids=frame.complaint_id.tolist())


# ---------------------------------------------------------------- orchestration (CFPB data; not used by tests)
def run(source_run, resume=None):
    from src.cfpb_prepare import check_resources
    from ..data import Encoder
    src = (ROOT / source_run).resolve()
    run_manifest = sc.validate_resume(src)
    if run_manifest["status"] != "completed":
        raise ValueError("Source run is not verified/completed")
    if sc._environment() != run_manifest["environment"]:
        raise ValueError("Environment differs from the source run; exact reproduction is not guaranteed")
    if resume:
        out = (ROOT / resume).resolve()
        manifest = json.loads((out / "manifest.json").read_text())
        if manifest["source_run_manifest_sha256"] != sha256(src / "manifest.json") or \
                manifest["source_sha256"] != sc._source_hashes():
            raise ValueError("Cannot resume: source run or code changed")
        for cid, h in manifest["completed_cells_sha256"].items():
            if sha256(out / "cells" / f"{cid}.json") != h:
                raise ValueError(f"Checkpoint integrity failure: {cid}")
    else:
        now = datetime.now(timezone.utc)
        out = ROOT / "results/classical_streaming" / f"scoped_test_{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        (out / "cells").mkdir(parents=True)
        import subprocess
        manifest = dict(run_id=out.name.removeprefix("scoped_test_"), started_utc=now.isoformat(),
                        interpretation=INTERPRETATION, source_run=str(src.relative_to(ROOT)),
                        source_run_id=run_manifest["run_id"], source_run_manifest_sha256=sha256(src / "manifest.json"),
                        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                        source_sha256=sc._source_hashes(), environment=sc._environment(),
                        confirmation_source_sha256={str(Path(__file__).relative_to(ROOT)): sha256(__file__)},
                        selection="frozen: validation-selected configuration per cell; no retuning; test labels "
                                  "never used for selection",
                        reproduction_gate="allocation, validation checkpoint scores and 5-pass validation "
                                          "predictions must match the source run exactly before test scoring",
                        completed_cells_sha256={}, wall_seconds=0.0, status="running")
        atomic_json(out / "manifest.json", manifest)
    prep = json.loads((src / "preparation.json").read_text())
    started = time.perf_counter()
    try:
        check_resources(before_stage=True)
        manifest["status"] = "running"
        refs_path = out / "test_reference.json"
        if refs_path.exists():
            refs = json.loads(refs_path.read_text())
        else:
            test_frame = load_test_rows()
            refs = dict(test_rows_sha256=hashlib.sha256("\n".join(
                f"{r.complaint_id}\t{r.label}\t{hashlib.sha256(r.text.encode()).hexdigest()}"
                for r in test_frame.itertuples()).encode()).hexdigest(), N_test_primary=len(test_frame),
                tasks={task: ridge_test_reference(src, prep, task, test_frame) for task in sc.SCOPED_TASKS})
            (out / "data_cache").mkdir(exist_ok=True)
            with open(out / "data_cache/test_primary.jsonl", "w") as f:
                for r in test_frame.itertuples():
                    f.write(json.dumps(dict(id=r.complaint_id, date=r.date_received, label=int(r.label),
                                            text=r.text), ensure_ascii=False) + "\n")
            atomic_json(refs_path, refs)
        manifest["test_reference_sha256"] = sha256(refs_path)
        atomic_json(out / "manifest.json", manifest)
        test_all = _read_spool(out / "data_cache/test_primary.jsonl")
        cells = {p.stem: json.loads(p.read_text()) for p in sorted((src / "cells").glob("*.json"))}
        for task in sc.SCOPED_TASKS:
            sp, spec = prep["spools"][task], sc.SCOPED_TASKS[task]
            labels = np.array([r["label"] for r in test_all])
            test_task = [r for r, keep in zip(test_all, task_mask(labels, task, sc.SCOPED_TASKS)) if keep]
            if [r["id"] for r in test_task] != refs["tasks"][task]["complaint_ids"]:
                raise ValueError("Test order differs from the Ridge test reference")
            for track in sc.TRACKS:
                todo = [c for c in sc.plan()["cells"] if c["task"] == task and c["representation"] == track
                        and not (out / "cells" / f"{c['id']}.json").exists()]
                if not todo:
                    continue
                enc = Encoder(track)
                train = sc._encode_rows(src / sp["train"]["path"], sp["train"]["N"], enc, "train", False)
                val = sc._encode_rows(src / sp["validation"]["path"], sp["validation"]["N"], enc, "validation", True)
                test = [(*enc.encode(r["text"]), 1 if r["label"] == spec["positive"] else -1) for r in test_task]
                for c in todo:
                    t0 = time.perf_counter()
                    result = evaluate_frozen_cell(cells[c["id"]], train, val, test, check=check_resources)
                    result.update(seconds=time.perf_counter() - t0, run_id=manifest["run_id"])
                    path = out / "cells" / f"{c['id']}.json"
                    atomic_json(path, result)
                    manifest["completed_cells_sha256"][c["id"]] = sha256(path)
                    atomic_json(out / "manifest.json", manifest)
                    print("cell", c["id"], f"{result['seconds']:.0f}s", flush=True)
        analyze(out)
        manifest["status"] = "completed"
    except (RuntimeError, MemoryError, ValueError, KeyboardInterrupt) as exc:
        manifest["status"], manifest["interruption"] = "interrupted", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["wall_seconds"] = manifest.get("wall_seconds", 0.0) + time.perf_counter() - started
        atomic_json(out / "manifest.json", manifest)
    return out


def analyze(out):
    out = Path(out)
    refs = json.loads((out / "test_reference.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    val_ks = {(r["task"], r["track"], r["method"], r["threshold"]): r
              for r in csv.DictReader(open(ROOT / manifest["source_run"] / "kstar.csv"))}
    cells = {p.stem: json.loads(p.read_text()) for p in sorted((out / "cells").glob("*.json"))}
    if len(cells) != len(sc.plan()["cells"]):
        raise ValueError("Incomplete test evaluation; analysis refuses partial grids")
    ks_rows, ret_rows = [], []
    for task in sc.SCOPED_TASKS:
        y, ref = np.array(refs["tasks"][task]["labels"]), np.array(refs["tasks"][task]["predictions"])
        for track in sc.TRACKS:
            for method in sc.METHODS:
                group = {c["budget_bytes"]: c for c in cells.values()
                         if (c["task"], c["representation"], c["method"]) == (task, track, method)}
                preds = {b: np.array([e["test_predictions"] for e in c["evaluation"]]) for b, c in group.items()}
                for t, r in sc.bootstrap_kstar(y, ref, preds).items():
                    v = val_ks[(task, track, method, str(t))]
                    ks_rows.append(dict(task=task, track=track, method=method, threshold=t,
                                        test_k_star_bytes=r["point"], test_ci95_low=r["ci95"][0],
                                        test_ci95_high=r["ci95"][1], test_not_reached_fraction=r["not_reached_fraction"],
                                        validation_k_star_bytes=v["k_star_bytes"], validation_ci95_low=v["ci95_low"],
                                        validation_ci95_high=v["ci95_high"],
                                        validation_not_reached_fraction=v["not_reached_fraction"]))
                for b in sorted(preds):
                    lo, hi = sc.bootstrap_retention_ci(y, ref, preds[b])
                    acc, ba = sc._scores(y, preds[b])
                    ret_rows.append(dict(task=task, track=track, method=method, budget_bytes=b,
                                         test_accuracy=acc, test_balanced_accuracy=ba,
                                         test_joint_retention=sc.joint_retention(y, preds[b], ref),
                                         ci95_low=lo, ci95_high=hi))
    for name, rows in (("kstar_test.csv", ks_rows), ("retention_test.csv", ret_rows)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return dict(kstar=ks_rows, retention=ret_rows)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("source_run"); r.add_argument("--resume")
    a = sub.add_parser("analyze"); a.add_argument("output")
    args = ap.parse_args()
    if args.cmd == "run":
        print("done", run(args.source_run, args.resume).relative_to(ROOT), flush=True)
    else:
        analyze(ROOT / args.output)


if __name__ == "__main__":
    main()
