"""Scoped classical streaming-state benchmark (approved ~5.7 h design; validation only).

108 cells = 3 tasks x 2 featurizer tracks x 3 methods x 6 learner budgets, chronological order.
Per cell: successive-halving tuning (8 candidates: eta0 {0.03,0.1,0.3,1} x {inverse_time,
inverse_sqrt}; 8 @1 pass -> 4 @2 -> 2 @5; tuning seed 2026; select by joint validation
retention, ties to lower candidate index), then the selected configuration on evaluation seeds
42/123/456 to 5 passes. k* at 95%/98% joint retention (min of accuracy and balanced-accuracy
retention vs frozen full Ridge) with a paired bootstrap over validation examples and seeds.
MISSION is excluded (recorded in the manifest). Learner, featurizer and stream state are
reported separately. Test data are never read.

Harness choice (declared): each (task, track) stream is encoded once into an in-memory cache.
This is research-harness infrastructure, like the on-disk spools, not learner state. Passes
replay the cache in chronological order (repeated passes are replay).

    .venv/bin/python -m experiments.cfpb_streaming_state.scoped run [--resume <dir>]
"""
import argparse
import csv
import json
import math
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from .data import ROOT, TASKS, atomic_json, sha256
from .methods import budgeted

SCOPED_TASKS = {
    "credit_vs_checking": TASKS["credit_vs_checking"],
    "checking_vs_mortgage": dict(positive=2, negative=3, N=4987, validation_N=1107,
                                 audit_id="checking_or_savings_account__vs__mortgage"),
    "mortgage_vs_rest": TASKS["mortgage_vs_rest"],
}
TRACKS = ("A_frozen", "B_causal")
METHODS = ("hash_sgd", "sparse_jl_sgd", "awm")
EXCLUDED = {"mission": "weakest method in every 128 KiB pilot cell; small-budget random/chronological order "
                       "effects up to 14 accuracy points need a separate multi-seed study; native MISSION has no "
                       "L2 term, so it does not optimize the Ridge objective"}
BUDGETS = tuple(k * 1024 for k in (4, 16, 32, 64, 128, 512))
CANDIDATES = [dict(eta=e, schedule=s) for s in ("inverse_time", "inverse_sqrt") for e in (0.03, 0.1, 0.3, 1.0)]
HALVING = [[8, 1], [4, 2], [2, 5]]
TUNING_SEED = 2026
EVALUATION_SEEDS = [42, 123, 456]
EVAL_CHECKPOINT_PASSES = (1, 2, 5)
THRESHOLDS = (0.95, 0.98)
N_BOOT, BOOT_SEED = 2000, 42
PILOT = ROOT / "results/classical_streaming/20261019T202457Z-4a8cdd1c"
INTERPRETATION = "measured validation score (classical streaming-state benchmark; validation only)"


def plan():
    cells = [dict(id=f"{task}_{track}_{method}_b{budget}", task=task, representation=track, method=method,
                  budget_bytes=budget)
             for task in SCOPED_TASKS for track in TRACKS for method in METHODS for budget in BUDGETS]
    return dict(cells=cells, excluded_methods=EXCLUDED, ordering="chronological", tuning_seed=TUNING_SEED,
                evaluation_seeds=EVALUATION_SEEDS, candidates=CANDIDATES, halving=HALVING,
                selection="joint validation retention (min of accuracy and balanced-accuracy retention vs frozen "
                          "full Ridge); ties to lower candidate index",
                inverse_sqrt_tau="N_train (one pass)", lambda_rule="1/N_train (Ridge alpha=1), as in the pilot",
                thresholds=list(THRESHOLDS), bootstrap=dict(n=N_BOOT, seed=BOOT_SEED,
                                                            resample="validation examples (shared across reference, "
                                                                     "budgets, methods) and seeds"),
                validation_only=True)


# ---------------------------------------------------------------- statistics
def _scores(y, P):
    P = np.atleast_2d(P)
    return (float(np.mean([accuracy_score(y, p) for p in P])),
            float(np.mean([balanced_accuracy_score(y, p) for p in P])))


def joint_retention(y, P, ref):
    acc, ba = _scores(y, P)
    racc, rba = _scores(y, ref)
    return min(acc / racc, ba / rba)


def k_star(retention_by_budget, threshold):
    return next((b for b in sorted(retention_by_budget) if retention_by_budget[b] >= threshold), None)


def bootstrap_kstar(y, ref, preds_by_budget, thresholds=THRESHOLDS, n_boot=N_BOOT, seed=BOOT_SEED):
    """Paired bootstrap: one index resample per replicate shared by the reference and every budget;
    seeds resampled with replacement. None (not reached) is treated as +inf for percentiles."""
    y, ref = np.asarray(y), np.asarray(ref)
    budgets = sorted(preds_by_budget)
    point = {b: joint_retention(y, preds_by_budget[b], ref) for b in budgets}
    rng = np.random.default_rng(seed)
    n = len(y)
    reps = {t: [] for t in thresholds}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ret = {}
        for b in budgets:
            P = np.atleast_2d(preds_by_budget[b])
            sidx = rng.integers(0, len(P), len(P))
            ret[b] = joint_retention(y[idx], P[sidx][:, idx], ref[idx])
        for t in thresholds:
            k = k_star(ret, t)
            reps[t].append(math.inf if k is None else k)
    out = {}
    for t in thresholds:
        arr = np.array(reps[t], float)
        lo, hi = np.percentile(arr, 2.5, method="lower"), np.percentile(arr, 97.5, method="higher")
        out[t] = dict(point=k_star(point, t), ci95=[None if math.isinf(lo) else int(lo), None if math.isinf(hi) else int(hi)],
                      not_reached_fraction=float(np.mean(np.isinf(arr))), point_retention=point)
    return out


def bootstrap_retention_ci(y, ref, P, n_boot=N_BOOT, seed=BOOT_SEED):
    y, ref, P = np.asarray(y), np.asarray(ref), np.atleast_2d(P)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y)); sidx = rng.integers(0, len(P), len(P))
        vals.append(joint_retention(y[idx], P[sidx][:, idx], ref[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ---------------------------------------------------------------- tuning
def successive_halving(candidates, advance, halving=HALVING):
    """advance(candidate, passes) -> score after bringing candidate to `passes` (continuation)."""
    survivors, rungs = list(candidates), []
    previous_passes, cost = 0, 0
    for k, passes in halving:
        survivors = survivors[:k]
        scores = {c: advance(c, passes) for c in survivors}
        cost += len(survivors) * (passes - previous_passes)
        previous_passes = passes
        rungs.append(dict(passes=passes, survivors=list(survivors), scores=scores))
        survivors = sorted(survivors, key=lambda c: (-scores[c], candidates.index(c)))
    return dict(selected=survivors[0], rungs=rungs, pass_equivalents=cost)


# ---------------------------------------------------------------- one cell
def _train(model, rows, passes_from, passes_to, check):
    for _ in range(passes_from, passes_to):
        for keys, x, y in rows:
            if check and model.t % 64 == 0:
                check()
            model.update(keys, x, y)


def _predict(model, rows):
    return np.array([1 if model.predict(k, x) >= 0 else -1 for k, x, _ in rows], dtype=int)


def run_cell(cell, train_rows, val_rows, *, ref_predictions, featurizer_bytes, candidates=CANDIDATES,
             halving=HALVING, tuning_seed=TUNING_SEED, evaluation_seeds=EVALUATION_SEEDS, check=None):
    n = len(train_rows)
    yv = np.array([r[2] for r in val_rows], dtype=int)
    ref = np.asarray(ref_predictions, dtype=int)
    t0 = time.perf_counter()

    def make(c, seed):
        spec = candidates[c]
        return budgeted(cell["method"], cell["budget_bytes"], n, seed=seed, eta=spec["eta"],
                        schedule=spec["schedule"], tau=n if spec["schedule"] == "inverse_sqrt" else None)

    models, done, diverged = {}, {}, set()

    def advance(c, passes):
        if c in diverged:
            return -math.inf
        if c not in models:
            models[c], done[c] = make(c, tuning_seed), 0
        try:
            _train(models[c], train_rows, done[c], passes, check)
        except FloatingPointError:
            diverged.add(c)
            return -math.inf
        done[c] = passes
        return joint_retention(yv, _predict(models[c], val_rows), ref)

    sh = successive_halving(list(range(len(candidates))), advance, halving)
    selected = sh["selected"]
    learner_bytes = max(m.state_bytes() for m in models.values())
    evaluation = []
    for seed in evaluation_seeds:
        m, passes_done, curve = make(selected, seed), 0, []
        status = "completed"
        try:
            for p in EVAL_CHECKPOINT_PASSES:
                _train(m, train_rows, passes_done, p, check)
                passes_done = p
                pred = _predict(m, val_rows)
                acc, ba = _scores(yv, pred)
                curve.append(dict(passes=p, accuracy=acc, balanced_accuracy=ba,
                                  joint_retention=joint_retention(yv, pred, ref)))
        except FloatingPointError:
            status, pred = "diverged", np.zeros(len(val_rows), dtype=int)
        learner_bytes = max(learner_bytes, m.state_bytes())
        evaluation.append(dict(seed=seed, status=status, curve=curve,
                               primary=dict(accuracy=curve[-1]["accuracy"] if curve else float("nan"),
                                            balanced_accuracy=curve[-1]["balanced_accuracy"] if curve else float("nan")),
                               predictions=pred.tolist()))
    if learner_bytes + 512 > cell["budget_bytes"]:
        raise MemoryError("learner state exceeded its byte ceiling")
    return dict(**cell, status="completed", selected_candidate=selected, selected_config=candidates[selected],
                tuning=dict(rungs=[dict(passes=r["passes"], survivors=r["survivors"],
                                        scores={str(k): v for k, v in r["scores"].items()}) for r in sh["rungs"]],
                            diverged=sorted(diverged)),
                pass_equivalents_tuning=sh["pass_equivalents"],
                pass_equivalents_evaluation=len(evaluation_seeds) * EVAL_CHECKPOINT_PASSES[-1],
                learner_state_bytes=int(learner_bytes), featurizer_state_bytes=int(featurizer_bytes),
                evaluation=evaluation, seconds=time.perf_counter() - t0)


# ---------------------------------------------------------------- orchestration (CFPB data; not used by tests)
def _encode_rows(spool, n, encoder, role, primary_only):
    from .data import Stream
    rows = []
    stream = Stream(spool, n, role=role)
    try:
        for row in stream:
            if primary_only and not row["primary"]:
                continue
            keys, x = encoder.encode(row["text"])
            rows.append((keys, x, int(row["y"])))
    finally:
        stream.close()
    return rows


def prepare_stage(output):
    from .prepare import prepare
    return prepare(output, tasks=SCOPED_TASKS, reference_predictions=True)


def run(resume=None):
    from src.cfpb_prepare import check_resources
    from .data import Encoder
    from .state import totals
    check_resources(before_stage=True)
    if resume:
        out = (ROOT / resume).resolve()
        manifest = json.loads((out / "manifest.json").read_text())
        if json.loads((out / "run_plan.json").read_text()) != json.loads(json.dumps(plan())):
            raise ValueError("Cannot change the approved scoped grid on resume")
    else:
        now = datetime.now(timezone.utc)
        run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        out = ROOT / "results/classical_streaming" / f"scoped_{run_id}"
        (out / "cells").mkdir(parents=True)
        (out / "data_cache").mkdir()
        srcs = sorted((ROOT / "experiments/cfpb_streaming_state").glob("*.py"))
        manifest = dict(run_id=run_id, started_utc=now.isoformat(), interpretation=INTERPRETATION,
                        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                        source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in srcs},
                        excluded_methods=EXCLUDED, test_access=False,
                        harness_cache="per (task, track) encoded-row cache in memory; harness infrastructure, "
                                      "not learner state (like the on-disk spools)",
                        featurizer_note="featurizer state (A: frozen vocabulary/IDF; B: causal tokenizer) is "
                                        "reported separately from learner state; it applies to QOS data streams too",
                        wall_seconds=0.0, status="running")
        atomic_json(out / "run_plan.json", plan())
        atomic_json(out / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        if not (out / "preparation.json").exists():
            prepare_stage(out)
        prep = json.loads((out / "preparation.json").read_text())
        featurizer = {track: totals(Encoder(track).state_inventory())["bytes"] for track in TRACKS}
        manifest["featurizer_state_bytes"] = featurizer
        for task in SCOPED_TASKS:
            sp = prep["spools"][task]
            ref = prep["references"][task]["selected_primary_predictions"]
            for track in TRACKS:
                cells = [c for c in plan()["cells"] if c["task"] == task and c["representation"] == track
                         and not (out / "cells" / f"{c['id']}.json").exists()]
                if not cells:
                    continue
                encoder = Encoder(track)
                train = _encode_rows(out / sp["train"]["path"], sp["train"]["N"], encoder, "train", False)
                val = _encode_rows(out / sp["validation"]["path"], sp["validation"]["N"], encoder, "validation", True)
                if [r[2] for r in val] != prep["references"][task]["primary_labels"]:
                    raise ValueError("Validation order/labels differ from the Ridge reference")
                for c in cells:
                    result = run_cell(c, train, val, ref_predictions=ref, featurizer_bytes=featurizer[track],
                                      check=check_resources)
                    result["stream_access_note"] = "chronological replay of the harness cache; no N-sized order array"
                    atomic_json(out / "cells" / f"{c['id']}.json", result)
                    print("cell", c["id"], f"{result['seconds']:.0f}s", flush=True)
        manifest["status"] = "cells_completed"
    except RuntimeError as exc:
        manifest["status"], manifest["interruption"] = "interrupted", str(exc)
        raise
    finally:
        manifest["wall_seconds"] = manifest.get("wall_seconds", 0.0) + time.perf_counter() - started
        atomic_json(out / "manifest.json", manifest)
    analyze(out)
    return out


def analyze(out):
    out = Path(out)
    prep = json.loads((out / "preparation.json").read_text())
    cells = {p.stem: json.loads(p.read_text()) for p in sorted((out / "cells").glob("*.json"))}
    kstar_rows, ret_rows, state_rows = [], [], []
    for task in SCOPED_TASKS:
        y = prep["references"][task]["primary_labels"]
        ref = prep["references"][task]["selected_primary_predictions"]
        for track in TRACKS:
            for method in METHODS:
                group = {c["budget_bytes"]: c for c in cells.values()
                         if c["task"] == task and c["representation"] == track and c["method"] == method}
                if len(group) != len(BUDGETS):
                    continue
                preds = {b: np.array([e["predictions"] for e in group[b]["evaluation"]]) for b in group}
                ks = bootstrap_kstar(y, ref, preds)
                for t, r in ks.items():
                    kstar_rows.append(dict(task=task, track=track, method=method, threshold=t, k_star_bytes=r["point"],
                                           ci95_low=r["ci95"][0], ci95_high=r["ci95"][1],
                                           not_reached_fraction=r["not_reached_fraction"]))
                for b, c in sorted(group.items()):
                    lo, hi = bootstrap_retention_ci(y, ref, preds[b])
                    ret_rows.append(dict(task=task, track=track, method=method, budget_bytes=b,
                                         joint_retention=joint_retention(np.array(y), preds[b], np.array(ref)),
                                         ci95_low=lo, ci95_high=hi, selected=json.dumps(c["selected_config"])))
                    state_rows.append(dict(task=task, track=track, method=method, budget_bytes=b,
                                           learner_state_bytes=c["learner_state_bytes"],
                                           featurizer_state_bytes=c["featurizer_state_bytes"],
                                           total_persistent_bytes=c["learner_state_bytes"] + c["featurizer_state_bytes"]))
    for name, rows in (("kstar.csv", kstar_rows), ("retention.csv", ret_rows), ("state.csv", state_rows)):
        if rows:
            with open(out / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return dict(kstar=kstar_rows, retention=ret_rows, state=state_rows)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--resume")
    a = sub.add_parser("analyze"); a.add_argument("output")
    args = ap.parse_args()
    if args.cmd == "run":
        print("done", run(args.resume).relative_to(ROOT), flush=True)
    else:
        analyze(ROOT / args.output)


if __name__ == "__main__":
    main()
