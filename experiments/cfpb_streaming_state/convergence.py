"""Full-dimensional multi-pass SGD convergence controls on frozen-feature track A.

Approved scope: 12 trajectories = 2 tasks x 2 orders x eta0 {0.1, 0.3, 1.0}; seed 42; up to five
passes; no intercept, no class weights; lambda = 1/N (Ridge alpha = 1); the pilot's inverse-time
schedule eta(t) = eta0 / (1 + eta0 * lambda * t); checkpoints ceil(N/8), ceil(N/4), ceil(N/2), N,
2N, 5N; validation only; 10-minute total cap. Purpose: does a suitable optimizer approach Ridge
before any gap is attributed to memory? This is an optimization control, NOT a bounded-memory method.

Declared before any fit: repeated passes REPLAY the stream (chronological passes are labelled
replay); random order replays the pilot's fixed seed-42 permutation each pass (cyclic replay, no
reshuffle). Pilot code (worker/methods/data) is reused unchanged and not modified.

Usage (main .venv): python -m experiments.cfpb_streaming_state.convergence --pilot <pilot-run-dir>
"""
import argparse
import csv
import json
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

TASKS = ("credit_vs_checking", "mortgage_vs_rest")
ORDERS = ("chronological", "random")
ETAS = (0.1, 0.3, 1.0)
MAX_PASSES = 5
SEED = 42
CAP_SECONDS = 600.0
D_FEATURES = 50_000
INTERPRETATION = "measured validation score (classical full-dimensional SGD convergence control; not bounded-memory)"


def checkpoints(n):
    return [math.ceil(n / 8), math.ceil(n / 4), math.ceil(n / 2), n, 2 * n, MAX_PASSES * n]


def pass_cursor(t, n):
    return t // n, t % n


def plan():
    return [dict(id=f"conv_{task}_{order}_eta{eta:g}", task=task, order=order, eta0=eta,
                 representation="A_frozen", method="identity_sgd", max_passes=MAX_PASSES, seed=SEED)
            for task in TASKS for order in ORDERS for eta in ETAS]


def retention(metrics, ref):
    acc = metrics["accuracy"] / ref["accuracy"]
    ba = metrics["balanced_accuracy"] / ref["balanced_accuracy"]
    return dict(accuracy=acc, balanced_accuracy=ba, joint=min(acc, ba))


def ridge_reference(preparation, task):
    """Frozen full-Ridge reference: alpha by primary validation accuracy, ties to larger alpha."""
    sweep = preparation["references"][task]["alpha_sweep"]
    best = max(sweep, key=lambda r: (r["primary"]["accuracy"], r["alpha"]))
    return dict(alpha=best["alpha"], accuracy=best["primary"]["accuracy"],
                balanced_accuracy=best["primary"]["balanced_accuracy"])


def run(pilot_dir, out_root, cap_seconds=CAP_SECONDS):
    from src.cfpb_prepare import check_resources
    from .data import ROOT, Encoder, Stream, sha256
    from .methods import Learner
    from .state import totals
    from .worker import Deadline, evaluate

    pilot_dir = Path(pilot_dir).resolve()
    prep = json.loads((pilot_dir / "preparation.json").read_text())
    for task in TASKS:
        for part in ("train", "validation"):
            info = prep["spools"][task][part]
            if sha256(pilot_dir / info["path"]) != info["sha256"]:
                raise ValueError(f"Pilot spool altered: {task}/{part}")
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    out = Path(out_root).resolve() / f"convergence_{run_id}"
    out.mkdir(parents=True, exist_ok=False)
    configs = plan()
    manifest = dict(run_id=run_id, date_utc=now.isoformat(), interpretation=INTERPRETATION,
                    pilot_run=str(pilot_dir.relative_to(ROOT)), pilot_preparation_sha256=sha256(pilot_dir / "preparation.json"),
                    configurations=configs, checkpoints_rule="ceil(N/8), ceil(N/4), ceil(N/2), N, 2N, 5N",
                    schedule="eta(t) = eta0 / (1 + eta0 * lambda * t), lambda = 1/N, half squared loss, no intercept",
                    replay="repeated passes are replay; random order cycles the pilot's fixed seed-42 permutation",
                    cap_seconds=cap_seconds, status="running")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    encoder = Encoder("A_frozen")
    featurizer_bytes = totals(encoder.state_inventory())["bytes"]
    deadline = time.monotonic() + cap_seconds
    stop = [False]
    rows, trajectories = [], []
    started = time.perf_counter()
    for c in configs:
        spec = prep["spools"][c["task"]]
        n, n_val = spec["train"]["N"], spec["validation"]["N"]
        ref = ridge_reference(prep, c["task"])
        model = Learner("identity_sgd", D_FEATURES, eta=c["eta0"], lam=1.0 / n)
        learner_bytes = model.state_bytes()
        marks = checkpoints(n)
        status, t0, reason = "completed", time.perf_counter(), None
        try:
            for p in range(MAX_PASSES):
                stream = Stream(pilot_dir / spec["train"]["path"], n, role="train", order=c["order"], seed=SEED)
                try:
                    for row in stream:
                        if model.t % 64 == 0:
                            check_resources()
                            if time.monotonic() >= deadline:
                                raise Deadline("10-minute cap reached at a training boundary")
                        keys, x = encoder.encode(row["text"])
                        model.update(keys, x, row["y"])
                        if model.t in marks:
                            metrics, _ = evaluate(model, encoder, pilot_dir / spec["validation"]["path"],
                                                  n_val, deadline, stop)
                            prim = metrics["primary"]
                            rows.append(dict(id=c["id"], task=c["task"], order=c["order"], eta0=c["eta0"],
                                             presentations=model.t, passes=model.t / n,
                                             unique_examples=min(model.t, n),
                                             accuracy=prim["accuracy"], balanced_accuracy=prim["balanced_accuracy"],
                                             macro_f1=prim["macro_f1"],
                                             **{f"retention_{k}": v for k, v in retention(prim, ref).items()},
                                             ridge_alpha=ref["alpha"], ridge_accuracy=ref["accuracy"],
                                             ridge_balanced_accuracy=ref["balanced_accuracy"],
                                             learner_state_bytes=learner_bytes, featurizer_state_bytes=featurizer_bytes,
                                             weight_norm=float(np.linalg.norm(model.weights * model.scale)),
                                             elapsed_seconds=time.perf_counter() - t0))
                finally:
                    stream.close()
        except Deadline as exc:
            status, reason = "stopped_at_cap", str(exc)
        except FloatingPointError as exc:
            status, reason = "diverged", str(exc)
        trajectories.append(dict(id=c["id"], status=status, reason=reason, presentations=model.t,
                                 seconds=time.perf_counter() - t0, learner_state_bytes=learner_bytes,
                                 featurizer_state_bytes=featurizer_bytes))
        print(c["id"], status, model.t, f"{time.perf_counter() - t0:.1f}s", flush=True)
        if status == "stopped_at_cap":
            break
    for name, recs in (("checkpoints.csv", rows), ("trajectories.csv", trajectories)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
    manifest.update(status="completed" if all(t["status"] == "completed" for t in trajectories) and
                    len(trajectories) == len(configs) else "partial",
                    wall_seconds=time.perf_counter() - started,
                    artifacts_sha256={p.name: sha256(p) for p in sorted(out.iterdir()) if p.name != "manifest.json"})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print("done", out.relative_to(ROOT), f"{manifest['wall_seconds']:.0f}s", manifest["status"], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--out-root", default="results/classical_streaming")
    args = ap.parse_args()
    run(args.pilot, args.out_root)


if __name__ == "__main__":
    main()
