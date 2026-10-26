"""Elliptic structural check (screen step 2): are there edges across time steps?

Graph-only encoding: row = labeled transaction, columns = ("out", neighbour)
and ("in", neighbour). A test row overlaps the training features only through
a neighbour it shares with a training row. Without cross-step edges, a test
step's rows share no neighbour with earlier steps, so under a step-aligned
chronological split every test row is cold-start (gamma = 0) and the case fails
the pre-registered cold-start gate. Measured on the public edge list.

    .venv/bin/python -m experiments.sparsity_screen.elliptic_structure run
    .venv/bin/python -m experiments.sparsity_screen.elliptic_structure publish results/screen/elliptic_structure_<run_id>
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import pandas as pd

from src import qos_audit as qa
from src.cfpb_prepare import ROOT, git_commit, write_json

from .controls import LEDGER, PROTOCOL, ledger_prefix, publish

RAW = ROOT / "data/raw/screen/elliptic/zip"
SOURCE = "https://data.pyg.org/datasets/elliptic/<file>.zip (PyTorch Geometric EllipticBitcoinDataset host)"
STANDARD_LAST_TRAIN_STEP = 34  # Weber et al. (2019) temporal split: train steps 1-34, test 35-49.
COLD_START_GATE = 0.10
INTERPRETATION = "measured structural property of the public Elliptic edge list (graph-only encoding)"


def cross_step_edges(edges: pd.DataFrame, step: pd.Series) -> int:
    """Number of edges whose endpoints lie in different time steps (KeyError on unknown nodes)."""
    a, b = step.loc[edges.txId1].to_numpy(), step.loc[edges.txId2].to_numpy()
    return int((a != b).sum())


def step_aligned_cold_start(edges: pd.DataFrame, step: pd.Series, labels: pd.Series,
                            last_train_step: int) -> float:
    """Share of labeled test rows (step > last_train_step) with no column shared with a training row."""
    keyed = pd.concat([pd.DataFrame({"tx": edges.txId1.to_numpy(), "col": "out:" + edges.txId2.astype(str)}),
                       pd.DataFrame({"tx": edges.txId2.to_numpy(), "col": "in:" + edges.txId1.astype(str)})])
    keyed = keyed[keyed.tx.isin(labels.index)]
    is_train = step.loc[keyed.tx].to_numpy() <= last_train_step
    support = set(keyed.col[is_train])
    test_rows = labels.index[step.loc[labels.index].to_numpy() > last_train_step]
    served = set(keyed.tx[~is_train & keyed.col.isin(support).to_numpy()])
    return float(1 - len(served & set(test_rows)) / len(test_rows))


def load(raw=RAW):
    edges = pd.read_csv(raw / "elliptic_txs_edgelist.csv.zip")
    step = pd.read_csv(raw / "elliptic_txs_features.csv.zip", header=None, usecols=[0, 1],
                       names=["txId", "step"]).set_index("txId").step
    classes = pd.read_csv(raw / "elliptic_txs_classes.csv.zip").set_index("txId")["class"].astype(str)
    return edges, step, classes


def run(output_dir, raw=RAW, ledger=LEDGER):
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; screen outputs are immutable")
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    edges, step, classes = load(raw)
    labels = classes[classes != "unknown"].map({"1": 1, "2": -1})  # 1 = illicit = +1, 2 = licit = -1.
    cross = cross_step_edges(edges, step)
    cold = step_aligned_cold_start(edges, step, labels, STANDARD_LAST_TRAIN_STEP)
    verdict = ("FAIL" if cross == 0 or cold > COLD_START_GATE else "NEEDS_CASE")
    structure = dict(
        run_id=run_id, date_utc=now.isoformat(), git_commit=git_commit(ROOT), source=SOURCE,
        licence="not verified: the Kaggle page (ellipticco/elliptic-data-set) shows no licence text without JavaScript",
        files_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(raw.glob("*.zip"))},
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        n_nodes=int(step.size), n_edges=int(len(edges)), n_steps=int(step.nunique()),
        labeled=dict(illicit=int((labels == 1).sum()), licit=int((labels == -1).sum()),
                     unknown=int((classes == "unknown").sum())),
        cross_step_edges=cross, within_step_edges=int(len(edges)) - cross,
        cold_start_step_aligned=dict(last_train_step=STANDARD_LAST_TRAIN_STEP, rate=cold,
                                     encoding="rows = labeled txs; columns = out:/in: neighbour ids"),
        cold_start_gate=COLD_START_GATE, verdict=verdict, ledger=ledger_prefix(ledger),
        reason=("no cross-step edges: every test row is cold-start (gamma = 0) under a step-aligned "
                "chronological split" if cross == 0 else f"{cross} cross-step edges; build a case before deciding"),
        interpretation=INTERPRETATION)
    write_json(out / "structure.json", qa._clean(structure))
    base = dict(dataset="Elliptic Bitcoin (graph only)", date=now.date().isoformat(), git_commit=structure["git_commit"],
                preprocessing="edge list + time step; rows = labeled txs, columns = in/out neighbour ids",
                n_samples=int(labels.size), n_features="", max_sparsity="", sketch_method="none",
                sketch_dimension="", seed=qa.SEED, model="none (structural check)")
    rel = str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out)
    notes = json.dumps(dict(interpretation=INTERPRETATION, run_id=run_id, artifact=f"{rel}/structure.json",
                            verdict=verdict), sort_keys=True)
    rows = [dict(base, experiment_id=f"screen-elliptic-{run_id}-{m}", metric=m, result=v, notes=notes)
            for m, v in (("cross_step_edges", cross), ("cold_start_rate_step_aligned_34_15", cold))]
    pd.DataFrame(rows).to_csv(out / "ledger_rows.csv", index=False)
    return out, structure


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run")
    p = sub.add_parser("publish")
    p.add_argument("output")
    args = parser.parse_args()
    if args.command == "run":
        out, s = run(ROOT / "results/screen" / ("elliptic_structure_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
        print(out, json.dumps({k: s[k] for k in ("n_nodes", "n_edges", "cross_step_edges", "cold_start_step_aligned",
                                                  "verdict", "labeled")}))
    else:
        print("appended", publish(args.output, LEDGER, record="structure.json"))


if __name__ == "__main__":
    main()
