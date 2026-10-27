"""Derived diagnostic (review fix): which feature family sets s and which serves test rows.

Rebuilds every (case, N, replicate) point of a committed screen run, checks that
s_row_or_col and cold start reproduce its scaling.csv exactly, and records per
family the largest training column count and the share of test rows that family
alone serves, plus the time span of each training prefix. Measured; it changes no
verdict. Written to a new directory next to the screen run.

    .venv/bin/python -m experiments.sparsity_screen.family_breakdown results/screen/screen_<ts>
"""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys

import pandas as pd

from src import qos_audit as qa
from src import screen_datasets as sd
from src import sparsity_screen as ss
from src.cfpb_prepare import ROOT, git_commit, write_json

FAMILIES = {"paysim_time_local": ["orig x 24-step window", "dest x 24-step window", "type x amount decile x step"],
            "aml_acct_window": ["from account x day", "to account x day"]}
WINDOWS = {"paysim_time_local": sd.PAYSIM_WINDOW, "aml_acct_window": sd.AML_WINDOW}


def breakdown_rows(case, build, names, scaling, window=None):
    """Per-family and time-span rows for every point of `scaling` (one case); raises on any drift."""
    fams, spans = [], []
    for point in scaling[scaling.case == case].itertuples(index=False):
        enc = build(int(point.N_requested), int(point.replicate))
        p = ss.sparsity_point(enc.X_train)
        cold = ss.cold_start_rate(enc.X_train, enc.X_test)
        if p["s_row_or_col"] != point.s_row_or_col or abs(cold - point.cold_start_rate) > 1e-12:  # CSV float round trip.
            raise ValueError(f"{case} N={point.N_requested} r={point.replicate} does not reproduce scaling.csv")
        key = dict(case=case, N_requested=int(point.N_requested), replicate=int(point.replicate))
        fams += [dict(key, **f) for f in sd.family_breakdown(enc, names)]
        (a, b), (c, d) = enc.meta["t_train"], enc.meta["t_test"]
        spans.append(dict(key, t_train_first=a, t_train_last=b, t_test_first=c, t_test_last=d, t_train_span=b - a,
                          test_shares_last_train_t=c == b,
                          test_after_train_window=None if window is None else bool(c // window > b // window)))
    return fams, spans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("screen_dir")
    args = parser.parse_args()
    screen = ROOT / args.screen_dir
    out = screen.parent / ("family_breakdown_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    scaling = pd.read_csv(screen / "scaling.csv")
    fams, spans = [], []
    for case, names in FAMILIES.items():
        f, s = breakdown_rows(case, sd.case(case), names, scaling, WINDOWS[case])
        fams += f
        spans += s
        print("rebuilt", case, len(s), "points", flush=True)
    pd.DataFrame(fams).to_csv(out / "families.csv", index=False)
    pd.DataFrame(spans).to_csv(out / "spans.csv", index=False)
    write_json(out / "manifest.json", qa._clean(dict(
        source_run=str(screen.relative_to(ROOT)), source_scaling_sha256=hashlib.sha256((screen / "scaling.csv").read_bytes()).hexdigest(),
        git_commit=git_commit(ROOT), date_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
        code_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                     for p in ("src/screen_datasets.py", "src/sparsity_screen.py", "experiments/sparsity_screen/family_breakdown.py")},
        interpretation="measured derived diagnostic; reproduces scaling.csv exactly; changes no verdict",
        time_units={"paysim_time_local": "step (hour)", "aml_acct_window": "minute"},
        artifacts_sha256={n: hashlib.sha256((out / n).read_bytes()).hexdigest() for n in ("families.csv", "spans.csv")})))
    print(out)


if __name__ == "__main__":
    main()
