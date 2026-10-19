"""Phase-1 runner: python -m qostight.run_phase1 run  (under .venv-qos-ref only).

Writes results/tightness/phase1_<run_id>/ with manifest, calibration, D.23 validation,
dense cross-check and ledger rows. Never touches the main environment.
"""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psutil

from .env import REFERENCE_COMMIT, REPO, load_reference, require_qos_ref_venv
from .matrices import IDEAL_SPARSITY_NOTE
from .resources import GIB, gate_ok

INTERPRETATION = "simulation (expected-unitary QOS, reference convention)"
LEDGER_COLUMNS = ["experiment_id", "dataset", "date", "git_commit", "preprocessing", "n_samples",
                  "n_features", "max_sparsity", "sketch_method", "sketch_dimension", "seed", "model",
                  "metric", "result", "notes"]
HERE = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit():
    out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else ""


def ledger_row(run_id, suffix, *, metric, result, N, s, notes, commit, date=None):
    return dict(experiment_id=f"qos-tightness-phase1-{run_id}-{suffix}",
                dataset="synthetic circulant sparse (uniform s; ideal-sparsity case)",
                date=date or datetime.now(timezone.utc).date().isoformat(), git_commit=commit or "",
                preprocessing="none (synthetic, ||X||=1)", n_samples=N, n_features=N, max_sparsity=s,
                sketch_method="QOS expected unitary (reference qos.py @ a3fe8d6)", sketch_dimension="",
                seed=42, model="sparse block encoding (Lemmas D.21/D.22) + Theorem D.23 amplification",
                metric=metric, result=result,
                notes=json.dumps(dict(notes, interpretation=INTERPRETATION), sort_keys=True))


def _write_csv(path, rows):
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def run_calibration(Ns, ss, M, timeout, pairs=None):
    rows = []
    pairs = pairs if pairs is not None else [(N, s) for s in ss for N in Ns if s <= N // 2]
    failed_s = set()
    for N, s in pairs:
        if s in failed_s:
            continue
        avail = psutil.virtual_memory().available
        if not gate_ok(avail, 0, before_stage=True):
            rows.append(dict(N=N, D=N, s=s, status="skipped: memory gate before stage",
                             available_bytes=avail)); continue
        t0 = time.perf_counter()
        try:
            out = subprocess.run([sys.executable, "-m", "qostight.run_phase1", "calibrate-one",
                                  "--N", str(N), "--s", str(s), "--M", str(M)], cwd=HERE,
                                 capture_output=True, text=True, timeout=timeout)
            r = (json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else
                 dict(N=N, D=N, s=s, status=f"failed rc={out.returncode}", stderr_tail=out.stderr[-300:]))
        except subprocess.TimeoutExpired:
            r = dict(N=N, D=N, s=s, status=f"timeout > {timeout}s")
        r["wall_seconds_incl_startup"] = time.perf_counter() - t0
        rows.append(r)
        print("calibrated", N, s, r["status"], f"{r.get('seconds', float('nan')):.1f}s",
              f"{r.get('peak_rss_bytes', 0) / GIB:.2f}GiB", flush=True)
        if r["status"] != "ok":
            failed_s.add(s)  # larger N for this s would only be worse
    return rows


PHASE_A_PLAN = dict(N=64, s_values=(2, 4, 8, 16, 32), M_values=12, seeds=(42, 123, 456, 789, 2026),
                    a2prime_degree=101)
RETIME_PAIRS = [(64, s) for s in PHASE_A_PLAN["s_values"]] + [(128, 32)]
TIME_BUDGET_HOURS = 3.0  # step-2 plan: (a) ~0.5-1 h index oracle + 1-2 h block encoding


def phase_a_budget(calibration):
    """Estimated phase (a) wall time from measured single-build times at N = 64 (time is M-independent)."""
    t = {int(r["s"]): float(r["seconds"]) for r in calibration if r.get("status") == "ok" and int(r["N"]) == 64}
    n = PHASE_A_PLAN["M_values"] * len(PHASE_A_PLAN["seeds"])
    missing = [s for s in PHASE_A_PLAN["s_values"] if s not in t]
    primary = sum(t.get(s, float("nan")) for s in PHASE_A_PLAN["s_values"]) * n
    return dict(plan=PHASE_A_PLAN, builds_per_s=n, per_build_seconds_N64=t, missing_s=missing,
                primary_sweep_hours=primary / 3600,
                note="a2 (index oracle) and a3 (block encoding) errors come from the same builds; a1 (element "
                     "oracle) is negligible; a2' (i)/(ii) at degree 101 add ~2 x the s<=8 build cost per s",
                budget_hours=TIME_BUDGET_HOURS, within_budget=bool(not missing and primary / 3600 <= TIME_BUDGET_HOURS))


def retime(args):
    """Calibration-only run from a light parent (no JAX in this process), for the phase (a) budget."""
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    out = REPO / "results" / "tightness" / f"phase1_retime_{run_id}"
    out.mkdir(parents=True, exist_ok=False)
    code = sorted(str(f.relative_to(REPO)) for f in (HERE / "qostight").glob("*.py"))
    manifest = dict(run_id=run_id, date_utc=now.isoformat(), interpretation=INTERPRETATION,
                    reason="re-timing after sign fix; the revalidation run's calibration was skipped by the "
                           "memory gate because its parent process held JAX state",
                    git_commit=git_commit(), reference_commit=REFERENCE_COMMIT, python=sys.version,
                    platform=platform.platform(), lock_sha256=_sha(HERE / "requirements-lock.txt"),
                    code_sha256={c: _sha(REPO / c) for c in code}, pairs=RETIME_PAIRS, unit_num_samples=10**6,
                    status="running")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    calib = run_calibration(None, None, M=10**6, timeout=args.timeout, pairs=RETIME_PAIRS)
    _write_csv(out / "calibration.csv", calib)
    (out / "phase_a_budget.json").write_text(json.dumps(phase_a_budget(calib), indent=2, sort_keys=True))
    manifest.update(status="completed", artifacts_sha256={f.name: _sha(f) for f in sorted(out.iterdir())
                                                           if f.name != "manifest.json"})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print("done", out.relative_to(REPO), flush=True)


def revalidate(args):
    """After the sign fix: D.23 validation (s up to 16), a2' (i)/(ii), and re-timing (incl. s=32)."""
    from .phase1 import a2prime_comparison, d23_validation
    ref = load_reference()
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    out = REPO / "results" / "tightness" / f"phase1_revalidation_{run_id}"
    out.mkdir(parents=True, exist_ok=False)
    code = sorted(str(f.relative_to(REPO)) for f in (HERE / "qostight").glob("*.py"))
    manifest = dict(run_id=run_id, date_utc=now.isoformat(), interpretation=INTERPRETATION,
                    supersedes_validation_of="results/tightness/phase1_20261019T125405Z-a6c2c22b (failed as found)",
                    git_commit=git_commit(), reference_commit=ref.commit, python=sys.version,
                    platform=platform.platform(), lock_sha256=_sha(HERE / "requirements-lock.txt"),
                    code_sha256={c: _sha(REPO / c) for c in code}, seed=42, data_note=IDEAL_SPARSITY_NOTE,
                    conventions=dict(sign="primary: verified-bounded Chebyshev erf polynomial at the D.23 oracle "
                                          "precision eps/(s ln(1/eps)), eps=1e-4; angles verified to 1e-10",
                                     a2prime="(i) reference pyqsp PolySign degree 101, failure recorded as found; "
                                             "(ii) verified-bounded polynomial at fixed degree 101",
                                     amplification="Theorem D.23 odd polynomial, eps=1e-4; comparison "
                                                   "'optimistic: reference convention' = classical multiply by s",
                                     phase_b_note="phase (b) query accounting must include the real-part "
                                                  "extraction (LCU) step of the reference QSVT convention"),
                    status="running")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    validation = []
    for N in (16, 32):
        for s in (2, 4, 8, 16):
            if s <= N:
                validation += d23_validation(N, s, M_grid=(10**4, 10**5, 10**6, 10**7, 10**8, 10**10))
                print("validated", N, s, flush=True)
    _write_csv(out / "d23_validation.csv", validation)
    a2p = []
    for N, s in ((32, 2), (32, 4), (32, 8), (32, 16), (64, 32)):
        a2p += a2prime_comparison(N, s)
        print("a2'", N, s, flush=True)
    _write_csv(out / "a2prime_comparison.csv", a2p)
    calib = run_calibration(None, None, M=10**6, timeout=args.timeout,
                            pairs=[(64, 2), (64, 4), (64, 8), (64, 16), (64, 32), (128, 32)])
    _write_csv(out / "calibration.csv", calib)
    budget = phase_a_budget(calib)
    (out / "phase_a_budget.json").write_text(json.dumps(budget, indent=2, sort_keys=True))
    commit = manifest["git_commit"]
    ledger = []
    for r in validation:
        if r["pass"] is not None:
            tag = "exact" if r["input"] == "exact A/s" else ("d23" if r["path"].startswith("primary") else "rescale")
            ledger.append(ledger_row(run_id, f"reval-N{r['N']}-s{r['s']}-{tag}", metric="block_encoding_error_spec",
                                     result=r["error_spec"], N=r["N"], s=r["s"], commit=commit,
                                     notes=dict(path=r["path"], input=r["input"], criterion=r["criterion"],
                                                passed=r["pass"], unit_num_samples=r["unit_num_samples"],
                                                sign="verified-bounded, D.23 precision",
                                                artifact=str((out / "d23_validation.csv").relative_to(REPO)))))
    for r in a2p:
        tag = "a2p-i" if r["sign_method"].startswith("pyqsp") else "a2p-ii"
        ledger.append(ledger_row(run_id, f"reval-N{r['N']}-s{r['s']}-{tag}", metric="d23_error_spec_fixed_degree",
                                 result=r["d23_error_spec"], N=r["N"], s=r["s"], commit=commit,
                                 notes=dict(comparison=r["comparison"], angles_verified=r["angles_verified"],
                                            angle_max_dev=r["angle_max_dev"], sign_degree=r["sign_degree"],
                                            artifact=str((out / "a2prime_comparison.csv").relative_to(REPO)))))
    with open(out / "ledger_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS); w.writeheader(); w.writerows(ledger)
    if args.append_ledger:
        log = REPO / "results" / "experiment_log.csv"
        before = log.read_bytes()
        if not before.endswith(b"\n"):
            raise ValueError("ledger does not end with newline")
        with open(log, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=LEDGER_COLUMNS).writerows(ledger)
        if log.read_bytes()[:len(before)] != before:
            raise RuntimeError("existing ledger bytes changed")
    manifest.update(status="completed", artifacts_sha256={f.name: _sha(f) for f in sorted(out.iterdir())
                                                           if f.name != "manifest.json"},
                    ledger_appended=bool(args.append_ledger))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print("done", out.relative_to(REPO), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    one = sub.add_parser("calibrate-one")
    one.add_argument("--N", type=int); one.add_argument("--s", type=int); one.add_argument("--M", type=int)
    run = sub.add_parser("run")
    run.add_argument("--timeout", type=int, default=900)
    run.add_argument("--append-ledger", action="store_true")
    rev = sub.add_parser("revalidate")
    rev.add_argument("--timeout", type=int, default=1800)
    rev.add_argument("--append-ledger", action="store_true")
    ret = sub.add_parser("retime")
    ret.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args(argv)
    require_qos_ref_venv()
    if args.cmd == "calibrate-one":
        from .phase1 import calibrate_one
        print(json.dumps(calibrate_one(args.N, args.s, args.M)))
        return
    if args.cmd == "revalidate":
        return revalidate(args)
    if args.cmd == "retime":
        return retime(args)
    from .phase1 import d23_validation, dense_crosscheck
    ref = load_reference()
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    out = REPO / "results" / "tightness" / f"phase1_{run_id}"
    out.mkdir(parents=True, exist_ok=False)
    code = sorted(str(f.relative_to(REPO)) for f in (HERE / "qostight").glob("*.py"))
    manifest = dict(run_id=run_id, date_utc=now.isoformat(), interpretation=INTERPRETATION,
                    git_commit=git_commit(), reference_commit=ref.commit, python=sys.version,
                    platform=platform.platform(), lock_sha256=_sha(HERE / "requirements-lock.txt"),
                    code_sha256={c: _sha(REPO / c) for c in code}, seed=42, data_note=IDEAL_SPARSITY_NOTE,
                    conventions=dict(model="expected-unitary oracles (reference qos.py); pessimistic per reference",
                                     sign_degree="primary: d = ceil(ln(1/eps)/sin(pi/(4s+2))), eps = 1e-4 (Lemma D.21); "
                                                 "reference fixed 101 is the a2' comparison",
                                     amplification="primary: Theorem D.23 odd polynomial, eps = 1e-4; comparison "
                                                   "'optimistic: reference convention' = classical multiply by s",
                                     qsvt="singular-value shortcut under Halmos completion; cross-checked against "
                                          "reference dense apply_qsvt"), status="running")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    calib = run_calibration(Ns=(16, 32, 64, 128, 256, 512), ss=(2, 4, 8, 16, 32), M=10**6,
                            timeout=args.timeout)
    _write_csv(out / "calibration.csv", calib)
    validation = []
    for N in (16, 32):
        for s in (2, 4, 8):
            validation += d23_validation(N, s, M_grid=(10**4, 10**5, 10**6, 10**7, 10**8, 10**10))
            print("validated", N, s, flush=True)
    _write_csv(out / "d23_validation.csv", validation)
    cross = []
    for N in (16, 32):
        for s in (2, 4):
            cross += dense_crosscheck(N, s)
            print("cross-checked", N, s, flush=True)
    _write_csv(out / "dense_crosscheck.csv", cross)
    commit = manifest["git_commit"]
    ledger = []
    for r in validation:
        if r["pass"] is not None:
            tag = "exact" if r["input"] == "exact A/s" else ("d23" if r["path"].startswith("primary") else "rescale")
            ledger.append(ledger_row(run_id, f"N{r['N']}-s{r['s']}-{tag}", metric="block_encoding_error_spec",
                                     result=r["error_spec"], N=r["N"], s=r["s"], commit=commit,
                                     notes=dict(path=r["path"], input=r["input"], criterion=r["criterion"],
                                                passed=r["pass"], unit_num_samples=r["unit_num_samples"],
                                                artifact=str((out / "d23_validation.csv").relative_to(REPO)))))
    for r in cross:
        ledger.append(ledger_row(run_id, f"N{r['N']}-s{r['s']}-dense-{r['polynomial'][:4].replace(' ', '')}",
                                 metric="dense_qsvt_vs_shortcut_max_abs_diff", result=r["max_abs_diff"],
                                 N=r["N"], s=r["s"], commit=commit,
                                 notes=dict(polynomial=r["polynomial"], degree=r["degree"],
                                            artifact=str((out / "dense_crosscheck.csv").relative_to(REPO)))))
    with open(out / "ledger_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS); w.writeheader(); w.writerows(ledger)
    if args.append_ledger:
        log = REPO / "results" / "experiment_log.csv"
        before = log.read_bytes()
        if not before.endswith(b"\n"):
            raise ValueError("ledger does not end with newline")
        with open(log, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=LEDGER_COLUMNS).writerows(ledger)
        if log.read_bytes()[:len(before)] != before:
            raise RuntimeError("existing ledger bytes changed")
    manifest.update(status="completed", artifacts_sha256={f.name: _sha(f) for f in sorted(out.iterdir())
                                                           if f.name != "manifest.json"},
                    ledger_appended=bool(args.append_ledger))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
