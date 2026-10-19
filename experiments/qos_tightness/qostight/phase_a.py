"""Phase (a): primitive sample-complexity sweeps vs s at N = 64 (expected-unitary QOS, reference convention).

Measured per (mode, s, seed, M): element-oracle, index-oracle and block-encoding (A/2) errors and the
samples each consumes. Analysis: M* at eps* in {1e-2, 1e-3} by log-log interpolation; alpha_s and
beta fits with 95% bootstrap CIs over instances; local slopes between consecutive s, raw and with the
theorem's polylog factors divided out. Synthetic circulant support is the ideal-sparsity case.
"""
import json
import math
from pathlib import Path

import numpy as np

N_FIXED = 64
S_VALUES = (2, 4, 8, 16, 32)
M_GRID = tuple(10**k for k in range(4, 12))
SEEDS = (42, 123, 456, 789, 2026)
EPS_TARGETS = (1e-2, 1e-3)
BOOT_SEED = 42


def seeds_for(s):
    """5 instances for s <= 16; 3 at s = 32 (budget; wider CI)."""
    return SEEDS if s <= 16 else SEEDS[:3]


def all_points(mode):
    return [(mode, s, seed, M) for s in S_VALUES for seed in seeds_for(s) for M in M_GRID]


# ---------------------------------------------------------------- checkpoints
def append_checkpoint(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()


def read_checkpoints(directory):
    rows = []
    for p in sorted(Path(directory).glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def point_key(row):
    return (row["mode"], int(row["s"]), int(row["seed"]), int(row["M"]))


def pending_points(points, done_rows):
    done = {point_key(r) for r in done_rows}
    return [p for p in points if p not in done]


# ---------------------------------------------------------------- measurement
def index_oracle_error(oracle, A, axis):
    """Max over valid (i, k) of || oracle[i, k] - e_{j(i,k)} ||_2 (reference benchmark metric)."""
    mat = A if axis == 0 else A.T
    oracle = np.asarray(oracle)
    worst = 0.0
    for i in range(mat.shape[0]):
        cols = np.flatnonzero(mat[i] != 0)
        for k, j in enumerate(cols):
            target = np.zeros(mat.shape[1]); target[j] = 1.0
            worst = max(worst, float(np.linalg.norm(oracle[i, k] - target)))
    return worst


def measure_point(mode, s, seed, M, N=N_FIXED):
    """One build. mode: 'primary' (verified sign, D.23 precision), 'a2p_i' (pyqsp, degree 101),
    'a2p_ii' (verified, degree 101)."""
    import time
    from .blockenc import classical_rescaling, d23_amplify, sketched_block_encoding
    from .matrices import circulant_sparse
    A = circulant_sparse(N, N, s, seed=seed)
    kw = dict(primary={}, a2p_i=dict(sign_method="pyqsp", sign_degree=101),
              a2p_ii=dict(sign_method="verified", sign_degree=101))[mode]
    t0 = time.perf_counter()
    be = sketched_block_encoding(A, int(M), return_oracles=True, **kw)
    amp = d23_amplify(be)
    seconds = time.perf_counter() - t0
    spec = lambda a: float(np.linalg.norm(a, 2))
    return dict(mode=mode, s=s, seed=seed, M=int(M), N=N, seconds=seconds,
                sign_method=be["sign_method"], sign_degree=be["sign_degree"],
                amplification_degree=amp["degree"],
                err_element=float(np.max(np.abs(be["oracles"]["element"] - A.reshape(-1)))),
                err_index=max(index_oracle_error(be["oracles"]["row_index"], A, 0),
                              index_oracle_error(be["oracles"]["col_index"], A, 1)),
                err_be_A_over_s=spec(be["B"] - A / s), err_d23=spec(amp["block"] - A / 2),
                err_rescale=spec(classical_rescaling(be)["block"] - A / 2),
                samples_element=be["samples"]["element"], samples_index=be["samples"]["row_index"],
                samples_be_query=be["samples"]["per_be_query"], samples_d23=amp["samples_total"],
                sign_verified=be["sign_info"]["row_index"]["verified"])


# ---------------------------------------------------------------- analysis
def crossing_M(Ms, errs, eps):
    """M at which error first drops to eps, log-log interpolated between bracketing grid points."""
    Ms, errs = np.asarray(Ms, float), np.asarray(errs, float)
    order = np.argsort(Ms); Ms, errs = Ms[order], errs[order]
    if errs[0] <= eps:
        return dict(M_star=float("nan"), status="below grid")
    for k in range(1, len(Ms)):
        if errs[k] <= eps:
            x0, x1, y0, y1 = np.log(Ms[k - 1]), np.log(Ms[k]), np.log(errs[k - 1]), np.log(errs[k])
            x = x0 + (np.log(eps) - y0) * (x1 - x0) / (y1 - y0)
            return dict(M_star=float(np.exp(x)), status="bracketed")
    return dict(M_star=float("nan"), status="not reached")


def _by_s(rows, eps, polylog=None):
    out = {}
    for r in rows:
        if r["eps"] != eps or not np.isfinite(r["M_star"]):
            continue
        v = math.log(r["M_star"] / (polylog(r["s"], eps) if polylog else 1.0))
        out.setdefault(r["s"], []).append(v)
    return {s: np.array(v) for s, v in sorted(out.items())}


def _boot(by_s, stat, n_boot, rng):
    vals = []
    for _ in range(n_boot):
        sample = {s: rng.choice(v, size=len(v), replace=True) for s, v in by_s.items()}
        vals.append(stat(sample))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _slope(by_s):
    s = np.log(np.array(list(by_s)))
    y = np.array([v.mean() for v in by_s.values()])
    return float(np.polyfit(s, y, 1)[0])


def fit_alpha(rows, eps, n_boot=2000, polylog=None):
    by_s = _by_s(rows, eps, polylog)
    if len(by_s) < 2:
        return dict(eps=eps, alpha=float("nan"), ci95=[float("nan")] * 2, n_s=len(by_s),
                    n_instances={int(s): int(len(v)) for s, v in by_s.items()}, status="insufficient s")
    rng = np.random.default_rng(BOOT_SEED)
    return dict(eps=eps, alpha=_slope(by_s), ci95=_boot(by_s, _slope, n_boot, rng), n_s=len(by_s),
                n_instances={int(s): int(len(v)) for s, v in by_s.items()})


def fit_beta(rows, n_boot=2000):
    """beta = log(M*(1e-3)/M*(1e-2)) / log(10) per (s, instance); mean with bootstrap over instances."""
    lo, hi = EPS_TARGETS
    m = {}
    for r in rows:
        if np.isfinite(r["M_star"]):
            m.setdefault((r["s"], r["seed"]), {})[r["eps"]] = r["M_star"]
    by_s = {}
    for (s, seed), d in sorted(m.items()):
        if lo in d and hi in d:
            by_s.setdefault(s, []).append(math.log(d[hi] / d[lo]) / math.log(lo / hi))
    by_s = {s: np.array(v) for s, v in by_s.items()}
    if not by_s:
        return dict(beta=float("nan"), ci95=[float("nan")] * 2, per_s={}, status="no instance bracketed at both eps*")
    mean = lambda b: float(np.mean(np.concatenate(list(b.values()))))
    rng = np.random.default_rng(BOOT_SEED)
    return dict(beta=mean(by_s), ci95=_boot(by_s, mean, n_boot, rng),
                per_s={int(s): float(v.mean()) for s, v in by_s.items()})


def local_slopes(rows, eps, n_boot=2000, polylog=None):
    by_s = _by_s(rows, eps, polylog)
    keys = list(by_s)
    rng = np.random.default_rng(BOOT_SEED)
    out = []
    for a, b in zip(keys, keys[1:]):
        pair = {a: by_s[a], b: by_s[b]}
        f = lambda d: float((d[b].mean() - d[a].mean()) / math.log(b / a))
        out.append(dict(pair=(int(a), int(b)), slope=f(pair), ci95=_boot(pair, f, n_boot, rng)))
    return out


def d23_polylog(s, N, eps):
    """s-dependent polylog of Theorem D.23 (Eq. D211): log^4(s log N / eps); log^2 N is s-independent."""
    return math.log(s * math.log2(N) / eps) ** 4


def d21_polylog(s, N, eps):
    """Lemma D.21 (Eq. D186): log^2(N) log^2(log(N)/eps) -- independent of s at fixed N."""
    return math.log2(N) ** 2 * math.log(math.log2(N) / eps) ** 2


# ---------------------------------------------------------------- orchestration
QUANTITIES = {  # name -> (samples column, error column, predicted alpha_s, polylog fn or None)
    "element": ("samples_element", "err_element", 1.0, None),
    "index": ("samples_index", "err_index", 3.0, d21_polylog),
    "d23": ("samples_d23", "err_d23", 5.0, d23_polylog),
    "rescale_optimistic": ("samples_be_query", "err_rescale", None, None),
}
PREDICTION_NOTES = {
    "element": "from the reference code: M ~ nnz/eps = N s/eps at fixed N (Lemma D.19 statement not read)",
    "index": "Lemma D.21: O~(R N s^3 / eps); its polylog log^2 N log^2(log N/eps) is s-independent at fixed N",
    "d23": "Theorem D.23: O~(R N s^5 / eps); s-dependent polylog log^4(s log N / eps) divided out",
    "rescale_optimistic": "comparison only (optimistic: reference convention); no theorem prediction",
}


def shard(points, workers):
    return [points[k::workers] for k in range(workers)]


def analyze(rows, mode="primary", n_boot=2000, N=N_FIXED):
    rows = [r for r in rows if r["mode"] == mode]
    mstar, fits, betas, slopes = [], [], [], []
    for q, (xcol, ycol, pred, poly) in QUANTITIES.items():
        qrows = []
        groups = {}
        for r in rows:
            groups.setdefault((r["s"], r["seed"]), []).append(r)
        for (s, seed), g in sorted(groups.items()):
            xs = np.array([r[xcol] for r in g], float); ys = np.array([r[ycol] for r in g], float)
            for eps in EPS_TARGETS:
                c = crossing_M(xs, ys, eps)
                rec = dict(quantity=q, s=s, seed=seed, eps=eps, M_star=c["M_star"], status=c["status"])
                mstar.append(rec); qrows.append(rec)
        for eps in EPS_TARGETS:
            for label, fn in (("raw", None), ("polylog_divided", (lambda s, e, f=poly: f(s, N, e)) if poly else None)):
                if label == "polylog_divided" and fn is None:
                    continue
                f = fit_alpha(qrows, eps, n_boot=n_boot, polylog=fn)
                fits.append(dict(quantity=q, polylog=label, predicted=pred, **f))
                for ls in local_slopes(qrows, eps, n_boot=n_boot, polylog=fn):
                    slopes.append(dict(quantity=q, eps=eps, polylog=label, predicted=pred, **ls))
        betas.append(dict(quantity=q, predicted=1.0 if pred is not None else None, **fit_beta(qrows, n_boot)))
    return dict(mstar=mstar, alpha_fits=fits, beta_fits=betas, local_slopes=slopes)


def a2prime_vs_s(rows):
    """Error vs s at the largest M (fixed degree 101): (i) reference pyqsp path, (ii) verified-bounded."""
    out = []
    Mmax = max(M_GRID)
    for mode in ("a2p_i", "a2p_ii", "primary"):
        for s in S_VALUES:
            g = [r for r in rows if r["mode"] == mode and r["s"] == s and r["M"] == Mmax]
            if not g:
                continue
            for k in ("err_index", "err_d23"):
                v = np.array([r[k] for r in g])
                out.append(dict(mode=mode, s=s, M=Mmax, metric=k, mean=float(v.mean()), min=float(v.min()),
                                max=float(v.max()), n_instances=len(v), sign_degree=g[0]["sign_degree"],
                                all_angles_verified=bool(all(r["sign_verified"] for r in g))))
    return out


def worker(run_dir, worker_id, points):
    """Process points sequentially, checkpointing each; skip points already done (resume)."""
    import psutil
    from .resources import gate_ok, measured
    N = json.loads((Path(run_dir) / "manifest.json").read_text())["N"]
    ck = Path(run_dir) / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    todo = pending_points([tuple(p) for p in points], read_checkpoints(ck))
    for mode, s, seed, M in todo:
        proc = psutil.Process()
        if not gate_ok(psutil.virtual_memory().available, proc.memory_info().rss):
            raise SystemExit(f"memory gate: available={psutil.virtual_memory().available}")
        with measured() as mem:
            row = measure_point(mode, s, seed, M, N=N)
        row.update(worker_id=worker_id, peak_rss_bytes=mem["peak_rss_bytes"])
        append_checkpoint(ck / f"w{worker_id}.jsonl", row)
        print("point", mode, s, seed, M, f"{row['seconds']:.1f}s", flush=True)


# ---------------------------------------------------------------- CLI
def _partest_worker(start_file, out_file, s=16, M=10**6):
    """Warm up (angles + one build), wait for the start signal, time one build."""
    import time, psutil
    measure_point("primary", s, 42, M)  # warm-up: computes and caches sign angles
    Path(str(out_file) + ".ready").write_text("ready")
    while not Path(start_file).exists():
        time.sleep(0.2)
    t0 = time.perf_counter()
    measure_point("primary", s, 123, M)
    Path(out_file).write_text(json.dumps(dict(seconds=time.perf_counter() - t0,
                                              rss_bytes=psutil.Process().memory_info().rss)))


def partest(out_dir, s=16):
    """Steady-state build time with 1 worker vs 2 concurrent workers (s=16, N=64)."""
    import subprocess, sys, time, psutil
    from .resources import gate_ok
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parents[1]
    result = {}
    for n in (1, 2):
        start = out_dir / f"start_{n}"
        files = [out_dir / f"t{n}_{k}.json" for k in range(n)]
        procs = [subprocess.Popen([sys.executable, "-m", "qostight.phase_a", "partest-worker", str(start), str(f),
                                   str(s)], cwd=here) for f in files]
        min_avail = psutil.virtual_memory().available
        while not all(Path(str(f) + ".ready").exists() for f in files):  # all workers warm
            if any(pr.poll() not in (None, 0) for pr in procs):
                raise RuntimeError("partest worker failed during warm-up")
            min_avail = min(min_avail, psutil.virtual_memory().available)
            time.sleep(0.5)
        start.write_text("go")
        while any(p.poll() is None for p in procs):
            min_avail = min(min_avail, psutil.virtual_memory().available)
            time.sleep(0.5)
        times = [json.loads(f.read_text())["seconds"] for f in files]
        rss = [json.loads(f.read_text())["rss_bytes"] for f in files]
        result[f"workers_{n}"] = dict(build_seconds=times, rss_bytes=rss, min_available_bytes=min_avail,
                         gate_held=bool(gate_ok(min_avail, max(rss))))
    t1, t2 = result["workers_1"]["build_seconds"][0], max(result["workers_2"]["build_seconds"])
    result["speedup"] = 2 * t1 / t2  # throughput ratio: 2 builds in parallel vs 2 sequential
    result["use_two_workers"] = bool(result["speedup"] >= 1.5 and result["workers_2"]["gate_held"])
    (out_dir / "partest.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv=None):
    import argparse
    from .env import require_qos_ref_venv
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pw = sub.add_parser("partest-worker"); pw.add_argument("start"); pw.add_argument("out"); pw.add_argument("s", type=int)
    pt = sub.add_parser("partest"); pt.add_argument("out_dir")
    w = sub.add_parser("worker"); w.add_argument("run_dir"); w.add_argument("worker_id", type=int); w.add_argument("points")
    r = sub.add_parser("run"); r.add_argument("--resume"); r.add_argument("--workers", type=int, default=1)
    r.add_argument("--partest"); r.add_argument("--append-ledger", action="store_true")
    r.add_argument("--n-check", action="store_true",
                   help="N=128 primary only, s=2,4,8,16, seeds=42,123,456")
    args = p.parse_args(argv)
    require_qos_ref_venv()
    if args.cmd == "partest-worker":
        return _partest_worker(args.start, args.out, args.s)
    if args.cmd == "partest":
        print(json.dumps(partest(args.out_dir), indent=1)); return
    if args.cmd == "worker":
        return worker(args.run_dir, args.worker_id, json.loads(Path(args.points).read_text()))
    return run(args)


def run(args):
    import csv, hashlib, platform, subprocess, sys, time, uuid
    from datetime import datetime, timezone
    from .env import REPO, REFERENCE_COMMIT
    from .matrices import IDEAL_SPARSITY_NOTE
    from .run_phase1 import INTERPRETATION, LEDGER_COLUMNS, git_commit
    here = Path(__file__).resolve().parents[1]
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    if args.resume:
        run_dir = REPO / args.resume
        manifest = json.loads((run_dir / "manifest.json").read_text())
    else:
        n_check = getattr(args, "n_check", False)
        now = datetime.now(timezone.utc)
        run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        run_dir = REPO / "results" / "tightness" / f"phase_a_{'n128_' if n_check else ''}{run_id}"
        run_dir.mkdir(parents=True)
        code = sorted(str(f.relative_to(REPO)) for f in (here / "qostight").glob("*.py"))
        manifest = dict(run_id=run_id, date_utc=now.isoformat(), interpretation=INTERPRETATION,
                        git_commit=git_commit(), reference_commit=REFERENCE_COMMIT, python=sys.version,
                        platform=platform.platform(), lock_sha256=sha(here / "requirements-lock.txt"),
                        code_sha256={c: sha(REPO / c) for c in code}, data_note=IDEAL_SPARSITY_NOTE,
                        N=128 if n_check else N_FIXED,
                        s_values=list(S_VALUES[:-1] if n_check else S_VALUES), M_grid=list(M_GRID),
                        seeds_per_s={str(s): list(SEEDS[:3] if n_check else seeds_for(s))
                                     for s in (S_VALUES[:-1] if n_check else S_VALUES)},
                        instances_note=("3 instances per s (N-dependence check)" if n_check else
                                        "s=32 has 3 instances (budget); its CIs are wider"),
                        eps_targets=list(EPS_TARGETS), bootstrap=dict(n=2000, seed=BOOT_SEED, resample="instances within s"),
                        workers=args.workers, partest=(json.loads(Path(args.partest).read_text()) if args.partest else None),
                        modes=dict(primary="verified sign at D.23 precision; Theorem D.23 amplification",
                                   a2p_i="a2' (i): reference pyqsp PolySign, degree 101, failure recorded as found",
                                   a2p_ii="a2' (ii): verified-bounded polynomial, fixed degree 101"),
                        predictions={q: dict(alpha=v[2], note=PREDICTION_NOTES[q]) for q, v in QUANTITIES.items()},
                        wall_seconds=0.0, stages=[], status="running")
        if n_check:
            manifest["modes"] = {"primary": manifest["modes"]["primary"]}
        manifest["git_dirty_at_start"] = bool(subprocess.check_output(
            ["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip())
    def save():
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    save()
    ck = run_dir / "checkpoints"; ck.mkdir(exist_ok=True)
    (run_dir / "assign").mkdir(exist_ok=True)
    for mode in (m for m in ("primary", "a2p_i", "a2p_ii") if m in manifest["modes"]):
        for s in manifest["s_values"]:
            pts = pending_points([(mode, s, seed, M) for seed in manifest["seeds_per_s"][str(s)]
                                  for M in manifest["M_grid"]], read_checkpoints(ck))
            if not pts:
                continue
            import psutil
            from .resources import gate_ok
            if not gate_ok(psutil.virtual_memory().available, psutil.Process().memory_info().rss,
                           before_stage=True):
                raise SystemExit("memory gate before stage; resume when >=4 GiB available")
            t0 = time.perf_counter()
            procs = []
            for k, part in enumerate(shard(pts, manifest["workers"])):
                if not part:
                    continue
                f = run_dir / "assign" / f"w{k}_{mode}_s{s}.json"
                f.write_text(json.dumps(part))
                procs.append(subprocess.Popen([sys.executable, "-m", "qostight.phase_a", "worker",
                                               str(run_dir), str(k), str(f)], cwd=here))
            codes = [pr.wait() for pr in procs]
            dt = time.perf_counter() - t0
            manifest["wall_seconds"] += dt
            manifest["stages"].append(dict(mode=mode, s=s, points=len(pts), seconds=dt, return_codes=codes))
            save()
            print(f"stage {mode} s={s}: {len(pts)} points in {dt:.0f}s rc={codes}", flush=True)
            if any(codes):
                manifest["status"] = "interrupted"; save()
                raise SystemExit(f"worker failure at {mode} s={s}; rerun with --resume {run_dir.relative_to(REPO)}")
    rows = read_checkpoints(ck)
    out = analyze(rows, N=manifest["N"])
    def write(name, recs):
        keys = sorted({k for r in recs for k in r})
        with open(run_dir / name, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(recs)
    write("points.csv", rows)
    write("mstar.csv", out["mstar"])
    write("alpha_fits.csv", out["alpha_fits"])
    write("beta_fits.csv", out["beta_fits"])
    write("local_slopes.csv", out["local_slopes"])
    write("a2prime_vs_s.csv", a2prime_vs_s(rows))
    ledger = []
    date = datetime.now(timezone.utc).date().isoformat()
    def lrow(suffix, metric, result, s, notes):
        return dict(experiment_id=f"qos-tightness-phase-a-{manifest['run_id']}-{suffix}",
                    dataset="synthetic circulant sparse (uniform s; ideal-sparsity case)", date=date,
                    git_commit=manifest["git_commit"], preprocessing="none (synthetic, ||X||=1)",
                    n_samples=manifest["N"], n_features=manifest["N"], max_sparsity=s,
                    sketch_method="QOS expected unitary (reference qos.py @ a3fe8d6)", sketch_dimension="", seed=42,
                    model="sparse oracles (Lemma D.21) + block encoding (Lemma D.22) + Theorem D.23 amplification",
                    metric=metric, result=result,
                    notes=json.dumps(dict(notes, interpretation=INTERPRETATION,
                                          artifact_dir=str(run_dir.relative_to(REPO))), sort_keys=True))
    for f in out["alpha_fits"]:
        ledger.append(lrow(f"alpha-{f['quantity']}-eps{f['eps']:g}-{f['polylog']}", "fitted_alpha_s", f["alpha"],
                           f"{min(manifest['s_values'])}-{max(manifest['s_values'])}",
                           dict(ci95=f["ci95"], predicted=f["predicted"], n_instances=f["n_instances"])))
    for f in out["beta_fits"]:
        ledger.append(lrow(f"beta-{f['quantity']}", "fitted_beta_eps", f["beta"],
                           f"{min(manifest['s_values'])}-{max(manifest['s_values'])}",
                           dict(ci95=f["ci95"], predicted=f["predicted"])))
    for r in a2prime_vs_s(rows):
        if r["mode"] != "primary":
            ledger.append(lrow(f"{r['mode']}-s{r['s']}-{r['metric']}", f"{r['metric']}_fixed_degree_101_at_Mmax",
                               r["mean"], r["s"], dict(min=r["min"], max=r["max"], n_instances=r["n_instances"],
                                                       all_angles_verified=r["all_angles_verified"])))
    with open(run_dir / "ledger_rows.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS); wr.writeheader(); wr.writerows(ledger)
    if args.append_ledger and not manifest.get("ledger_appended"):
        log = REPO / "results" / "experiment_log.csv"
        before = log.read_bytes()
        assert before.endswith(b"\n")
        with open(log, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=LEDGER_COLUMNS).writerows(ledger)
        assert log.read_bytes()[:len(before)] == before
        manifest["ledger_appended"] = True
    manifest.update(status="completed", artifacts_sha256={p.name: sha(p) for p in sorted(run_dir.iterdir())
                                                           if p.is_file() and p.name != "manifest.json"})
    save()
    print("done", run_dir.relative_to(REPO), f"wall {manifest['wall_seconds'] / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
