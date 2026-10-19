# Phase (a): N-dependence check

Date: 2026-10-19. Status: complete; stopped before phase (b).

Every numerical result is a **simulation (expected-unitary QOS, reference
convention)**. Circulant support is the ideal-sparsity case. These results are
not measured quantum hardware usage or evidence of quantum advantage.

## Handoff verification

The repository was clean at `2b5554b`. The original phase-(a) artifact and
source hashes match its manifest; the saved fits agree with the handoff.
Reference checkout: `a3fe8d6fe777750749f5e90131180e016088bdaf`.
The audit summary and re-validation artifacts are present and agree with the
reported findings. The original phase-(a) run has 36 ledger rows with distinct
experiment IDs.

Corrections to the handoff:

- Its history omits `be789eb` (dataset provenance), `acc0a4d` (path
  sanitization), and `741ddd0` (phase-(a) runner).
- The existing tightness suite has **54 passing tests**, rather than 41;
  phase-(a) added 13. The main suite collects 186 tests, consistent with
  the stated 177 + 9 total. Main-environment `pip check` is clean and JAX
  is absent.
- The original sample grid does not support an element-oracle exponent
  estimate; its saved fits explicitly report insufficient sparsity points.

## Protocol and provenance

- N = D = 128; s = {2, 4, 8, 16}; seeds = {42, 123, 456}.
- Primary verified-sign / D.23 path only; unchanged M = 10^4, ..., 10^11.
- ε* = 10^-2 and 10^-3. M* is interpolated in log-log coordinates between
  bracketing points, using each primitive's sample accounting.
- Same measurement, matrix generation, precision, normalization, sample
  accounting and fit functions as N = 64. The runner passes N from its
  manifest to measurement and polylog normalization. The element/index/D.23
  values retain the reference sample-count convention; no phase-(b) circuit
  query accounting is introduced here.
- Two worker processes under `caffeinate -i`, per-point checkpoints, 50 ms
  RSS sampling and the existing memory gate (4 GiB maximum RSS per process;
  4 GiB available before a stage, 2 GiB while running).
- Global fits and local slopes use 2,000 bootstrap replicates, seed 42,
  resampling instances within each s. CIs capture instance variation only,
  not grid interpolation, degree-search or finite-size systematic errors.
- Compare with both the published N = 64 fit (s = 2–32, five instances
  except three at s = 32) and a matched fit from its saved points
  (s = 2–16, seeds 42/123/456). No N = 64 simulation is rerun.
- Code runs only in `.venv-qos-ref`: JAX 0.8.1, pyqsp 0.2.0, NumPy 2.3.5,
  SciPy 1.16.3. The run manifest records Python, platform, dependency-lock
  hash, base commit, dirty-worktree status and numerical source hashes.
  The comparison manifest separately records its source and input hashes.

New run: [`phase_a_n128_20261019T153934Z-75edf052`](../../results/tightness/phase_a_n128_20261019T153934Z-75edf052/).
Frozen baseline: [`phase_a_20261019T145729Z-a1644ff3`](../../results/tightness/phase_a_20261019T145729Z-a1644ff3/).

## Results

**No global exponent difference exceeds 0.3**, against either the published
or the matched N = 64 baseline. For index and D.23, the largest absolute
change is 0.126 against the published fits and 0.071 against matched fits.
This supports stability of the fitted sparsity exponents over N = 64–128
under this simulation protocol. It does not establish an asymptotic exponent
or show that the theorem bound is loose. The D.23 change is small relative to
0.3 but exceeds some narrow instance-only CIs; it is not exactly zero.

| Quantity | ε* | Normalization | N64 published | N64 matched | N128 [95% CI] | Δ vs matched |
|---|---:|---|---:|---:|---:|---:|
| Index | 0.01 | raw = polylog-divided | 2.200 | 2.233 | 2.225 [2.221, 2.231] | -0.007 |
| Index | 0.001 | raw = polylog-divided | 2.191 | 2.223 | 2.207 [2.203, 2.213] | -0.016 |
| D.23 A/2 | 0.01 | raw | 3.147 | 3.210 | 3.273 [3.247, 3.298] | +0.062 |
| D.23 A/2 | 0.01 | polylog-divided | 2.671 | 2.716 | 2.787 [2.761, 2.813] | +0.071 |
| D.23 A/2 | 0.001 | raw | 3.128 | 3.190 | 3.251 [3.226, 3.275] | +0.061 |
| D.23 A/2 | 0.001 | polylog-divided | 2.755 | 2.805 | 2.872 [2.847, 2.896] | +0.066 |

The comparison targets remain 3 (index) and 5 (D.23), with the original
polylog conventions. Read these finite-grid fits at roughly tenths-level
precision, rather than treating the displayed decimal places as systematic
accuracy. The index polylog factor is s-independent at fixed N, so its
raw and polylog-divided slopes are equal.

At N = 128, β = 1.012 (index) and 1.020 (D.23), consistent with the
previous approximately inverse-linear error dependence. The optimistic
classical-rescaling comparison has raw α = 2.204 / 2.191 for the two
targets; it also raises no 0.3 flag. It remains a separately labelled
comparison, not the primary block-encoding path.

**Element-oracle limitation.** At ε* = 0.01, all crossings are below
the sample grid. At ε* = 0.001, only s = 2 and 4 are bracketed at
N = 128, yielding a two-point slope of −0.975; s = 8 and 16 remain
below-grid. N = 64 has no estimable corresponding exponent. Retain this
diagnostic in the CSV and ledger, but do not interpret it as the
element-oracle scaling law or an N-dependence result. The grid was not
extended in this check.

### Local slopes

The triples below correspond to s = 2→4, 4→8, 8→16. The degree-search
zig-zag persists at both sizes; global fits summarize a visibly nonuniform
finite-grid curve. Local-slope bootstrap CIs are retained in the CSVs.

| Quantity | ε* | Normalization | N64 matched | N128 |
|---|---:|---|---|---|
| index | 0.01 | raw | 2.441, 1.854, 2.530 | 2.435, 1.865, 2.496 |
| index | 0.001 | raw | 2.443, 1.837, 2.519 | 2.398, 1.850, 2.494 |
| d23 | 0.01 | raw | 3.388, 2.834, 3.535 | 3.382, 2.882, 3.683 |
| d23 | 0.01 | polylog-divided | 2.850, 2.341, 3.081 | 2.855, 2.399, 3.238 |
| d23 | 0.001 | raw | 3.449, 2.748, 3.520 | 3.459, 2.777, 3.674 |
| d23 | 0.001 | polylog-divided | 3.038, 2.364, 3.161 | 3.054, 2.399, 3.320 |

Full tables: [comparison](../../results/tightness/phase_a_n128_20261019T153934Z-75edf052/comparison/).

The run completed all 96 unique points in **16.76 minutes** (sum of stage
wall times), with **0.646 GiB** maximum sampled worker RSS. All sign-angle
checks passed; every worker stage returned zero. There were no restarts.

## Validation and stop point

- All 139 pre-existing data/result/test/notebook files in the pre-run hash
  inventory are byte-identical, including Bank data, fixtures, results and
  notebook 05, plus the original tightness results.
- The existing ledger bytes are an exact prefix of the updated file;
  exactly 16 N = 128 rows were appended.
- Numerical-source and output hashes match the run manifest. Comparison
  input manifests and output tables have separate SHA-256 provenance.
- Main-environment dependencies and `src/` were not changed. No Bank
  rerun was needed; the main regression suite was collected but not run.
- Updated tightness suite: **57 passed in 105.23 s** under `.venv-qos-ref`.
  New coverage checks N forwarding and checkpoint resume, N-specific polylog
  normalization, and matched comparison without modifying the baseline.

**Stopped before phase (b).** No end-to-end study, empirical F61
extrapolation, streaming benchmark, or later phase has been started.

## Reproduction

From `experiments/qos_tightness/`:

```sh
caffeinate -i ../../.venv-qos-ref/bin/python -m qostight.phase_a run --n-check --workers 2 --append-ledger
```

Resume the recorded run with `--resume results/tightness/phase_a_n128_20261019T153934Z-75edf052`.
The manifest defines the resumed configuration. Generate the comparison once
(the command refuses to overwrite an existing comparison directory):

```sh
../../.venv-qos-ref/bin/python -m qostight.n_check \
  ../../results/tightness/phase_a_n128_20261019T153934Z-75edf052 \
  ../../results/tightness/phase_a_20261019T145729Z-a1644ff3
```

The required stop point is immediately after this check; phase (b) requires
separate approval.
