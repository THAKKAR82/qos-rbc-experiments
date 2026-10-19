"""Integration check: use the matched baseline and preserve its bytes."""
import csv
import json

import pytest

from qostight import n_check, phase_a


def test_comparison_matches_grid_and_preserves_baseline(tmp_path, monkeypatch):
    # Reduce bootstrap count for this exact power-law integration fixture.
    real_analyze = phase_a.analyze
    monkeypatch.setattr(n_check, 'analyze', lambda rows, N: real_analyze(rows, N=N, n_boot=10))
    grid = list(phase_a.M_GRID)
    for N in (64, 128):
        directory = tmp_path / str(N)
        (directory / 'checkpoints').mkdir(parents=True)
        ss = (2, 4, 8, 16, 32) if N == 64 else (2, 4, 8, 16)
        seeds = (42, 123, 456, 789, 2026) if N == 64 else (42, 123, 456)
        rows = []
        for s in ss:
            for seed in seeds:
                for M in grid:
                    # The extra N64 s=32 points deliberately bend the global fit.
                    error = 1000*s*(8 if s == 32 else 1)/M
                    rows.append(dict(mode='primary', N=N, s=s, seed=seed, M=M,
                                     sign_verified=True, peak_rss_bytes=100,
                                     samples_element=M, err_element=error,
                                     samples_index=M, err_index=error,
                                     samples_d23=M, err_d23=error,
                                     samples_be_query=M, err_rescale=error))
        for row in rows:
            phase_a.append_checkpoint(directory / 'checkpoints' / 'w0.jsonl', row)
        n_check.write_csv(directory / 'alpha_fits.csv', real_analyze(rows, N=N, n_boot=10)['alpha_fits'])
        meta = dict(status='completed', N=N, M_grid=grid, eps_targets=list(phase_a.EPS_TARGETS),
                    s_values=list(ss), seeds_per_s={str(s): list(seeds) for s in ss},
                    interpretation='simulation (expected-unitary QOS, reference convention)',
                    artifacts_sha256={'alpha_fits.csv': n_check.sha(directory / 'alpha_fits.csv')})
        (directory / 'manifest.json').write_text(json.dumps(meta))
    baseline = tmp_path / '64'
    before = {p.relative_to(baseline): p.read_bytes() for p in baseline.rglob('*') if p.is_file()}
    n_check.compare(tmp_path / '128', baseline)
    with (tmp_path / '128/comparison/alpha_comparison.csv').open() as f:
        comparisons = list(csv.DictReader(f))
    raw = [r for r in comparisons if r['quantity'] == 'index' and r['polylog'] == 'raw']
    assert all(float(r['delta_alpha']) == pytest.approx(0) for r in raw if r['baseline'] == 'matched')
    assert all(r['flag_gt_0_3'] == 'True' for r in raw if r['baseline'] == 'published')
    assert before == {p.relative_to(baseline): p.read_bytes() for p in baseline.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):
        n_check.compare(tmp_path / '128', baseline)
