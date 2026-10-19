"""Compare the N=128 check with frozen N=64 results; never rewrite the baseline."""
import csv
import hashlib
import json
import math
from pathlib import Path

from .phase_a import analyze, read_checkpoints


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)


def compare(run_dir, baseline):
    run_dir, baseline = Path(run_dir), Path(baseline)
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    original = json.loads((baseline / 'manifest.json').read_text())
    assert manifest['status'] == original['status'] == 'completed'
    assert manifest['N'] == 128 and original['N'] == 64
    assert manifest['M_grid'] == original['M_grid']
    assert manifest['eps_targets'] == original['eps_targets']
    for directory, meta in ((run_dir, manifest), (baseline, original)):
        for file, digest in meta['artifacts_sha256'].items():
            assert sha(directory / file) == digest, file
    all_new = read_checkpoints(run_dir / 'checkpoints')
    expected = {('primary', s, seed, M) for s in manifest['s_values']
                for seed in manifest['seeds_per_s'][str(s)] for M in manifest['M_grid']}
    keys = [(r['mode'], r['s'], r['seed'], r['M']) for r in all_new]
    assert len(keys) == len(set(keys)) and set(keys) == expected
    assert all(r['N'] == 128 and r['sign_verified'] for r in all_new)
    assert max(r['peak_rss_bytes'] for r in all_new) <= 4 * 2**30
    rows64 = read_checkpoints(baseline / 'checkpoints')
    matched = [r for r in rows64 if r['mode'] == 'primary' and r['s'] in manifest['s_values']
               and r['seed'] in manifest['seeds_per_s'][str(r['s'])]]
    assert len(matched) == len(all_new)
    results = {'N128': analyze(all_new, N=128), 'N64_matched': analyze(matched, N=64)}
    # Retain the original published fits, including s=32 and its unequal instance count.
    with (baseline / 'alpha_fits.csv').open() as f:
        published = list(csv.DictReader(f))
    def indexed(fits):
        return {(r['quantity'], float(r['eps']), r['polylog']): r for r in fits}
    new = indexed(results['N128']['alpha_fits'])
    comparisons = []
    for label, fits in [('published', published), ('matched', results['N64_matched']['alpha_fits'])]:
        for key, old in indexed(fits).items():
            current = new[key]
            delta = float(current['alpha']) - float(old['alpha'])
            comparable = math.isfinite(delta)
            comparisons.append(dict(baseline=label, quantity=key[0], eps=key[1], polylog=key[2],
                                    alpha_N64=float(old['alpha']), ci95_N64=old['ci95'],
                                    alpha_N128=current['alpha'], ci95_N128=current['ci95'],
                                    delta_alpha=delta,
                                    status='comparable' if comparable else 'exponent not estimable',
                                    flag_gt_0_3=(abs(delta) > 0.3) if comparable else ''))
    out = run_dir / 'comparison'
    out.mkdir()  # refuse to overwrite a completed comparison
    for name, result in results.items():
        for table, rows in result.items():
            write_csv(out / f'{name}_{table}.csv', rows)
    write_csv(out / 'alpha_comparison.csv', comparisons)
    provenance = dict(
        interpretation=manifest['interpretation'], baseline_dir=baseline.name,
        baseline_manifest_sha256=sha(baseline / 'manifest.json'),
        run_manifest_sha256=sha(run_dir / 'manifest.json'),
        analysis_code_sha256=sha(Path(__file__)),
        matched_note='N64 restricted to s=2,4,8,16 and seeds=42,123,456; no new N64 simulation',
        uncertainty_note='95% bootstrap CIs resample instances within s; do not cover finite-grid, degree-search, or finite-size systematic error',
        artifacts_sha256={p.name: sha(p) for p in sorted(out.glob('*.csv'))})
    (out / 'manifest.json').write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    for r in comparisons:
        if r['quantity'] in ('index', 'd23'):
            print(r['baseline'], r['quantity'], r['eps'], r['polylog'],
                  f"{r['alpha_N64']:.4f} -> {r['alpha_N128']:.4f}; delta {r['delta_alpha']:+.4f}",
                  'FLAG' if r['flag_gt_0_3'] else '')


if __name__ == '__main__':
    import argparse
    from .env import require_qos_ref_venv
    require_qos_ref_venv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    parser.add_argument('baseline')
    args = parser.parse_args()
    compare(args.run_dir, args.baseline)
