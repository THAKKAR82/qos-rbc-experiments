"""Render saved gate measurements; no quantum or scaling experiment is run."""
import csv
import hashlib
import json
from pathlib import Path
import sys

from qostight.env import require_qos_ref_venv


def main(directory):
    require_qos_ref_venv()
    root = Path(directory)
    results = {job:json.loads((root/job/'result.json').read_text())
               for job in ('baseline','cold','warm','kappa4')}
    rows = []
    for job, result in results.items():
        rows.append(dict(job=job, kappa=result['kappa_reg'],
                         seconds=result['runtime_seconds'], peak_GiB=result['peak_rss_bytes']/2**30,
                         fixed_point_seconds=result['timings']['fixed_point_polynomial_and_angles'],
                         inverse_degree=result['degrees']['inverse'], fixed_point_degree=result['degrees']['fixed_point'],
                         full_state_error=result['gates']['full_state_error'],
                         max_dense_error=max(result['gates'][k] for k in
                                             ('d23_dense_error','inverse_dense_error','fixed_point_dense_error')),
                         fixed_point_angle_error=result['fixed_point']['angle_error']))
    with (root/'benchmark.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
    comparisons = []
    base = results['baseline']
    for job in ('cold','warm'):
        r = results[job]
        comparisons.append(dict(job=job, total_speedup=base['runtime_seconds']/r['runtime_seconds'],
                                phase_stage_speedup=base['timings']['fixed_point_polynomial_and_angles']/r['timings']['fixed_point_polynomial_and_angles'],
                                peak_RSS_ratio=r['peak_rss_bytes']/base['peak_rss_bytes'],
                                state_error_change=r['gates']['full_state_error']-base['gates']['full_state_error'],
                                mean_signal_change=r['mean_signal']-base['mean_signal'],
                                degrees_equal=r['degrees']==base['degrees'],
                                resources_equal=r['resources']==base['resources']))
    (root/'comparison.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    labels = ['6d25652\nfresh','batched\ncold','batched\ncache hit','kappa=4\ncap 8191']
    for ax, key, title in zip(axes, ['seconds','peak_GiB','max_dense_error'],
                              ['Measured wall time (s)','Measured peak RSS (GiB)','Max dense-vs-shortcut error']):
        ax.bar(labels, [r[key] for r in rows]); ax.set_title(title); ax.tick_params(axis='x', labelsize=8)
        if key != 'peak_GiB': ax.set_yscale('log')
    axes[1].axhline(4, color='red', linestyle='--', label='RSS limit'); axes[1].legend(fontsize=8)
    fig.suptitle('N=4 fixed-test simulation gate; single runs, not scaling or hardware measurements')
    fig.tight_layout(); fig.savefig(root/'benchmark.png', dpi=180); plt.close(fig)
    (root/'summary_manifest.json').write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs_sha256={job:hashlib.sha256((root/job/'result.json').read_bytes()).hexdigest() for job in results},
        outputs_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                       for name in ('benchmark.csv','comparison.json','benchmark.png')}), indent=2)+'\n')
    print(json.dumps(dict(rows=rows, comparisons=comparisons), indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
