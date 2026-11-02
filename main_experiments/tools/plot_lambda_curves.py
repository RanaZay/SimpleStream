#!/usr/bin/env python3
"""Accuracy-cost figure of PRISM-VoI: one curve per benchmark from its trace run, replayed at every lambda.

Each point is a lambda; x is the cost per question (semantic-entropy evaluations plus the final answer), y is the
accuracy change over Recent-6 (lambda = infinity, the leftmost point) with its paired-bootstrap 95% CI as a band.

Usage: python main_experiments/tools/plot_lambda_curves.py --out reports/figures/lambda_curves \\
           --run "StreamingBench=streamingbench:reports/.../prism_voi_345764" --run "OVO=ovo:reports/.../prism_voi_345763"
Writes <out>.pdf, <out>.png and <out>.csv (the plotted numbers).
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from main_experiments.tools.benchmark_tables import paired_test
from main_experiments.tools.lambda_sweep import LAMBDAS, depth_for, load_items, replay


def curve(benchmark, run, tasks=None):
    items = load_items(run, tasks)
    bench = 'ovo' if benchmark == 'ovo' else 'streamingbench'
    recent = replay(items, benchmark, lambda t: depth_for(t, float('inf')))[0]
    points = []
    for lam in LAMBDAS:
        answers, frames, used, cost = replay(items, benchmark, lambda t: depth_for(t, lam))
        test = paired_test(recent, answers, bench)
        points.append(dict(lam=lam, cost=statistics.mean(cost) / 1000, frames=statistics.mean(frames),
                           memory_share=used / len(items), delta=test['delta'], low=test['low'], high=test['high'],
                           p=test['p']))
    return points


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', action='append', required=True, help='Label=benchmark:trace run dir[:TASK,TASK]')
    p.add_argument('--out', required=True, type=Path, help='output path without extension')
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    curves = []
    for spec in args.run:
        label, rest = spec.split('=', 1)
        benchmark, run, *tasks = rest.split(':')
        curves.append((label, benchmark, curve(benchmark, Path(run), tasks[0].split(',') if tasks else None)))
    rows = [dict(label=label, benchmark=benchmark, **q) for label, benchmark, points in curves for q in points]
    with args.out.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        import matplotlib
    except ImportError:
        raise SystemExit(f'Wrote {args.out}.csv; install matplotlib for the figure (pip install matplotlib)')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for label, _benchmark, points in curves:
        xs, ys = [q['cost'] for q in points], [q['delta'] for q in points]
        line, = ax.plot(xs, ys, marker='o', markersize=3.5, linewidth=1.6, label=label)
        ax.fill_between(xs, [q['low'] for q in points], [q['high'] for q in points], color=line.get_color(), alpha=0.15,
                        linewidth=0)
        for q in points:
            if q['lam'] in (0.5, 0.1, 0.0):
                ax.annotate(f"λ={q['lam']:g}", (q['cost'], q['delta']), textcoords='offset points', xytext=(4, -10),
                            fontsize=7, color=line.get_color())
    ax.axhline(0, color='0.5', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Cost per question (s, MI210)')
    ax.set_ylabel('Accuracy change vs Recent-6 (pts)')
    ax.legend(frameon=False, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(args.out.with_suffix('.' + ext), dpi=200)
    print(f'Wrote {args.out}.pdf, .png and .csv')


if __name__ == '__main__':
    main()
