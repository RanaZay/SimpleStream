#!/usr/bin/env python3
"""Replay PRISM-VoI at many lambda values from one trace run (PRISM_VOI_TRACE=1) on a multiple-choice benchmark.

For each lambda the greedy rule is re-applied to the recorded semantic entropies: retrieve only if SE(R) > lambda,
keep adding the next ranked frame while it removes more than lambda nats, answer from the last accepted context.
Accuracy uses the benchmark's metric; cost counts the contexts that rule would have evaluated plus the final answer.

Usage: python main_experiments/tools/lambda_sweep.py --benchmark streamingbench --run reports/.../prism_voi_<job>
"""
import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from main_experiments.tools.benchmark_tables import headline, table
from main_experiments.tools.summarize_progressive_arbitration import correctness, profile, records

LAMBDAS = (float('inf'), 1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.0)


def depth_for(trace, lam):
    if trace[0]['se'] <= lam:
        return 0, 1
    depth = 0
    for row in trace[1:]:
        if row['gain'] > lam and row['depth'] == depth + 1:
            depth = row['depth']
        else:
            return depth, min(len(trace), depth + 2)
    return depth, len(trace)


def correct_at(row, record, benchmark):
    pred = row.get('prediction')
    if benchmark == 'ovo':
        return bool(correctness({**record, 'response': pred, 'correct': None, 'error': None}, 'ovo'))
    return pred is not None and pred == str(record.get('answer_gt', '')).strip().upper()[:1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--benchmark', required=True, choices=['ovo', 'streamingbench', 'videomme'])
    p.add_argument('--run', required=True, type=Path)
    args = p.parse_args()
    items = []
    for record in records(args.run):
        meta, timing = profile(record)[1], profile(record)[2]
        trace = meta.get('trace')
        if not trace or meta.get('format') != 'mcq':
            continue
        task = record.get('task') or record.get('task_type') or 'all'
        items.append((task, record, trace, timing.get('final_generation_ms') or 0.0))
    if not items:
        raise SystemExit('No traced multiple-choice questions found (was the run made with PRISM_VOI_TRACE=1?)')
    bench = 'ovo' if args.benchmark == 'ovo' else 'streamingbench'
    rows = []
    for lam in LAMBDAS:
        answers, frames, used, cost = [], [], 0, []
        for task, record, trace, final_ms in items:
            depth, evaluated = depth_for(trace, lam)
            answers.append((task, correct_at(trace[depth], record, args.benchmark)))
            frames.append(depth)
            used += depth > 0
            cost.append(sum(r.get('ms', 0.0) for r in trace[:evaluated]) + final_ms)
        rows.append([('∞ (= Recent-6)' if lam == float('inf') else f'{lam:g}'), f"{headline(answers, bench):.2f}",
                     f"{statistics.mean(frames):.2f}", str(used), f"{statistics.mean(cost) / 1000:.3f}"])
    base = float(rows[0][1])
    for r in rows:
        r.insert(2, f"{float(r[1]) - base:+.2f}")
    print(f'## {args.benchmark}: PRISM-VoI accuracy-cost curve from one trace run ({len(items)} questions)\n')
    print(table(['λ (nats/frame)', 'Accuracy', 'Δ vs Recent-6', 'Memory frames (mean)', 'Questions using memory',
                 'Cost per question (s)'], rows))
    print('\nCost: semantic-entropy evaluations the rule would have run, plus the final answer; decoding excluded.')


if __name__ == '__main__':
    main()
