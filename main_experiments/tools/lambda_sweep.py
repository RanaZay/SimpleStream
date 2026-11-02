#!/usr/bin/env python3
"""Replay PRISM-VoI at many lambda values from one trace run (PRISM_VOI_TRACE=1) on a multiple-choice benchmark.

For each lambda the greedy rule is re-applied to the recorded semantic entropies: retrieve only if SE(R) > lambda,
keep adding the next ranked frame while it removes more than lambda nats, answer from the last accepted context.
Accuracy uses the benchmark's metric; cost counts the contexts that rule would have evaluated plus the final answer.

Ablations from the same trace (no extra GPU time):
  always add k frames  memory used on every question, no uncertainty rule (k = 1, 2, 3)
  oracle               a question counts as correct if any traced context answers it correctly (upper bound)
  signal               the same rule driven by another uncertainty signal (1 - answer margin, 1 - top probability),
                       compared at the memory budget semantic entropy uses at each lambda

Usage: python main_experiments/tools/lambda_sweep.py --benchmark streamingbench --run reports/.../prism_voi_<job>
"""
import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from main_experiments.tools.benchmark_tables import headline, paired_test, table
from main_experiments.tools.summarize_progressive_arbitration import correctness, profile, records
from lib.shared.recent_window import extract_mcq_answer

LAMBDAS = (float('inf'), 1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.0)


def margin_uncertainty(row):
    top = sorted(row['probabilities'].values(), reverse=True) + [0.0]
    return 1.0 - (top[0] - top[1])


def top_uncertainty(row):
    return 1.0 - max(row['probabilities'].values())


SIGNALS = {'1 - answer margin': margin_uncertainty, '1 - top probability': top_uncertainty}


def depth_for(trace, lam, uncertainty=None):
    """Greedy rule on the traced chain; `uncertainty` replaces semantic entropy (gain = drop in that signal)."""
    u = uncertainty or (lambda row: row['se'])
    if u(trace[0]) <= lam:
        return 0, 1
    depth = 0
    for j, row in enumerate(trace[1:], 1):
        gain = row['gain'] if uncertainty is None else u(trace[j - 1]) - u(row)
        if gain > lam and row['depth'] == depth + 1:
            depth = row['depth']
        else:
            return depth, min(len(trace), depth + 2)
    return depth, len(trace)


def matched_threshold(items, signal, target, steps=30):
    """Threshold of `signal` whose mean memory frames is closest to `target` (frames fall as the threshold rises)."""
    cache = {}

    def frames(theta):
        if theta not in cache:
            cache[theta] = statistics.mean(depth_for(t, theta, signal)[0] for _task, _r, t, _ms in items)
        return cache[theta]
    low, high = 0.0, 1.0
    best = min((low, high), key=lambda th: abs(frames(th) - target))
    for _ in range(steps):
        mid = (low + high) / 2
        value = frames(mid)
        best = min((best, mid), key=lambda th: abs(frames(th) - target))
        low, high = (mid, high) if value > target else (low, mid)
    return best


def replay(items, benchmark, choose):
    """Score a depth rule: answers, memory frames per question, questions using memory, cost per question (ms)."""
    answers, frames, cost = {}, [], []
    for task, record, trace, final_ms in items:
        depth, evaluated = choose(trace)
        answers[len(answers)] = (task, correct_at(trace[depth], record, benchmark))
        frames.append(depth)
        cost.append(sum(r.get('ms', 0.0) for r in trace[:evaluated]) + final_ms)
    return answers, frames, sum(f > 0 for f in frames), cost


def correct_at(row, record, benchmark):
    if row.get('answer') is not None:  # Generated answer of that context, scored like the benchmark scores answers.
        if benchmark == 'ovo':
            return bool(correctness({**record, 'response': row['answer'], 'correct': None, 'error': None}, 'ovo'))
        return extract_mcq_answer(row['answer']) == str(record.get('answer_gt', '')).strip().upper()[:1]
    pred = row.get('prediction')  # Older traces: option-logit argmax only.
    if benchmark == 'ovo':
        return bool(correctness({**record, 'response': pred, 'correct': None, 'error': None}, 'ovo'))
    return pred is not None and pred == str(record.get('answer_gt', '')).strip().upper()[:1]


def load_items(run, tasks=None):
    """(task, record, trace, final generation ms) for every traced multiple-choice question of a run."""
    items = []
    for record in records(run):
        meta, timing = profile(record)[1], profile(record)[2]
        trace = meta.get('trace')
        if not trace or meta.get('format') != 'mcq':
            continue
        task = record.get('task') or record.get('task_type') or 'all'
        if tasks and task not in tasks:
            continue
        items.append((task, record, trace, timing.get('final_generation_ms') or 0.0))
    if not items:
        raise SystemExit(f'No traced multiple-choice questions in {run} (was it made with PRISM_VOI_TRACE=1?)')
    return items


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--benchmark', required=True, choices=['ovo', 'streamingbench', 'videomme'])
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--tasks', nargs='+', help='only these task types, e.g. EPM ASI HLD for OVO Backward')
    args = p.parse_args()
    items = load_items(args.run, args.tasks)
    bench = 'ovo' if args.benchmark == 'ovo' else 'streamingbench'
    # Check: at the lambda the run was made with, the replay must reproduce the run's own decisions and scores.
    depth_off = score_off = 0
    for _task, record, trace, _ms in items:
        meta = profile(record)[1]
        depth, _n = depth_for(trace, float(meta['config']['lam']))
        depth_off += depth != meta.get('selected_memory_depth')
        actual = correctness(record, 'ovo') if args.benchmark == 'ovo' else record.get('correct')
        score_off += isinstance(actual, bool) and correct_at(trace[depth], record, args.benchmark) != actual

    def row(label, answers, frames, used, cost):
        test = paired_test(recent, answers, bench)
        return [label, f"{headline(list(answers.values()), bench):.2f}", f"{test['delta']:+.2f}",
                f"[{test['low']:+.2f}, {test['high']:+.2f}]", f"{test['gained']} / {test['lost']}", f"{test['p']:.3f}",
                '—' if frames is None else f"{statistics.mean(frames):.2f}", '—' if used is None else str(used),
                '—' if cost is None else f"{statistics.mean(cost) / 1000:.3f}"]

    recent = replay(items, args.benchmark, lambda t: depth_for(t, float('inf')))[0]
    rows, budgets = [], {}
    for lam in LAMBDAS:
        answers, frames, used, cost = replay(items, args.benchmark, lambda t: depth_for(t, lam))
        budgets[lam] = statistics.mean(frames)
        rows.append(row('∞ (= Recent-6)' if lam == float('inf') else f'{lam:g}', answers, frames, used, cost))
    scope = f", tasks {' '.join(args.tasks)}" if args.tasks else ''
    print(f'## {args.benchmark}: PRISM-VoI accuracy-cost curve from one trace run ({len(items)} questions{scope})\n')
    print(table(['λ (nats/frame)', 'Accuracy', 'Δ vs Recent-6', '95% CI (paired bootstrap)', 'Fixed / broken',
                 'McNemar p', 'Memory frames (mean)', 'Questions using memory', 'Cost per question (s)'], rows))
    print(f'\nReplay check at the run lambda: {depth_off} memory-depth and {score_off} correctness mismatches '
          f'with the recorded answers (both 0 for a trace made with per-context answers).')
    print('\nCost: semantic-entropy evaluations the rule would have run, plus the final answer; decoding excluded.')

    header = ['Rule', 'Accuracy', 'Δ vs Recent-6', '95% CI (paired bootstrap)', 'Fixed / broken', 'McNemar p',
              'Memory frames (mean)', 'Questions using memory', 'Cost per question (s)']
    depth = max(len(t) for _task, _r, t, _ms in items) - 1
    rows = []
    for k in range(1, depth + 1):
        answers, frames, used, _cost = replay(items, args.benchmark, lambda t: (min(k, len(t) - 1), 1))
        rows.append(row(f'always add {k} memory frame{"s" if k > 1 else ""}', answers, frames, used, None))
    oracle = {}
    for i, (task, record, trace, _ms) in enumerate(items):
        oracle[i] = (task, any(correct_at(r, record, args.benchmark) for r in trace))
    rows.append(row('oracle: any traced context correct', oracle, None, None, None))
    print(f'\n## {args.benchmark}: memory without the uncertainty rule, and the oracle upper bound\n')
    print(table(header, rows))

    rows = []
    for lam in (0.5, 0.2, 0.1, 0.0):
        target = budgets[lam]
        answers, frames, used, cost = replay(items, args.benchmark, lambda t: depth_for(t, lam))
        rows.append(row(f'semantic entropy, λ = {lam:g}', answers, frames, used, cost))
        for name, signal in SIGNALS.items():
            # Threshold of the other signal whose mean memory frames is closest to semantic entropy's at this lambda.
            theta = matched_threshold(items, signal, target)
            answers, frames, used, cost = replay(items, args.benchmark, lambda t: depth_for(t, theta, signal))
            rows.append(row(f'  {name}, θ = {theta:.4f}', answers, frames, used, cost))
    print(f'\n## {args.benchmark}: decision signal at a matched memory budget\n')
    print(table(header, rows))
    print('\nEach other signal drives the same rule (retrieve if uncertainty > θ, keep a frame while it lowers uncertainty '
          'by more than θ), with θ chosen so it uses as many memory frames as semantic entropy at that λ.')


if __name__ == '__main__':
    main()
