#!/usr/bin/env python3
"""Decision-signal swap on PRISM-VoI trace runs made with PRISM_VOI_TRACE=1 PRISM_VOI_SIGNALS=1 (no GPU).

Everything but the decision signal stays fixed: the same backbone, recent window, ranked memory frames and traced
answers. Each signal is compared at the memory budget semantic entropy uses at each lambda.

  full swap   the signal drives both decisions: retrieve if u(R) > θ, keep a frame while it lowers u by more than θ
              u = 1 - top option probability, 1 - answer margin (confidence baselines),
                  P(evidence insufficient) from asking the VLM (StreamScout-style self-assessment)
  gate swap   the signal only decides whether to retrieve (at semantic entropy's retrieval rate); the frames are then
              either all added (+K, as a gate-only method would) or kept by semantic entropy (+SE)
              gates: WRWS retrieval entropy (retrieve when relevance is peaked), self-assessment, confidence, random

Usage: python main_experiments/tools/signal_swap.py --benchmark ovo --run reports/.../prism_voi_<job>
"""
import argparse
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from main_experiments.tools.benchmark_tables import headline, paired_test, table
from main_experiments.tools.lambda_sweep import (SIGNALS, correct_at, depth_for, matched_threshold, top_uncertainty)
from main_experiments.tools.summarize_progressive_arbitration import profile, records

LAMS = (0.2, 0.1, 0.0)
RANDOM_SEEDS = 200


def insufficient(row):
    return row['p_insufficient']


FULL = {**SIGNALS, 'self-assessment P(insufficient)': insufficient}


def load(run):
    """(task, record, trace, retrieval entropy) per traced multiple-choice question that has the extra signals."""
    items = []
    for record in records(run):
        meta = profile(record)[1]
        trace = meta.get('trace')
        if not trace or meta.get('format') != 'mcq':
            continue
        if 'p_insufficient' not in trace[0] or meta.get('retrieval_entropy') is None:
            continue
        items.append((record.get('task') or record.get('task_type') or 'all', record, trace, meta['retrieval_entropy']))
    if not items:
        raise SystemExit(f'No traced questions with signals in {run} (made with PRISM_VOI_TRACE=1 PRISM_VOI_SIGNALS=1?)')
    return items


def keep_by_se(trace, lam):
    """Depth reached by semantic entropy's per-frame rule once retrieval has been decided."""
    depth = 0
    for j, row in enumerate(trace[1:], 1):
        if row['gain'] > lam:
            depth = j
        else:
            break
    return depth


def score(items, benchmark, depths):
    answers = {i: (task, correct_at(trace[d], record, benchmark))
               for i, ((task, record, trace, _a), d) in enumerate(zip(items, depths))}
    return answers, statistics.mean(depths), sum(d > 0 for d in depths) / len(depths)


def gate_depths(items, lam, retrieve, add):
    """Depth per question when `retrieve[i]` decides retrieval and `add` ('all' or 'se') decides the frames."""
    out = []
    for (_t, _r, trace, _a), go in zip(items, retrieve):
        out.append(0 if not go else len(trace) - 1 if add == 'all' else keep_by_se(trace, lam))
    return out


def top_fraction(scores, rate):
    """Retrieve for the `rate` fraction of questions with the highest score."""
    n = round(rate * len(scores))
    chosen = set(sorted(range(len(scores)), key=lambda i: -scores[i])[:n])
    return [i in chosen for i in range(len(scores))]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--benchmark', required=True, choices=['ovo', 'streamingbench', 'videomme'])
    p.add_argument('--run', required=True, type=Path)
    args = p.parse_args()
    items = load(args.run)
    bench = 'ovo' if args.benchmark == 'ovo' else 'streamingbench'
    traces = [(t, r, tr, 0.0) for t, r, tr, _a in items]  # lambda_sweep's item shape
    recent = score(items, args.benchmark, [0] * len(items))[0]
    header = ['Rule', 'Accuracy', 'Δ vs Recent-6', 'Δ vs semantic entropy', 'Fixed / broken vs ours', 'McNemar p (vs ours)',
              'Memory frames', 'Questions retrieving']

    def row(label, answers, frames, rate, ours):
        vs_r, vs_o = paired_test(recent, answers, bench), paired_test(ours, answers, bench)
        return [label, f"{headline(list(answers.values()), bench):.2f}", f"{vs_r['delta']:+.2f}",
                '—' if answers is ours else f"{vs_o['delta']:+.2f} [{vs_o['low']:+.2f}, {vs_o['high']:+.2f}]",
                '—' if answers is ours else f"{vs_o['gained']} / {vs_o['lost']}",
                '—' if answers is ours else f"{vs_o['p']:.3f}", f'{frames:.2f}', f'{100 * rate:.1f}%']

    print(f'# {args.benchmark}: decision-signal swap ({len(items)} traced questions, same frames and answers)\n')
    for lam in LAMS:
        se_depths = [depth_for(tr, lam)[0] for _t, _r, tr, _a in items]
        ours, frames, rate = score(items, args.benchmark, se_depths)
        rows = [row(f'semantic entropy (ours), λ = {lam:g}', ours, frames, rate, ours)]
        for name, signal in FULL.items():
            theta = matched_threshold(traces, signal, frames)
            depths = [depth_for(tr, theta, signal)[0] for _t, _r, tr, _a in items]
            rows.append(row(f'full swap: {name}, θ = {theta:.4f}', *score(items, args.benchmark, depths), ours))
        gates = {'WRWS retrieval entropy': [-a for _t, _r, _tr, a in items],
                 'self-assessment': [insufficient(tr[0]) for _t, _r, tr, _a in items],
                 '1 - top probability': [top_uncertainty(tr[0]) for _t, _r, tr, _a in items]}
        for name, scores in gates.items():
            retrieve = top_fraction(scores, rate)
            for add in ('all', 'se'):
                label = f"gate swap: {name} {'+K' if add == 'all' else '+SE'}"
                rows.append(row(label, *score(items, args.benchmark, gate_depths(items, lam, retrieve, add)), ours))
        for add in ('all', 'se'):  # Random retrieval at the same rate, averaged over seeds.
            accs, fr = [], []
            for seed in range(RANDOM_SEEDS):
                rng = random.Random(seed)
                retrieve = top_fraction([rng.random() for _ in items], rate)
                answers, f, _rate = score(items, args.benchmark, gate_depths(items, lam, retrieve, add))
                accs.append(headline(list(answers.values()), bench))
                fr.append(f)
            rows.append([f"gate swap: random {'+K' if add == 'all' else '+SE'} (mean ± sd, {RANDOM_SEEDS} seeds)",
                         f'{statistics.mean(accs):.2f} ± {statistics.stdev(accs):.2f}',
                         f"{statistics.mean(accs) - headline(list(recent.values()), bench):+.2f}",
                         f"{statistics.mean(accs) - headline(list(ours.values()), bench):+.2f}", '—', '—',
                         f'{statistics.mean(fr):.2f}', f'{100 * rate:.1f}%'])
        print(f'## λ = {lam:g}: every signal at semantic entropy\'s budget\n')
        print(table(header, rows))
        print()
    print('Full swap: θ is set so the signal uses as many memory frames as semantic entropy. Gate swap: the signal '
          'retrieves for the same fraction of questions; +K then adds every ranked frame, +SE keeps frames by '
          'semantic entropy. WRWS retrieval entropy is computed over the same memory frames and R with our encoder.')


if __name__ == '__main__':
    main()
