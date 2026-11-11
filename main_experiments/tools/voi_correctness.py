#!/usr/bin/env python3
"""Does lower semantic entropy mean a correct answer? Offline on PRISM-VoI trace runs (PRISM_VOI_TRACE=1, no GPU).

  1. Every traced step (one more ranked memory frame) by how much semantic entropy it removed: how often the answer
     goes wrong -> right, right -> wrong, or keeps its correctness.
  2. The steps PRISM-VoI accepts at each lambda: the same counts, i.e. how often the frames it keeps help or hurt.
  3. The final answer at each lambda by its semantic entropy: accuracy per bin and the confident-but-wrong rate,
     next to Recent-6's answer under the same bins.

Usage: python main_experiments/tools/voi_correctness.py --benchmark ovo --run reports/.../prism_voi_<job>
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from main_experiments.tools.benchmark_tables import table
from main_experiments.tools.lambda_sweep import correct_at, depth_for, load_items

GAIN_BINS = ((float('-inf'), 0.0, 'raises SE (< 0)'), (0.0, 0.05, '0 – 0.05'), (0.05, 0.1, '0.05 – 0.1'),
             (0.1, 0.3, '0.1 – 0.3'), (0.3, float('inf'), '> 0.3'))
SE_BINS = ((0.0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, float('inf')))
CONFIDENT = 0.05  # nats: "confident" final answer


def transitions(steps):
    n = len(steps)
    fixed = sum(not a and b for a, b in steps)
    broken = sum(a and not b for a, b in steps)
    pct = lambda x: f'{100 * x / n:.1f}%' if n else '—'
    return [str(n), f'{fixed} ({pct(fixed)})', f'{broken} ({pct(broken)})',
            f'{n - fixed - broken} ({pct(n - fixed - broken)})', f'{fixed - broken:+d}',
            f'{fixed / (fixed + broken):.2f}' if fixed + broken else '—']


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--benchmark', required=True, choices=['ovo', 'streamingbench', 'videomme'])
    p.add_argument('--run', required=True, type=Path)
    args = p.parse_args()
    items = load_items(args.run)
    ok = [[correct_at(r, record, args.benchmark) for r in trace] for _t, record, trace, _ms in items]
    head = ['', 'Steps', 'Wrong → right', 'Right → wrong', 'Unchanged', 'Net', 'Share of changes that fix']

    print(f'# {args.benchmark}: semantic-entropy reduction vs correctness ({len(items)} traced questions)\n')
    rows = []
    for low, high, label in GAIN_BINS:
        steps = [(c[j - 1], c[j]) for (_t, _r, trace, _ms), c in zip(items, ok)
                 for j in range(1, len(trace)) if low <= trace[j]['gain'] < high]
        rows.append([f'SE removed {label}'] + transitions(steps))
    print('## 1. Every traced step, by the semantic entropy the added frame removed (nats)\n')
    print(table(head, rows))

    rows = []
    for lam in (0.2, 0.1, 0.0):
        steps = []
        for (_t, _r, trace, _ms), c in zip(items, ok):
            depth = depth_for(trace, lam)[0]
            steps += [(c[j - 1], c[j]) for j in range(1, depth + 1)]
        rows.append([f'accepted at λ = {lam:g}'] + transitions(steps))
    print('\n## 2. Frames PRISM-VoI keeps: does each accepted frame help or hurt?\n')
    print(table(head, rows))

    for lam in (0.1, 0.0):
        rows = []
        finals = []
        for (_t, _r, trace, _ms), c in zip(items, ok):
            depth = depth_for(trace, lam)[0]
            finals.append((trace[depth]['se'], c[depth], trace[0]['se'], c[0]))
        for low, high in SE_BINS:
            mine = [x for se, x, _s0, _c0 in finals if low <= se < high]
            base = [x for _se, _x, s0, x in finals if low <= s0 < high]
            rows.append([f'{low:g} – {high:g}' if high != float('inf') else f'> {low:g}', str(len(mine)),
                         f'{100 * sum(mine) / len(mine):.1f}' if mine else '—', str(len(base)),
                         f'{100 * sum(base) / len(base):.1f}' if base else '—'])
        cw = sum(se < CONFIDENT and not x for se, x, _s, _c in finals) / len(finals)
        cw0 = sum(s0 < CONFIDENT and not c0 for _se, _x, s0, c0 in finals) / len(finals)
        print(f'\n## 3. Final answer by its semantic entropy, λ = {lam:g} (vs Recent-6 by SE(R))\n')
        print(table(['Final SE (nats)', 'PRISM-VoI questions', 'PRISM-VoI accuracy (%)', 'Recent-6 questions',
                     'Recent-6 accuracy (%)'], rows))
        print(f'\nConfident (SE < {CONFIDENT} nats) but wrong: PRISM-VoI {100 * cw:.1f}% of questions, '
              f'Recent-6 {100 * cw0:.1f}%.')


if __name__ == '__main__':
    main()
