#!/usr/bin/env python3
"""Offline comparison of memory-selection rules on PRISM-VoI trace runs (no GPU): choose on a development
benchmark, then check the chosen rule on the test benchmarks.

Rules (R = recent frames; R+k = R plus the top-k ranked memory frames, the traced contexts):
  greedy λ           PRISM-VoI: retrieve if SE(R) > λ, keep adding while each frame removes > λ nats
  gate λ + k         retrieve if SE(R) > λ, then always answer from R+k
  min-SE λ           retrieve if SE(R) > λ, then answer from the traced context with the lowest SE
  always k           answer from R+k on every question (no uncertainty rule)
Cost (≈) = semantic-entropy evaluations the rule needs + the final answer; ranking and decoding excluded.

Usage: python main_experiments/tools/voi_variants.py --dev videomme:<trace run> \\
           --test streamingbench:<trace run> --test ovo:<trace run>
"""
import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from main_experiments.tools.benchmark_tables import headline, paired_test, table
from main_experiments.tools.lambda_sweep import correct_at, depth_for, load_items

LAMS = (0.5, 0.2, 0.1, 0.05, 0.0)


def rules(max_depth):
    out = {}
    for lam in LAMS:
        out[f'greedy λ={lam:g}'] = lambda t, lam=lam: depth_for(t, lam)
        for k in range(1, max_depth + 1):
            out[f'gate λ={lam:g} + {k}'] = (lambda t, lam=lam, k=k:
                                            (0, 1) if t[0]['se'] <= lam else (min(k, len(t) - 1), 1))
        out[f'min-SE λ={lam:g}'] = (lambda t, lam=lam:
                                    (0, 1) if t[0]['se'] <= lam else (min(range(len(t)), key=lambda d: t[d]['se']), len(t)))
    for k in range(1, max_depth + 1):
        out[f'always {k}'] = lambda t, k=k: (min(k, len(t) - 1), 0)
    return out


def score(items, benchmark, rule):
    answers, frames, cost = {}, [], []
    for task, record, trace, final_ms in items:
        depth, evaluated = rule(trace)
        answers[len(answers)] = (task, correct_at(trace[depth], record, benchmark))
        frames.append(depth)
        # SE evaluations are traced per context: a gate needs SE(R) only, an 'always' rule none.
        cost.append(sum(r.get('ms', 0.0) for r in trace[:evaluated]) + final_ms)
    return answers, statistics.mean(frames), statistics.mean(cost) / 1000


def evaluate(spec):
    benchmark, run = spec.split(':', 1)
    items = load_items(Path(run))
    bench = 'ovo' if benchmark == 'ovo' else 'streamingbench'
    recent = score(items, benchmark, lambda t: (0, 1))[0]
    depth = max(len(t) for _task, _r, t, _ms in items) - 1
    results = {}
    for name, rule in rules(depth).items():
        answers, frames, cost = score(items, benchmark, rule)
        results[name] = dict(acc=headline(list(answers.values()), bench), test=paired_test(recent, answers, bench),
                             frames=frames, cost=cost)
    return benchmark, results


def show(benchmark, results, names=None):
    rows = []
    for name, r in results.items():
        if names and name not in names:
            continue
        t = r['test']
        rows.append([name, f"{r['acc']:.2f}", f"{t['delta']:+.2f}", f"[{t['low']:+.2f}, {t['high']:+.2f}]",
                     f"{t['gained']} / {t['lost']}", f"{t['p']:.3f}", f"{r['frames']:.2f}", f"{r['cost']:.3f}"])
    print(table(['Rule', 'Accuracy', 'Δ vs Recent-6', '95% CI', 'Fixed / broken', 'McNemar p', 'Memory frames',
                 'Cost ≈ (s)'], rows))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dev', required=True, help='benchmark:trace run used to choose the rule')
    p.add_argument('--test', action='append', default=[], help='benchmark:trace run to check it on')
    p.add_argument('--top', type=int, default=8, help='rules carried from dev to test, best dev accuracy first')
    args = p.parse_args()
    benchmark, dev = evaluate(args.dev)
    print(f'## Development ({benchmark}): every rule\n')
    show(benchmark, dev)
    best = sorted(dev, key=lambda n: -dev[n]['acc'])[:args.top]
    keep = list(dict.fromkeys(best + ['greedy λ=0', 'always 3']))
    for spec in args.test:
        benchmark, res = evaluate(spec)
        print(f'\n## Test ({benchmark}): the {args.top} best rules on development, plus greedy λ=0 and always 3\n')
        show(benchmark, res, keep)


if __name__ == '__main__':
    main()
