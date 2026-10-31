#!/usr/bin/env python3
"""Per-benchmark comparison tables: every subtask, plus a latency breakdown that excludes video decoding.

Usage:
  python main_experiments/tools/benchmark_tables.py --benchmark ovo \
      --run "Recent-6=reports/.../recent6_X" --run "OASIS=reports/.../oasis_paper_Y" ...
  python main_experiments/tools/benchmark_tables.py --benchmark streambench \
      --run "PRISM=reports/progressive_arbitration/streambench_v03/prism_unified_X" \
      --judged "PRISM=reports/streambench_judging/prism_unified_X" ...

Accuracy counts inference errors as wrong. OASIS runs also get a lenient-parse row (answers that MiniCPM did not
wrap in OASIS's <answer> tag are re-parsed). StreamBench accuracy comes from the LLaMA-3 judge summaries.
Latency is seconds per question (mean), from question arrival to the final answer; decoding is never included.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from main_experiments.tools.summarize_progressive_arbitration import correctness, profile, records

OVO_GROUPS = (('Real-Time', ('OCR', 'ACR', 'ATR', 'STU', 'FPD', 'OJR')),
              ('Backward', ('EPM', 'ASI', 'HLD')),
              ('Forward', ('REC', 'SSR', 'CRR')))
STREAMBENCH_TASKS = ('OS', 'LM', 'SM', 'CI', 'KG', 'SF')
LATENCY_ROWS = (
    ('Question → first answer token', 'first_token'),
    ('Algorithmic E2E (question → final answer)', 'PRISM_end_to_end_latency_ms'),
    ('  retrieval / scoring', 'retrieval_total_ms'),
    ('  extra VLM passes before the answer', 'vlm_selection_total_ms'),
    ('  decision logic', 'decision_logic_total_ms'),
    ('  context assembly / preprocessing', 'context_total_ms'),
    ('  final answer generation', 'final_generation_ms'),
    ('    of which time to first token', 'final_ttft_ms'),
    ('Memory upkeep (summaries, indexing)', 'memory_maintenance_total_ms'),
    ('Algorithmic E2E + upkeep', 'with_upkeep'),
)


def pairs(values):
    out = []
    for value in values or []:
        label, _, path = value.partition('=')
        if not path:
            raise SystemExit(f'Expected Label=path, got {value!r}')
        out.append((label, Path(path)))
    return out


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def lenient_rows(rows, benchmark):
    """OASIS rows with unparsed answers re-parsed leniently (same rule as rescore_oasis_answers)."""
    from main_experiments.tools.rescore_oasis_answers import find, lenient
    out = []
    for r in rows:
        if find(r, 'answer_parse_source') == 'unparsed':
            pred = lenient(find(r, 'fine_response')) or lenient(find(r, 'coarse_response'))
            if pred:
                r = {k: v for k, v in r.items() if k != 'correct'}
                r['response'] = pred
                if benchmark != 'ovo':
                    r['correct'] = pred == str(r.get('answer_gt', '')).strip().upper()[:1]
        out.append(r)
    return out


def is_oasis(rows):
    return any(str(profile(r)[1].get('mode', '')).startswith(('oasis', 'recent6_oasis')) for r in rows[:50])


def ovo_scores(rows):
    tasks = {}
    for r in rows:
        tasks.setdefault(r.get('task'), []).append(bool(correctness(r, 'ovo')))
    scores = {task: 100 * statistics.mean(v) for task, v in tasks.items() if task}
    groups = {name: mean([scores.get(t) for t in members]) for name, members in OVO_GROUPS}
    present = [g for g in groups.values() if g is not None]
    groups['Real-Time + Backward'] = mean([groups['Real-Time'], groups['Backward']])
    groups['Total'] = statistics.mean(present) if len(present) == 3 else None
    return {**scores, **groups}


def streamingbench_scores(rows):
    tasks = {}
    for r in rows:
        tasks.setdefault(r.get('task_type') or 'unknown', []).append(bool(r.get('correct')) and not r.get('error'))
    scores = {task: 100 * statistics.mean(v) for task, v in tasks.items()}
    scores['Overall'] = 100 * statistics.mean(v for vs in tasks.values() for v in vs)
    return scores


def streambench_scores(judged_dir):
    summary = json.loads((judged_dir / 'summary.json').read_text())
    scores = {task: group['accuracy_percent'] for task, group in summary['subtasks'].items()}
    scores['Overall'] = summary['accuracy_percent']
    return scores


def latency(rows):
    timings = [profile(r)[2] for r in rows if profile(r)[2]]
    def value(t, key):
        if key == 'first_token':
            if t.get('request_to_answer_first_token_ms') is not None:
                return t['request_to_answer_first_token_ms']
            if t.get('PRISM_selection_latency_ms') is not None and t.get('final_ttft_ms') is not None:
                return t['PRISM_selection_latency_ms'] + t['final_ttft_ms']
            return None
        if key == 'with_upkeep':
            e2e = t.get('PRISM_end_to_end_latency_ms')
            return None if e2e is None else e2e + (t.get('memory_maintenance_total_ms') or 0.0)
        if key == 'memory_maintenance_total_ms':
            return t.get(key) or 0.0
        return t.get(key)
    out = {key: mean([value(t, key) for t in timings]) for _label, key in LATENCY_ROWS}
    e2e = sorted(v for v in (t.get('PRISM_end_to_end_latency_ms') for t in timings) if v is not None)
    out['p50'] = e2e[len(e2e) // 2] if e2e else None
    out['p90'] = e2e[int(0.9 * (len(e2e) - 1))] if e2e else None
    frames = []
    for r in rows:
        t = profile(r)[2]
        if t:
            frames.append(t.get('final_frame_count') if t.get('final_frame_count') is not None
                          else r.get('num_frames', profile(r)[0].get('num_frames')))
    out['frames'] = mean(frames)
    out['peak_mb'] = mean([t.get('gpu_peak_allocated_mb') for t in timings])
    out['n'] = len(timings)
    return out


def per_question(rows, benchmark):
    """{question key: (task, correct)} for paired tests; inference errors count as wrong."""
    out = {}
    for r in rows:
        if benchmark == 'ovo':
            out[str(r['pair_key'])] = (r.get('task'), bool(correctness(r, 'ovo')))
        else:
            key = str(r['_index']) if isinstance(r.get('_index'), int) else str(r['pair_key'])
            out[key] = (r.get('task_type'), bool(r.get('correct')) and not r.get('error'))
    return out


def per_question_judged(judged_dir):
    out = {}
    for row in json.loads((judged_dir / 'judge' / 'judged.json').read_text()):
        out[str(row['id'])] = (row['class'], row.get('llama_pred') == 'yes')
    for failure in json.loads((judged_dir / 'coverage.json').read_text()).get('failures', []):
        out[str(failure['id'])] = (failure['subtask'], False)
    return out


def headline(items, benchmark):
    """Benchmark metric on a list of (task, correct): OVO uses the mean of category averages."""
    if benchmark != 'ovo':
        return 100 * statistics.mean(c for _t, c in items)
    tasks = {}
    for task, c in items:
        tasks.setdefault(task, []).append(c)
    groups = [statistics.mean(statistics.mean(tasks[t]) for t in members if t in tasks)
              for _name, members in OVO_GROUPS if any(t in tasks for t in members)]
    return 100 * statistics.mean(groups)


def paired_test(base, other, benchmark, resamples=2000, seed=0):
    """Accuracy difference (other - base) on shared questions: paired bootstrap 95% CI and exact McNemar p."""
    import math
    import random
    keys = sorted(set(base) & set(other))
    if not keys:
        return None
    a = [base[k] for k in keys]
    b = [other[k] for k in keys]
    delta = headline(b, benchmark) - headline(a, benchmark)
    rng = random.Random(seed)
    diffs = []
    for _ in range(resamples):
        idx = [rng.randrange(len(keys)) for _ in keys]
        diffs.append(headline([b[i] for i in idx], benchmark) - headline([a[i] for i in idx], benchmark))
    diffs.sort()
    gained = sum(1 for x, y in zip(a, b) if not x[1] and y[1])
    lost = sum(1 for x, y in zip(a, b) if x[1] and not y[1])
    n = gained + lost
    tail = sum(math.comb(n, i) for i in range(0, min(gained, lost) + 1)) / 2 ** n if n else 1.0
    return dict(n=len(keys), delta=delta, low=diffs[int(0.025 * resamples)], high=diffs[int(0.975 * resamples) - 1],
                gained=gained, lost=lost, p=min(1.0, 2 * tail))


def fmt(value, digits=2, scale=1.0):
    return '—' if value is None else f'{value / scale:.{digits}f}'


def table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|---|' + '---:|' * (len(header) - 1)]
    lines += ['| ' + ' | '.join(row) + ' |' for row in rows]
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--benchmark', required=True, choices=['ovo', 'streamingbench', 'streambench'])
    p.add_argument('--run', action='append', default=[], help='Label=inference result dir (accuracy and latency)')
    p.add_argument('--judged', action='append', default=[], help='Label=StreamBench judging dir (accuracy)')
    p.add_argument('--output', help='Also write markdown and JSON here (directory)')
    p.add_argument('--baseline', default='Recent-6', help='Label every other method is paired against')
    args = p.parse_args()
    runs, judged = pairs(args.run), dict(pairs(args.judged))

    accuracy, timing, answers = {}, {}, {}
    for label, directory in runs:
        rows = records(directory)
        timing[label] = latency(rows)
        if args.benchmark == 'streambench':
            continue
        score = ovo_scores if args.benchmark == 'ovo' else streamingbench_scores
        accuracy[label] = score(rows)
        answers[label] = per_question(rows, args.benchmark)
        if is_oasis(rows):
            accuracy[label + ' (lenient parse)'] = score(lenient_rows(rows, args.benchmark))
            answers[label + ' (lenient parse)'] = per_question(lenient_rows(rows, args.benchmark), args.benchmark)
    for label, directory in judged.items():
        accuracy[label] = streambench_scores(directory)
        answers[label] = per_question_judged(directory)

    if args.benchmark == 'ovo':
        columns = [t for _g, members in OVO_GROUPS for t in members] + [g for g, _m in OVO_GROUPS] + ['Real-Time + Backward', 'Total']
    elif args.benchmark == 'streamingbench':
        columns = sorted({k for s in accuracy.values() for k in s if k != 'Overall'}) + ['Overall']
    else:
        columns = list(STREAMBENCH_TASKS) + ['Overall']
    out = [f'## {args.benchmark}: accuracy (%)', '',
           table(['Method'] + columns, [[label] + [fmt(s.get(c)) for c in columns] for label, s in accuracy.items()])]
    tests = {}
    if args.baseline in answers:
        for label, values in answers.items():
            if label != args.baseline:
                tests[label] = paired_test(answers[args.baseline], values, args.benchmark)
        metric = 'mean of category averages' if args.benchmark == 'ovo' else 'overall accuracy'
        sig_rows = [[label, str(t['n']), f"{t['delta']:+.2f}", f"[{t['low']:+.2f}, {t['high']:+.2f}]",
                     f"{t['gained']} / {t['lost']}", f"{t['p']:.3f}"] for label, t in tests.items() if t]
        out += ['', f'## {args.benchmark}: paired comparison with {args.baseline} ({metric}, shared questions)', '',
                table(['Method', 'Questions', 'Δ (pts)', '95% CI (paired bootstrap)', 'Fixed / broken', 'McNemar p'], sig_rows),
                '', 'A difference is statistically supported when the CI excludes 0 (and p < 0.05). '
                '“Fixed / broken” counts questions the method gets right where the baseline is wrong, and vice versa.']
    labels = list(timing)
    lat_rows = [[name] + [fmt(timing[l].get(key), 3, 1000) for l in labels] for name, key in LATENCY_ROWS]
    lat_rows += [['Algorithmic E2E p50 / p90'] + [f"{fmt(timing[l]['p50'], 2, 1000)} / {fmt(timing[l]['p90'], 2, 1000)}" for l in labels],
                 ['Frames in the answer context (mean)'] + [fmt(timing[l]['frames'], 1) for l in labels],
                 ['Peak GPU allocated (GB)'] + [fmt(timing[l]['peak_mb'], 2, 1024) for l in labels],
                 ['Questions timed'] + [str(timing[l]['n']) for l in labels]]
    out += ['', f'## {args.benchmark}: latency, seconds per question (decoding excluded)', '',
            table(['Component'] + labels, lat_rows)]
    text = '\n'.join(out) + '\n'
    print(text)
    if args.output:
        target = Path(args.output)
        target.mkdir(parents=True, exist_ok=True)
        (target / f'{args.benchmark}_tables.md').write_text(text)
        (target / f'{args.benchmark}_tables.json').write_text(json.dumps(dict(accuracy=accuracy, latency=timing, paired_tests=tests), indent=2))


if __name__ == '__main__':
    main()
