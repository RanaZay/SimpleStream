#!/usr/bin/env python3
"""Matched accuracy and latency table: PRISM vs Recent-6 + OASIS (optionally the Recent-6 control).

Methods are compared on the intersection of question keys (OASIS may run a subset of OVO splits),
and the exact Recent-6 frame hashes are verified per question so only the memory differs.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from main_experiments.tools.summarize_progressive_arbitration import accuracy, distribution, profile, records

# (label, function of the per-question timing dict -> ms or None)
ROWS = [
    ('Final-answer generation', lambda t: t.get('final_generation_ms')),
    ('Final-answer TTFT (inside final generation)', lambda t: t.get('final_ttft_ms')),
    ('Question -> first answer token', lambda t: _sum(t, 'PRISM_selection_latency_ms', 'final_ttft_ms')),
    ('Algorithmic E2E (excl. decoding and memory upkeep)', lambda t: t.get('PRISM_end_to_end_latency_ms')),
    ('Memory upkeep (OASIS summaries, merges, QA summary)', lambda t: t.get('memory_maintenance_total_ms') or 0.0),
    ('Algorithmic E2E + memory upkeep', lambda t: _sum(t, 'PRISM_end_to_end_latency_ms', 'memory_maintenance_total_ms')),
    ('Video preparation / decoding', lambda t: t.get('total_visual_preparation_ms')),
    ('Full system (everything)', lambda t: t.get('full_system_latency_ms')),
]
OASIS_ROWS = ['oasis_coarse_generation_ms', 'retrieval_total_ms', 'oasis_fine_generation_ms', 'context_total_ms',
              'memory_update_before_query_ms', 'memory_event_summary_ms', 'memory_merge_ms', 'memory_embedding_ms',
              'memory_qa_update_ms', 'broad_history_decode_ms', 'recent6_decode_ms']


def _sum(t, *keys):
    values = [t.get(k) for k in keys]
    return None if values[0] is None else sum(v or 0.0 for v in values)


def load(directory):
    rows = records(Path(directory))
    return {r['pair_key']: r for r in rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', required=True, choices=['ovo', 'streamingbench'])
    p.add_argument('--prism', required=True)
    p.add_argument('--oasis', required=True)
    p.add_argument('--recent6')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    methods = {'PRISM (ours)': load(args.prism), 'Recent-6 + OASIS': load(args.oasis)}
    if args.recent6:
        methods = {'Recent-6': load(args.recent6), **methods}
    keys = set.intersection(*(set(m) for m in methods.values()))
    if not keys:
        raise SystemExit('No common question keys; check that the runs use the same annotations')
    coverage = {name: dict(total=len(m), used=len(keys)) for name, m in methods.items()}

    prism, oasis = methods['PRISM (ours)'], methods['Recent-6 + OASIS']
    checked = mismatched = 0
    for k in keys:
        a, b = profile(prism[k])[1].get('recent_frame_hashes'), profile(oasis[k])[1].get('recent_frame_hashes')
        if a and b:
            checked += 1
            mismatched += a != b
    errors = {name: sum(bool(m[k].get('error')) for k in keys) for name, m in methods.items()}

    result = dict(benchmark=args.benchmark, questions=len(keys), coverage=coverage, errors=errors,
                  recent6_hash_pairs_checked=checked, recent6_hash_mismatches=mismatched,
                  accuracy={name: accuracy([m[k] for k in sorted(keys)], args.benchmark) for name, m in methods.items()},
                  latency={}, oasis_components={})
    for label, fn in ROWS:
        result['latency'][label] = {name: distribution([fn(profile(m[k])[2]) for k in keys if profile(m[k])[2]])
                                    for name, m in methods.items()}
    timings = [profile(oasis[k])[2] for k in keys]
    for key in OASIS_ROWS:
        values = [t.get(key) for t in timings]
        result['oasis_components'][key] = dict(conditional=distribution(values),
                                               amortized_mean=sum(v or 0 for v in values) / max(len(values), 1))
    adaptive = [profile(oasis[k])[1] for k in keys]
    result['oasis_fine_trigger_rate'] = sum(bool(t.get('oasis_fine_triggered')) for t in timings) / len(timings)
    result['oasis_answer_parse'] = {s: sum(a.get('answer_parse_source') == s for a in adaptive)
                                    for s in sorted({a.get('answer_parse_source') for a in adaptive}, key=str)}
    result['oasis_mean_generated_tokens'] = dict(
        coarse=distribution([t.get('oasis_coarse_generated_tokens') for t in timings])['mean'],
        fine=distribution([t.get('oasis_fine_generated_tokens') for t in timings])['mean'],
        memory=distribution([t.get('memory_generated_tokens') for t in timings])['mean'])

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / f'{args.benchmark}_oasis_vs_prism.json').write_text(json.dumps(result, indent=2))
    names = list(methods)
    sec = lambda d: '-' if d['mean'] is None else f"{d['mean'] / 1000:.3f}"
    acc = lambda v: '-' if v is None else f'{v:.2f}%'
    lines = [f'# {args.benchmark}: PRISM vs Recent-6 + OASIS (MiniCPM-V-4.6)', '',
             f'Matched questions: {len(keys)}. Errors (counted wrong): '
             + ', '.join(f'{n} {e}' for n, e in errors.items()) + '.',
             f'Recent-6 frame hashes PRISM vs OASIS: {checked - mismatched}/{checked} identical.', '',
             '| Metric | ' + ' | '.join(names) + ' |', '|---|' + '---:|' * len(names),
             '| Accuracy | ' + ' | '.join(acc(result['accuracy'][n]) for n in names) + ' |']
    lines += [f'| {label} (s/question, mean) | ' + ' | '.join(sec(result['latency'][label][n]) for n in names) + ' |'
              for label, _ in ROWS]
    lines += ['', f"OASIS fine stage triggered on {100 * result['oasis_fine_trigger_rate']:.1f}% of questions; "
              f"answer parsing: {result['oasis_answer_parse']}; mean generated tokens: {result['oasis_mean_generated_tokens']}.",
              '', '| OASIS component | Conditional mean s | Amortized mean s |', '|---|---:|---:|']
    lines += [f"| {k} | {sec(v['conditional'])} | {v['amortized_mean'] / 1000:.3f} |" for k, v in result['oasis_components'].items()]
    lines += ['', 'Algorithmic E2E starts once the question arrives and memory is current; it excludes video decoding.',
              'OASIS memory upkeep is asynchronous in the paper; here it runs inline and is reported as its own row.',
              'PRISM has no upkeep stage: its CLIP retrieval and scoring are inside its algorithmic E2E.']
    (output / f'{args.benchmark}_oasis_vs_prism.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
