"""Prepare saved StreamBench answers, run the author judge, and report coverage.

No videos or MiniCPM generation are rerun. Inference failures are never silently
excluded; reporting them as wrong requires --allow-inference-errors.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .run_author_judge import SUBTASKS, audit


def prepare(payload, method, expected=1838):
    rows = payload['results']
    if len(rows) != expected:
        raise ValueError(f'Expected {expected} recorded questions, found {len(rows)}')
    seen, inputs, failures, modes = set(), [], [], set()
    totals = Counter()
    for row in rows:
        indices = (row.get('video_index'), row.get('breakpoint_index'))
        if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in indices):
            raise ValueError('Missing/invalid video_index or breakpoint_index')
        key = ':'.join(map(str, indices))
        if key in seen:
            raise ValueError(f'Duplicate question: {key}')
        seen.add(key)
        task = row.get('subtask')
        if task not in SUBTASKS:
            raise ValueError(f'Invalid subtask for {key}: {task!r}')
        totals[task] += 1
        matches = [r for r in row.get('methods', []) if r.get('method') == method]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one {method} result for {key}')
        result = matches[0]
        adaptive = result.get('adaptive') or (result.get('profile') or {}).get('adaptive') or {}
        mode = adaptive.get('mode')
        if mode:
            modes.add(mode)
        for field in ('question', 'answer_gt'):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f'Missing {field} for {key}')
        prediction = result.get('prediction')
        error = result.get('error')
        if error or not isinstance(prediction, str) or not prediction.strip():
            failures.append(dict(id=key, subtask=task, error=error or 'empty_prediction'))
            continue
        inputs.append(dict(id=key, question=row['question'], label=row['answer_gt'],
                           predict=prediction, **{'class': task}))
    if len(modes) > 1:
        raise ValueError(f'Mixed inference modes: {sorted(modes)}')
    mode = next(iter(modes), None)
    label = 'Recent-6' if mode == 'progressive_arbitration_exact_recent6_control' else mode or method
    coverage = dict(total_questions=expected, valid_predictions=len(inputs),
                    inference_errors=len(failures), failures=failures,
                    totals_by_subtask=dict(totals), stored_method=method,
                    inference_mode=mode, display_method=label)
    return inputs, coverage


def report(inputs, judged, coverage):
    audited = audit(inputs, judged, len(inputs))
    correct = audited['correct']
    total = coverage['total_questions']
    subtasks = {}
    for task, n in sorted(coverage['totals_by_subtask'].items()):
        group = audited['subtasks'].get(task, {'correct': 0, 'n': 0})
        subtasks[task] = dict(total=n, judged=group['n'], correct=group['correct'],
                              inference_errors=n-group['n'],
                              accuracy_percent=100*group['correct']/n)
    return dict(method=coverage['display_method'], total_questions=total,
                judged_questions=len(inputs), inference_errors=coverage['inference_errors'],
                correct=correct, accuracy_percent=100*correct/total,
                valid_predictions_accuracy_percent=audited['accuracy_percent'],
                error_free_evaluation=coverage['inference_errors'] == 0,
                denominator_policy='all_recorded_questions_inference_failures_count_as_wrong',
                semantic_metric='author_llama3_yes_fraction', subtasks=subtasks,
                judge_time_in_inference_latency=False)


def markdown(summary):
    lines = [f"# StreamBench: {summary['method']}", '',
             f"Overall semantic accuracy: **{summary['accuracy_percent']:.2f}%** "
             f"({summary['correct']}/{summary['total_questions']}).",
             f"Judged: {summary['judged_questions']}; inference errors: {summary['inference_errors']}.",
             'Inference failures count as wrong; judge time is excluded from inference latency.', '',
             '| Subtask | Correct / Total | Accuracy | Inference Errors |',
             '|---|---:|---:|---:|']
    for task, group in summary['subtasks'].items():
        lines.append(f"| {task} | {group['correct']}/{group['total']} | "
                     f"{group['accuracy_percent']:.2f}% | {group['inference_errors']} |")
    if summary['inference_errors']:
        lines += ['', '**This is not an error-free benchmark result.**',
                  f"Diagnostic accuracy on valid predictions only: "
                  f"{summary['valid_predictions_accuracy_percent']:.2f}%."]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--method', default='prism', help='Stored methods-list key, not the display label')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source', type=Path, help='Pinned StreamChat author-code checkout')
    parser.add_argument('--judge-model', help='Fixed local Meta-Llama-3-8B-Instruct snapshot')
    parser.add_argument('--expected', type=int, default=1838)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--allow-inference-errors', action='store_true')
    args = parser.parse_args()
    if not args.prepare_only and (not args.source or not args.judge_model):
        parser.error('--source and --judge-model are required to run judging')
    raw = args.results.read_bytes()
    inputs, coverage = prepare(json.loads(raw), args.method, args.expected)
    coverage['results_sha256'] = hashlib.sha256(raw).hexdigest()
    if not args.prepare_only and coverage['inference_errors'] and not args.allow_inference_errors:
        parser.error(f"{coverage['inference_errors']} inference failures. Repair them or explicitly "
                     'use --allow-inference-errors to count them as wrong.')
    if not inputs:
        parser.error('No valid predictions to judge')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    input_file = args.output_dir / 'judge_input.json'
    input_file.write_text(json.dumps(inputs, indent=2, ensure_ascii=False))
    (args.output_dir / 'coverage.json').write_text(json.dumps(coverage, indent=2))
    print(f"{coverage['display_method']}: {len(inputs)}/{args.expected} predictions; "
          f"{coverage['inference_errors']} inference errors", flush=True)
    if args.prepare_only:
        print('Prepared only; no semantic accuracy computed.', flush=True)
        return
    subprocess.run([sys.executable, '-m',
                    'main_experiments.minicpm_v46.streambench_v03.run_author_judge',
                    '--source', str(args.source.resolve()), '--input', str(input_file.resolve()),
                    '--output-dir', str((args.output_dir / 'judge').resolve()),
                    '--judge-model', args.judge_model, '--expected', str(len(inputs))], check=True)
    judged = [json.loads(line) for line in (args.output_dir/'judge/judged.json').read_text().splitlines() if line.strip()]
    summary = report(inputs, judged, coverage)
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2))
    (args.output_dir/'summary.md').write_text(markdown(summary))
    print(markdown(summary), flush=True)


if __name__ == '__main__':
    main()
