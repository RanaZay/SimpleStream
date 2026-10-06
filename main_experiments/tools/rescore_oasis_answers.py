#!/usr/bin/env python3
"""Lenient re-score of Recent-6 + OASIS multiple-choice answers that the strict parser left unparsed.

Only questions with answer_parse_source == 'unparsed' are touched. The saved fine response (the paper's
final answer) is tried first, then the coarse response. Reports strict and re-scored accuracy side by side;
the result files are not modified.
"""
import argparse
import collections
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from main_experiments.tools.summarize_progressive_arbitration import records

PATTERNS = [
    re.compile(r'"answer"\s*:\s*"\s*([A-E])\b'),                                     # {"answer": "E. ..."}
    re.compile(r'(?:^|\n)\s*(?:option\s+)?([A-E])\s*(?:[.):]|$)', re.I),             # line starting "A." / bare "C"
    re.compile(r'\b(?:option\s+)?([A-E])\s+(?:best fits|is (?:the )?(?:most )?(?:plausible|likely|correct|best))'),
]


def find(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            x = find(v, key)
            if x is not None:
                return x
    return None


def lenient(text):
    text = re.sub(r'<tool_call>.*?(</tool_call>|$)', ' ', text or '', flags=re.S)
    for pattern in PATTERNS:
        hits = pattern.findall(text)
        if hits:
            return hits[-1].upper()
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('result_dir')
    args = p.parse_args()
    rows = records(Path(args.result_dir))
    strict = sum(bool(r.get('correct')) for r in rows)
    rescored, sources, recovered = strict, collections.Counter(), []
    for r in rows:
        source = find(r, 'answer_parse_source')
        sources[source] += 1
        if source != 'unparsed':
            continue
        pred = lenient(find(r, 'fine_response')) or lenient(find(r, 'coarse_response'))
        if pred:
            recovered.append((r.get('video_idx'), pred, r.get('answer_gt')))
            rescored += pred == r.get('answer_gt')
    n = len(rows)
    print(f'questions {n}; parse sources {dict(sources)}')
    print(f'strict accuracy   {100 * strict / n:.2f}% ({strict}/{n})')
    print(f'unparsed recovered {len(recovered)}/{sources["unparsed"]}; '
          f'still unanswered {sources["unparsed"] - len(recovered)}')
    print(f'rescored accuracy {100 * rescored / n:.2f}% ({rescored}/{n})')
    fine = sum(bool(find(r, 'oasis_fine_triggered')) for r in rows)
    print(f'fine stage (memory retrieval) triggered on {fine}/{n} ({100 * fine / n:.1f}%)')


if __name__ == '__main__':
    main()
