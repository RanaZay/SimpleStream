#!/usr/bin/env python3
"""Replace failed StreamBench answers with the answers from a rerun of just those questions.

Writes <output>/streambench_v0_3_smoke_results.json (plus the original run_config.json) and leaves both inputs untouched.
"""
import argparse
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('original', type=Path)
    p.add_argument('rerun', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    name = 'streambench_v0_3_smoke_results.json'
    base = json.loads((args.original / name).read_text())
    fresh = {f"{r['video_index']}:{r['breakpoint_index']}": r for r in json.loads((args.rerun / name).read_text())['results']}
    replaced, still_failed = 0, []
    rows = []
    for row in base['results']:
        key = f"{row['video_index']}:{row['breakpoint_index']}"
        if key in fresh:
            row = fresh[key]
            replaced += 1
        if any(m.get('error') for m in row.get('methods', [])):
            still_failed.append(key)
        rows.append(row)
    args.output.mkdir(parents=True, exist_ok=False)
    base.update(results=rows, merged_from=dict(original=str(args.original), rerun=str(args.rerun), replaced=replaced,
                                               still_failed=still_failed))
    (args.output / name).write_text(json.dumps(base, indent=2, ensure_ascii=False))
    for extra in ('run_config.json',):
        if (args.original / extra).exists():
            shutil.copy(args.original / extra, args.output / extra)
    print(f'replaced {replaced} answers; still failing: {len(still_failed)} {still_failed[:10]}')


if __name__ == '__main__':
    main()
