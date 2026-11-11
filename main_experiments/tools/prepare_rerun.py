#!/usr/bin/env python3
"""Prepare a finished or interrupted run so that only missing or failed questions are run again.

StreamingBench / OVO (per-rank checkpoints): rows with an inference error are removed from each
rank_*/results_incremental.jsonl (the original is kept as *.before_rerun) and the per-rank `done` markers are
deleted. Resuming the run with RESUME_DIR=<dir> then answers exactly the missing questions.

StreamBench (one results file): failed question keys are written to <dir>/failed_keys.json for STREAMBENCH_ONLY_KEYS;
merge the rerun back with merge_streambench_rerun.py.
"""
import argparse
import json
from pathlib import Path


def failed(row):
    """An inference error on the question, or on any of its probes (OVO Forward questions record one per probe)."""
    children = row.get('test_info') if isinstance(row.get('test_info'), list) else []
    return bool(row.get('error')) or any(isinstance(c, dict) and c.get('error') for c in children)


def prune_checkpoints(directory):
    files = sorted(directory.glob('rank_*/results_incremental.jsonl')) or sorted(directory.glob('results_incremental.jsonl'))
    if not files:
        raise SystemExit(f'No checkpoints in {directory}')
    kept = dropped = 0
    for path in files:
        rows = [line for line in path.read_text().splitlines() if line.strip()]
        good = [line for line in rows if not failed(json.loads(line))]
        backup = path.with_name(path.name + '.before_rerun')
        if not backup.exists():
            backup.write_text(path.read_text())
        path.write_text(''.join(line + '\n' for line in good))
        kept += len(good)
        dropped += len(rows) - len(good)
        done = path.with_name('done')
        if done.exists():
            done.unlink()
    print(f'{directory}: kept {kept} answered questions, removed {dropped} failed ones; resume with RESUME_DIR={directory}')


def streambench_failed(directory):
    payload = json.loads((directory / 'streambench_v0_3_smoke_results.json').read_text())
    failed = [f"{row['video_index']}:{row['breakpoint_index']}" for row in payload['results']
              if any(m.get('error') or not str(m.get('prediction') or '').strip() for m in row.get('methods', []))]
    (directory / 'failed_keys.json').write_text(json.dumps(failed))
    print(f'{directory}: {len(failed)} failed questions -> {directory / "failed_keys.json"}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dir', type=Path)
    args = p.parse_args()
    if (args.run_dir / 'streambench_v0_3_smoke_results.json').exists() and not list(args.run_dir.glob('rank_*')):
        streambench_failed(args.run_dir)
    else:
        prune_checkpoints(args.run_dir)


if __name__ == '__main__':
    main()
