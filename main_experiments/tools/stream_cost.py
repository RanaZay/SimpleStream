#!/usr/bin/env python3
"""Total compute per stream as a function of question density (memory upkeep is paid per streamed second,
answering per question).

  cost per stream-minute = 60 * u + d * q
    u: memory upkeep per streamed second (ms), estimated from the logged runs
       (OASIS: event summaries and merges for newly streamed seconds; ShallowStream: shallow indexing of new frames)
    q: per-question cost (ms), algorithmic (question -> answer) or full-system (+ history decoding)
    d: questions per minute

Usage: python main_experiments/tools/stream_cost.py --run "Recent-6=dir" --run "PRISM=dir" ... [--densities 0.1 0.5 1 2 6]
"""
import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from main_experiments.tools.benchmark_tables import pairs, table
from main_experiments.tools.summarize_progressive_arbitration import profile, records


def new_stream_seconds(meta, timing):
    """Seconds of video newly streamed (and upkept) before this question."""
    if timing.get('shallow_index_new_frames') is not None:  # ShallowStream: 1 fps frames indexed for the first time
        return float(timing['shallow_index_new_frames'])
    if meta.get('query_time') is not None:  # OASIS: stream advanced from the last reused point to the question
        return max(0.0, float(meta['query_time']) - float(timing.get('oasis_reused_stream_until_s') or 0.0))
    return None


def estimate(rows):
    upkeep, seconds, answer, full = 0.0, 0.0, [], []
    for r in rows:
        _p, meta, t = profile(r)
        if not t or t.get('PRISM_end_to_end_latency_ms') is None:
            continue
        answer.append(t['PRISM_end_to_end_latency_ms'])
        full.append(t['PRISM_end_to_end_latency_ms'] + (t.get('total_visual_preparation_ms') or 0.0))
        spent = t.get('memory_maintenance_total_ms') or 0.0
        streamed = new_stream_seconds(meta, t)
        if spent and streamed:
            upkeep += spent
            seconds += streamed
    return dict(u=upkeep / seconds if seconds else 0.0, q=statistics.mean(answer), q_full=statistics.mean(full),
                n=len(answer), streamed_s=seconds)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', action='append', default=[], help='Label=inference result dir')
    p.add_argument('--densities', type=float, nargs='+', default=[0.1, 0.5, 1.0, 2.0, 6.0], help='questions per minute')
    args = p.parse_args()
    est = {label: estimate(records(directory)) for label, directory in pairs(args.run)}
    labels = list(est)
    rows = [['Upkeep per streamed minute (s)'] + [f"{60 * est[l]['u'] / 1000:.2f}" for l in labels],
            ['Per-question cost, algorithmic (s)'] + [f"{est[l]['q'] / 1000:.3f}" for l in labels],
            ['Per-question cost, full system incl. decoding (s)'] + [f"{est[l]['q_full'] / 1000:.3f}" for l in labels]]
    for d in args.densities:
        rows.append([f'Cost per stream-minute at {d:g} q/min, algorithmic (s)']
                    + [f"{(60 * est[l]['u'] + d * est[l]['q']) / 1000:.2f}" for l in labels])
    for d in args.densities:
        rows.append([f'Cost per stream-minute at {d:g} q/min, full system (s)']
                    + [f"{(60 * est[l]['u'] + d * est[l]['q_full']) / 1000:.2f}" for l in labels])
    base = labels[0]
    rows.append([f'Per-question cost relative to {base}'] + [f"{est[l]['q'] / est[base]['q']:.2f}×" for l in labels])
    print(table(['GPU seconds'] + labels, rows))
    sources = ', '.join('%s: %.0f streamed s' % (l, est[l]['streamed_s']) for l in labels)
    print(f'\nUpkeep is estimated from {sources}.')
    print('Full system adds video decoding as measured in these runs; on a phone, camera frames arrive already decoded.')


if __name__ == '__main__':
    main()
