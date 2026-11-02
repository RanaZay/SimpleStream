#!/usr/bin/env python3
"""Qualitative PRISM-VoI examples from a run: where memory fixed the answer, where it broke one, and where the
recent frames were confident enough to skip memory. Writes a markdown file and, with --frames, one image strip per
example (the recent frames, then the memory frames that were added, in time order).

Usage: python main_experiments/tools/voi_examples.py --run reports/.../prism_voi_<job> --out reports/figures/examples
       [--lam 0] [--per-kind 5] [--frames]
Trace runs are replayed at --lam; other runs use their own decisions. Frame strips need a run that saved
recent_times and memory times (made after this tool was added).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from main_experiments.tools.lambda_sweep import correct_at, depth_for
from main_experiments.tools.summarize_progressive_arbitration import correctness, profile, records

KINDS = {'fixed': 'Memory fixed the answer', 'broken': 'Memory broke a correct answer',
         'skipped': 'Confident from the recent frames; memory skipped'}


def first(record, keys):
    return next((record[k] for k in keys if record.get(k) not in (None, '')), None)


def classify(record, meta, benchmark, lam):
    trace = meta.get('trace') or []
    if trace and meta.get('format') == 'mcq':
        depth = depth_for(trace, lam)[0]
        right = correct_at(trace[depth], record, benchmark)
        base = correct_at(trace[0], record, benchmark)
        answers = [r.get('answer') or r.get('prediction') for r in trace]
    else:
        depth = int(meta.get('selected_memory_depth') or 0)
        right = correctness(record, 'ovo') if benchmark == 'ovo' else record.get('correct')
        base, answers = None, [record.get('response') or record.get('pred')]
    if depth > 0 and right and base is False:
        kind = 'fixed'
    elif depth > 0 and base and right is False:
        kind = 'broken'
    elif depth == 0 and right and trace and trace[0]['se'] < 0.05:
        kind = 'skipped'
    else:
        return None
    drop = trace[0]['se'] - trace[depth]['se'] if trace else 0.0
    return kind, depth, drop, answers


def strip(video, recent_times, memory_times, path):
    """Recent frames then memory frames, decoded with the evaluation's own 1 fps decoder."""
    from PIL import Image, ImageDraw
    from lib.minicpm.prism_retrieval_variants import representative_frame
    from lib.minicpm.progressive_sufficiency import _chunk_temporal_bounds
    from lib.shared.recent_window import decode_video_to_chunks_qwen
    times = sorted(memory_times) + list(recent_times)
    chunks, _backend = decode_video_to_chunks_qwen(video, 1.0, 1.0, None, video_start=max(0.0, min(times) - 1.0),
                                                   video_end=max(times) + 1.0)
    tiles = []
    for t in times:
        chunk = min(chunks, key=lambda c: abs(_chunk_temporal_bounds(c)[1] - t))
        tile = representative_frame(chunk).convert('RGB')
        tile.thumbnail((256, 256))
        canvas = Image.new('RGB', (tile.width, tile.height + 18), 'white')
        canvas.paste(tile, (0, 18))
        label = f"{'memory' if t in memory_times else 'recent'} {t:.0f}s"
        ImageDraw.Draw(canvas).text((4, 3), label, fill=(200, 40, 40) if t in memory_times else (40, 40, 40))
        tiles.append(canvas)
    out = Image.new('RGB', (sum(t.width for t in tiles) + 4 * len(tiles), max(t.height for t in tiles)), 'white')
    x = 0
    for t in tiles:
        out.paste(t, (x, 0))
        x += t.width + 4
    out.save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--benchmark', required=True, choices=['ovo', 'streamingbench', 'videomme'])
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--lam', type=float, default=0.0, help='lambda to replay trace runs at')
    p.add_argument('--per-kind', type=int, default=5)
    p.add_argument('--frames', action='store_true', help='also cut image strips from the videos')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    found = {k: [] for k in KINDS}
    for record in records(args.run):
        meta = profile(record)[1]
        result = classify(record, meta, args.benchmark, args.lam)
        if result:
            found[result[0]].append((record, meta, *result[1:]))
    lines = [f'# PRISM-VoI examples: {args.run.name} (λ = {args.lam:g})\n']
    for kind, title in KINDS.items():
        # Largest uncertainty drops first for memory cases; most confident first for skipped ones.
        chosen = sorted(found[kind], key=lambda x: -abs(x[3]))[:args.per_kind]
        lines.append(f'## {title} ({len(found[kind])} questions)\n')
        for n, (record, meta, depth, drop, answers) in enumerate(chosen, 1):
            trace = meta.get('trace') or []
            question = first(record, ['question', 'query', 'prompt']) or meta.get('question_text')
            lines += [f'### {kind} {n}: {record.get("task") or record.get("task_type") or ""}',
                      f'- Video: `{first(record, ["video_path", "video", "video_file"])}`',
                      f'- Question: {question}',
                      f'- Ground truth: {first(record, ["answer_gt", "ground_truth", "answer", "gt"])}',
                      f'- Memory frames added: {depth}; semantic entropy {trace[0]["se"]:.3f} → '
                      f'{trace[depth]["se"]:.3f} nats' if trace else f'- Memory frames added: {depth}',
                      '- Answer per context: ' + ', '.join(f'R+{d}: {a!r}' for d, a in enumerate(answers))]
            recent_times = meta.get('recent_times') or []
            memory_times = [r['time'] for r in trace[1:depth + 1] if 'time' in r] or meta.get('memory_times') or []
            video = first(record, ['video_path', 'video_file'])
            if args.frames and video and recent_times and len(memory_times) == depth:
                image = args.out / f'{kind}_{n}.jpg'
                try:
                    strip(video, recent_times, memory_times, image)
                    lines.append(f'\n![{kind} {n}]({image.name})')
                except Exception as exc:  # A missing or unreadable video should not stop the report.
                    lines.append(f'- (frames not cut: {exc})')
            lines.append('')
    (args.out / 'examples.md').write_text('\n'.join(lines))
    (args.out / 'counts.json').write_text(json.dumps({k: len(v) for k, v in found.items()}, indent=2))
    print(f"Wrote {args.out / 'examples.md'}: " + ', '.join(f'{k} {len(v)}' for k, v in found.items()))


if __name__ == '__main__':
    main()
