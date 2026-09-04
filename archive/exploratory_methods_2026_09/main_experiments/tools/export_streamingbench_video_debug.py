#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from main_experiments.tools.visualize_selected_frames import (  # noqa: E402
    _adaptive_meta,
    _decode_frames,
    _format_score,
    _load_records,
    _make_sheet,
    _resolve_video_path,
    _safe_name,
)


def _video_name(record: dict[str, Any]) -> str:
    value = record.get("video") or record.get("video_path") or record.get("video_path_raw")
    return Path(str(value)).name


def _choose_video(records: list[dict[str, Any]], requested: str | None) -> str:
    if requested:
        wanted = Path(requested).name
        names = {_video_name(record) for record in records}
        if wanted not in names:
            raise ValueError(f"Video {wanted!r} not found in result records. Available examples: {sorted(names)[:10]}")
        return wanted

    triggered_counts: Counter[str] = Counter()
    total_counts: Counter[str] = Counter()
    for record in records:
        name = _video_name(record)
        total_counts[name] += 1
        if _adaptive_meta(record).get("memory_triggered"):
            triggered_counts[name] += 1

    if triggered_counts:
        return triggered_counts.most_common(1)[0][0]
    if total_counts:
        return total_counts.most_common(1)[0][0]
    raise ValueError("No records found.")


def _load_matching_annotation(anno_path: Path, video_name: str) -> dict[str, Any] | None:
    if not anno_path.exists():
        return None
    with anno_path.open() as handle:
        data = json.load(handle)
    for entry in data:
        if Path(str(entry.get("video_path", ""))).name == video_name:
            return entry
    return None


def _record_to_html(record: dict[str, Any], sheet_rel: str, selected_scores: list[str]) -> str:
    meta = _adaptive_meta(record)
    return f"""
    <section>
      <h2>Question {_html(record.get('_index'))}: {_html(record.get('task_type'))}</h2>
      <p><b>Timestamp:</b> {_html(record.get('time_stamp'))}
         <b>Correct:</b> {_html(record.get('correct'))}
         <b>GT:</b> {_html(record.get('answer_gt'))}
         <b>Response:</b> {_html(record.get('response'))}</p>
      <p><b>Question:</b> {_html(record.get('question'))}</p>
      <p><b>Mode:</b> {_html(meta.get('mode'))}
         <b>Triggered:</b> {_html(meta.get('memory_triggered'))}
         <b>Gate:</b> {_html(json.dumps(meta.get('memory_gate'), ensure_ascii=False))}</p>
      <p><b>Recent chunks:</b> {_html(meta.get('recent_chunk_ids'))}
         <b>Memory chunks:</b> {_html(meta.get('memory_chunk_ids'))}
         <b>Selected chunks:</b> {_html(meta.get('selected_chunk_ids', record.get('final_chunk_ids')))}</p>
      <img src="{html.escape(sheet_rel)}" style="max-width:100%;border:1px solid #ccd3dd;border-radius:8px;">
      <details>
        <summary>Selected memory scores</summary>
        <pre>{html.escape(chr(10).join(selected_scores) if selected_scores else 'No selected memory scores.')}</pre>
      </details>
    </section>
    """


def _html(value: Any) -> str:
    return html.escape(str(value))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a one-video StreamingBench debug bundle: copied video, matching questions, "
            "result records, and selected-frame contact sheets for every question."
        )
    )
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--anno-path", type=Path, default=Path("data/streamingbench/questions_real.json"))
    parser.add_argument("--video-dir", type=Path, default=Path("data/streamingbench/videos"))
    parser.add_argument("--video", default="", help="Video basename, e.g. sample_123_real.mp4. Defaults to a video with many memory-triggered records.")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/one_streamingbench_video_debug"))
    parser.add_argument("--copy-video", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    records = _load_records(args.result_dir)
    video_name = _choose_video(records, args.video or None)
    video_records = [record for record in records if _video_name(record) == video_name]
    video_records.sort(key=lambda item: int(item.get("_index", 0)))
    if not video_records:
        raise ValueError(f"No records found for {video_name}")

    video_path = _resolve_video_path(video_records[0], args.video_dir)
    bundle_dir = args.out_dir / Path(video_name).stem
    frames_dir = bundle_dir / "selected_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    copied_video = None
    if args.copy_video:
        copied_video = bundle_dir / video_path.name
        if not copied_video.exists():
            shutil.copy2(video_path, copied_video)

    annotation = _load_matching_annotation(args.anno_path, video_name)
    if annotation is not None:
        (bundle_dir / "questions.json").write_text(json.dumps(annotation, indent=2, ensure_ascii=False), encoding="utf-8")

    with (bundle_dir / "records.jsonl").open("w") as handle:
        for record in video_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    sections: list[str] = []
    for i, record in enumerate(video_records, start=1):
        meta = _adaptive_meta(record)
        timestamps = [float(x) for x in meta.get("selected_timestamps", [])]
        if not timestamps:
            chunk_duration = float(meta.get("chunk_duration", 1.0) or 1.0)
            timestamps = [float(x) * chunk_duration for x in meta.get("selected_chunk_ids", record.get("final_chunk_ids", []))]
        chunk_ids = [int(x) for x in meta.get("selected_chunk_ids", record.get("final_chunk_ids", []))]
        recent_ids = set(int(x) for x in meta.get("recent_chunk_ids", []))
        memory_ids = set(int(x) for x in meta.get("memory_chunk_ids", []))

        frames = _decode_frames(video_path, timestamps)
        labels: list[str] = []
        for j, chunk_id in enumerate(chunk_ids):
            role = "memory" if chunk_id in memory_ids else "recent" if chunk_id in recent_ids else "selected"
            ts = timestamps[j] if j < len(timestamps) else float("nan")
            labels.append(f"{j + 1}. {role} | chunk {chunk_id} | {ts:.1f}s")

        sheet_name = _safe_name(f"{i:02d}_{record.get('_index')}_{record.get('task_type')}.jpg")
        sheet_path = frames_dir / sheet_name
        _make_sheet(frames, labels, sheet_path)

        selected_scores = [
            _format_score(score)
            for score in meta.get("memory_scores", [])
            if isinstance(score, dict) and score.get("selected")
        ]
        sections.append(_record_to_html(record, f"selected_frames/{sheet_name}", selected_scores))

    task_counts = Counter(str(record.get("task_type", "")) for record in video_records)
    triggered = sum(1 for record in video_records if _adaptive_meta(record).get("memory_triggered"))
    correct = sum(1 for record in video_records if record.get("correct"))
    html_path = bundle_dir / "index.html"
    html_path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(video_name)} selected-frame debug</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #142033; }}
    section {{ margin-bottom: 38px; padding-bottom: 26px; border-bottom: 1px solid #d9dee8; }}
    h1, h2 {{ margin-bottom: 8px; }}
    p {{ line-height: 1.45; }}
    pre {{ background: #f5f7fb; padding: 12px; border-radius: 8px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>StreamingBench Selected-Frame Debug</h1>
  <p><b>Video:</b> {html.escape(video_name)}</p>
  <p><b>Source video:</b> {html.escape(str(video_path))}</p>
  <p><b>Copied video:</b> {html.escape(str(copied_video.name if copied_video else 'not copied'))}</p>
  <p><b>Questions:</b> {len(video_records)}
     <b>Correct:</b> {correct}/{len(video_records)}
     <b>Memory-triggered:</b> {triggered}/{len(video_records)}</p>
  <p><b>Task counts:</b> {html.escape(json.dumps(dict(task_counts), ensure_ascii=False))}</p>
  {''.join(sections)}
</body>
</html>
""",
        encoding="utf-8",
    )

    print(f"video: {video_name}")
    print(f"bundle: {bundle_dir}")
    print(f"records: {len(video_records)} correct: {correct}/{len(video_records)} triggered: {triggered}/{len(video_records)}")
    print(f"html: {html_path}")
    if annotation is None:
        print(f"warning: no matching annotation found in {args.anno_path}")


if __name__ == "__main__":
    main()
