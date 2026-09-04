#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from lib.minicpm.adaptive import AdaptiveWindowConfig, select_adaptive_frames  # noqa: E402
from lib.shared.recent_window import EvalChunk  # noqa: E402
from main_experiments.minicpm_v46.streamingbench.eval_baseline import (  # noqa: E402
    build_prompt,
    timestamp_to_seconds,
)
from main_experiments.tools.visualize_selected_frames import (  # noqa: E402
    _format_score,
    _make_sheet,
    _safe_name,
)


def _seconds_to_timestamp(seconds: float) -> str:
    seconds_i = max(0, int(round(seconds)))
    h, rem = divmod(seconds_i, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _normalise_options(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except Exception:
            pass
        return [part.strip() for part in re.split(r"\s*[ABCD]\.\s*", text) if part.strip()]
    return []


def _normalise_question(row: dict[str, Any], index: int) -> dict[str, Any]:
    question = dict(row)
    question.setdefault("question_id", question.get("id", f"debug_{index}"))
    question.setdefault("task_type", question.get("task", "Debug"))
    question.setdefault("time_stamp", question.get("timestamp", "00:00:00"))
    question["options"] = _normalise_options(question.get("options", []))
    if "answer" not in question and "answer_gt" in question:
        question["answer"] = question["answer_gt"]
    return question


def _load_questions(path: Path, video_name: str, max_questions: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    else:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and isinstance(data.get("questions"), list):
            rows = list(data["questions"])
        elif isinstance(data, list) and data and isinstance(data[0], dict) and "questions" in data[0]:
            target_stem = Path(video_name).stem
            sample_match = re.search(r"sample_(\d+)", target_stem)
            sample_key = f"sample_{sample_match.group(1)}" if sample_match else target_stem
            chosen = None
            for entry in data:
                entry_name = Path(str(entry.get("video_path", ""))).stem
                if entry_name == target_stem or sample_key in entry_name:
                    chosen = entry
                    break
            chosen = chosen or data[0]
            rows = list(chosen.get("questions", []))
        elif isinstance(data, list):
            rows = data
        else:
            raise ValueError(f"Unsupported question file format: {path}")

    video_stem = Path(video_name).stem
    sample_match = re.search(r"sample_(\d+)", video_stem)
    sample_key = f"sample_{sample_match.group(1)}" if sample_match else ""
    if sample_key:
        sample_id_re = re.compile(rf"(^|[_/\\-]){re.escape(sample_key)}([_/\\-]|$)")

        def _matches_sample(row: dict[str, Any]) -> bool:
            return any(
                sample_id_re.search(str(row.get(key, "")))
                for key in ("question_id", "video_path", "video", "file")
            )

        matched = [
            row
            for row in rows
            if _matches_sample(row)
        ]
        if matched:
            rows = matched

    questions = [_normalise_question(row, i) for i, row in enumerate(rows)]
    questions.sort(key=lambda item: timestamp_to_seconds(str(item.get("time_stamp", "00:00:00"))))
    if max_questions > 0:
        questions = questions[:max_questions]
    return questions


def _ffprobe_duration(video_path: Path) -> float:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        text=True,
    ).strip()
    return float(output)


def _extract_frame(video_path: Path, timestamp: float, out_path: Path) -> Image.Image:
    if not out_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{max(0.0, timestamp):.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(out_path),
            ],
            check=True,
        )
    return Image.open(out_path).convert("RGB")


def _decode_chunks_with_ffmpeg(
    video_path: Path,
    frame_cache_dir: Path,
    fps: float,
    chunk_duration: float,
    video_start: float,
    video_end: float,
) -> list[EvalChunk]:
    duration = _ffprobe_duration(video_path)
    start = max(0.0, min(video_start, duration))
    end = max(start, min(video_end, duration))
    step = 1.0 / max(fps, 1e-6)

    timestamps: list[float] = []
    current = start
    while current <= end + 1e-6:
        timestamps.append(current)
        current += step
    if not timestamps:
        timestamps = [end]

    chunks: list[EvalChunk] = []
    for ts in timestamps:
        chunk_index = int(math.floor(ts / chunk_duration)) if chunk_duration > 0 else int(round(ts))
        cache_name = f"{Path(video_path).stem}_{ts:.3f}.jpg".replace(".", "_", 1)
        frame = _extract_frame(video_path, ts, frame_cache_dir / cache_name)
        chunks.append(
            EvalChunk(
                frames=[frame],
                frame_timestamps=[ts],
                start_time=ts,
                end_time=ts,
                chunk_index=chunk_index,
                fps=fps,
            )
        )
    return chunks


def _score_lines(scores: list[dict[str, Any]]) -> str:
    selected = [_format_score(score) for score in scores if isinstance(score, dict) and score.get("selected")]
    top = [_format_score(score) for score in scores[:5] if isinstance(score, dict)]
    return "SELECTED\n" + ("\n".join(selected) or "None") + "\n\nTOP-5\n" + ("\n".join(top) or "None")


def _answer_text(question: dict[str, Any]) -> str:
    answer = str(question.get("answer", "")).strip()
    options = question.get("options", [])
    if answer and isinstance(options, list):
        letters = ["A", "B", "C", "D", "E"]
        for letter, option in zip(letters, options):
            if answer.upper() == letter:
                return f"{letter}. {option}"
    return answer or "N/A"


def _write_html(
    out_path: Path,
    video_path: Path,
    rows: list[dict[str, Any]],
    copied_video: Path | None = None,
) -> None:
    sections: list[str] = []
    for row in rows:
        score_text = _score_lines(row["memory_scores"])
        sections.append(
            f"""
<section>
  <h2>{html.escape(str(row['index']))}. {html.escape(row['task_type'])}</h2>
  <p><b>Timestamp:</b> {html.escape(row['time_stamp'])}
     <b>Mode:</b> {html.escape(row['mode'])}
     <b>Triggered:</b> {html.escape(str(row['memory_triggered']))}</p>
  <p><b>Question:</b> {html.escape(row['question'])}</p>
  <p><b>Ground truth answer:</b> {html.escape(str(row['answer_text']))}</p>
  <p><b>Options:</b> {html.escape(' | '.join(row.get('options', [])) or 'N/A')}</p>
  <p><b>Recent chunks:</b> {html.escape(str(row['recent_chunk_ids']))}
     <b>Memory chunks:</b> {html.escape(str(row['memory_chunk_ids']))}
     <b>Selected chunks:</b> {html.escape(str(row['selected_chunk_ids']))}</p>
  <p><b>Selected timestamps:</b> {html.escape(str([round(x, 2) for x in row['selected_timestamps']]))}</p>
  <img src="{html.escape(row['sheet_rel'])}" style="max-width:100%;border:1px solid #ccd3dd;border-radius:8px;">
  <details><summary>Memory scores</summary><pre>{html.escape(score_text)}</pre></details>
</section>
"""
        )
    out_path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Adaptive Selection Debug</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #172033; }}
    section {{ border-bottom: 1px solid #d8dee8; margin-bottom: 34px; padding-bottom: 24px; }}
    pre {{ background: #f5f7fb; padding: 12px; border-radius: 8px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>Adaptive Frame Selection Debug</h1>
  <p><b>Video:</b> {html.escape(str(video_path))}</p>
  {
      f'<video controls src="{html.escape(copied_video.name)}" style="max-width:100%;border:1px solid #ccd3dd;border-radius:8px;"></video>'
      if copied_video is not None
      else ''
  }
  {''.join(sections)}
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize adaptive MiniCPM frame selection for one local video and a small question file. "
            "This does not run MiniCPM; it only runs the frame-selection policy."
        )
    )
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--questions-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/adaptive_selection_one_video"))
    parser.add_argument("--mode", default="question_aware_memory")
    parser.add_argument("--recent-window", type=int, default=6)
    parser.add_argument("--memory-anchors", type=int, default=3)
    parser.add_argument("--memory-search-chunks", type=int, default=32)
    parser.add_argument("--context-time", type=float, default=60.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--max-questions", type=int, default=8)
    parser.add_argument("--copy-video", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not args.video_path.exists():
        raise FileNotFoundError(args.video_path)
    questions = _load_questions(args.questions_path, args.video_path.name, args.max_questions)
    if not questions:
        raise ValueError(f"No questions found in {args.questions_path}")

    config = AdaptiveWindowConfig(
        mode=args.mode,
        min_window=args.recent_window,
        mid_window=args.recent_window,
        max_window=args.recent_window,
        memory_anchors=args.memory_anchors,
        memory_search_chunks=args.memory_search_chunks,
    )
    config.validate()

    frames_dir = args.out_dir / "selected_frames"
    cache_dir = args.out_dir / "_frame_cache"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    copied_video = None
    if args.copy_video:
        copied_video = args.out_dir / args.video_path.name
        if copied_video.resolve() != args.video_path.resolve():
            shutil.copy2(args.video_path, copied_video)

    records: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        timestamp = float(timestamp_to_seconds(str(question.get("time_stamp", "00:00:00"))))
        video_start = max(0.0, timestamp - args.context_time) if args.context_time >= 0 else 0.0
        chunks = _decode_chunks_with_ffmpeg(
            args.video_path,
            cache_dir,
            fps=args.fps,
            chunk_duration=args.chunk_duration,
            video_start=video_start,
            video_end=timestamp,
        )
        prompt = build_prompt(question)
        selection = select_adaptive_frames(chunks, prompt, config)
        meta = selection.metadata

        labels = []
        memory_ids = set(int(x) for x in meta.get("memory_chunk_ids", []))
        recent_ids = set(int(x) for x in meta.get("recent_chunk_ids", []))
        for frame_index, chunk_id in enumerate(meta.get("selected_chunk_ids", [])):
            role = "memory" if int(chunk_id) in memory_ids else "recent" if int(chunk_id) in recent_ids else "selected"
            ts = meta.get("selected_timestamps", [])[frame_index]
            labels.append(f"{frame_index + 1}. {role} | chunk {chunk_id} | {_seconds_to_timestamp(float(ts))}")

        sheet_name = _safe_name(f"{index:02d}_{question.get('task_type', 'debug')}_{question.get('question', '')}") + ".jpg"
        sheet_path = frames_dir / sheet_name
        answer_text = _answer_text(question)
        options_text = " | ".join(str(x) for x in question.get("options", [])) or "N/A"
        _make_sheet(
            selection.frames,
            labels,
            sheet_path,
            title=f"Q{index}: {question.get('question', '')}",
            subtitle=(
                f"GT: {answer_text} | Task: {question.get('task_type', 'Debug')} | "
                f"t={question.get('time_stamp', '')} | Options: {options_text}"
            ),
        )

        record = {
            "index": index,
            "question_id": question.get("question_id"),
            "task_type": str(question.get("task_type", "Debug")),
            "time_stamp": str(question.get("time_stamp", "")),
            "question": str(question.get("question", "")),
            "answer": question.get("answer"),
            "answer_text": answer_text,
            "options": question.get("options", []),
            "mode": config.mode,
            "memory_triggered": meta.get("memory_triggered"),
            "memory_gate": meta.get("memory_gate"),
            "recent_chunk_ids": meta.get("recent_chunk_ids", []),
            "memory_chunk_ids": meta.get("memory_chunk_ids", []),
            "selected_chunk_ids": meta.get("selected_chunk_ids", []),
            "selected_timestamps": meta.get("selected_timestamps", []),
            "memory_scores": meta.get("memory_scores", []),
            "sheet_rel": f"selected_frames/{sheet_name}",
        }
        records.append(record)

    with (args.out_dir / "selection_records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    _write_html(args.out_dir / "index.html", args.video_path, records, copied_video)

    print(f"video: {args.video_path}")
    if copied_video is not None:
        print(f"copied_video: {copied_video}")
    print(f"questions: {len(records)}")
    print(f"out_dir: {args.out_dir}")
    print(f"html: {args.out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
