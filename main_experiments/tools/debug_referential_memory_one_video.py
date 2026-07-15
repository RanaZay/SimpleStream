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
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from lib.minicpm.baseline import RecentWindowQAModel, query_recent_window  # noqa: E402
from lib.minicpm.referential_memory import (  # noqa: E402
    AnswerGroundedFrameScorer,
    ReferentialMemoryEntry,
    make_memory_entry,
    query_referential_memory_window,
)
from lib.shared.recent_window import extract_mcq_answer  # noqa: E402
from main_experiments.minicpm_v46.streamingbench.eval_baseline import (  # noqa: E402
    build_prompt,
    timestamp_to_seconds,
)


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
            rows = list((chosen or data[0]).get("questions", []))
        elif isinstance(data, list):
            rows = data
        else:
            raise ValueError(f"Unsupported question file format: {path}")

    video_stem = Path(video_name).stem
    sample_match = re.search(r"sample_(\d+)", video_stem)
    sample_key = f"sample_{sample_match.group(1)}" if sample_match else ""
    if sample_key:
        sample_id_re = re.compile(rf"(^|[_/\\-]){re.escape(sample_key)}([_/\\-]|$)")

        matched = [
            row
            for row in rows
            if any(
                sample_id_re.search(str(row.get(key, "")))
                for key in ("question_id", "video_path", "video", "file")
            )
        ]
        if matched:
            rows = matched

    questions = [_normalise_question(row, index) for index, row in enumerate(rows)]
    questions.sort(key=lambda item: timestamp_to_seconds(str(item.get("time_stamp", "00:00:00"))))
    if max_questions > 0:
        questions = questions[:max_questions]
    return questions


def _safe_name(text: str, max_len: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return text[:max_len] or "example"


def _seconds_to_timestamp(seconds: float) -> str:
    seconds_i = max(0, int(round(seconds)))
    h, rem = divmod(seconds_i, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _wrap_text(text: str, width: int) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join([*current, word])
        if len(trial) <= width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines or [""]


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        ["DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf"]
        if bold
        else ["DejaVuSans.ttf", "Arial.ttf"]
    )
    paths = [
        *(f"/usr/share/fonts/truetype/dejavu/{name}" for name in names),
        *(f"/usr/share/fonts/truetype/msttcorefonts/{name}" for name in names),
    ]
    for path in paths:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _make_sheet(
    frames: list[Image.Image],
    labels: list[str],
    out_path: Path,
    *,
    title_lines: list[str] | None = None,
    thumb_width: int = 320,
    thumb_height: int = 200,
) -> None:
    if not frames:
        return
    cols = min(4, len(frames))
    rows = int(math.ceil(len(frames) / cols))
    pad = 12
    title_lines = title_lines or []
    title_font = _load_font(18, bold=True)
    body_font = _load_font(14)
    small_font = _load_font(13)
    title_step = 24
    label_step = 18
    title_h = 34 + title_step * len(title_lines) if title_lines else 0
    label_h = 168
    width = cols * thumb_width + (cols + 1) * pad
    height = title_h + rows * (thumb_height + label_h) + (rows + 1) * pad
    sheet = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(sheet)

    y_offset = pad
    if title_lines:
        draw.rounded_rectangle(
            [pad, pad, width - pad, pad + title_h - 10],
            radius=10,
            fill=(255, 255, 255),
            outline=(203, 213, 225),
            width=1,
        )
    for line_index, line in enumerate(title_lines):
        fill = (21, 128, 61) if "Correct: True" in line else (15, 23, 42)
        if "Correct: False" in line:
            fill = (185, 28, 28)
        font = title_font if line_index == 0 else body_font
        draw.text((pad + 12, y_offset + 10 + line_index * title_step), line, fill=fill, font=font)
    y_offset += title_h

    for index, frame in enumerate(frames):
        row, col = divmod(index, cols)
        x = pad + col * (thumb_width + pad)
        y = y_offset + row * (thumb_height + label_h + pad)
        label_lines = str(labels[index]).split("\n")
        label_text = " ".join(label_lines).lower()
        if "anchor" in label_text:
            border = (22, 163, 74)
            fill_label = (236, 253, 245)
        elif "reference" in label_text:
            border = (37, 99, 235)
            fill_label = (239, 246, 255)
        else:
            border = (71, 85, 105)
            fill_label = (255, 255, 255)
        image = frame.copy()
        image.thumbnail((thumb_width, thumb_height))
        bg = Image.new("RGB", (thumb_width, thumb_height), (226, 232, 240))
        bg.paste(image, ((thumb_width - image.width) // 2, (thumb_height - image.height) // 2))
        sheet.paste(bg, (x, y))
        draw.rectangle([x, y, x + thumb_width, y + thumb_height], outline=border, width=3)
        label_y = y + thumb_height + 6
        draw.rounded_rectangle(
            [x, label_y, x + thumb_width, label_y + label_h - 8],
            radius=8,
            fill=fill_label,
            outline=(203, 213, 225),
            width=1,
        )
        for line_index, line in enumerate(label_lines[:9]):
            fill = (21, 128, 61) if "ANCHOR" in line else (15, 23, 42)
            font = body_font if line_index == 0 else small_font
            draw.text((x + 8, label_y + 8 + line_index * label_step), line, fill=fill, font=font)
    sheet.save(out_path)


def _memory_entry_to_json(entry: ReferentialMemoryEntry) -> dict[str, Any]:
    return {
        "question_index": entry.question_index,
        "question": entry.question,
        "response": entry.response,
        "task_type": entry.task_type,
        "time_stamp": entry.time_stamp,
        "selected_chunk_ids": entry.selected_chunk_ids,
        "selected_timestamps": entry.selected_timestamps,
        "anchor_chunk_ids": entry.anchor_chunk_ids or [],
        "anchor_timestamps": entry.anchor_timestamps or [],
        "anchor_scores": entry.anchor_scores or [],
        "anchor_candidate_chunk_ids": entry.anchor_candidate_chunk_ids or [],
        "anchor_candidate_timestamps": entry.anchor_candidate_timestamps or [],
        "anchor_candidate_scores": entry.anchor_candidate_scores or [],
        "memory_text": entry.memory_text,
        "anchor_scoring": entry.anchor_scoring,
        "anchor_scoring_error": entry.anchor_scoring_error,
    }


def _format_candidate(candidate: dict[str, Any]) -> str:
    keys = (
        "question_index",
        "score",
        "lexical_overlap",
        "recency_rank",
        "selected",
        "reason",
        "question",
        "response",
    )
    compact = {key: candidate[key] for key in keys if key in candidate}
    return json.dumps(compact, indent=2, ensure_ascii=False)


def _write_html(
    out_path: Path,
    *,
    video_path: Path,
    copied_video: Path | None,
    rows: list[dict[str, Any]],
) -> None:
    sections: list[str] = []
    for row in rows:
        ref = row.get("referential_memory") or {}
        reference_question = ref.get("reference_question")
        candidate_scores = ref.get("reference_candidate_scores") or []
        candidate_text = "\n\n".join(
            _format_candidate(score) for score in candidate_scores if isinstance(score, dict)
        ) or "No reference candidates."

        baseline_block = ""
        if row.get("baseline"):
            baseline = row["baseline"]
            baseline_block = f"""
  <div class="baseline">
    <h3>SimpleStream Recent-{html.escape(str(row['recent_window']))} baseline</h3>
    <p><b>Response:</b> {html.escape(str(baseline.get('response')))}
       <b>Pred:</b> {html.escape(str(baseline.get('pred')))}
       <b>Correct:</b> {html.escape(str(baseline.get('correct')))}</p>
    <p><b>Chunks:</b> {html.escape(str(baseline.get('final_chunk_ids')))}</p>
  </div>
"""

        reference_question_block = ""
        if isinstance(reference_question, dict):
            q_number = int(reference_question.get("question_index", -1)) + 1
            reference_question_block = f"""
  <p><b>Referenced question:</b> Q{html.escape(str(q_number))}
     at {html.escape(str(reference_question.get('time_stamp')))}:
     {html.escape(str(reference_question.get('question')))}</p>
  <p><b>Referenced answer:</b> {html.escape(str(reference_question.get('response')))}</p>
  <p><b>Memory text used for anchor scoring:</b>
     {html.escape(str(reference_question.get('memory_text') or 'N/A'))}</p>
"""

        sections.append(
            f"""
<section>
  <h2>{html.escape(str(row['index']))}. {html.escape(row['task_type'])}</h2>
  <p><b>Time:</b> {html.escape(row['time_stamp'])}
     <b>GT:</b> {html.escape(row['answer_gt'])}
     <b>Response:</b> {html.escape(str(row['response']))}
     <b>Correct:</b> {html.escape(str(row['correct']))}</p>
  <p><b>Question:</b> {html.escape(row['question'])}</p>
  {baseline_block}
  <div class="memory">
    <h3>Referential Memory</h3>
    <p><b>Triggered:</b> {html.escape(str(ref.get('memory_triggered')))}
       <b>Gate:</b> {html.escape(json.dumps(ref.get('memory_gate'), ensure_ascii=False))}
       <b>Memory before:</b> {html.escape(str(ref.get('memory_size_before')))}</p>
    {reference_question_block}
    <p><b>Reference chunks:</b> {html.escape(str(ref.get('reference_chunk_ids')))}
       <b>Current chunks:</b> {html.escape(str(ref.get('current_chunk_ids')))}
       <b>Selected chunks:</b> {html.escape(str(ref.get('selected_chunk_ids')))}</p>
    <p><b>Selected timestamps:</b> {html.escape(str([round(float(x), 2) for x in ref.get('selected_timestamps', [])]))}</p>
    <img src="{html.escape(row['sheet_rel'])}" alt="selected frames">
    <details><summary>Reference candidate scores</summary><pre>{html.escape(candidate_text)}</pre></details>
    <details><summary>Stored memory after this question</summary><pre>{html.escape(json.dumps(row.get('stored_memory_entry'), indent=2, ensure_ascii=False))}</pre></details>
  </div>
</section>
"""
        )

    video_block = ""
    if copied_video is not None:
        video_block = f"""
<video controls src="{html.escape(copied_video.name)}"></video>
"""

    out_path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Referential Memory Debug</title>
  <style>
    body {{
      color: #162033;
      font-family: Arial, sans-serif;
      line-height: 1.45;
      margin: 28px;
      max-width: 1180px;
    }}
    video {{
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      max-width: 100%;
      width: 900px;
    }}
    section {{
      border-bottom: 1px solid #d8dee8;
      margin-bottom: 36px;
      padding-bottom: 28px;
    }}
    img {{
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      max-width: 100%;
    }}
    pre {{
      background: #f5f7fb;
      border-radius: 8px;
      overflow-x: auto;
      padding: 12px;
      white-space: pre-wrap;
    }}
    .baseline, .memory {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      margin: 14px 0;
      padding: 12px 14px;
    }}
  </style>
</head>
<body>
  <h1>MiniCPM Referential Memory Debug</h1>
  <p><b>Video:</b> {html.escape(str(video_path))}</p>
  {video_block}
  {''.join(sections)}
</body>
</html>
""",
        encoding="utf-8",
    )


def _labels_from_metadata(
    meta: dict[str, Any],
    *,
    memory_entry: ReferentialMemoryEntry | None = None,
) -> list[str]:
    reference_ids = list(meta.get("reference_chunk_ids", []))
    current_ids = list(meta.get("current_chunk_ids", []))
    timestamps = [float(ts) for ts in meta.get("selected_timestamps", [])]
    reference_question = meta.get("reference_question") or {}
    source_question = str(reference_question.get("question") or "")
    source_response = str(reference_question.get("response") or "")
    source_q_number = None
    if "question_index" in reference_question:
        try:
            source_q_number = int(reference_question.get("question_index")) + 1
        except Exception:
            source_q_number = reference_question.get("question_index")
    ref_anchor_ids = list(reference_question.get("stored_anchor_chunk_ids", []) or [])
    ref_anchor_timestamps = list(reference_question.get("stored_anchor_timestamps", []) or [])
    ref_anchor_scores = list(reference_question.get("stored_anchor_scores", []) or [])
    current_scores = list(memory_entry.anchor_candidate_scores or []) if memory_entry is not None else []
    current_anchor_pairs = set()
    if memory_entry is not None:
        for chunk_id, timestamp in zip(memory_entry.anchor_chunk_ids or [], memory_entry.anchor_timestamps or []):
            current_anchor_pairs.add((int(chunk_id), round(float(timestamp), 3)))
    labels: list[str] = []
    for index, chunk_id in enumerate([*reference_ids, *current_ids]):
        role = "reference" if index < len(reference_ids) else "current"
        ts_text = _seconds_to_timestamp(timestamps[index]) if index < len(timestamps) else "unknown"
        lines = [f"{index + 1}. {role}", f"chunk {chunk_id} | {ts_text}"]
        if role == "reference":
            if source_q_number is not None:
                lines.append(f"FROM Q{source_q_number}")
            if source_response:
                lines.append(f"prev answer: {source_response[:46]}")
            if source_question:
                question_hint = source_question[:54] + ("..." if len(source_question) > 54 else "")
                lines.append(f"source: {question_hint}")
            if index < len(ref_anchor_ids):
                anchor_id = int(ref_anchor_ids[index])
                anchor_score = (
                    float(ref_anchor_scores[index])
                    if index < len(ref_anchor_scores)
                    else None
                )
                anchor_ts = (
                    _seconds_to_timestamp(float(ref_anchor_timestamps[index]))
                    if index < len(ref_anchor_timestamps)
                    else "unknown"
                )
                lines.append(f"REF ANCHOR chunk {anchor_id}")
                lines.append(f"stored time {anchor_ts}")
                if anchor_score is not None:
                    lines.append(f"anchor score={anchor_score:.3f}")
        else:
            current_index = index - len(reference_ids)
            if current_index < len(current_scores):
                score = float(current_scores[current_index])
                timestamp = round(float(timestamps[index]), 3) if index < len(timestamps) else None
                is_anchor = timestamp is not None and (int(chunk_id), timestamp) in current_anchor_pairs
                lines.append(("ANCHOR " if is_anchor else "") + f"score={score:.3f}")
        labels.append("\n".join(lines))
    return labels


def _sheet_title_lines(row: dict[str, Any]) -> list[str]:
    lines = [
        f"Q{row['index']} | {row['task_type']} | t={row['time_stamp']} | Correct: {row['correct']}",
        f"GT: {row['answer_gt']} | Pred: {row.get('pred')} | Response: {row.get('response')}",
    ]
    for wrapped in _wrap_text(f"Question: {row['question']}", 118)[:3]:
        lines.append(wrapped)
    ref = row.get("referential_memory") or {}
    reference_question = ref.get("reference_question")
    if isinstance(reference_question, dict):
        try:
            q_number = int(reference_question.get("question_index", -1)) + 1
        except Exception:
            q_number = reference_question.get("question_index")
        source = str(reference_question.get("question") or "")
        response = str(reference_question.get("response") or "")
        for wrapped in _wrap_text(f"Uses memory from Q{q_number}: {source}", 118)[:2]:
            lines.append(wrapped)
        if response:
            lines.append(f"Previous answer: {response}")
    stored = row.get("stored_memory_entry") or {}
    memory_text = stored.get("memory_text")
    if memory_text:
        for wrapped in _wrap_text(f"Stored after this answer: {memory_text}", 118)[:2]:
            lines.append(wrapped)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run MiniCPM+referential memory on one real StreamingBench sample and export "
            "a visual report of reference/current frames."
        )
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=Path("reports/streamingbench_real_sqa_sample_1/sample_1_sqa.mp4"),
    )
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=Path("reports/streamingbench_real_sqa_sample_1/Sequential_Question_Answering.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--model-name", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"))
    parser.add_argument("--max-questions", type=int, default=5)
    parser.add_argument("--max-qa-tokens", type=int, default=64)
    parser.add_argument("--recent-window", type=int, default=6)
    parser.add_argument("--reference-frames", type=int, default=2)
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--context-time", type=float, default=-1.0)
    parser.add_argument("--answer-grounded-memory", action="store_true")
    parser.add_argument("--entity-grounded-memory", action="store_true")
    parser.add_argument("--memory-anchor-frames", type=int, default=1)
    parser.add_argument("--memory-clip-model", default=os.environ.get("MINICPM_REF_CLIP_MODEL", "openai/clip-vit-base-patch32"))
    parser.add_argument("--memory-clip-device", default=os.environ.get("MINICPM_REF_CLIP_DEVICE", ""))
    parser.add_argument("--compare-baseline", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--copy-video", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.out_dir is None:
        if args.entity_grounded_memory:
            debug_name = "entity_grounded_referential_memory_model_debug"
        elif args.answer_grounded_memory:
            debug_name = "answer_grounded_referential_memory_model_debug"
        else:
            debug_name = "referential_memory_model_debug"
        args.out_dir = Path("reports/streamingbench_real_sqa_sample_1") / debug_name

    if not args.video_path.exists():
        raise FileNotFoundError(args.video_path)
    questions = _load_questions(args.questions_path, args.video_path.name, args.max_questions)
    if not questions:
        raise ValueError(f"No questions found in {args.questions_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out_dir / "selected_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    copied_video = None
    if args.copy_video:
        copied_video = args.out_dir / args.video_path.name
        if copied_video.resolve() != args.video_path.resolve():
            shutil.copy2(args.video_path, copied_video)

    qa = RecentWindowQAModel(
        model_name=args.model_name,
        device=args.device,
        max_new_tokens=args.max_qa_tokens,
        attn_implementation=args.attn_implementation,
    )
    frame_scorer = None
    if args.answer_grounded_memory or args.entity_grounded_memory:
        frame_scorer = AnswerGroundedFrameScorer(
            model_name=args.memory_clip_model,
            device=args.memory_clip_device.strip() or args.device,
        )

    memory: list[ReferentialMemoryEntry] = []
    rows: list[dict[str, Any]] = []
    result_path = args.out_dir / "results_incremental.jsonl"
    with result_path.open("w", encoding="utf-8") as handle:
        for index, question in enumerate(questions):
            ts_sec = float(timestamp_to_seconds(str(question["time_stamp"])))
            prompt = build_prompt(question)
            answer_gt = (
                extract_mcq_answer(str(question.get("answer", "")))
                or str(question.get("answer", "")).strip().upper()
            )
            window_seconds = (
                float(args.context_time)
                if args.context_time > 0
                else float(args.recent_window) * float(args.chunk_duration)
            )
            video_start = max(0.0, ts_sec - max(window_seconds, float(args.chunk_duration)))
            effective_recent_chunks = max(
                int(args.recent_window),
                int(math.ceil(window_seconds / max(float(args.chunk_duration), 1e-6))),
            )

            baseline_record: dict[str, Any] | None = None
            if args.compare_baseline:
                baseline_result, _baseline_backend = query_recent_window(
                    qa=qa,
                    video_path=str(args.video_path),
                    prompt=prompt,
                    chunk_duration=args.chunk_duration,
                    fps=args.fps,
                    recent_frames_only=effective_recent_chunks,
                    video_start=video_start,
                    video_end=ts_sec + 1e-4,
                )
                baseline_pred = extract_mcq_answer(baseline_result.answer)
                baseline_record = {
                    "response": baseline_result.answer,
                    "pred": baseline_pred,
                    "correct": bool(baseline_pred is not None and baseline_pred == answer_gt),
                    "final_chunk_ids": baseline_result.final_chunk_ids,
                    "num_frames": baseline_result.num_frames,
                    "num_vision_tokens": baseline_result.num_vision_tokens,
                }

            result, decode_backend, selection = query_referential_memory_window(
                qa=qa,
                video_path=str(args.video_path),
                prompt=prompt,
                question_text=str(question.get("question", "")),
                memory=memory,
                chunk_duration=args.chunk_duration,
                fps=args.fps,
                recent_frames_only=effective_recent_chunks,
                reference_frames=args.reference_frames,
                video_start=video_start,
                video_end=ts_sec + 1e-4,
            )
            response = result.answer
            pred = extract_mcq_answer(response)
            correct = bool(pred is not None and pred == answer_gt)
            metadata = getattr(result, "referential_memory_metadata", selection.metadata)

            memory_entry = make_memory_entry(
                question_index=index,
                question_text=str(question.get("question", "")),
                response=response,
                task_type=str(question.get("task_type", "")),
                time_stamp=str(question.get("time_stamp", "")),
                selection=selection,
                options=list(question.get("options", []) or []),
                answer_grounded=bool(args.answer_grounded_memory or args.entity_grounded_memory),
                frame_scorer=frame_scorer,
                anchor_frames=int(args.memory_anchor_frames),
                anchor_text_mode="entity" if args.entity_grounded_memory else "answer",
            )
            memory.append(memory_entry)

            sheet_name = _safe_name(
                f"{index + 1:02d}_{question.get('task_type', 'debug')}_{question.get('question', '')}"
            ) + ".jpg"
            sheet_path = frames_dir / sheet_name
            row = {
                "index": index + 1,
                "question_id": question.get("question_id"),
                "task_type": str(question.get("task_type", "")),
                "time_stamp": str(question.get("time_stamp", "")),
                "question": str(question.get("question", "")),
                "answer_gt": answer_gt,
                "response": response,
                "pred": pred,
                "correct": correct,
                "decode_backend": decode_backend,
                "recent_window": int(args.recent_window),
                "reference_frames_requested": int(args.reference_frames),
                "final_chunk_ids": result.final_chunk_ids,
                "num_frames": result.num_frames,
                "num_vision_tokens": result.num_vision_tokens,
                "referential_memory": metadata,
                "stored_memory_entry": _memory_entry_to_json(memory_entry),
                "baseline": baseline_record,
                "sheet_rel": f"selected_frames/{sheet_name}",
            }
            _make_sheet(
                selection.frames,
                _labels_from_metadata(metadata, memory_entry=memory_entry),
                sheet_path,
                title_lines=_sheet_title_lines(row),
            )
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{index + 1}/{len(questions)}] {question['time_stamp']} "
                f"ref={metadata.get('memory_triggered')} -> {response} (gt={answer_gt})",
                flush=True,
            )

    _write_html(
        args.out_dir / "index.html",
        video_path=args.video_path,
        copied_video=copied_video,
        rows=rows,
    )
    print(f"video: {args.video_path}")
    if copied_video is not None:
        print(f"copied_video: {copied_video}")
    print(f"questions: {len(rows)}")
    print(f"results: {result_path}")
    print(f"html: {args.out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
