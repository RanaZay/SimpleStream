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

from lib.minicpm.recent_description import RecentDescriptionQAModel  # noqa: E402
from lib.shared.recent_window import decode_video_to_chunks_qwen, extract_mcq_answer  # noqa: E402
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


def _load_questions(path: Path, max_questions: int, question_id_prefix: str | None) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]

    questions: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = dict(row)
        item.setdefault("question_id", item.get("id", f"debug_{index}"))
        if question_id_prefix and not str(item.get("question_id", "")).startswith(question_id_prefix):
            continue
        item.setdefault("task_type", item.get("task", "Debug"))
        item.setdefault("time_stamp", item.get("timestamp", "00:00:00"))
        item["options"] = _normalise_options(item.get("options", []))
        questions.append(item)
    questions.sort(key=lambda item: timestamp_to_seconds(str(item.get("time_stamp", "00:00:00"))))
    return questions[:max_questions] if max_questions > 0 else questions


def _safe_name(text: str, max_len: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return text[:max_len] or "question"


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size=size)
    except OSError:
        return ImageFont.load_default()


def _wrap(text: str, width: int) -> list[str]:
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


def _make_recent_description_sheet(
    entries: list[dict[str, Any]],
    out_path: Path,
    *,
    title_lines: list[str],
) -> None:
    cols = min(3, max(1, len(entries)))
    rows = int(math.ceil(max(1, len(entries)) / cols))
    pad = 16
    thumb_w, thumb_h = 300, 188
    label_h = 150
    title_h = 36 + 26 * len(title_lines)
    width = cols * thumb_w + (cols + 1) * pad
    height = title_h + rows * (thumb_h + label_h + pad) + pad

    sheet = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(sheet)
    title_font = _load_font(18, bold=True)
    body_font = _load_font(13)
    small_font = _load_font(12)

    draw.rounded_rectangle(
        [pad, pad, width - pad, title_h - 6],
        radius=10,
        fill=(255, 255, 255),
        outline=(203, 213, 225),
        width=2,
    )
    for i, line in enumerate(title_lines):
        draw.text(
            (pad + 14, pad + 10 + i * 26),
            line,
            fill=(15, 23, 42),
            font=title_font if i == 0 else body_font,
        )

    y0 = title_h
    for idx, entry in enumerate(entries):
        row, col = divmod(idx, cols)
        x = pad + col * (thumb_w + pad)
        y = y0 + row * (thumb_h + label_h + pad)

        frame: Image.Image = entry["frame"]
        image = frame.copy()
        image.thumbnail((thumb_w, thumb_h))
        bg = Image.new("RGB", (thumb_w, thumb_h), (226, 232, 240))
        bg.paste(image, ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2))
        sheet.paste(bg, (x, y))
        draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=(37, 99, 235), width=3)

        label_y = y + thumb_h + 8
        draw.rounded_rectangle(
            [x, label_y, x + thumb_w, label_y + label_h - 8],
            radius=8,
            fill=(239, 246, 255),
            outline=(191, 219, 254),
            width=2,
        )
        header = f"Frame {idx + 1} | chunk={entry['chunk_id']} | t={entry['timestamp']:.1f}s"
        draw.text((x + 10, label_y + 10), header, fill=(30, 64, 175), font=body_font)
        draw.text((x + 10, label_y + 34), "Generated description:", fill=(15, 23, 42), font=small_font)
        for j, line in enumerate(_wrap(entry["description"], 38)[:6]):
            draw.text((x + 10, label_y + 54 + j * 15), line, fill=(15, 23, 42), font=small_font)

    sheet.save(out_path)


def _write_html(
    out_path: Path,
    *,
    video_path: Path,
    copied_video: Path | None,
    rows: list[dict[str, Any]],
) -> None:
    video_block = ""
    if copied_video is not None:
        video_block = f'<video controls src="{html.escape(copied_video.name)}"></video>'

    sections: list[str] = []
    for row in rows:
        notes_text = "\n".join(
            f"Frame {i + 1} chunk={item['chunk_id']} t={item['timestamp']:.1f}s: {item['description']}"
            for i, item in enumerate(row["recent_entries"])
        )
        sections.append(
            f"""
<section>
  <h2>Q{row['index']}. {html.escape(row['task_type'])}</h2>
  <p><b>Time:</b> {html.escape(row['time_stamp'])}
     <b>GT:</b> {html.escape(row['answer_gt'])}
     <b>Response:</b> {html.escape(row['response'])}
     <b>Pred:</b> {html.escape(str(row['pred']))}
     <b>Correct:</b> {html.escape(str(row['correct']))}</p>
  <p><b>Question:</b> {html.escape(row['question'])}</p>
  <p><b>Recent visual chunks:</b> {html.escape(str(row['recent_chunk_ids']))}</p>
  <img src="{html.escape(row['sheet_rel'])}" alt="recent description debug">
  <details open><summary>Generated Recent-Frame Descriptions</summary><pre>{html.escape(notes_text)}</pre></details>
  <details open><summary>Exact Final Prompt Passed To MiniCPM</summary><pre>{html.escape(row['final_prompt'])}</pre></details>
  <details><summary>Raw JSON</summary><pre>{html.escape(json.dumps(row['json'], indent=2, ensure_ascii=False))}</pre></details>
</section>
"""
        )

    out_path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Recent Description Proposal Debug</title>
  <style>
    body {{ color: #162033; font-family: Arial, sans-serif; line-height: 1.45; margin: 28px; max-width: 1120px; }}
    video {{ border: 1px solid #cbd5e1; border-radius: 8px; max-width: 100%; width: 900px; }}
    section {{ border-bottom: 1px solid #d8dee8; margin-bottom: 36px; padding-bottom: 28px; }}
    img {{ border: 1px solid #cbd5e1; border-radius: 8px; max-width: 100%; }}
    pre {{ background: #f5f7fb; border-radius: 8px; overflow-x: auto; padding: 12px; white-space: pre-wrap; }}
    summary {{ cursor: pointer; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>MiniCPM Proposal 2: Recent Frames + Generated Descriptions</h1>
  <p><b>Video:</b> {html.escape(str(video_path))}</p>
  {video_block}
  {''.join(sections)}
</body>
</html>
""",
        encoding="utf-8",
    )


def _chunk_timestamp(chunk: Any) -> float:
    values = getattr(chunk, "frame_timestamps", None)
    if values:
        return float(values[-1])
    return float(getattr(chunk, "start_time", 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize Proposal 2 on one real StreamingBench video: recent frames plus generated recent-frame descriptions."
    )
    parser.add_argument("--video-path", type=Path, default=Path("reports/streamingbench_real_sqa_sample_1/sample_1_sqa.mp4"))
    parser.add_argument("--questions-path", type=Path, default=Path("reports/streamingbench_real_sqa_sample_1/Sequential_Question_Answering.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/streamingbench_real_sqa_sample_1/recent_description_debug"))
    parser.add_argument("--model", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"))
    parser.add_argument("--max-questions", type=int, default=3)
    parser.add_argument("--question-id-prefix", default="", help="Optional question_id prefix filter.")
    parser.add_argument("--recent-frames", type=int, default=6)
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--copy-video", action="store_true")
    args = parser.parse_args()

    if not args.video_path.exists():
        raise FileNotFoundError(args.video_path)
    if not args.questions_path.exists():
        raise FileNotFoundError(args.questions_path)

    os.environ["MINICPM_RECENT_DESC_FRAMES"] = str(args.recent_frames)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_dir = args.out_dir / "selected_frames"
    selected_dir.mkdir(parents=True, exist_ok=True)

    copied_video = None
    if args.copy_video:
        copied_video = args.out_dir / args.video_path.name
        if copied_video.resolve() != args.video_path.resolve():
            shutil.copy2(args.video_path, copied_video)

    qa = RecentDescriptionQAModel(
        model_name=args.model,
        device=args.device,
        max_new_tokens=256,
        attn_implementation=args.attn_implementation,
    )

    question_id_prefix = args.question_id_prefix or None
    questions = _load_questions(args.questions_path, args.max_questions, question_id_prefix)
    if not questions:
        raise ValueError(f"No questions matched {args.questions_path}")

    rows: list[dict[str, Any]] = []
    jsonl_path = args.out_dir / "recent_description_debug_records.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for q_index, question in enumerate(questions, start=1):
            ts_sec = float(timestamp_to_seconds(str(question.get("time_stamp", "00:00:00"))))
            chunks, _backend = decode_video_to_chunks_qwen(
                video_path=str(args.video_path),
                chunk_duration=args.chunk_duration,
                fps=args.fps,
                recent_frames_only=None,
                video_start=0.0,
                video_end=ts_sec + 1e-4,
            )
            if not chunks:
                continue

            recent_chunks = list(chunks[-max(1, args.recent_frames) :])
            recent_frames = [frame for chunk in recent_chunks for frame in chunk.frames]
            recent_chunk_ids = [chunk.chunk_index for chunk in recent_chunks]

            notes, description_time = qa.describe_recent_frames(recent_frames)
            prompt = build_prompt(question)
            final_prompt = qa.build_recent_description_prompt(
                original_prompt=prompt,
                notes=notes,
                chunk_ids=recent_chunk_ids,
            )
            response = qa.generate_from_frames(recent_frames, final_prompt)
            pred = extract_mcq_answer(response)
            answer_gt = str(question.get("answer", "")).strip()
            correct = bool(pred is not None and pred == answer_gt)

            recent_entries = []
            for chunk, note in zip(recent_chunks, notes):
                if not chunk.frames:
                    continue
                recent_entries.append(
                    {
                        "chunk_id": chunk.chunk_index,
                        "timestamp": _chunk_timestamp(chunk),
                        "description": note,
                        "frame": chunk.frames[-1],
                    }
                )

            sheet_name = f"{q_index:02d}_{_safe_name(question.get('question', 'question'))}.jpg"
            sheet_path = selected_dir / sheet_name
            _make_recent_description_sheet(
                recent_entries,
                sheet_path,
                title_lines=[
                    f"Q{q_index} | t={question.get('time_stamp')} | Correct: {correct}",
                    f"Proposal 2: recent-{args.recent_frames} frames + generated descriptions",
                    f"Response: {response} | GT: {answer_gt}",
                    f"Question: {question.get('question')}",
                ],
            )

            serial_entries = [
                {
                    "chunk_id": item["chunk_id"],
                    "timestamp": item["timestamp"],
                    "description": item["description"],
                }
                for item in recent_entries
            ]
            record = {
                "index": q_index,
                "task_type": str(question.get("task_type", "")),
                "question": str(question.get("question", "")),
                "time_stamp": str(question.get("time_stamp", "")),
                "answer_gt": answer_gt,
                "response": response,
                "pred": pred,
                "correct": correct,
                "recent_chunk_ids": recent_chunk_ids,
                "recent_entries": serial_entries,
                "description_time_seconds": description_time,
                "final_prompt": final_prompt,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows.append(
                {
                    **record,
                    "sheet_rel": f"selected_frames/{sheet_name}",
                    "json": record,
                }
            )
            print(
                f"Q{q_index}: response={response!r} gt={answer_gt!r} "
                f"correct={correct} desc_time={description_time:.2f}s sheet={sheet_path}"
            )

    _write_html(
        args.out_dir / "index.html",
        video_path=args.video_path,
        copied_video=copied_video,
        rows=rows,
    )
    print(f"\nWrote: {args.out_dir / 'index.html'}")
    print(f"Wrote: {jsonl_path}")


if __name__ == "__main__":
    main()
