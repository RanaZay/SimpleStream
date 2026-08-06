#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from lib.minicpm.recent_description import RecentDescriptionQAModel  # noqa: E402
from lib.shared.recent_window import (  # noqa: E402
    build_ovo_prompt,
    decode_video_to_chunks_qwen,
    extract_mcq_answer,
)


def _safe_name(text: str, max_len: int = 110) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return text[:max_len] or "sample"


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = ["DejaVuSans-Bold.ttf", "Arial Bold.ttf"] if bold else ["DejaVuSans.ttf", "Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size=size)
        except OSError:
            continue
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


def _find_annotation(
    annotations: list[dict[str, Any]],
    *,
    chunked_dir: Path,
    sample_id: str | None,
    task: str | None,
) -> dict[str, Any]:
    for anno in annotations:
        if "test_info" in anno:
            continue
        if sample_id and str(anno.get("id")) != str(sample_id):
            continue
        if task and str(anno.get("task")) != str(task):
            continue
        video_path = chunked_dir / f"{anno['id']}.mp4"
        if video_path.exists():
            return anno
    filters = []
    if sample_id:
        filters.append(f"id={sample_id}")
    if task:
        filters.append(f"task={task}")
    suffix = f" matching {' '.join(filters)}" if filters else ""
    raise FileNotFoundError(f"No single-video OVO annotation with an existing mp4 was found{suffix}.")


def _make_sheet(
    out_path: Path,
    *,
    frames: list[Image.Image],
    chunk_ids: list[int],
    notes: list[str],
    title_lines: list[str],
) -> None:
    cols = min(3, max(1, len(frames)))
    rows = (len(frames) + cols - 1) // cols
    pad = 18
    thumb_w, thumb_h = 320, 190
    label_h = 150
    title_h = 44 + 26 * len(title_lines)
    width = cols * thumb_w + (cols + 1) * pad
    height = title_h + rows * (thumb_h + label_h + pad) + pad
    sheet = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(sheet)
    title_font = _load_font(20, bold=True)
    body_font = _load_font(14)
    small_font = _load_font(12)

    draw.rounded_rectangle([pad, pad, width - pad, title_h - 6], radius=12, fill=(255, 255, 255), outline=(203, 213, 225), width=2)
    for i, line in enumerate(title_lines):
        draw.text((pad + 14, pad + 12 + i * 26), line, fill=(15, 23, 42), font=title_font if i == 0 else body_font)

    y0 = title_h
    for idx, frame in enumerate(frames):
        row, col = divmod(idx, cols)
        x = pad + col * (thumb_w + pad)
        y = y0 + row * (thumb_h + label_h + pad)
        image = frame.copy()
        image.thumbnail((thumb_w, thumb_h))
        bg = Image.new("RGB", (thumb_w, thumb_h), (226, 232, 240))
        bg.paste(image, ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2))
        sheet.paste(bg, (x, y))
        draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=(37, 99, 235), width=3)
        label_y = y + thumb_h + 7
        draw.rounded_rectangle([x, label_y, x + thumb_w, label_y + label_h - 8], radius=10, fill=(239, 246, 255), outline=(203, 213, 225))
        label_lines = [
            f"Frame {idx + 1} | OVO chunk {chunk_ids[idx] if idx < len(chunk_ids) else '?'}",
            f"Generated note: {notes[idx] if idx < len(notes) else '(missing)'}",
        ]
        cursor_y = label_y + 10
        for j, line in enumerate(label_lines):
            wrapped = _wrap(line, 42)[:5 if j else 1]
            for wrapped_line in wrapped:
                draw.text((x + 10, cursor_y), wrapped_line, fill=(15, 23, 42), font=body_font if j == 0 else small_font)
                cursor_y += 17
            cursor_y += 4
    sheet.save(out_path)


def _write_html(out_path: Path, *, row: dict[str, Any], copied_video: Path | None) -> None:
    video_block = ""
    if copied_video is not None:
        video_block = f'<video controls src="{html.escape(copied_video.name)}"></video>'
    notes = "\n".join(
        f"Frame {idx + 1} chunk={chunk}: {note}"
        for idx, (chunk, note) in enumerate(zip(row["chunk_ids"], row["notes"]))
    )
    out_path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>OVO Proposal 2 Debug</title>
  <style>
    body {{ color: #172033; font-family: Arial, sans-serif; line-height: 1.45; margin: 28px; max-width: 1180px; }}
    video {{ border: 1px solid #cbd5e1; border-radius: 8px; max-width: 100%; width: 900px; }}
    img {{ border: 1px solid #cbd5e1; border-radius: 8px; max-width: 100%; }}
    pre {{ background: #f5f7fb; border-radius: 8px; overflow-x: auto; padding: 12px; white-space: pre-wrap; }}
    .pill {{ background: #e0f2fe; border-radius: 999px; display: inline-block; margin-right: 8px; padding: 3px 10px; }}
    summary {{ cursor: pointer; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>OVO Proposal 2: Recent Frames + Generated Descriptions</h1>
  <p>
    <span class="pill">sample: {html.escape(str(row["id"]))}</span>
    <span class="pill">task: {html.escape(str(row["task"]))}</span>
    <span class="pill">GT: {html.escape(str(row["ground_truth"]))}</span>
    <span class="pill">prediction: {html.escape(str(row["prediction"]))}</span>
    <span class="pill">correct: {html.escape(str(row["correct"]))}</span>
  </p>
  <p><b>Question:</b> {html.escape(str(row["question"]))}</p>
  <p><b>Response:</b> {html.escape(str(row["response"]))}</p>
  <p><b>Selected chunks:</b> {html.escape(str(row["chunk_ids"]))}</p>
  {video_block}
  <h2>Selected Recent Frames And Their Generated Notes</h2>
  <img src="{html.escape(row["sheet_name"])}" alt="selected recent frames">
  <details open><summary>Generated frame descriptions</summary><pre>{html.escape(notes)}</pre></details>
  <details open><summary>Final prompt passed to MiniCPM</summary><pre>{html.escape(row["final_prompt"])}</pre></details>
  <details><summary>Raw JSON</summary><pre>{html.escape(json.dumps(row, indent=2, ensure_ascii=False, default=str))}</pre></details>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize one OVO sample using Proposal 2: recent visual frames plus generated descriptions."
    )
    parser.add_argument("--anno-path", type=Path, default=Path("data/ovo_bench/ovo_bench_new.json"))
    parser.add_argument("--chunked-dir", type=Path, default=Path("data/ovo_bench/chunked_videos"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/ovo_recent_description_debug"))
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--model", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"))
    parser.add_argument("--recent-frames", type=int, default=6)
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-qa-tokens", type=int, default=256)
    parser.add_argument("--copy-video", action="store_true")
    args = parser.parse_args()

    if not args.anno_path.exists():
        raise FileNotFoundError(args.anno_path)
    if not args.chunked_dir.exists():
        raise FileNotFoundError(args.chunked_dir)

    with args.anno_path.open(encoding="utf-8") as handle:
        annotations = json.load(handle)
    if not isinstance(annotations, list):
        raise ValueError(f"Expected a list of OVO annotations in {args.anno_path}")

    anno = _find_annotation(
        annotations,
        chunked_dir=args.chunked_dir,
        sample_id=args.sample_id,
        task=args.task,
    )
    video_path = args.chunked_dir / f"{anno['id']}.mp4"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    copied_video = None
    if args.copy_video:
        copied_video = args.out_dir / video_path.name
        if copied_video.resolve() != video_path.resolve():
            shutil.copy2(video_path, copied_video)

    qa = RecentDescriptionQAModel(
        model_name=args.model,
        device=args.device,
        max_new_tokens=args.max_qa_tokens,
        attn_implementation=args.attn_implementation,
    )
    chunks, decode_backend = decode_video_to_chunks_qwen(
        video_path=str(video_path),
        chunk_duration=args.chunk_duration,
        fps=args.fps,
        recent_frames_only=args.recent_frames,
    )
    if not chunks:
        raise ValueError(f"No decoded frames for {video_path}")

    recent_chunks = chunks[-max(1, args.recent_frames):]
    frames = [frame for chunk in recent_chunks for frame in chunk.frames]
    chunk_ids = [chunk.chunk_index for chunk in recent_chunks]

    prompt = build_ovo_prompt(str(anno["task"]), anno)
    notes, description_time = qa.describe_recent_frames(frames)
    final_prompt = qa.build_recent_description_prompt(
        original_prompt=prompt,
        notes=notes,
        chunk_ids=chunk_ids,
    )
    response = qa.generate_from_frames(frames, final_prompt)
    prediction = extract_mcq_answer(response)
    ground_truth = chr(65 + int(anno["gt"])) if "gt" in anno else None
    correct = bool(prediction and ground_truth and prediction == ground_truth)

    sheet_name = f"{_safe_name(str(anno['id']))}_proposal2_selected_frames.jpg"
    _make_sheet(
        args.out_dir / sheet_name,
        frames=frames,
        chunk_ids=chunk_ids,
        notes=notes,
        title_lines=[
            "OVO Proposal 2: recent frames plus generated descriptions",
            f"sample={anno['id']} | task={anno['task']} | chunks={chunk_ids}",
        ],
    )

    row = {
        "id": anno.get("id"),
        "video": anno.get("video"),
        "task": anno.get("task"),
        "question": anno.get("question"),
        "ground_truth": ground_truth,
        "prediction": prediction,
        "correct": correct,
        "response": response,
        "chunk_ids": chunk_ids,
        "notes": notes,
        "final_prompt": final_prompt,
        "decode_backend": decode_backend,
        "description_time_seconds": description_time,
        "num_frames": len(frames),
        "num_vision_tokens": getattr(qa, "_last_num_vision_tokens", None),
        "sheet_name": sheet_name,
    }
    with (args.out_dir / "debug_record.json").open("w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2, ensure_ascii=False)
    _write_html(args.out_dir / "index.html", row=row, copied_video=copied_video)
    print(f"Wrote: {args.out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
