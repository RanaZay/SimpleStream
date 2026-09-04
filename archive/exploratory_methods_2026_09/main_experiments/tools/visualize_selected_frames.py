#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import html
import json
import math
import os
import re
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def _load_records(result_dir: Path) -> list[dict[str, Any]]:
    paths = []
    direct = result_dir / "results_incremental.jsonl"
    if direct.exists():
        paths.append(direct)
    paths.extend(sorted(result_dir.glob("rank_*/results_incremental.jsonl")))
    if not paths:
        raise FileNotFoundError(f"No results_incremental.jsonl files found under {result_dir}")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = str(row.get("_key") or row.get("_index") or len(records))
                if key in seen:
                    continue
                seen.add(key)
                row["_source_file"] = str(path)
                records.append(row)
    records.sort(key=lambda item: int(item.get("_index", 0)))
    return records


def _adaptive_meta(record: dict[str, Any]) -> dict[str, Any]:
    profile = record.get("profile")
    if isinstance(profile, dict) and isinstance(profile.get("adaptive"), dict):
        return profile["adaptive"]
    adaptive = record.get("adaptive")
    if isinstance(adaptive, dict):
        return adaptive
    referential = record.get("referential_memory")
    if isinstance(referential, dict):
        return referential
    return {}


def _safe_name(text: str, max_len: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return text[:max_len] or "example"


def _resolve_video_path(record: dict[str, Any], video_dir: Path) -> Path:
    candidates: list[Path] = []
    explicit = record.get("_video_path_override")
    if isinstance(explicit, str) and explicit:
        candidates.append(Path(explicit))
    for key in ("video_path", "video_path_raw", "video"):
        value = record.get(key)
        if isinstance(value, str) and value:
            p = Path(value)
            if p.is_absolute():
                candidates.append(p)
            candidates.append(video_dir / p.name)
            candidates.append(video_dir / value)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not resolve video for record "
        f"{record.get('_key') or record.get('_index')}: tried {[str(p) for p in candidates[:8]]}"
    )


def _decode_frames(video_path: Path, timestamps: list[float]) -> list[Image.Image]:
    if not timestamps:
        return []
    try:
        import decord  # type: ignore

        reader = decord.VideoReader(str(video_path))
        fps = float(reader.get_avg_fps())
        total = len(reader)
        indices = [min(max(int(round(ts * fps)), 0), total - 1) for ts in timestamps]
        batch = reader.get_batch(indices).asnumpy()
        return [Image.fromarray(frame).convert("RGB") for frame in batch]
    except Exception:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install decord or opencv-python to extract video frames.") from exc

        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or total <= 0:
            cap.release()
            raise RuntimeError(f"Could not read FPS/frame count from {video_path}")
        frames: list[Image.Image] = []
        for ts in timestamps:
            idx = min(max(int(round(ts * fps)), 0), total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame).convert("RGB"))
        cap.release()
        return frames


def _make_sheet(
    frames: list[Image.Image],
    labels: list[str],
    out_path: Path,
    thumb_width: int = 240,
    thumb_height: int = 150,
    title: str = "",
    subtitle: str = "",
    status: str = "",
    status_fill: tuple[int, int, int] = (244, 248, 252),
) -> None:
    if not frames:
        return
    cols = min(4, len(frames))
    rows = int(math.ceil(len(frames) / cols))
    pad = 12
    label_h = 42
    width = cols * thumb_width + (cols + 1) * pad
    font = ImageFont.load_default()
    title_lines = textwrap.wrap(title, width=115) if title else []
    subtitle_lines = textwrap.wrap(subtitle, width=115) if subtitle else []
    status_h = 24 if status else 0
    header_lines = title_lines + subtitle_lines
    header_h = (len(header_lines) * 15 + pad + status_h) if header_lines or status else 0
    height = header_h + rows * (thumb_height + label_h) + (rows + 1) * pad
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    y_offset = 0
    if header_lines or status:
        draw.rectangle([0, 0, width, header_h], fill=(244, 248, 252))
        y_text = 0
        if status:
            draw.rectangle([0, 0, width, status_h], fill=status_fill)
            draw.text((pad, 6), status, fill=(255, 255, 255), font=font)
            y_text = status_h + 8
        else:
            y_text = 8
        for line_index, line in enumerate(header_lines):
            fill = (18, 35, 55) if line_index < len(title_lines) else (55, 70, 90)
            draw.text((pad, y_text), line, fill=fill, font=font)
            y_text += 15
        y_offset = header_h

    for i, frame in enumerate(frames):
        row, col = divmod(i, cols)
        x = pad + col * (thumb_width + pad)
        y = y_offset + pad + row * (thumb_height + label_h + pad)
        image = frame.copy()
        image.thumbnail((thumb_width, thumb_height))
        bg = Image.new("RGB", (thumb_width, thumb_height), (245, 247, 250))
        ix = (thumb_width - image.width) // 2
        iy = (thumb_height - image.height) // 2
        bg.paste(image, (ix, iy))
        sheet.paste(bg, (x, y))
        draw.rectangle([x, y, x + thumb_width, y + thumb_height], outline=(70, 85, 105), width=1)
        draw.text((x, y + thumb_height + 6), labels[i], fill=(20, 30, 45), font=font)
    sheet.save(out_path)


def _format_score(score: Any) -> str:
    if not isinstance(score, dict):
        return ""
    keys = [
        "chunk_id",
        "selected",
        "question_aware_memory_score",
        "question_window_similarity",
        "semantic_memory_score",
        "bound_memory_score",
        "online_memory_score",
        "episodic_score",
        "event_change_norm",
        "text_detail_norm",
        "temporal_relevance_score",
        "temporal_position",
    ]
    compact = {key: score[key] for key in keys if key in score}
    return json.dumps(compact, indent=2, ensure_ascii=False)


def _format_options(record: dict[str, Any]) -> str:
    options = record.get("options", record.get("choices"))
    if isinstance(options, str):
        try:
            parsed = ast.literal_eval(options)
            if isinstance(parsed, list):
                options = parsed
        except Exception:
            pass
    if isinstance(options, dict):
        items = []
        for key in sorted(options):
            items.append(f"{key}. {options[key]}")
        return " | ".join(items)
    if isinstance(options, list):
        labels = ["A", "B", "C", "D", "E", "F"]
        items = []
        for idx, value in enumerate(options):
            text = str(value)
            if re.match(r"^[A-Z][.)]\s+", text):
                items.append(text)
            else:
                prefix = labels[idx] if idx < len(labels) else str(idx + 1)
                items.append(f"{prefix}. {text}")
        return " | ".join(items)
    return ""


def _pick_records(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    needle = args.question_contains.lower() if args.question_contains else ""
    for record in records:
        meta = _adaptive_meta(record)
        if args.only == "correct" and not record.get("correct"):
            continue
        if args.only == "incorrect" and record.get("correct"):
            continue
        if args.memory == "triggered" and not meta.get("memory_triggered"):
            continue
        if args.memory == "not-triggered" and meta.get("memory_triggered"):
            continue
        if args.task_type and record.get("task_type") != args.task_type:
            continue
        if needle and needle not in str(record.get("question", "")).lower():
            continue
        if args.index is not None and int(record.get("_index", -1)) != args.index:
            continue
        picked.append(record)
        if len(picked) >= args.num_examples:
            break
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and visualize the exact frames selected by adaptive MiniCPM StreamingBench runs."
    )
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--video-dir", type=Path, default=Path("data/streamingbench/videos"))
    parser.add_argument("--video-path", type=Path, help="Fallback video path when records do not store video names.")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/selected_frames_debug"))
    parser.add_argument("--num-examples", type=int, default=8)
    parser.add_argument("--only", choices=["all", "correct", "incorrect"], default="incorrect")
    parser.add_argument("--memory", choices=["all", "triggered", "not-triggered"], default="all")
    parser.add_argument("--task-type", default="")
    parser.add_argument("--question-contains", default="")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()

    records = _load_records(args.result_dir)
    picked = _pick_records(records, args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    for n, record in enumerate(picked, start=1):
        if args.video_path is not None:
            record["_video_path_override"] = str(args.video_path)
        meta = _adaptive_meta(record)
        timestamps = [float(x) for x in meta.get("selected_timestamps", [])]
        chunk_ids = [int(x) for x in meta.get("selected_chunk_ids", record.get("final_chunk_ids", []))]
        recent_ids = set(int(x) for x in meta.get("recent_chunk_ids", []))
        memory_ids = set(int(x) for x in meta.get("memory_chunk_ids", []))
        video_path = _resolve_video_path(record, args.video_dir)
        frames = _decode_frames(video_path, timestamps)

        labels = []
        for i, chunk_id in enumerate(chunk_ids):
            role = "memory" if chunk_id in memory_ids else "recent" if chunk_id in recent_ids else "selected"
            ts = timestamps[i] if i < len(timestamps) else float("nan")
            labels.append(f"{i + 1}. {role} | chunk {chunk_id} | {ts:.1f}s")

        name = _safe_name(f"{n:02d}_{record.get('_index')}_{record.get('task_type')}_{record.get('video')}")
        sheet_path = args.out_dir / f"{name}.jpg"
        is_correct = bool(record.get("correct"))
        pred = record.get("response", record.get("pred"))
        gt = record.get("answer_gt", record.get("answer"))
        options_text = _format_options(record)
        _make_sheet(
            frames,
            labels,
            sheet_path,
            title=f"Q: {record.get('question', '')}",
            subtitle=(
                f"Predicted: {pred} | Ground truth: {gt} | "
                f"Task: {record.get('task_type', '')} | "
                f"t={record.get('time_stamp', '')}"
                + (f" | Options: {options_text}" if options_text else "")
            ),
            status=f"{'CORRECT' if is_correct else 'WRONG'}   Predicted: {pred}   GT: {gt}",
            status_fill=(34, 139, 86) if is_correct else (196, 55, 55),
        )

        selected_scores = []
        for score in meta.get("memory_scores", []):
            if isinstance(score, dict) and score.get("selected"):
                selected_scores.append(_format_score(score))
        selected_score_text = "\n\n".join(selected_scores) if selected_scores else "No selected memory scores."

        rows.append(
            f"""
            <section>
              <h2>{n}. index={html.escape(str(record.get('_index')))} | {html.escape(str(record.get('task_type', '')))}</h2>
              <p><b>Video:</b> {html.escape(str(video_path))}</p>
              <p><b>Time:</b> {html.escape(str(record.get('time_stamp', '')))}
                 <b>Correct:</b> {html.escape(str(record.get('correct')))}
                 <b>GT:</b> {html.escape(str(record.get('answer_gt')))}
                 <b>Response:</b> {html.escape(str(record.get('response')))}</p>
              <p><b>Question:</b> {html.escape(str(record.get('question')))}</p>
              <p><b>Options:</b> {html.escape(options_text or 'N/A')}</p>
              <p><b>Mode:</b> {html.escape(str(meta.get('mode')))}
                 <b>Gate:</b> {html.escape(json.dumps(meta.get('memory_gate'), ensure_ascii=False))}
                 <b>Triggered:</b> {html.escape(str(meta.get('memory_triggered')))}</p>
              <p><b>Recent chunks:</b> {html.escape(str(sorted(recent_ids)))}
                 <b>Memory chunks:</b> {html.escape(str(sorted(memory_ids)))}
                 <b>Selected chunks:</b> {html.escape(str(chunk_ids))}</p>
              <img src="{html.escape(sheet_path.name)}" style="max-width:100%;border:1px solid #ccd3dd;border-radius:8px;">
              <details>
                <summary>Selected memory scores</summary>
                <pre>{html.escape(selected_score_text)}</pre>
              </details>
            </section>
            """
        )

        print("=" * 88)
        print(f"[{n}] index={record.get('_index')} task={record.get('task_type')} correct={record.get('correct')}")
        print(f"video={video_path}")
        print(f"question={record.get('question')}")
        print(f"response={record.get('response')} gt={record.get('answer_gt')}")
        print(f"recent={sorted(recent_ids)} memory={sorted(memory_ids)} selected={chunk_ids}")
        print(f"sheet={sheet_path}")

    html_path = args.out_dir / "index.html"
    html_path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Selected Frame Debug</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #142033; }}
    section {{ margin-bottom: 38px; padding-bottom: 26px; border-bottom: 1px solid #d9dee8; }}
    h1, h2 {{ margin-bottom: 8px; }}
    p {{ line-height: 1.45; }}
    pre {{ background: #f5f7fb; padding: 12px; border-radius: 8px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>Selected Frame Debug</h1>
  <p><b>Result dir:</b> {html.escape(str(args.result_dir))}</p>
  <p><b>Filters:</b> only={args.only}, memory={args.memory}, task_type={html.escape(args.task_type or 'any')}</p>
  {''.join(rows)}
</body>
</html>
""",
        encoding="utf-8",
    )
    print("=" * 88)
    print(f"records={len(records)} shown={len(picked)}")
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
