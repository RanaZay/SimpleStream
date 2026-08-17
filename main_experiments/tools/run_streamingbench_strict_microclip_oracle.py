#!/usr/bin/env python3
"""Strict frame-level microclip oracle for a saved StreamingBench-100 run.

This tool does not modify PRISM. It reuses the saved PRISM candidate queue,
then evaluates controller-free contexts:

  K0: Recent-6
  Anchor: exactly one top-anchor image + Recent-6
  Exact Microclip: previous + anchor + next historical images + Recent-6
  Random/local: deterministic random local historical packet + Recent-6
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

HISTORY_INSTRUCTION = (
    "Historical evidence appears before the six recent frames.\n\n"
)
MICROCLIP_INSTRUCTION = (
    "A short historical event clip appears before the six recent frames.\n"
    "The historical clip frames are ordered from earlier to later.\n\n"
)

build_prompt = None
decode_video_to_chunks_qwen = None
resolve_video_path = None
select_recent_window_frames = None
timestamp_to_seconds = None
RecentWindowQAModel = None


@dataclass(frozen=True)
class FrameRecord:
    image: Image.Image
    timestamp: float
    chunk_id: int
    frame_index: int

    @property
    def key(self) -> tuple[int, int, float]:
        return (self.chunk_id, self.frame_index, round(self.timestamp, 6))


def default_qa_device() -> str:
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_results(path: Path) -> tuple[str, list[dict[str, Any]]]:
    if path.is_file():
        payload = json.load(path.open(encoding="utf-8"))
        if isinstance(payload, dict):
            return str(path), list(payload.get("results", []))
        return str(path), list(payload)

    merged = sorted(
        path.glob("streaming_bench_minicpmv46_results_*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if merged:
        return load_results(merged[0])

    rank_files = sorted(path.glob("rank_*/results_incremental.jsonl"))
    if not rank_files:
        raise FileNotFoundError(f"No StreamingBench results found under {path}")
    rows: list[dict[str, Any]] = []
    for rank_file in rank_files:
        rows.extend(read_jsonl(rank_file))
    dedup: dict[Any, dict[str, Any]] = {}
    for row in rows:
        dedup[row.get("_index", row.get("_key"))] = row
    return "\n  ".join(str(item) for item in rank_files), sorted(
        dedup.values(),
        key=lambda item: int(item.get("_index", 0)),
    )


def adaptive(row: dict[str, Any]) -> dict[str, Any]:
    profile = row.get("profile") if isinstance(row.get("profile"), dict) else {}
    return row.get("adaptive") or profile.get("adaptive") or {}


def load_annotations(path: Path, video_dir: str) -> dict[int, dict[str, Any]]:
    if build_prompt is None or resolve_video_path is None or timestamp_to_seconds is None:
        raise RuntimeError("StreamingBench helpers were not initialized")
    data = json.load(path.open(encoding="utf-8"))
    tasks: list[dict[str, Any]] = []
    for entry in data:
        video_path_raw = entry["video_path"]
        video_path = resolve_video_path(video_path_raw, video_dir)
        questions = sorted(entry.get("questions", []), key=lambda item: timestamp_to_seconds(item.get("time_stamp")))
        for question in questions:
            tasks.append(
                {
                    "_index": len(tasks),
                    "video": entry.get("video_id") or Path(video_path_raw).stem,
                    "video_path": video_path,
                    "video_path_raw": video_path_raw,
                    "question_obj": question,
                    "prompt": build_prompt(question),
                    "task_type": question.get("task_type") or entry.get("task_type"),
                }
            )
    return {int(task["_index"]): task for task in tasks}


def extract_mcq_answer(text: Any) -> str | None:
    if text is None:
        return None
    match = re.search(r"\b([A-E])\b", str(text).upper())
    return match.group(1) if match else None


def answer_gt_from(saved: dict[str, Any], question: dict[str, Any]) -> str | None:
    for value in (
        saved.get("answer_gt"),
        saved.get("ground_truth"),
        question.get("answer"),
        question.get("answer_gt"),
    ):
        answer = extract_mcq_answer(str(value)) if value is not None else None
        if answer:
            return answer
    return None


def number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def chunk_id(chunk: Any) -> int:
    return int(getattr(chunk, "chunk_index"))


def chunk_bounds(chunk: Any) -> tuple[float, float]:
    timestamps = getattr(chunk, "frame_timestamps", None) or []
    numeric = [float(ts) for ts in timestamps if isinstance(ts, (int, float)) and math.isfinite(float(ts))]
    if numeric:
        return min(numeric), max(numeric)
    start = number(getattr(chunk, "start_time", None))
    end = number(getattr(chunk, "end_time", None))
    if start is None:
        start = float(chunk_id(chunk))
    if end is None:
        end = start
    return (end, start) if end < start else (start, end)


def chunk_anchor_time(chunk: Any) -> float:
    timestamps = getattr(chunk, "frame_timestamps", None) or []
    numeric = [float(ts) for ts in timestamps if isinstance(ts, (int, float)) and math.isfinite(float(ts))]
    if numeric:
        return numeric[len(numeric) // 2]
    start, end = chunk_bounds(chunk)
    return 0.5 * (start + end)


def recent_start_time(chunks: list[Any]) -> float:
    if not chunks:
        raise ValueError("No recent chunks")
    return min(chunk_bounds(chunk)[0] for chunk in chunks)


def frame_records_from_chunks(chunks: list[Any], recent_start: float, epsilon: float) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for chunk in chunks:
        frames = list(getattr(chunk, "frames", []) or [])
        if not frames:
            continue
        timestamps = getattr(chunk, "frame_timestamps", None) or []
        numeric = [float(ts) for ts in timestamps if isinstance(ts, (int, float)) and math.isfinite(float(ts))]
        start, end = chunk_bounds(chunk)
        if len(numeric) != len(frames):
            if len(frames) == 1:
                numeric = [chunk_anchor_time(chunk)]
            elif end > start:
                step = (end - start) / max(1, len(frames) - 1)
                numeric = [start + index * step for index in range(len(frames))]
            else:
                numeric = [start + index * 1e-4 for index in range(len(frames))]
        for index, (frame, ts) in enumerate(zip(frames, numeric)):
            if float(ts) < float(recent_start) - epsilon:
                records.append(
                    FrameRecord(
                        image=frame,
                        timestamp=float(ts),
                        chunk_id=chunk_id(chunk),
                        frame_index=index,
                    )
                )
    records.sort(key=lambda item: (item.timestamp, item.chunk_id, item.frame_index))
    return records


def choose_anchor_record(anchor_chunk: Any, records: list[FrameRecord], anchor_time: float) -> FrameRecord | None:
    candidates = [record for record in records if record.chunk_id == chunk_id(anchor_chunk)]
    if not candidates:
        return None
    return min(candidates, key=lambda item: (abs(item.timestamp - anchor_time), item.frame_index))


def strict_microclip_packet(
    anchor_chunk: Any,
    records: list[FrameRecord],
    anchor_time: float,
) -> list[FrameRecord]:
    anchor = choose_anchor_record(anchor_chunk, records, anchor_time)
    if anchor is None:
        return []
    previous = [
        record for record in records if record.timestamp < anchor.timestamp and record.key != anchor.key
    ]
    following = [
        record for record in records if record.timestamp > anchor.timestamp and record.key != anchor.key
    ]
    selected: list[FrameRecord] = []
    if previous:
        selected.append(max(previous, key=lambda item: (item.timestamp, item.chunk_id, item.frame_index)))
    selected.append(anchor)
    if following:
        selected.append(min(following, key=lambda item: (item.timestamp, item.chunk_id, item.frame_index)))

    unique: dict[tuple[int, int, float], FrameRecord] = {}
    for record in selected:
        unique[record.key] = record
    return sorted(unique.values(), key=lambda item: (item.timestamp, item.chunk_id, item.frame_index))


def random_local_packet(
    records: list[FrameRecord],
    qid: int,
    seed: int,
    exclude_key: tuple[int, int, float] | None = None,
) -> list[FrameRecord]:
    if not records:
        return []
    rng = random.Random(seed + qid * 9973)
    candidates = [record for record in records if record.key != exclude_key] or list(records)
    center = rng.choice(candidates)
    previous = [record for record in records if record.timestamp < center.timestamp and record.key != center.key]
    following = [record for record in records if record.timestamp > center.timestamp and record.key != center.key]
    selected: list[FrameRecord] = []
    if previous:
        selected.append(max(previous, key=lambda item: (item.timestamp, item.chunk_id, item.frame_index)))
    selected.append(center)
    if following:
        selected.append(min(following, key=lambda item: (item.timestamp, item.chunk_id, item.frame_index)))
    unique = {record.key: record for record in selected}
    return sorted(unique.values(), key=lambda item: (item.timestamp, item.chunk_id, item.frame_index))


def run_generation(
    qa: RecentWindowQAModel,
    *,
    prompt: str,
    gt: str | None,
    history_records: list[FrameRecord],
    recent_frames: list[Image.Image],
    prompt_variant: str,
) -> dict[str, Any]:
    if prompt_variant == "microclip":
        final_prompt = f"{MICROCLIP_INSTRUCTION}{prompt}" if history_records else prompt
    elif prompt_variant == "anchor":
        final_prompt = f"{HISTORY_INSTRUCTION}{prompt}" if history_records else prompt
    else:
        final_prompt = prompt
    frames = [record.image for record in history_records] + list(recent_frames)
    t0 = time.perf_counter()
    response = qa.generate_from_frames(frames, final_prompt)
    generate_seconds = time.perf_counter() - t0
    prediction = extract_mcq_answer(response)
    return {
        "prediction": prediction,
        "correct": bool(gt and prediction == gt),
        "response": response,
        "history_timestamps": [record.timestamp for record in history_records],
        "history_chunk_ids": [record.chunk_id for record in history_records],
        "history_frame_indices": [record.frame_index for record in history_records],
        "history_images": len(history_records),
        "total_images": len(frames),
        "num_vision_tokens": int(getattr(qa, "_last_num_vision_tokens", 0) or 0),
        "ttft_seconds": float(getattr(qa, "_last_ttft_seconds", 0.0) or 0.0),
        "generate_seconds": generate_seconds,
        "prompt_variant": prompt_variant,
    }


def bin_count(value: int) -> str:
    if value <= 3:
        return str(value)
    if value <= 6:
        return "4-6"
    if value <= 9:
        return "7-9"
    return ">9"


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "min": ordered[0],
        "p25": ordered[int(0.25 * (len(ordered) - 1))],
        "median": statistics.median(ordered),
        "p75": ordered[int(0.75 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def audit_existing_microclip(rows: list[dict[str, Any]]) -> dict[str, Any]:
    microclip_num_frames: list[float] = []
    final_historical_frames: list[float] = []
    num_memory_frames: list[float] = []
    vision_tokens: list[float] = []
    actual_image_bins: Counter[str] = Counter()
    final_state: Counter[str] = Counter()
    for row in rows:
        meta = adaptive(row)
        for target, key in (
            (microclip_num_frames, "microclip_num_frames"),
            (final_historical_frames, "final_historical_frames"),
            (num_memory_frames, "num_memory_frames"),
        ):
            value = number(meta.get(key))
            if value is not None:
                target.append(value)
        vt = number(row.get("num_vision_tokens"))
        if vt is None:
            profile = row.get("profile") if isinstance(row.get("profile"), dict) else {}
            vt = number(profile.get("num_vision_tokens"))
        if vt is None:
            vt = number(meta.get("num_vision_tokens"))
        if vt is not None:
            vision_tokens.append(vt)
        actual_images = int(number(meta.get("final_historical_frames")) or number(meta.get("microclip_num_frames")) or 0)
        actual_image_bins[bin_count(actual_images)] += 1
        final_state[str(meta.get("final_state", "unknown"))] += 1
    return {
        "samples": len(rows),
        "microclip_num_frames": distribution(microclip_num_frames),
        "final_historical_frames": distribution(final_historical_frames),
        "num_memory_frames": distribution(num_memory_frames),
        "vision_tokens": distribution(vision_tokens),
        "actual_historical_image_bins": dict(actual_image_bins),
        "final_state_distribution": dict(final_state),
    }


def branch_correct(row: dict[str, Any], name: str) -> bool:
    branch = row.get("branches", {}).get(name, {})
    return bool(branch.get("correct"))


def summarize_oracle(rows: list[dict[str, Any]], existing_rows_by_id: dict[int, dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    branches = ("K0", "anchor", "microclip", "random_local")
    acc = {}
    for branch in branches:
        correct_count = sum(branch_correct(row, branch) for row in rows)
        acc[f"{branch}_correct"] = correct_count
        acc[f"{branch}_accuracy"] = correct_count / total if total else None

    oracle_k0_anchor = sum(branch_correct(row, "K0") or branch_correct(row, "anchor") for row in rows)
    oracle_k0_micro = sum(branch_correct(row, "K0") or branch_correct(row, "microclip") for row in rows)
    oracle_all = sum(
        branch_correct(row, "K0") or branch_correct(row, "anchor") or branch_correct(row, "microclip")
        for row in rows
    )

    k0_wrong = [row for row in rows if not branch_correct(row, "K0")]
    rescue_classes = Counter()
    microclip_only_examples: list[dict[str, Any]] = []
    for row in k0_wrong:
        anchor_ok = branch_correct(row, "anchor")
        micro_ok = branch_correct(row, "microclip")
        if anchor_ok and micro_ok:
            rescue_classes["both_rescue"] += 1
        elif anchor_ok:
            rescue_classes["anchor_rescue_only"] += 1
        elif micro_ok:
            rescue_classes["microclip_rescue_only"] += 1
            microclip_only_examples.append(
                {
                    "question_id": row["question_id"],
                    "category": row.get("category"),
                    "question": row.get("question"),
                    "ground_truth": row.get("ground_truth"),
                    "K0_prediction": row["branches"]["K0"].get("prediction"),
                    "anchor_prediction": row["branches"]["anchor"].get("prediction"),
                    "microclip_prediction": row["branches"]["microclip"].get("prediction"),
                    "anchor_timestamp": row.get("anchor_timestamp"),
                    "neighbor_timestamps": row["branches"]["microclip"].get("history_timestamps"),
                    "candidate_total_score": row.get("candidate_total_score"),
                    "candidate_semantic_score": row.get("candidate_semantic_score"),
                }
            )
        else:
            rescue_classes["neither"] += 1

    k0_correct = [row for row in rows if branch_correct(row, "K0")]
    damage = {
        "anchor_damage": sum(not branch_correct(row, "anchor") for row in k0_correct),
        "microclip_damage": sum(not branch_correct(row, "microclip") for row in k0_correct),
        "random_local_damage": sum(not branch_correct(row, "random_local") for row in k0_correct),
    }
    existing_whole_chunk_damage = 0
    existing_whole_chunk_available = 0
    for row in k0_correct:
        existing = existing_rows_by_id.get(int(row["question_id"]))
        if existing is not None:
            existing_whole_chunk_available += 1
            existing_whole_chunk_damage += int(not bool(existing.get("correct")))

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row.get("category", "unknown"))].append(row)
    category_summary: dict[str, Any] = {}
    for category, group in sorted(by_category.items()):
        group_k0_correct = [row for row in group if branch_correct(row, "K0")]
        group_k0_wrong = [row for row in group if not branch_correct(row, "K0")]
        category_summary[category] = {
            "total": len(group),
            "K0_correct": sum(branch_correct(row, "K0") for row in group),
            "anchor_rescue_only": sum(
                branch_correct(row, "anchor") and not branch_correct(row, "microclip")
                for row in group_k0_wrong
            ),
            "microclip_rescue_only": sum(
                branch_correct(row, "microclip") and not branch_correct(row, "anchor")
                for row in group_k0_wrong
            ),
            "both_rescue": sum(
                branch_correct(row, "anchor") and branch_correct(row, "microclip")
                for row in group_k0_wrong
            ),
            "neither": sum(
                not branch_correct(row, "anchor") and not branch_correct(row, "microclip")
                for row in group_k0_wrong
            ),
            "anchor_damage": sum(not branch_correct(row, "anchor") for row in group_k0_correct),
            "microclip_damage": sum(not branch_correct(row, "microclip") for row in group_k0_correct),
        }

    return {
        "samples": total,
        **acc,
        "Oracle_K0_Anchor_correct": oracle_k0_anchor,
        "Oracle_K0_Anchor_accuracy": oracle_k0_anchor / total if total else None,
        "Oracle_K0_Microclip_correct": oracle_k0_micro,
        "Oracle_K0_Microclip_accuracy": oracle_k0_micro / total if total else None,
        "Oracle_K0_Anchor_Microclip_correct": oracle_all,
        "Oracle_K0_Anchor_Microclip_accuracy": oracle_all / total if total else None,
        "K0_wrong_samples": len(k0_wrong),
        "K0_wrong_rescue_classification": dict(rescue_classes),
        "K0_correct_samples": len(k0_correct),
        "damage": damage,
        "existing_whole_chunk_microclip_damage_on_K0_correct": existing_whole_chunk_damage,
        "existing_whole_chunk_microclip_damage_available": existing_whole_chunk_available,
        "category_breakdown": category_summary,
        "microclip_only_rescue_examples": microclip_only_examples,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "question_id",
        "category",
        "ground_truth",
        "K0_prediction",
        "K0_correct",
        "anchor_prediction",
        "anchor_correct",
        "microclip_prediction",
        "microclip_correct",
        "random_local_prediction",
        "random_local_correct",
        "anchor_timestamp",
        "microclip_timestamps",
        "candidate_total_score",
        "candidate_semantic_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "question_id": row["question_id"],
                    "category": row.get("category"),
                    "ground_truth": row.get("ground_truth"),
                    "K0_prediction": row["branches"]["K0"].get("prediction"),
                    "K0_correct": row["branches"]["K0"].get("correct"),
                    "anchor_prediction": row["branches"]["anchor"].get("prediction"),
                    "anchor_correct": row["branches"]["anchor"].get("correct"),
                    "microclip_prediction": row["branches"]["microclip"].get("prediction"),
                    "microclip_correct": row["branches"]["microclip"].get("correct"),
                    "random_local_prediction": row["branches"]["random_local"].get("prediction"),
                    "random_local_correct": row["branches"]["random_local"].get("correct"),
                    "anchor_timestamp": row.get("anchor_timestamp"),
                    "microclip_timestamps": row["branches"]["microclip"].get("history_timestamps"),
                    "candidate_total_score": row.get("candidate_total_score"),
                    "candidate_semantic_score": row.get("candidate_semantic_score"),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", required=True, help="Saved temporal_microclip StreamingBench-100 result dir/JSON.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--qa-model", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--qa-device", default=None)
    parser.add_argument("--attn-implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"))
    parser.add_argument("--max-qa-tokens", type=int, default=256)
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--recent-window", type=int, default=6)
    parser.add_argument("--context-time", type=float, default=70.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    global RecentWindowQAModel, build_prompt, decode_video_to_chunks_qwen
    global resolve_video_path, select_recent_window_frames, timestamp_to_seconds

    from lib.minicpm.baseline import RecentWindowQAModel as _RecentWindowQAModel
    from lib.minicpm.baseline import select_recent_window_frames as _select_recent_window_frames
    from lib.shared.recent_window import decode_video_to_chunks_qwen as _decode_video_to_chunks_qwen
    from main_experiments.minicpm_v46.streamingbench.eval_baseline import build_prompt as _build_prompt
    from main_experiments.minicpm_v46.streamingbench.eval_baseline import resolve_video_path as _resolve_video_path
    from main_experiments.minicpm_v46.streamingbench.eval_baseline import timestamp_to_seconds as _timestamp_to_seconds

    RecentWindowQAModel = _RecentWindowQAModel
    build_prompt = _build_prompt
    decode_video_to_chunks_qwen = _decode_video_to_chunks_qwen
    resolve_video_path = _resolve_video_path
    select_recent_window_frames = _select_recent_window_frames
    timestamp_to_seconds = _timestamp_to_seconds

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_path, saved_rows = load_results(Path(args.source_results))
    if args.max_samples > 0:
        saved_rows = saved_rows[: args.max_samples]
    saved_rows_by_id = {int(row.get("_index", index)): row for index, row in enumerate(saved_rows)}
    annotations = load_annotations(Path(args.annotations), args.video_dir)
    audit = audit_existing_microclip(saved_rows)
    (out_dir / "existing_microclip_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    qa = RecentWindowQAModel(
        model_name=args.qa_model,
        device=args.qa_device or default_qa_device(),
        max_new_tokens=args.max_qa_tokens,
        attn_implementation=args.attn_implementation,
    )

    completed: dict[int, dict[str, Any]] = {}
    jsonl_path = out_dir / "strict_microclip_oracle_results.jsonl"
    if jsonl_path.exists():
        for row in read_jsonl(jsonl_path):
            completed[int(row["question_id"])] = row

    epsilon = 1e-6
    with jsonl_path.open("a", encoding="utf-8") as handle:
        for ordinal, saved in enumerate(saved_rows, start=1):
            qid = int(saved.get("_index", ordinal - 1))
            if qid in completed:
                print(f"[{ordinal}/{len(saved_rows)}] skip qid={qid}", flush=True)
                continue
            task = annotations.get(qid)
            if task is None:
                raise KeyError(f"Question index {qid} is absent from annotations")
            question = task["question_obj"]
            prompt = task["prompt"]
            gt = answer_gt_from(saved, question)
            ts_sec = float(timestamp_to_seconds(question["time_stamp"]))
            video_end = ts_sec + 1e-4
            broad_start = max(0.0, ts_sec - max(float(args.context_time), float(args.chunk_duration)))
            decode_recent_hint = max(
                int(math.ceil(float(args.context_time) / max(float(args.chunk_duration), 1e-6))),
                int(args.recent_window) + 64,
            )

            print(f"[{ordinal}/{len(saved_rows)}] qid={qid}", flush=True)
            broad_chunks, broad_backend = decode_video_to_chunks_qwen(
                video_path=task["video_path"],
                chunk_duration=args.chunk_duration,
                fps=args.fps,
                recent_frames_only=decode_recent_hint,
                video_start=broad_start,
                video_end=video_end,
            )
            broad_by_id = {chunk_id(chunk): chunk for chunk in broad_chunks}

            recent_video_start = max(0.0, video_end - float(args.recent_window) * float(args.chunk_duration))
            recent_selection = select_recent_window_frames(
                qa=qa,
                video_path=task["video_path"],
                chunk_duration=args.chunk_duration,
                fps=args.fps,
                recent_frames_only=args.recent_window,
                video_start=recent_video_start,
                video_end=video_end,
                cdas_config=None,
            )
            recent_chunks = list(recent_selection.selected_chunks)
            recent_frames = list(recent_selection.frames)
            r_start = recent_start_time(recent_chunks)
            historical_chunks = [
                chunk for chunk in broad_chunks if chunk_bounds(chunk)[1] < r_start - epsilon
            ]
            records = frame_records_from_chunks(historical_chunks, r_start, epsilon)

            source_meta = adaptive(saved)
            candidate_queue = list(source_meta.get("candidate_queue") or [])
            anchor_meta = candidate_queue[0] if candidate_queue else {}
            anchor_chunk = broad_by_id.get(int(anchor_meta["chunk_id"])) if "chunk_id" in anchor_meta else None
            anchor_time = number(anchor_meta.get("timestamp"))
            if anchor_chunk is not None and anchor_time is None:
                anchor_time = chunk_anchor_time(anchor_chunk)

            anchor_records: list[FrameRecord] = []
            microclip_records: list[FrameRecord] = []
            if anchor_chunk is not None and anchor_time is not None:
                anchor_record = choose_anchor_record(anchor_chunk, records, anchor_time)
                if anchor_record is not None:
                    anchor_records = [anchor_record]
                microclip_records = strict_microclip_packet(anchor_chunk, records, anchor_time)
            random_records = random_local_packet(
                records,
                qid=qid,
                seed=args.random_seed,
                exclude_key=anchor_records[0].key if anchor_records else None,
            )

            for record in [*anchor_records, *microclip_records, *random_records]:
                if not record.timestamp < r_start - epsilon:
                    raise AssertionError(
                        f"Temporal violation qid={qid}: history frame {record.timestamp} recent_start={r_start}"
                    )

            branches = {
                "K0": run_generation(
                    qa,
                    prompt=prompt,
                    gt=gt,
                    history_records=[],
                    recent_frames=recent_frames,
                    prompt_variant="recent",
                ),
                "anchor": run_generation(
                    qa,
                    prompt=prompt,
                    gt=gt,
                    history_records=anchor_records,
                    recent_frames=recent_frames,
                    prompt_variant="anchor",
                ),
                "microclip": run_generation(
                    qa,
                    prompt=prompt,
                    gt=gt,
                    history_records=microclip_records,
                    recent_frames=recent_frames,
                    prompt_variant="microclip",
                ),
                "random_local": run_generation(
                    qa,
                    prompt=prompt,
                    gt=gt,
                    history_records=random_records,
                    recent_frames=recent_frames,
                    prompt_variant="microclip",
                ),
            }
            output = {
                "question_id": qid,
                "key": saved.get("_key"),
                "video_id": task["video"],
                "timestamp": question.get("time_stamp"),
                "timestamp_seconds": ts_sec,
                "category": saved.get("task_type") or task.get("task_type"),
                "question": question.get("question"),
                "options": question.get("options"),
                "ground_truth": gt,
                "decode": {
                    "source_path": source_path,
                    "broad_backend": broad_backend,
                    "broad_video_start": broad_start,
                    "broad_video_end": video_end,
                    "broad_recent_frames_only": decode_recent_hint,
                    "recent_start_time_seconds": r_start,
                    "recent_chunk_ids": [chunk_id(chunk) for chunk in recent_chunks],
                    "historical_frame_records": len(records),
                },
                "candidate_anchor_id": anchor_meta.get("chunk_id"),
                "anchor_timestamp": anchor_time,
                "candidate_total_score": anchor_meta.get("total_score"),
                "candidate_semantic_score": anchor_meta.get("semantic_score"),
                "candidate_queue_prefix": candidate_queue[:3],
                "strict_anchor_images": len(anchor_records),
                "strict_microclip_images": len(microclip_records),
                "random_local_images": len(random_records),
                "branches": branches,
                "existing_whole_chunk_microclip_correct": bool(saved.get("correct")),
                "existing_whole_chunk_microclip_prediction": extract_mcq_answer(saved.get("response")),
                "existing_whole_chunk_adaptive": source_meta,
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            handle.flush()
            completed[qid] = output

    rows = [completed[key] for key in sorted(completed)]
    summary = {
        "existing_microclip_audit": audit,
        "strict_oracle": summarize_oracle(rows, saved_rows_by_id),
    }
    (out_dir / "strict_microclip_oracle_results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "strict_microclip_oracle_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(out_dir / "strict_microclip_oracle_results.csv", rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
