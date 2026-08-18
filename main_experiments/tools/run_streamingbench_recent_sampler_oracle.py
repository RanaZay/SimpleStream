#!/usr/bin/env python3
"""StreamingBench-100 oracle over fixed six-frame recent samplers.

This is an isolated diagnostic. It does not use PRISM memory and does not
modify any official method. Every evaluated branch feeds exactly six recent
frames when six decoded frames are available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

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
    dedup: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        dedup[int(row.get("_index", index))] = row
    return "\n  ".join(str(item) for item in rank_files), [
        dedup[key] for key in sorted(dedup)
    ]


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


def frame_records_from_chunks(chunks: list[Any]) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for chunk in chunks:
        frames = list(getattr(chunk, "frames", []) or [])
        timestamps = getattr(chunk, "frame_timestamps", None) or []
        numeric = [float(ts) for ts in timestamps if isinstance(ts, (int, float)) and math.isfinite(float(ts))]
        start, end = chunk_bounds(chunk)
        if len(numeric) != len(frames):
            if len(frames) == 1:
                numeric = [0.5 * (start + end)]
            elif end > start:
                step = (end - start) / max(1, len(frames) - 1)
                numeric = [start + index * step for index in range(len(frames))]
            else:
                numeric = [start + index * 1e-4 for index in range(len(frames))]
        for index, (frame, ts) in enumerate(zip(frames, numeric)):
            records.append(FrameRecord(frame, float(ts), chunk_id(chunk), index))
    records.sort(key=lambda item: (item.timestamp, item.chunk_id, item.frame_index))
    return records


def unique_chronological(records: list[FrameRecord], limit: int = 6) -> list[FrameRecord]:
    seen: set[tuple[int, int, float]] = set()
    out: list[FrameRecord] = []
    for record in sorted(records, key=lambda item: (item.timestamp, item.chunk_id, item.frame_index)):
        if record.key in seen:
            continue
        seen.add(record.key)
        out.append(record)
        if len(out) >= limit:
            break
    return out


def nearest_to_targets(records: list[FrameRecord], targets: list[float], count: int = 6) -> list[FrameRecord]:
    selected: list[FrameRecord] = []
    used: set[tuple[int, int, float]] = set()
    for target in targets:
        candidates = [record for record in records if record.key not in used]
        if not candidates:
            break
        chosen = min(candidates, key=lambda item: (abs(item.timestamp - target), -item.timestamp))
        selected.append(chosen)
        used.add(chosen.key)
    if len(selected) < min(count, len(records)):
        for record in sorted(records, key=lambda item: item.timestamp, reverse=True):
            if record.key not in used:
                selected.append(record)
                used.add(record.key)
            if len(selected) >= min(count, len(records)):
                break
    return unique_chronological(selected, count)


def frame_signature(frame: Image.Image, resize: int) -> np.ndarray:
    gray = frame.convert("L").resize((resize, resize), Image.BILINEAR)
    return np.asarray(gray, dtype=np.float32)


def structural_change(left: np.ndarray, right: np.ndarray) -> float:
    c1 = 6.5025
    c2 = 58.5225
    mu_left = float(np.mean(left))
    mu_right = float(np.mean(right))
    var_left = float(np.var(left))
    var_right = float(np.var(right))
    cov = float(np.mean((left - mu_left) * (right - mu_right)))
    numerator = (2.0 * mu_left * mu_right + c1) * (2.0 * cov + c2)
    denominator = (mu_left * mu_left + mu_right * mu_right + c1) * (var_left + var_right + c2)
    if denominator <= 0:
        return 0.0
    return float(1.0 - max(0.0, min(1.0, numerator / denominator)))


def change_aware(records: list[FrameRecord], query_time: float, count: int = 6, resize: int = 64) -> list[FrameRecord]:
    if len(records) <= count:
        return unique_chronological(records, count)

    latest = max(records, key=lambda item: item.timestamp)
    selected: dict[tuple[int, int, float], FrameRecord] = {latest.key: latest}
    signatures = [frame_signature(record.image, resize) for record in records]
    changes: list[tuple[float, FrameRecord]] = []
    for index in range(1, len(records)):
        score = structural_change(signatures[index - 1], signatures[index])
        changes.append((score, records[index]))

    # Region quota keeps the selector from collapsing onto a single burst.
    start = max(0.0, query_time - 6.0)
    width = max((query_time - start) / 5.0, 1e-6)
    region_counts: Counter[int] = Counter()
    for record in selected.values():
        region_counts[min(4, max(0, int((record.timestamp - start) / width)))] += 1

    for _score, record in sorted(changes, key=lambda item: (item[0], item[1].timestamp), reverse=True):
        if record.key in selected:
            continue
        region = min(4, max(0, int((record.timestamp - start) / width)))
        if region_counts[region] >= 2:
            continue
        selected[record.key] = record
        region_counts[region] += 1
        if len(selected) >= min(count, len(records)):
            break

    if len(selected) < min(count, len(records)):
        for record in sorted(records, key=lambda item: item.timestamp, reverse=True):
            if record.key not in selected:
                selected[record.key] = record
            if len(selected) >= min(count, len(records)):
                break
    return unique_chronological(list(selected.values()), count)


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


def temporal_stats(timestamps: list[float]) -> tuple[float | None, float | None]:
    if not timestamps:
        return None, None
    span = max(timestamps) - min(timestamps)
    if len(timestamps) < 2:
        return span, None
    gaps = [b - a for a, b in zip(sorted(timestamps), sorted(timestamps)[1:])]
    return span, statistics.mean(gaps)


def run_generation(
    qa: Any,
    *,
    prompt: str,
    gt: str | None,
    records: list[FrameRecord],
) -> dict[str, Any]:
    frames = [record.image for record in records]
    t0 = time.perf_counter()
    response = qa.generate_from_frames(frames, prompt)
    generate_seconds = time.perf_counter() - t0
    prediction = extract_mcq_answer(response)
    timestamps = [record.timestamp for record in records]
    span, mean_gap = temporal_stats(timestamps)
    return {
        "prediction": prediction,
        "correct": bool(gt and prediction == gt),
        "response": response,
        "frame_count": len(frames),
        "frame_timestamps": timestamps,
        "frame_chunk_ids": [record.chunk_id for record in records],
        "frame_indices": [record.frame_index for record in records],
        "temporal_span_seconds": span,
        "mean_adjacent_spacing_seconds": mean_gap,
        "num_vision_tokens": int(getattr(qa, "_last_num_vision_tokens", 0) or 0),
        "ttft_seconds": float(getattr(qa, "_last_ttft_seconds", 0.0) or 0.0),
        "generate_seconds": generate_seconds,
    }


def branch_correct(row: dict[str, Any], name: str) -> bool:
    return bool(row.get("branches", {}).get(name, {}).get("correct"))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    variants = [
        "current_recent6",
        "dense_recent6",
        "hybrid_recent6",
        "change_aware6",
        "uniform_dense6",
    ]
    total = len(rows)
    variant_summary: dict[str, Any] = {}
    current_correct = {int(row["question_id"]): branch_correct(row, "current_recent6") for row in rows}
    for variant in variants:
        correct = sum(branch_correct(row, variant) for row in rows)
        rescued = sum(
            (not current_correct[int(row["question_id"])]) and branch_correct(row, variant)
            for row in rows
        )
        damaged = sum(
            current_correct[int(row["question_id"])] and not branch_correct(row, variant)
            for row in rows
        )
        branch_rows = [row["branches"][variant] for row in rows]
        variant_summary[variant] = {
            "correct": correct,
            "accuracy": correct / total if total else None,
            "rescued_vs_current": rescued,
            "damaged_vs_current": damaged,
            "net_rescue": rescued - damaged,
            "temporal_span_seconds": distribution(
                [float(item["temporal_span_seconds"]) for item in branch_rows if item.get("temporal_span_seconds") is not None]
            ),
            "mean_adjacent_spacing_seconds": distribution(
                [float(item["mean_adjacent_spacing_seconds"]) for item in branch_rows if item.get("mean_adjacent_spacing_seconds") is not None]
            ),
            "vision_tokens": distribution([float(item["num_vision_tokens"]) for item in branch_rows]),
            "latency_seconds": distribution([float(item["generate_seconds"]) for item in branch_rows]),
            "ttft_seconds": distribution([float(item["ttft_seconds"]) for item in branch_rows]),
        }

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row.get("category", "unknown"))].append(row)
    category_summary: dict[str, Any] = {}
    for category, group in sorted(by_category.items()):
        current_group_correct = {
            int(row["question_id"]): branch_correct(row, "current_recent6") for row in group
        }
        per_variant: dict[str, Any] = {}
        for variant in variants:
            correct = sum(branch_correct(row, variant) for row in group)
            rescued = sum(
                (not current_group_correct[int(row["question_id"])]) and branch_correct(row, variant)
                for row in group
            )
            damaged = sum(
                current_group_correct[int(row["question_id"])] and not branch_correct(row, variant)
                for row in group
            )
            per_variant[variant] = {
                "correct": correct,
                "total": len(group),
                "accuracy": correct / len(group) if group else None,
                "rescued_vs_current": rescued,
                "damaged_vs_current": damaged,
                "net_rescue": rescued - damaged,
            }
        category_summary[category] = per_variant

    oracle_correct = sum(any(branch_correct(row, variant) for variant in variants) for row in rows)
    current_wrong_rows = [row for row in rows if not branch_correct(row, "current_recent6")]
    current_wrong_report = []
    for row in current_wrong_rows:
        rescuers = [variant for variant in variants[1:] if branch_correct(row, variant)]
        current_wrong_report.append(
            {
                "question_id": row["question_id"],
                "category": row.get("category"),
                "question": row.get("question"),
                "ground_truth": row.get("ground_truth"),
                "current_prediction": row["branches"]["current_recent6"].get("prediction"),
                "rescuing_samplers": rescuers,
                "num_rescuing_samplers": len(rescuers),
            }
        )

    best_variant = max(variants, key=lambda name: variant_summary[name]["correct"])
    return {
        "samples": total,
        "variants": variant_summary,
        "category_summary": category_summary,
        "current_recent6_accuracy": variant_summary["current_recent6"]["accuracy"],
        "best_individual_sampler": best_variant,
        "best_individual_accuracy": variant_summary[best_variant]["accuracy"],
        "oracle_recent_sampling_correct": oracle_correct,
        "oracle_recent_sampling_accuracy": oracle_correct / total if total else None,
        "current_recent6_wrong_samples": len(current_wrong_rows),
        "current_wrong_rescue_report": current_wrong_report,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    variants = ["current_recent6", "dense_recent6", "hybrid_recent6", "change_aware6", "uniform_dense6"]
    fields = ["question_id", "video_id", "timestamp", "category", "ground_truth", "question"]
    for variant in variants:
        fields.extend([f"{variant}_prediction", f"{variant}_correct", f"{variant}_timestamps"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {
                "question_id": row["question_id"],
                "video_id": row.get("video_id"),
                "timestamp": row.get("timestamp"),
                "category": row.get("category"),
                "ground_truth": row.get("ground_truth"),
                "question": row.get("question"),
            }
            for variant in variants:
                branch = row["branches"][variant]
                out[f"{variant}_prediction"] = branch.get("prediction")
                out[f"{variant}_correct"] = branch.get("correct")
                out[f"{variant}_timestamps"] = branch.get("frame_timestamps")
            writer.writerow(out)


def print_summary(summary: dict[str, Any]) -> None:
    print("\n================================================================================")
    print("StreamingBench-100 Recent Sampler Oracle")
    print("================================================================================")
    for name, item in summary["variants"].items():
        print(
            f"{name}: {100.0 * item['accuracy']:.2f}% ({item['correct']}/{summary['samples']}) "
            f"rescue={item['rescued_vs_current']} damage={item['damaged_vs_current']} net={item['net_rescue']}"
        )
        print(
            f"  span_mean={item['temporal_span_seconds'].get('mean')} "
            f"gap_mean={item['mean_adjacent_spacing_seconds'].get('mean')} "
            f"tokens_mean={item['vision_tokens'].get('mean')} "
            f"latency_mean={item['latency_seconds'].get('mean')} "
            f"ttft_mean={item['ttft_seconds'].get('mean')}"
        )
    print(
        f"\nBest individual: {summary['best_individual_sampler']} "
        f"{100.0 * summary['best_individual_accuracy']:.2f}%"
    )
    print(
        f"Oracle over recent samplers: {100.0 * summary['oracle_recent_sampling_accuracy']:.2f}% "
        f"({summary['oracle_recent_sampling_correct']}/{summary['samples']})"
    )
    print("\nPer-category accuracy:")
    for category, per_variant in summary["category_summary"].items():
        pieces = []
        for name, item in per_variant.items():
            pieces.append(f"{name}={100.0 * item['accuracy']:.2f}%")
        print(f"  {category}: " + ", ".join(pieces))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", required=True, help="Saved SB-100 result dir/JSON defining the exact subset.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--qa-model", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--qa-device", default=None)
    parser.add_argument("--attn-implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"))
    parser.add_argument("--max-qa-tokens", type=int, default=256)
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--current-fps", type=float, default=1.0)
    parser.add_argument("--candidate-fps", type=float, default=4.0)
    parser.add_argument("--recent-window", type=int, default=6)
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
    annotations = load_annotations(Path(args.annotations), args.video_dir)

    qa = RecentWindowQAModel(
        model_name=args.qa_model,
        device=args.qa_device or default_qa_device(),
        max_new_tokens=args.max_qa_tokens,
        attn_implementation=args.attn_implementation,
    )

    completed: dict[int, dict[str, Any]] = {}
    jsonl_path = out_dir / "recent_sampler_oracle_results.jsonl"
    if jsonl_path.exists():
        for row in read_jsonl(jsonl_path):
            completed[int(row["question_id"])] = row

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
            recent_start = max(0.0, video_end - float(args.recent_window) * float(args.chunk_duration))

            print(f"[{ordinal}/{len(saved_rows)}] qid={qid}", flush=True)
            current_selection = select_recent_window_frames(
                qa=qa,
                video_path=task["video_path"],
                chunk_duration=args.chunk_duration,
                fps=args.current_fps,
                recent_frames_only=args.recent_window,
                video_start=recent_start,
                video_end=video_end,
                cdas_config=None,
            )
            current_records = frame_records_from_chunks(list(current_selection.selected_chunks))

            candidate_chunks, candidate_backend = decode_video_to_chunks_qwen(
                video_path=task["video_path"],
                chunk_duration=args.chunk_duration,
                fps=args.candidate_fps,
                recent_frames_only=max(1, int(math.ceil(float(args.recent_window) * args.candidate_fps))),
                video_start=recent_start,
                video_end=video_end,
            )
            candidate_records = frame_records_from_chunks(candidate_chunks)
            if not candidate_records:
                candidate_records = current_records

            dense_targets = [ts_sec - offset for offset in (3.0, 2.0, 1.0, 0.5, 0.25, 0.0)]
            hybrid_targets = [ts_sec - offset for offset in (5.0, 3.0, 1.5, 0.75, 0.25, 0.0)]
            uniform_dense_targets = [
                ts_sec - 3.0 + i * (3.0 / max(1, args.recent_window - 1))
                for i in range(args.recent_window)
            ]

            selections = {
                "current_recent6": current_records,
                "dense_recent6": nearest_to_targets(candidate_records, dense_targets, args.recent_window),
                "hybrid_recent6": nearest_to_targets(candidate_records, hybrid_targets, args.recent_window),
                "change_aware6": change_aware(candidate_records, ts_sec, args.recent_window),
                "uniform_dense6": nearest_to_targets(candidate_records, uniform_dense_targets, args.recent_window),
            }
            for name, records in selections.items():
                if len(candidate_records) >= args.recent_window and len(records) != args.recent_window:
                    raise AssertionError(f"{name} selected {len(records)} frames for qid={qid}, expected {args.recent_window}")
                if any(b.timestamp < a.timestamp for a, b in zip(records, records[1:])):
                    raise AssertionError(f"{name} is not chronologically ordered for qid={qid}")

            branches = {
                name: run_generation(qa, prompt=prompt, gt=gt, records=records)
                for name, records in selections.items()
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
                    "current_decode_backend": current_selection.decode_backend,
                    "candidate_decode_backend": candidate_backend,
                    "recent_start": recent_start,
                    "video_end": video_end,
                    "current_fps": args.current_fps,
                    "candidate_fps": args.candidate_fps,
                    "candidate_decoded_frames": len(candidate_records),
                    "candidate_decoded_chunks": len(candidate_chunks),
                },
                "branches": branches,
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            handle.flush()
            completed[qid] = output

    rows = [completed[key] for key in sorted(completed)]
    summary = summarize(rows)
    (out_dir / "recent_sampler_oracle_results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "recent_sampler_oracle_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(out_dir / "recent_sampler_oracle_results.csv", rows)
    print_summary(summary)


if __name__ == "__main__":
    main()
