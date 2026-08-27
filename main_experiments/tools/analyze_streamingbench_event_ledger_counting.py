#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from lib.minicpm.prism_event_ledger import (  # noqa: E402
    EventLedgerClipScorer,
    chunk_bounds,
    event_boundary_scores,
    event_to_json,
    ledger_grid,
    numeric_options,
    score_events_for_question,
    segment_events,
)
from lib.minicpm.prism_retrieval_variants import representative_frame  # noqa: E402
from lib.shared.recent_window import decode_video_to_chunks_qwen  # noqa: E402


def timestamp_to_seconds(ts: Any) -> float:
    parts = [float(part) for part in re.findall(r"\d+(?:\.\d+)?", str(ts))]
    if len(parts) >= 3:
        return parts[-3] * 3600.0 + parts[-2] * 60.0 + parts[-1]
    if len(parts) == 2:
        return parts[0] * 60.0 + parts[1]
    return parts[0] if parts else 0.0


def extract_answer(text: Any) -> str | None:
    if text is None:
        return None
    match = re.search(r"\b([A-E])\b", str(text).upper())
    return match.group(1) if match else None


def load_results(path: Path) -> tuple[str, list[dict[str, Any]]]:
    if path.is_file():
        payload = json.load(path.open(encoding="utf-8"))
        rows = payload.get("results", payload if isinstance(payload, list) else [])
        return str(path), rows
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
        with rank_file.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    dedup: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("_index") is not None:
            dedup[int(row["_index"])] = row
    return "\n  ".join(str(path) for path in rank_files), [dedup[key] for key in sorted(dedup)]


def load_counting_annotations(path: Path, video_dir: Path) -> list[dict[str, Any]]:
    data = json.load(path.open(encoding="utf-8"))
    tasks: list[dict[str, Any]] = []
    index = 0
    for video_entry in data:
        questions = sorted(video_entry.get("questions", []), key=lambda item: timestamp_to_seconds(item.get("time_stamp")))
        video_name = video_entry.get("video") or video_entry.get("video_name") or video_entry.get("video_id")
        video_basename = Path(str(video_name)).name
        if not video_basename.endswith(".mp4"):
            video_basename = f"{Path(video_basename).stem}_real.mp4" if not Path(video_basename).stem.endswith("_real") else f"{Path(video_basename).stem}.mp4"
        for question in questions:
            if str(question.get("task_type", "")).strip() == "Counting":
                tasks.append(
                    {
                        "question_id": index,
                        "video_id": Path(video_basename).stem,
                        "video_path": str(video_dir / video_basename),
                        "question": str(question.get("question", "")),
                        "timestamp": question.get("time_stamp"),
                        "timestamp_seconds": timestamp_to_seconds(question.get("time_stamp")),
                        "answer_gt": extract_answer(question.get("answer")) or str(question.get("answer", "")).strip().upper(),
                        "options": parse_options(question),
                    }
                )
            index += 1
    return tasks


def parse_options(question: dict[str, Any]) -> list[dict[str, str]]:
    raw = question.get("options") or question.get("choices") or []
    out: list[dict[str, str]] = []
    if isinstance(raw, dict):
        for letter in "ABCDE":
            if letter in raw:
                out.append({"letter": letter, "text": str(raw[letter])})
    elif isinstance(raw, list):
        for index, value in enumerate(raw):
            letter = chr(65 + index)
            text = str(value).strip()
            text = re.sub(r"^[A-E]\s*[\.\)]\s*", "", text)
            out.append({"letter": letter, "text": text})
    return out


def row_by_index(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["_index"]): row for row in rows if row.get("_index") is not None}


def correct(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("correct"))


def prediction(row: dict[str, Any] | None) -> str | None:
    if not row:
        return None
    return extract_answer(row.get("response"))


def parse_float_list(raw: str) -> list[float]:
    if not raw:
        return []
    return [float(item) for item in raw.split(",") if item.strip()]


def option_numeric_gt(task: dict[str, Any]) -> int | None:
    gt = str(task["answer_gt"])
    values = numeric_options(task["options"])
    if gt in values:
        return values[gt]
    return None


def decode_history_chunks(task: dict[str, Any], args: argparse.Namespace) -> tuple[list[Any], dict[str, Any]]:
    ts = float(task["timestamp_seconds"])
    recent_start = max(0.0, ts + 1e-4 - float(args.recent_window) * float(args.chunk_duration))
    history_start = max(0.0, ts - max(float(args.context_time), float(args.chunk_duration)))
    chunks, backend = decode_video_to_chunks_qwen(
        video_path=task["video_path"],
        chunk_duration=float(args.chunk_duration),
        fps=float(args.fps),
        recent_frames_only=max(int(args.recent_window), int(math.ceil(float(args.context_time) * max(float(args.fps), 1.0)))),
        video_start=history_start,
        video_end=ts + 1e-4,
    )
    historical = []
    for chunk in chunks:
        start, end = chunk_bounds(chunk)
        if float(end) < recent_start - 1e-6:
            historical.append(chunk)
    history_search = int(args.history_search_chunks)
    if history_search > 0:
        historical = historical[-history_search:]
    return historical, {
        "decode_backend": backend,
        "decoded_chunks": len(chunks),
        "historical_chunks": len(historical),
        "history_start_time": history_start,
        "recent_start_time": recent_start,
        "query_time": ts,
    }


def evaluate_grid_record(
    grid: dict[str, dict[str, Any]],
    gt_option: str,
    gt_numeric: int | None,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, item in grid.items():
        mapped = item.get("mapped_option")
        count = item.get("ledger_count")
        out[key] = {
            "ledger_count": count,
            "mapped_option": mapped,
            "exact_count_correct": bool(gt_numeric is not None and int(count) == int(gt_numeric)),
            "absolute_count_error": None if gt_numeric is None else abs(int(count) - int(gt_numeric)),
            "mcq_correct": bool(mapped is not None and str(mapped) == str(gt_option)),
        }
    return out


def summarize(records: list[dict[str, Any]], recent_by_id: dict[int, dict[str, Any]], prism_by_id: dict[int, dict[str, Any]]) -> dict[str, Any]:
    grid_keys = sorted(records[0]["grid_eval"]) if records else []
    grid_summary: dict[str, Any] = {}
    for key in grid_keys:
        with_numeric = [row for row in records if row["ground_truth_numeric_count"] is not None]
        exact_correct = sum(1 for row in with_numeric if row["grid_eval"][key]["exact_count_correct"])
        abs_errors = [
            float(row["grid_eval"][key]["absolute_count_error"])
            for row in with_numeric
            if row["grid_eval"][key]["absolute_count_error"] is not None
        ]
        mcq_correct = sum(1 for row in records if row["grid_eval"][key]["mcq_correct"])
        recent_or_ledger = sum(
            1
            for row in records
            if correct(recent_by_id.get(int(row["question_id"]))) or row["grid_eval"][key]["mcq_correct"]
        )
        all_oracle = sum(
            1
            for row in records
            if correct(recent_by_id.get(int(row["question_id"])))
            or correct(prism_by_id.get(int(row["question_id"])))
            or row["grid_eval"][key]["mcq_correct"]
        )
        grid_summary[key] = {
            "exact_count_accuracy": exact_correct / len(with_numeric) if with_numeric else None,
            "exact_count_correct": exact_correct,
            "numeric_gt_count": len(with_numeric),
            "mean_absolute_error": mean(abs_errors) if abs_errors else None,
            "ledger_mcq_accuracy": mcq_correct / len(records) if records else None,
            "ledger_mcq_correct": mcq_correct,
            "oracle_recent_ledger_correct": recent_or_ledger,
            "oracle_recent_ledger_accuracy": recent_or_ledger / len(records) if records else None,
            "oracle_recent_prism_ledger_correct": all_oracle,
            "oracle_recent_prism_ledger_accuracy": all_oracle / len(records) if records else None,
        }
    recent_correct = sum(1 for row in records if correct(recent_by_id.get(int(row["question_id"]))))
    prism_correct = sum(1 for row in records if correct(prism_by_id.get(int(row["question_id"]))))
    return {
        "samples": len(records),
        "recent6_correct": recent_correct,
        "recent6_accuracy": recent_correct / len(records) if records else None,
        "prism_correct": prism_correct,
        "prism_accuracy": prism_correct / len(records) if records else None,
        "grid": grid_summary,
        "mean_historical_chunks": mean([row["num_historical_chunks"] for row in records]) if records else None,
        "mean_raw_ledger_events": mean([row["num_raw_ledger_events"] for row in records]) if records else None,
        "mean_clip_embedding_ms_per_chunk": mean(
            [row["timing"]["clip_embedding_ms_per_chunk"] for row in records if row["timing"]["clip_embedding_ms_per_chunk"] is not None]
        )
        if records
        else None,
        "mean_segmentation_ms": mean([row["timing"]["segmentation_ms"] for row in records]) if records else None,
        "mean_event_scoring_ms": mean([row["timing"]["event_scoring_ms"] for row in records]) if records else None,
        "mean_total_ledger_ms": mean([row["timing"]["total_ledger_ms"] for row in records]) if records else None,
    }


def print_damage_rescue(records: list[dict[str, Any]], key: str) -> None:
    print("\nRECENT CORRECT / PRISM WRONG COUNTING CASES")
    cases = [row for row in records if row["recent_correct"] and not row["prism_correct"]]
    for row in cases:
        item = row["grid_eval"][key]
        print("-" * 80)
        print(f"Q{row['question_id']} {row['video_id']} {row['timestamp']}")
        print(row["question"])
        print(f"GT={row['ground_truth']} recent={row['recent_prediction']} prism={row['prism_prediction']}")
        print(f"ledger_count={item['ledger_count']} mapped={item['mapped_option']} correct={item['mcq_correct']}")
        print(f"clusters={row['grid'][key]['clusters']}")

    print("\nRECENT WRONG / PRISM CORRECT COUNTING CASES")
    cases = [row for row in records if not row["recent_correct"] and row["prism_correct"]]
    for row in cases:
        item = row["grid_eval"][key]
        print("-" * 80)
        print(f"Q{row['question_id']} {row['video_id']} {row['timestamp']}")
        print(row["question"])
        print(f"GT={row['ground_truth']} recent={row['recent_prediction']} prism={row['prism_prediction']}")
        print(f"ledger_count={item['ledger_count']} mapped={item['mapped_option']} correct={item['mcq_correct']}")


def write_flat_csv(path: Path, records: list[dict[str, Any]], best_key: str) -> None:
    fields = [
        "question_id",
        "video_id",
        "timestamp",
        "question",
        "ground_truth",
        "ground_truth_numeric_count",
        "recent_prediction",
        "recent_correct",
        "prism_prediction",
        "prism_correct",
        "num_historical_chunks",
        "num_raw_ledger_events",
        "ledger_count",
        "ledger_mapped_option",
        "ledger_mcq_correct",
        "ledger_absolute_count_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            item = row["grid_eval"][best_key]
            writer.writerow(
                {
                    "question_id": row["question_id"],
                    "video_id": row["video_id"],
                    "timestamp": row["timestamp"],
                    "question": row["question"],
                    "ground_truth": row["ground_truth"],
                    "ground_truth_numeric_count": row["ground_truth_numeric_count"],
                    "recent_prediction": row["recent_prediction"],
                    "recent_correct": row["recent_correct"],
                    "prism_prediction": row["prism_prediction"],
                    "prism_correct": row["prism_correct"],
                    "num_historical_chunks": row["num_historical_chunks"],
                    "num_raw_ledger_events": row["num_raw_ledger_events"],
                    "ledger_count": item["ledger_count"],
                    "ledger_mapped_option": item["mapped_option"],
                    "ledger_mcq_correct": item["mcq_correct"],
                    "ledger_absolute_count_error": item["absolute_count_error"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostic CLIP event ledger for StreamingBench Counting.")
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--recent", required=True, type=Path)
    parser.add_argument("--prism", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--context-time", type=float, default=70.0)
    parser.add_argument("--recent-window", type=int, default=6)
    parser.add_argument("--history-search-chunks", type=int, default=64)
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--gamma-events", default="0.24,0.26,0.28,0.30,0.32")
    parser.add_argument("--dedup-deltas", default="1.0,2.0,3.0")
    parser.add_argument("--dedup-iou", type=float, default=0.20)
    parser.add_argument("--clip-device", default=os.environ.get("MINICPM_EVENT_LEDGER_CLIP_DEVICE", ""))
    parser.add_argument("--clip-model", default=os.environ.get("MINICPM_EVENT_LEDGER_CLIP_MODEL", "openai/clip-vit-base-patch32"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recent_source, recent_rows = load_results(args.recent)
    prism_source, prism_rows = load_results(args.prism)
    recent_by_id = row_by_index(recent_rows)
    prism_by_id = row_by_index(prism_rows)
    tasks = load_counting_annotations(args.annotations, args.video_dir)
    scorer = EventLedgerClipScorer(model_name=args.clip_model, device=args.clip_device or None)
    gamma_values = parse_float_list(args.gamma_events)
    delta_values = parse_float_list(args.dedup_deltas)

    records: list[dict[str, Any]] = []
    for offset, task in enumerate(tasks, start=1):
        print(f"[{offset}/{len(tasks)}] Q{task['question_id']} {task['video_id']} {task['timestamp']}", flush=True)
        total_t0 = time.perf_counter()
        historical_chunks, decode_meta = decode_history_chunks(task, args)
        frames = [representative_frame(chunk) for chunk in historical_chunks]
        embed_t0 = time.perf_counter()
        image_embeddings = scorer.image_embeddings(frames)
        embed_ms = (time.perf_counter() - embed_t0) * 1000.0
        visual_change = None
        boundary_rows = event_boundary_scores(historical_chunks, image_embeddings, visual_change)
        seg_t0 = time.perf_counter()
        events = segment_events(historical_chunks, image_embeddings, boundary_rows)
        segmentation_ms = (time.perf_counter() - seg_t0) * 1000.0
        score_t0 = time.perf_counter()
        event_scores, query_meta = score_events_for_question(scorer, events, task["question"], task["options"])
        event_scoring_ms = (time.perf_counter() - score_t0) * 1000.0
        grid = ledger_grid(event_scores, task["options"], gamma_values, delta_values, iou_threshold=args.dedup_iou)
        gt_numeric = option_numeric_gt(task)
        grid_eval = evaluate_grid_record(grid, task["answer_gt"], gt_numeric)
        recent_row = recent_by_id.get(int(task["question_id"]))
        prism_row = prism_by_id.get(int(task["question_id"]))
        records.append(
            {
                **task,
                "ground_truth": task["answer_gt"],
                "ground_truth_numeric_count": gt_numeric,
                "recent_prediction": prediction(recent_row),
                "recent_correct": correct(recent_row),
                "prism_prediction": prediction(prism_row),
                "prism_correct": correct(prism_row),
                "num_historical_chunks": len(historical_chunks),
                "num_raw_ledger_events": len(events),
                "decode": decode_meta,
                "query_meta": query_meta,
                "boundary_scores": boundary_rows,
                "events": [event_to_json(event) for event in events],
                "event_scores": event_scores,
                "grid": grid,
                "grid_eval": grid_eval,
                "timing": {
                    "clip_embedding_ms": embed_ms,
                    "clip_embedding_ms_per_chunk": embed_ms / len(historical_chunks) if historical_chunks else None,
                    "segmentation_ms": segmentation_ms,
                    "event_scoring_ms": event_scoring_ms,
                    "total_ledger_ms": (time.perf_counter() - total_t0) * 1000.0,
                    "estimated_embedding_bytes": int(image_embeddings.numel() * image_embeddings.element_size()),
                },
            }
        )

    summary = summarize(records, recent_by_id, prism_by_id)
    summary["sources"] = {"recent": recent_source, "prism": prism_source, "annotations": str(args.annotations)}
    summary["counting_damage_rescue_counts"] = dict(
        Counter(
            "recent_correct_prism_wrong"
            if row["recent_correct"] and not row["prism_correct"]
            else "recent_wrong_prism_correct"
            if (not row["recent_correct"] and row["prism_correct"])
            else "both_correct"
            if row["recent_correct"] and row["prism_correct"]
            else "both_wrong"
            for row in records
        )
    )
    best_key = max(summary["grid"], key=lambda key: summary["grid"][key]["ledger_mcq_correct"]) if summary["grid"] else ""
    summary["best_grid_by_ledger_mcq"] = best_key

    (args.out_dir / "event_ledger_counting_records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out_dir / "event_ledger_counting_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if best_key:
        write_flat_csv(args.out_dir / "event_ledger_counting_best_grid.csv", records, best_key)

    print("\n" + "=" * 80)
    print("StreamingBench Counting Event Ledger Diagnostic")
    print("=" * 80)
    print(f"samples: {summary['samples']}")
    print(f"Recent-6: {summary['recent6_correct']}/{summary['samples']} = {100*summary['recent6_accuracy']:.2f}%")
    print(f"PRISM: {summary['prism_correct']}/{summary['samples']} = {100*summary['prism_accuracy']:.2f}%")
    print(f"mean historical chunks: {summary['mean_historical_chunks']}")
    print(f"mean raw ledger events: {summary['mean_raw_ledger_events']}")
    print("\nGRID")
    print("key,ledger_mcq,exact_count,MAE,Oracle(Recent,Ledger),Oracle(Recent,PRISM,Ledger)")
    for key, item in summary["grid"].items():
        print(
            f"{key},"
            f"{item['ledger_mcq_correct']}/{summary['samples']},"
            f"{item['exact_count_correct']}/{item['numeric_gt_count']},"
            f"{item['mean_absolute_error']},"
            f"{item['oracle_recent_ledger_correct']}/{summary['samples']},"
            f"{item['oracle_recent_prism_ledger_correct']}/{summary['samples']}"
        )
    if best_key:
        print(f"\nBest grid by ledger MCQ: {best_key}")
        print_damage_rescue(records, best_key)
    print(f"\nSaved: {args.out_dir}")


if __name__ == "__main__":
    main()
