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
    cluster_count_evidence,
    is_cumulative_count_question,
    numeric_options,
    option_for_count,
    score_chunks_for_count_target,
)
from lib.shared.recent_window import decode_video_to_chunks_qwen  # noqa: E402
from main_experiments.minicpm_v46.streamingbench.eval_baseline import resolve_video_path  # noqa: E402


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
            text = re.sub(r"^[A-E]\s*[\.\)]\s*", "", str(value).strip())
            out.append({"letter": letter, "text": text})
    return out


def load_counting_tasks(path: Path, video_dir: Path) -> list[dict[str, Any]]:
    data = json.load(path.open(encoding="utf-8"))
    tasks: list[dict[str, Any]] = []
    index = 0
    for video_entry in data:
        questions = sorted(video_entry.get("questions", []), key=lambda item: timestamp_to_seconds(item.get("time_stamp")))
        video_path_raw = video_entry.get("video_path") or video_entry.get("video") or video_entry.get("video_name")
        if not video_path_raw:
            raise KeyError(f"Missing video_path for annotation entry with keys: {sorted(video_entry)}")
        video_path = resolve_video_path(str(video_path_raw), str(video_dir))
        for question in questions:
            if str(question.get("task_type", "")).strip() == "Counting":
                options = parse_options(question)
                tasks.append(
                    {
                        "question_id": index,
                        "video_id": Path(video_path).stem,
                        "video_path": video_path,
                        "timestamp": question.get("time_stamp"),
                        "timestamp_seconds": timestamp_to_seconds(question.get("time_stamp")),
                        "category": "Counting",
                        "question": str(question.get("question", "")),
                        "options": options,
                        "ground_truth": extract_answer(question.get("answer")) or str(question.get("answer", "")).strip().upper(),
                        "numeric_options": numeric_options(options),
                    }
                )
            index += 1
    return tasks


def load_results(path: Path) -> tuple[str, list[dict[str, Any]]]:
    if path.is_file():
        payload = json.load(path.open(encoding="utf-8"))
        rows = payload.get("results", payload if isinstance(payload, list) else [])
        return str(path), rows
    merged = sorted(path.glob("streaming_bench_minicpmv46_results_*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    if merged:
        return load_results(merged[0])
    rank_files = sorted(path.glob("rank_*/results_incremental.jsonl"))
    if not rank_files:
        raise FileNotFoundError(f"No StreamingBench results found under {path}")
    rows: list[dict[str, Any]] = []
    for rank_file in rank_files:
        for line in rank_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return "\n  ".join(str(item) for item in rank_files), rows


def row_by_id(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("_index") is not None:
            out[int(row["_index"])] = row
        elif row.get("question_id") is not None:
            out[int(row["question_id"])] = row
    return out


def row_prediction(row: dict[str, Any] | None) -> str | None:
    if not row:
        return None
    for key in ("predicted_answer", "prediction", "model_answer", "parsed_answer", "answer_pred"):
        if row.get(key):
            answer = extract_answer(row.get(key))
            if answer:
                return answer
    return extract_answer(row.get("response"))


def row_correct(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("correct"))


def parse_float_list(raw: str) -> list[float]:
    return [float(item) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item) for item in raw.split(",") if item.strip()]


def decode_history(task: dict[str, Any], args: argparse.Namespace) -> tuple[list[Any], dict[str, Any]]:
    query_time = float(task["timestamp_seconds"])
    recent_start = max(0.0, query_time + 1e-4 - float(args.recent_window) * float(args.chunk_duration))
    history_start = max(0.0, query_time - max(float(args.context_time), float(args.chunk_duration)))
    chunks, backend = decode_video_to_chunks_qwen(
        video_path=task["video_path"],
        chunk_duration=float(args.chunk_duration),
        fps=float(args.fps),
        recent_frames_only=max(int(args.recent_window), int(math.ceil(float(args.context_time) * max(float(args.fps), 1.0)))),
        video_start=history_start,
        video_end=query_time + 1e-4,
    )
    historical = []
    violations = 0
    for chunk in chunks:
        timestamps = [float(ts) for ts in (getattr(chunk, "frame_timestamps", None) or []) if isinstance(ts, (int, float))]
        end = max(timestamps) if timestamps else float(getattr(chunk, "end_time", 0.0))
        if end < recent_start - 1e-6:
            historical.append(chunk)
        elif end >= recent_start and float(getattr(chunk, "start_time", 0.0)) < recent_start:
            violations += 1
    if args.history_search_chunks > 0:
        historical = historical[-int(args.history_search_chunks) :]
    return historical, {
        "decode_backend": backend,
        "decoded_chunks": len(chunks),
        "historical_chunks": len(historical),
        "recent_start_time": recent_start,
        "query_time": query_time,
        "history_start_time": history_start,
        "overlap_violations_excluded": violations,
    }


def evaluate_policy(
    records: list[dict[str, Any]],
    *,
    base_key: str,
    score_threshold: float,
    merge_gap_seconds: float,
    min_positive_run: int,
) -> dict[str, Any]:
    total = len(records)
    base_correct = 0
    final_correct = 0
    triggered = 0
    rescued = 0
    damaged = 0
    rejected_no_map = 0
    rejected_not_cumulative = 0
    for row in records:
        base_ok = bool(row[f"{base_key}_correct"])
        base_correct += int(base_ok)
        ledger = row["ledger_grid"][f"s{score_threshold:.3f}_g{merge_gap_seconds:.1f}_r{min_positive_run}"]
        use_ledger = bool(row["cumulative_count_required"] and ledger.get("mapped_option"))
        if not row["cumulative_count_required"]:
            rejected_not_cumulative += 1
        elif not ledger.get("mapped_option"):
            rejected_no_map += 1
        final_pred = ledger["mapped_option"] if use_ledger else row[f"{base_key}_prediction"]
        final_ok = bool(final_pred == row["ground_truth"])
        final_correct += int(final_ok)
        triggered += int(use_ledger)
        rescued += int((not base_ok) and final_ok)
        damaged += int(base_ok and not final_ok)
    return {
        "base_key": base_key,
        "score_threshold": score_threshold,
        "merge_gap_seconds": merge_gap_seconds,
        "min_positive_run": min_positive_run,
        "base_correct": base_correct,
        "base_accuracy": base_correct / total if total else None,
        "final_correct": final_correct,
        "final_accuracy": final_correct / total if total else None,
        "triggered": triggered,
        "trigger_rate": triggered / total if total else None,
        "rescued": rescued,
        "damaged": damaged,
        "net": rescued - damaged,
        "rejected_no_map": rejected_no_map,
        "rejected_not_cumulative": rejected_not_cumulative,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted StreamingBench Counting ledger override diagnostic.")
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
    parser.add_argument("--score-thresholds", default="-0.02,0.00,0.02,0.04,0.06,0.08")
    parser.add_argument("--merge-gaps", default="2.0,3.0,4.0,6.0")
    parser.add_argument("--min-positive-runs", default="1,2")
    parser.add_argument("--clip-device", default=os.environ.get("MINICPM_EVENT_LEDGER_CLIP_DEVICE", ""))
    parser.add_argument("--clip-model", default=os.environ.get("MINICPM_EVENT_LEDGER_CLIP_MODEL", "openai/clip-vit-base-patch32"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recent_source, recent_rows = load_results(args.recent)
    prism_source, prism_rows = load_results(args.prism)
    recent_by_id = row_by_id(recent_rows)
    prism_by_id = row_by_id(prism_rows)
    tasks = load_counting_tasks(args.annotations, args.video_dir)
    if args.max_samples > 0:
        tasks = tasks[: int(args.max_samples)]
    all_task_count = len(tasks)
    if args.num_shards > 1:
        tasks = [task for task_index, task in enumerate(tasks) if task_index % int(args.num_shards) == int(args.shard_index)]

    scorer = EventLedgerClipScorer(model_name=args.clip_model, device=args.clip_device or None)
    score_thresholds = parse_float_list(args.score_thresholds)
    merge_gaps = parse_float_list(args.merge_gaps)
    min_runs = parse_int_list(args.min_positive_runs)

    records: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] Q{task['question_id']} {task['video_id']} {task['timestamp']}", flush=True)
        t0 = time.perf_counter()
        historical_chunks, decode = decode_history(task, args)
        scored_chunks, score_meta = score_chunks_for_count_target(scorer, historical_chunks, task["question"])
        ledger_grid: dict[str, Any] = {}
        for threshold in score_thresholds:
            for gap in merge_gaps:
                for run in min_runs:
                    key = f"s{threshold:.3f}_g{gap:.1f}_r{run}"
                    clusters = cluster_count_evidence(
                        scored_chunks,
                        score_threshold=threshold,
                        merge_gap_seconds=gap,
                        min_positive_run=run,
                    )
                    count = len(clusters)
                    ledger_grid[key] = {
                        "ledger_count": int(count),
                        "mapped_option": option_for_count(count, task["options"]),
                        "clusters": clusters,
                        "score_threshold": float(threshold),
                        "merge_gap_seconds": float(gap),
                        "min_positive_run": int(run),
                        **score_meta,
                    }
        recent_row = recent_by_id.get(int(task["question_id"]))
        prism_row = prism_by_id.get(int(task["question_id"]))
        records.append(
            {
                **task,
                "recent_prediction": row_prediction(recent_row),
                "recent_correct": row_correct(recent_row),
                "prism_prediction": row_prediction(prism_row),
                "prism_correct": row_correct(prism_row),
                "cumulative_count_required": is_cumulative_count_question(task["question"], task["options"]),
                "decode": decode,
                "scored_chunks": scored_chunks,
                "ledger_grid": ledger_grid,
                "timing_ms": (time.perf_counter() - t0) * 1000.0,
            }
        )

    policies: list[dict[str, Any]] = []
    for base_key in ("recent", "prism"):
        for threshold in score_thresholds:
            for gap in merge_gaps:
                for run in min_runs:
                    policies.append(
                        evaluate_policy(
                            records,
                            base_key=base_key,
                            score_threshold=threshold,
                            merge_gap_seconds=gap,
                            min_positive_run=run,
                        )
                    )
    policies.sort(key=lambda item: (item["final_correct"], item["net"], -item["triggered"]), reverse=True)

    summary = {
        "sources": {"recent": recent_source, "prism": prism_source, "annotations": str(args.annotations)},
        "samples": len(records),
        "all_task_count": all_task_count,
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "counting_questions": len(records),
        "cumulative_count_required": sum(1 for row in records if row["cumulative_count_required"]),
        "recent_correct": sum(1 for row in records if row["recent_correct"]),
        "prism_correct": sum(1 for row in records if row["prism_correct"]),
        "policies": policies,
        "best_policy": policies[0] if policies else None,
        "mean_timing_ms": mean([row["timing_ms"] for row in records]) if records else None,
        "mean_historical_chunks": mean([row["decode"]["historical_chunks"] for row in records]) if records else None,
        "pattern_counts": dict(Counter("movie_clip" if "movie clip" in row["question"].lower() else "other" for row in records)),
    }

    (args.out_dir / "count_ledger_override_records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out_dir / "count_ledger_override_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.out_dir / "count_ledger_override_policies.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(policies[0]) if policies else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(policies)

    print("\n" + "=" * 80)
    print("StreamingBench Targeted Count Ledger Override")
    print("=" * 80)
    if args.num_shards > 1:
        print(f"shard: {args.shard_index}/{args.num_shards} ({summary['samples']}/{summary['all_task_count']} tasks)")
    print(f"samples: {summary['samples']}")
    print(f"cumulative_count_required: {summary['cumulative_count_required']}")
    print(f"Recent-6 Counting: {summary['recent_correct']}/{summary['samples']} = {100*summary['recent_correct']/summary['samples']:.2f}%")
    print(f"PRISM Counting: {summary['prism_correct']}/{summary['samples']} = {100*summary['prism_correct']/summary['samples']:.2f}%")
    print(f"mean historical chunks: {summary['mean_historical_chunks']:.2f}")
    print(f"mean timing ms/sample: {summary['mean_timing_ms']:.1f}")
    print("\nTOP POLICIES")
    print("base,threshold,gap,minrun,final,acc,rescue,damage,net,trigger")
    for item in policies[:15]:
        print(
            f"{item['base_key']},"
            f"{item['score_threshold']:.3f},"
            f"{item['merge_gap_seconds']:.1f},"
            f"{item['min_positive_run']},"
            f"{item['final_correct']}/{summary['samples']},"
            f"{100*item['final_accuracy']:.2f}%,"
            f"{item['rescued']},"
            f"{item['damaged']},"
            f"{item['net']},"
            f"{item['triggered']}"
        )
    print(f"\nSaved: {args.out_dir}")


if __name__ == "__main__":
    main()
