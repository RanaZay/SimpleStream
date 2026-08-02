#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_records(result_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(result_dir.glob("rank_*/results_incremental.jsonl"))
    if not paths:
        paths = sorted(result_dir.glob("**/results_incremental.jsonl"))
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record["_file"] = str(path)
                records.append(record)
    dedup: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        key = record.get("_key") or f"{record.get('_file')}:{index}"
        dedup[str(key)] = record
    return list(dedup.values())


def _adaptive(record: dict[str, Any]) -> dict[str, Any]:
    profile = record.get("profile")
    if isinstance(profile, dict) and isinstance(profile.get("adaptive"), dict):
        return profile["adaptive"]
    adaptive = record.get("adaptive")
    if isinstance(adaptive, dict):
        return adaptive
    return {}


def _task(record: dict[str, Any]) -> str:
    return str(record.get("task_type") or record.get("task") or record.get("video_categories") or "unknown")


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _pct(numer: int, denom: int) -> float:
    return 100.0 * numer / denom if denom else 0.0


def _time_seconds(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        nums = [float(part) for part in parts]
    except ValueError:
        return None
    if len(nums) == 3:
        return nums[0] * 3600.0 + nums[1] * 60.0 + nums[2]
    if len(nums) == 2:
        return nums[0] * 60.0 + nums[1]
    if len(nums) == 1:
        return nums[0]
    return None


def _memory_scores(adaptive: dict[str, Any]) -> list[dict[str, Any]]:
    scores = adaptive.get("memory_scores")
    if isinstance(scores, list):
        return [item for item in scores if isinstance(item, dict)]
    return []


def _selected_score(scores: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in scores:
        if item.get("selected"):
            return item
    return None


def _best_score(scores: list[dict[str, Any]]) -> float | None:
    vals = [float(item["event_summary_score"]) for item in scores if item.get("event_summary_score") is not None]
    return max(vals) if vals else None


def _margin(scores: list[dict[str, Any]]) -> float | None:
    for item in scores:
        if item.get("event_summary_margin") is not None:
            return float(item["event_summary_margin"])
    vals = sorted(
        [float(item["event_summary_score"]) for item in scores if item.get("event_summary_score") is not None],
        reverse=True,
    )
    if len(vals) >= 2:
        return vals[0] - vals[1]
    if len(vals) == 1:
        return vals[0]
    return None


def _print_metric(name: str, value: Any) -> None:
    if value is None:
        print(f"{name}: n/a")
    elif isinstance(value, float):
        print(f"{name}: {value:.4f}")
    else:
        print(f"{name}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Conditional Event Bookmark Memory diagnostics.")
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of text.")
    args = parser.parse_args()

    records = _load_records(args.result_dir)
    total = len(records)
    selected_records = []
    gate_activated = 0
    gate_enabled = 0
    frames: list[float] = []
    best_scores: list[float] = []
    margins: list[float] = []
    anchor_ages: list[float] = []
    scan_ms: list[float] = []
    retrieval_ms: list[float] = []
    by_task: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "gate": 0, "retrieved": 0, "correct": 0})
    retrieved_debug: list[dict[str, Any]] = []

    for record in records:
        adaptive = _adaptive(record)
        task = _task(record)
        by_task[task]["total"] += 1
        if bool(record.get("correct")):
            by_task[task]["correct"] += 1
        if record.get("num_frames") is not None:
            frames.append(float(record["num_frames"]))

        gate = adaptive.get("memory_gate")
        gate_on = bool(isinstance(gate, dict) and gate.get("activated"))
        if isinstance(gate, dict) and gate.get("enabled"):
            gate_enabled += 1
        if gate_on:
            gate_activated += 1
            by_task[task]["gate"] += 1

        memory_chunk_ids = adaptive.get("memory_chunk_ids") or []
        retrieved = bool(memory_chunk_ids)
        if retrieved:
            selected_records.append(record)
            by_task[task]["retrieved"] += 1

        scores = _memory_scores(adaptive)
        best = _best_score(scores)
        if best is not None and math.isfinite(best):
            best_scores.append(best)
        margin = _margin(scores)
        if margin is not None and math.isfinite(margin):
            margins.append(margin)
        for item in scores:
            if item.get("event_scan_latency_ms") is not None:
                scan_ms.append(float(item["event_scan_latency_ms"]))
                break
        for item in scores:
            if item.get("bookmark_retrieval_ms") is not None:
                retrieval_ms.append(float(item["bookmark_retrieval_ms"]))
                break

        selected = _selected_score(scores)
        if selected is not None:
            q_time = _time_seconds(record.get("time_stamp"))
            anchor_time = selected.get("timestamp")
            if q_time is not None and anchor_time is not None:
                anchor_ages.append(max(0.0, q_time - float(anchor_time)))
            retrieved_debug.append(
                {
                    "question": record.get("question"),
                    "task": task,
                    "correct": bool(record.get("correct")),
                    "answer_gt": record.get("answer_gt"),
                    "response": record.get("response"),
                    "gate": gate,
                    "best_score": best,
                    "margin": margin,
                    "retrieved_chunk_id": selected.get("chunk_id"),
                    "retrieved_timestamp": selected.get("timestamp"),
                    "summary": selected.get("summary"),
                    "recent_chunk_ids": adaptive.get("recent_chunk_ids"),
                    "selected_chunk_ids": adaptive.get("selected_chunk_ids"),
                }
            )

    diagnostics = {
        "result_dir": str(args.result_dir),
        "records": total,
        "accuracy": _pct(sum(1 for record in records if record.get("correct")), total),
        "retrieval_trigger_rate": _pct(len(selected_records), total),
        "gate_activation_rate": _pct(gate_activated, total),
        "gate_enabled_rate": _pct(gate_enabled, total),
        "average_frames_per_question": _mean(frames),
        "average_best_score": _mean(best_scores),
        "average_score_margin": _mean(margins),
        "average_anchor_age_seconds": _mean(anchor_ages),
        "event_scan_latency_ms": _mean(scan_ms),
        "bookmark_retrieval_latency_ms": _mean(retrieval_ms),
        "retrieval_rate_by_question_category": {
            task: {
                "total": counts["total"],
                "accuracy": _pct(counts["correct"], counts["total"]),
                "gate_activation_rate": _pct(counts["gate"], counts["total"]),
                "retrieval_trigger_rate": _pct(counts["retrieved"], counts["total"]),
            }
            for task, counts in sorted(by_task.items())
        },
        "retrieved_examples": retrieved_debug[:20],
    }

    if args.json:
        print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
        return

    print("=" * 80)
    print("Conditional Event Bookmark Memory Diagnostics")
    print("=" * 80)
    _print_metric("records", diagnostics["records"])
    _print_metric("accuracy (%)", diagnostics["accuracy"])
    _print_metric("retrieval_trigger_rate (%)", diagnostics["retrieval_trigger_rate"])
    _print_metric("gate_activation_rate (%)", diagnostics["gate_activation_rate"])
    _print_metric("average_frames_per_question", diagnostics["average_frames_per_question"])
    _print_metric("average_best_score", diagnostics["average_best_score"])
    _print_metric("average_score_margin", diagnostics["average_score_margin"])
    _print_metric("average_anchor_age_seconds", diagnostics["average_anchor_age_seconds"])
    _print_metric("event_scan_latency_ms", diagnostics["event_scan_latency_ms"])
    _print_metric("bookmark_retrieval_latency_ms", diagnostics["bookmark_retrieval_latency_ms"])
    print("\nBy category:")
    for task, values in diagnostics["retrieval_rate_by_question_category"].items():
        print(
            f"- {task}: acc={values['accuracy']:.2f}%, "
            f"gate={values['gate_activation_rate']:.2f}%, "
            f"retrieval={values['retrieval_trigger_rate']:.2f}% "
            f"({values['total']} samples)"
        )
    print("\nFirst retrieved examples:")
    for example in diagnostics["retrieved_examples"][:5]:
        print("-" * 80)
        print(f"task: {example['task']}")
        print(f"question: {example['question']}")
        print(f"response: {example['response']} | gt: {example['answer_gt']} | correct: {example['correct']}")
        print(f"best_score: {example['best_score']} | margin: {example['margin']}")
        print(f"retrieved: chunk={example['retrieved_chunk_id']} t={example['retrieved_timestamp']}")
        print(f"summary: {example['summary']}")


if __name__ == "__main__":
    main()
