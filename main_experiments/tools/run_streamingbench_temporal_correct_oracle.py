#!/usr/bin/env python3
"""Temporal-correct PRISM oracle over a saved StreamingBench diagnostic set.

This runs MiniCPM generation for K=0..3 fixed prefixes of the saved, timestamp-
aligned PRISM candidate queue. It does not tune or modify PRISM.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

PSM_HISTORY_INSTRUCTION = (
    "When historical frames are present, they appear before the six recent frames.\n\n"
)

build_prompt = None
decode_video_to_chunks_qwen = None
resolve_video_path = None
select_recent_window_frames = None
timestamp_to_seconds = None
RecentWindowQAModel = None


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


def extract_mcq_answer(text: Any) -> str | None:
    if text is None:
        return None
    match = re.search(r"\b([A-E])\b", str(text).upper())
    return match.group(1) if match else None


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


def chunk_id(chunk: Any) -> int:
    return int(getattr(chunk, "chunk_index"))


def int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            out.append(int(item))
    return out


def as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def chunk_bounds(chunk: Any) -> tuple[float, float]:
    timestamps = getattr(chunk, "frame_timestamps", None) or []
    numeric = [float(ts) for ts in timestamps if isinstance(ts, (int, float)) and math.isfinite(float(ts))]
    if numeric:
        return min(numeric), max(numeric)
    start = as_float(getattr(chunk, "start_time", None))
    end = as_float(getattr(chunk, "end_time", None))
    if start is None:
        start = float(chunk_id(chunk))
    if end is None:
        end = start
    return (end, start) if end < start else (start, end)


def recent_start_time(chunks: list[Any]) -> float:
    if not chunks:
        raise ValueError("No recent chunks")
    return min(chunk_bounds(chunk)[0] for chunk in chunks)


def select_chunks_by_id(by_id: dict[int, Any], ids: list[int]) -> list[Any]:
    missing = [chunk_id_ for chunk_id_ in ids if chunk_id_ not in by_id]
    if missing:
        raise KeyError(f"Missing decoded chunk IDs: {missing}")
    return [by_id[chunk_id_] for chunk_id_ in ids]


def run_generation(
    qa: RecentWindowQAModel,
    prompt: str,
    gt: str | None,
    recent_chunks: list[Any],
    memory_chunks: list[Any],
) -> dict[str, Any]:
    ordered_memory = sorted(memory_chunks, key=lambda chunk: (chunk_bounds(chunk)[0], chunk_bounds(chunk)[1], chunk_id(chunk)))
    context_chunks = [*ordered_memory, *recent_chunks]
    final_prompt = f"{PSM_HISTORY_INSTRUCTION}{prompt}" if ordered_memory else prompt
    frames = [frame for chunk in context_chunks for frame in chunk.frames]
    t0 = time.perf_counter()
    response = qa.generate_from_frames(frames, final_prompt)
    generate_seconds = time.perf_counter() - t0
    prediction = extract_mcq_answer(response)
    correct = bool(gt and prediction == gt)
    return {
        "prediction": prediction,
        "correct": correct,
        "response": response,
        "chunk_ids": [chunk_id(chunk) for chunk in context_chunks],
        "memory_chunk_ids": [chunk_id(chunk) for chunk in ordered_memory],
        "num_frames": len(frames),
        "generate_seconds": generate_seconds,
    }


def numeric_auc(labels: list[bool], scores: list[float]) -> float | None:
    pairs = [(bool(label), float(score)) for label, score in zip(labels, scores) if math.isfinite(float(score))]
    positives = [score for label, score in pairs if label]
    negatives = [score for label, score in pairs if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    ties = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                ties += 1.0
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def safe_mean(values: list[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    return mean(clean) if clean else None


def candidate_signal(row: dict[str, Any], signal: str) -> float | None:
    candidates = row.get("candidate_prefix") or []
    first = candidates[0] if candidates else {}
    if signal == "candidate_total_score":
        return as_float(first.get("total_score"))
    if signal == "candidate_semantic_score":
        return as_float(first.get("semantic_score"))
    if signal == "temporal_distance_seconds":
        return as_float(first.get("candidate_temporal_distance_seconds"))
    if signal == "candidate_total_score_times_normalized_temporal_distance":
        total = as_float(first.get("total_score"))
        distance = as_float(first.get("candidate_temporal_distance_seconds"))
        if total is None or distance is None:
            return None
        return total * min(1.0, max(0.0, distance / 64.0))
    if signal == "candidate_semantic_minus_current_support":
        semantic = as_float(first.get("semantic_score"))
        support = as_float(row.get("visual_support_norm"))
        if semantic is None or support is None:
            return None
        return semantic - support
    if signal == "current_sufficiency":
        return as_float(row.get("current_sufficiency"))
    return None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    by_k_acc = {}
    for k in range(4):
        correct = sum(bool(row["branches"][str(k)]["correct"]) for row in rows)
        by_k_acc[f"K{k}_accuracy"] = correct / total if total else None
        by_k_acc[f"K{k}_correct"] = correct
    oracle_correct = sum(any(bool(row["branches"][str(k)]["correct"]) for k in range(4)) for row in rows)

    k0_wrong = [row for row in rows if not bool(row["branches"]["0"]["correct"])]
    rescue_counts = Counter()
    for row in k0_wrong:
        minimum = row.get("minimum_k_to_correct")
        if minimum == 1:
            rescue_counts["rescuable_with_K1"] += 1
        elif minimum == 2:
            rescue_counts["additionally_rescuable_with_K2"] += 1
        elif minimum == 3:
            rescue_counts["additionally_rescuable_with_K3"] += 1
        else:
            rescue_counts["not_rescuable"] += 1

    confidently_wrong = [row for row in rows if row.get("confidently_wrong")]
    confident_rescuable = [row for row in confidently_wrong if row.get("minimum_k_to_correct") in {1, 2, 3}]
    confident_not = [row for row in confidently_wrong if row.get("minimum_k_to_correct") not in {1, 2, 3}]
    signals = [
        "candidate_total_score",
        "candidate_semantic_score",
        "temporal_distance_seconds",
        "candidate_total_score_times_normalized_temporal_distance",
        "candidate_semantic_minus_current_support",
        "current_sufficiency",
    ]
    aucs: dict[str, Any] = {}
    labels = [row in confident_rescuable for row in confidently_wrong]
    for signal in signals:
        values = [candidate_signal(row, signal) for row in confidently_wrong]
        valid_labels = [label for label, value in zip(labels, values) if value is not None]
        valid_values = [float(value) for value in values if value is not None]
        aucs[signal] = numeric_auc(valid_labels, valid_values) if valid_values else None

    k0_correct = [row for row in rows if bool(row["branches"]["0"]["correct"])]
    damage = {
        f"K{k}_damage_from_K0_correct": sum(not bool(row["branches"][str(k)]["correct"]) for row in k0_correct)
        for k in (1, 2, 3)
    }

    category_summary: dict[str, Any] = {}
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row.get("category", "unknown"))].append(row)
    for category, group in sorted(by_category.items()):
        g_total = len(group)
        g_k0 = sum(bool(row["branches"]["0"]["correct"]) for row in group)
        g_oracle = sum(any(bool(row["branches"][str(k)]["correct"]) for k in range(4)) for row in group)
        category_summary[category] = {
            "total": g_total,
            "K0_accuracy": g_k0 / g_total if g_total else None,
            "Oracle_K_accuracy": g_oracle / g_total if g_total else None,
            "oracle_headroom": (g_oracle - g_k0) / g_total if g_total else None,
        }

    return {
        "samples": total,
        **by_k_acc,
        "Oracle_K_correct": oracle_correct,
        "Oracle_K_accuracy": oracle_correct / total if total else None,
        "K0_wrong_samples": len(k0_wrong),
        "K0_wrong_rescue_breakdown": dict(rescue_counts),
        "confidently_wrong_samples": len(confidently_wrong),
        "confidently_wrong_rescuable_any_K": len(confident_rescuable),
        "confidently_wrong_not_rescuable": len(confident_not),
        "confidently_wrong_minimum_K_distribution": dict(Counter(str(row.get("minimum_k_to_correct")) for row in confidently_wrong)),
        "confidently_wrong_rescue_signal_auc": aucs,
        "K0_correct_samples": len(k0_correct),
        "K0_correct_damage": damage,
        "category_summary": category_summary,
        "mean_candidates": safe_mean([float(len(row.get("candidate_prefix") or [])) for row in rows]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "question_id",
        "video_id",
        "category",
        "timestamp",
        "ground_truth",
        "answer_margin",
        "visual_support_norm",
        "current_sufficiency",
        "confidently_wrong",
        "minimum_k_to_correct",
        "K0_prediction",
        "K0_correct",
        "K1_prediction",
        "K1_correct",
        "K2_prediction",
        "K2_correct",
        "K3_prediction",
        "K3_correct",
        "candidate1_total_score",
        "candidate1_semantic_score",
        "candidate1_temporal_distance_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            candidate1 = (row.get("candidate_prefix") or [{}])[0]
            writer.writerow(
                {
                    "question_id": row["question_id"],
                    "video_id": row.get("video_id"),
                    "category": row.get("category"),
                    "timestamp": row.get("timestamp"),
                    "ground_truth": row.get("ground_truth"),
                    "answer_margin": row.get("answer_margin"),
                    "visual_support_norm": row.get("visual_support_norm"),
                    "current_sufficiency": row.get("current_sufficiency"),
                    "confidently_wrong": row.get("confidently_wrong"),
                    "minimum_k_to_correct": row.get("minimum_k_to_correct"),
                    "K0_prediction": row["branches"]["0"]["prediction"],
                    "K0_correct": row["branches"]["0"]["correct"],
                    "K1_prediction": row["branches"]["1"]["prediction"],
                    "K1_correct": row["branches"]["1"]["correct"],
                    "K2_prediction": row["branches"]["2"]["prediction"],
                    "K2_correct": row["branches"]["2"]["correct"],
                    "K3_prediction": row["branches"]["3"]["prediction"],
                    "K3_correct": row["branches"]["3"]["correct"],
                    "candidate1_total_score": candidate1.get("total_score"),
                    "candidate1_semantic_score": candidate1.get("semantic_score"),
                    "candidate1_temporal_distance_seconds": candidate1.get("candidate_temporal_distance_seconds"),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Fixed PRISM StreamingBench diagnostic result dir or JSON.")
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
    source_path, saved_rows = load_results(Path(args.results))
    annotations = load_annotations(Path(args.annotations), args.video_dir)
    if args.max_samples > 0:
        saved_rows = saved_rows[: args.max_samples]

    qa = RecentWindowQAModel(
        model_name=args.qa_model,
        device=args.qa_device or default_qa_device(),
        max_new_tokens=args.max_qa_tokens,
        attn_implementation=args.attn_implementation,
    )

    completed: dict[int, dict[str, Any]] = {}
    jsonl_path = out_dir / "temporal_correct_oracle_results.jsonl"
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
            adaptive = saved.get("adaptive") or {}
            iterations = adaptive.get("iterations") or []
            iter0 = iterations[0] if iterations and isinstance(iterations[0], dict) else {}
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
            r_start = recent_start_time(recent_chunks)

            candidate_meta = list(adaptive.get("candidate_queue") or [])[:3]
            candidate_ids = int_list([candidate.get("chunk_id") for candidate in candidate_meta])
            candidate_chunks = select_chunks_by_id(broad_by_id, candidate_ids) if candidate_ids else []
            for meta, chunk in zip(candidate_meta, candidate_chunks):
                start, end = chunk_bounds(chunk)
                distance = r_start - end
                meta["start_time_seconds"] = start
                meta["end_time_seconds"] = end
                meta["candidate_temporal_distance_seconds"] = distance
                if not end < r_start:
                    raise AssertionError(
                        f"Temporal violation qid={qid}: candidate={chunk_id(chunk)} end={end} recent_start={r_start}"
                    )

            branches: dict[str, dict[str, Any]] = {}
            for k in range(4):
                memory = candidate_chunks[:k]
                branches[str(k)] = run_generation(qa, prompt, gt, recent_chunks, memory)

            minimum_k = None
            if not branches["0"]["correct"]:
                for k in (1, 2, 3):
                    if branches[str(k)]["correct"]:
                        minimum_k = k
                        break
            elif branches["0"]["correct"]:
                minimum_k = 0

            answer_margin = as_float(iter0.get("answer_margin"))
            visual_support = as_float(iter0.get("visual_support_norm"))
            current_sufficiency = as_float(iter0.get("sufficiency"))
            output = {
                "question_id": qid,
                "key": saved.get("_key"),
                "video_id": task["video"],
                "video_path": task["video_path"],
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
                    "recent_video_start": recent_video_start,
                    "recent_video_end": video_end,
                    "recent_start_time_seconds": r_start,
                    "recent_chunk_ids": [chunk_id(chunk) for chunk in recent_chunks],
                },
                "candidate_prefix": candidate_meta,
                "branches": branches,
                "minimum_k_to_correct": minimum_k,
                "answer_margin": answer_margin,
                "visual_support_norm": visual_support,
                "current_sufficiency": current_sufficiency,
                "confidently_wrong": bool(
                    not branches["0"]["correct"]
                    and answer_margin is not None
                    and answer_margin >= 0.90
                ),
                "saved_psm_correct": bool(saved.get("correct")),
                "saved_psm_response": saved.get("response"),
                "saved_psm_adaptive": adaptive,
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            handle.flush()
            completed[qid] = output

    rows = [completed[key] for key in sorted(completed)]
    summary = summarize(rows)
    (out_dir / "temporal_correct_oracle_results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "temporal_correct_oracle_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(out_dir / "temporal_correct_oracle_results.csv", rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
