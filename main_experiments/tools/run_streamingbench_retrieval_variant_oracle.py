#!/usr/bin/env python3
"""StreamingBench diagnostic oracle for isolated PRISM retrieval variants."""

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

from lib.minicpm.prism_retrieval_variants import rank_candidates
from main_experiments.minicpm_v46.streamingbench.eval_prism_exact_recent_dist import (
    select_exact_current_recent_frames,
)

PSM_HISTORY_INSTRUCTION = (
    "When historical frames are present, they appear before the six recent frames.\n\n"
)
VARIANTS = ["current", "clip_question", "clip_question_options", "clip_mmr"]


def extract_mcq_answer(text: Any) -> str | None:
    if text is None:
        return None
    match = re.search(r"\b([A-E])\b", str(text).upper())
    return match.group(1) if match else None


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
    for row in rows:
        dedup[int(row.get("_index", len(dedup)))] = row
    return "\n  ".join(str(item) for item in rank_files), [dedup[key] for key in sorted(dedup)]


def load_annotations(path: Path, video_dir: str) -> dict[int, dict[str, Any]]:
    from main_experiments.minicpm_v46.streamingbench.eval_baseline import (
        build_prompt,
        resolve_video_path,
        timestamp_to_seconds,
    )

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
                    "timestamp_seconds": float(timestamp_to_seconds(question["time_stamp"])),
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


def as_float(value: Any) -> float | None:
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
    start = as_float(getattr(chunk, "start_time", None))
    end = as_float(getattr(chunk, "end_time", None))
    if start is None:
        start = float(chunk_id(chunk))
    if end is None:
        end = start
    return (end, start) if end < start else (start, end)


def context_bounds(chunks: list[Any]) -> tuple[float, float]:
    starts = [chunk_bounds(chunk)[0] for chunk in chunks]
    ends = [chunk_bounds(chunk)[1] for chunk in chunks]
    return min(starts), max(ends)


def run_generation(qa: Any, prompt: str, gt: str | None, recent_chunks: list[Any], memory_chunks: list[Any]) -> dict[str, Any]:
    ordered_memory = sorted(memory_chunks, key=lambda chunk: (chunk_bounds(chunk)[0], chunk_bounds(chunk)[1], chunk_id(chunk)))
    frames = [frame for chunk in [*ordered_memory, *recent_chunks] for frame in chunk.frames]
    final_prompt = f"{PSM_HISTORY_INSTRUCTION}{prompt}" if ordered_memory else prompt
    t0 = time.perf_counter()
    response = qa.generate_from_frames(frames, final_prompt)
    elapsed = time.perf_counter() - t0
    prediction = extract_mcq_answer(response)
    return {
        "prediction": prediction,
        "correct": bool(gt and prediction == gt),
        "response": response,
        "memory_chunk_ids": [chunk_id(chunk) for chunk in ordered_memory],
        "final_chunk_ids": [chunk_id(chunk) for chunk in [*ordered_memory, *recent_chunks]],
        "num_frames": len(frames),
        "generate_seconds": elapsed,
        "num_vision_tokens": getattr(qa, "_last_num_vision_tokens", None),
        "ttft_seconds": getattr(qa, "_last_ttft_seconds", None),
    }


def load_completed(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[tuple[int, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        completed[(int(row["question_id"]), str(row["retrieval_variant"]))] = row
    return completed


def summarize_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if not rows:
        return {"samples": 0}
    by_k: dict[str, Any] = {}
    for k in range(4):
        correct = sum(bool(row["branches"][str(k)]["correct"]) for row in rows)
        by_k[f"K{k}_correct"] = correct
        by_k[f"K{k}_accuracy"] = correct / total
    oracle_correct = sum(any(bool(row["branches"][str(k)]["correct"]) for k in range(4)) for row in rows)
    k0_wrong = [row for row in rows if not bool(row["branches"]["0"]["correct"])]
    rescue = Counter()
    for row in k0_wrong:
        minimum = row.get("minimum_k_to_correct")
        if minimum == 1:
            rescue["rescuable_with_K1"] += 1
        elif minimum == 2:
            rescue["additional_rescue_at_K2"] += 1
        elif minimum == 3:
            rescue["additional_rescue_at_K3"] += 1
        else:
            rescue["not_rescuable"] += 1
    k0_correct = [row for row in rows if bool(row["branches"]["0"]["correct"])]
    damage = {
        f"K{k}_damage_from_K0_correct": sum(not bool(row["branches"][str(k)]["correct"]) for row in k0_correct)
        for k in (1, 2, 3)
    }
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row.get("category", "unknown"))].append(row)
    category_summary = {}
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
        **by_k,
        "Oracle_K_correct": oracle_correct,
        "Oracle_K_accuracy": oracle_correct / total,
        "K0_wrong_samples": len(k0_wrong),
        "K0_wrong_rescue_breakdown": dict(rescue),
        "K0_correct_samples": len(k0_correct),
        "K0_correct_damage": damage,
        "category_summary": category_summary,
        "retrieval_quality": aggregate_quality(rows),
    }


def safe_mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return mean(clean) if clean else None


def aggregate_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    top1 = []
    top3 = []
    distances = []
    redundancy = []
    gap_fallbacks = []
    option_counts: Counter[str] = Counter()
    for row in rows:
        fallback = as_float((row.get("retrieval_quality") or {}).get("temporal_gap_fallback_count"))
        if fallback is not None:
            gap_fallbacks.append(fallback)
        queue = row.get("candidate_queue") or []
        if queue:
            top1.append(float(queue[0].get("retrieval_relevance", queue[0].get("semantic_score", 0.0))))
            for item in queue[:3]:
                top3.append(float(item.get("retrieval_relevance", item.get("semantic_score", 0.0))))
                distance = as_float(item.get("candidate_temporal_distance_seconds"))
                if distance is not None:
                    distances.append(distance)
                red = as_float(item.get("visual_redundancy"))
                if red is not None:
                    redundancy.append(red)
                option = item.get("best_supported_option")
                if option is not None:
                    option_counts[str(option)] += 1
    return {
        "mean_top1_relevance": safe_mean(top1),
        "mean_top3_relevance": safe_mean(top3),
        "mean_temporal_distance_seconds": safe_mean(distances),
        "mean_visual_embedding_redundancy": safe_mean(redundancy),
        "temporal_gap_fallback_count_total": int(sum(gap_fallbacks)) if gap_fallbacks else 0,
        "temporal_gap_fallback_count_mean": safe_mean(gap_fallbacks),
        "best_supported_option_distribution": dict(option_counts),
    }


def newly_rescuable(rows_by_variant: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    current = rows_by_variant.get("current", [])
    current_rescued = {
        int(row["question_id"])
        for row in current
        if row.get("minimum_k_to_correct") in {1, 2, 3}
    }
    out: dict[str, list[dict[str, Any]]] = {}
    for variant, rows in rows_by_variant.items():
        if variant == "current":
            continue
        cases = []
        for row in rows:
            if row.get("minimum_k_to_correct") in {1, 2, 3} and int(row["question_id"]) not in current_rescued:
                cases.append(
                    {
                        "question_id": row["question_id"],
                        "category": row.get("category"),
                        "timestamp": row.get("timestamp"),
                        "question": row.get("question"),
                        "ground_truth": row.get("ground_truth"),
                        "minimum_k_to_correct": row.get("minimum_k_to_correct"),
                        "candidate_chunk_ids": [item.get("chunk_id") for item in row.get("candidate_queue", [])[:3]],
                    }
                )
        out[variant] = cases
    return out


def queue_ids(row: dict[str, Any], k: int) -> set[int]:
    ids: set[int] = set()
    for item in (row.get("candidate_queue") or [])[:k]:
        value = item.get("chunk_id")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            ids.add(int(value))
    return ids


def queue_overlap_stats(rows_by_variant: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    pairs = [
        ("current", "clip_question"),
        ("current", "clip_question_options"),
        ("current", "clip_mmr"),
        ("clip_question_options", "clip_mmr"),
    ]
    by_variant_id = {
        variant: {int(row["question_id"]): row for row in rows}
        for variant, rows in rows_by_variant.items()
    }
    out: dict[str, dict[str, Any]] = {}
    for left, right in pairs:
        left_rows = by_variant_id.get(left, {})
        right_rows = by_variant_id.get(right, {})
        common = sorted(set(left_rows) & set(right_rows))
        overlap1 = []
        overlap3 = []
        for qid in common:
            left1 = queue_ids(left_rows[qid], 1)
            right1 = queue_ids(right_rows[qid], 1)
            left3 = queue_ids(left_rows[qid], 3)
            right3 = queue_ids(right_rows[qid], 3)
            overlap1.append(1.0 if left1 and left1 == right1 else 0.0)
            denom = min(len(left3), len(right3), 3)
            overlap3.append((len(left3 & right3) / denom) if denom else 0.0)
        out[f"{left}_vs_{right}"] = {
            "samples": len(common),
            "overlap@1": safe_mean(overlap1),
            "overlap@3": safe_mean(overlap3),
        }
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "question_id",
        "retrieval_variant",
        "category",
        "ground_truth",
        "K0_prediction",
        "K0_correct",
        "K1_prediction",
        "K1_correct",
        "K2_prediction",
        "K2_correct",
        "K3_prediction",
        "K3_correct",
        "minimum_k_to_correct",
        "candidate1_chunk_id",
        "candidate1_relevance",
        "candidate1_total_score",
        "candidate1_temporal_distance_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            candidate1 = (row.get("candidate_queue") or [{}])[0]
            writer.writerow(
                {
                    "question_id": row["question_id"],
                    "retrieval_variant": row["retrieval_variant"],
                    "category": row.get("category"),
                    "ground_truth": row.get("ground_truth"),
                    "K0_prediction": row["branches"]["0"]["prediction"],
                    "K0_correct": row["branches"]["0"]["correct"],
                    "K1_prediction": row["branches"]["1"]["prediction"],
                    "K1_correct": row["branches"]["1"]["correct"],
                    "K2_prediction": row["branches"]["2"]["prediction"],
                    "K2_correct": row["branches"]["2"]["correct"],
                    "K3_prediction": row["branches"]["3"]["prediction"],
                    "K3_correct": row["branches"]["3"]["correct"],
                    "minimum_k_to_correct": row.get("minimum_k_to_correct"),
                    "candidate1_chunk_id": candidate1.get("chunk_id"),
                    "candidate1_relevance": candidate1.get("retrieval_relevance"),
                    "candidate1_total_score": candidate1.get("total_score"),
                    "candidate1_temporal_distance_seconds": candidate1.get("candidate_temporal_distance_seconds"),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-source", required=True, help="Saved SB-100 result JSON/dir that defines sample IDs.")
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
    parser.add_argument("--history-search-chunks", type=int, default=64)
    parser.add_argument("--candidate-pool", type=int, default=12)
    parser.add_argument("--min-temporal-gap", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    args = parser.parse_args()

    from lib.minicpm.baseline import RecentWindowQAModel
    from lib.shared.recent_window import decode_video_to_chunks_qwen
    from lib.minicpm.adaptive import AdaptiveWindowConfig

    try:
        import torch

        default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        default_device = "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_path, sample_rows = load_results(Path(args.sample_source))
    if args.max_samples > 0:
        sample_rows = sample_rows[: args.max_samples]
    annotations = load_annotations(Path(args.annotations), args.video_dir)
    qa = RecentWindowQAModel(
        model_name=args.qa_model,
        device=args.qa_device or default_device,
        max_new_tokens=args.max_qa_tokens,
        attn_implementation=args.attn_implementation,
    )
    config = AdaptiveWindowConfig.from_env()

    jsonl_path = out_dir / "retrieval_variant_oracle_results.jsonl"
    completed = load_completed(jsonl_path)
    all_rows: list[dict[str, Any]] = list(completed.values())
    with jsonl_path.open("a", encoding="utf-8") as handle:
        for ordinal, saved in enumerate(sample_rows, start=1):
            qid = int(saved.get("_index", ordinal - 1))
            task = annotations[qid]
            question = task["question_obj"]
            gt = answer_gt_from(saved, question)
            prompt = task["prompt"]
            ts_sec = float(task["timestamp_seconds"])
            video_end = ts_sec + 1e-4
            broad_start = max(0.0, ts_sec - max(float(args.context_time), float(args.chunk_duration)))
            decode_recent_hint = max(
                int(math.ceil(float(args.context_time) / max(float(args.chunk_duration), 1e-6))),
                int(args.recent_window) + int(args.history_search_chunks),
            )
            print(f"[{ordinal}/{len(sample_rows)}] qid={qid}", flush=True)
            broad_chunks, broad_backend = decode_video_to_chunks_qwen(
                video_path=task["video_path"],
                chunk_duration=args.chunk_duration,
                fps=args.fps,
                recent_frames_only=decode_recent_hint,
                video_start=broad_start,
                video_end=video_end,
            )
            recent_video_start = max(0.0, video_end - float(args.recent_window) * float(args.chunk_duration))
            recent_selection = select_exact_current_recent_frames(
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
            recent_start, recent_end = context_bounds(recent_chunks)
            older_chunks = [
                chunk
                for chunk in broad_chunks
                if chunk_bounds(chunk)[1] < float(recent_start) - 1e-6
            ][-int(args.history_search_chunks) :]

            k0 = None
            for variant in args.variants:
                key = (qid, variant)
                if key in completed:
                    continue
                queue, ranking_ms, quality = rank_candidates(
                    qa=qa,
                    older_chunks=older_chunks,
                    prompt=prompt,
                    config=config,
                    candidate_pool=args.candidate_pool,
                    min_temporal_gap=args.min_temporal_gap,
                    variant=variant,
                    recent_start_time=recent_start,
                )
                violations = [
                    item for item in queue if bool(item.get("history_temporal_violation"))
                ]
                if violations:
                    first = violations[0]
                    raise AssertionError(
                        f"Temporal violation qid={qid} variant={variant}: "
                        f"candidate={first.get('chunk_id')} end={first.get('end_time_seconds')} "
                        f"recent_start={recent_start}"
                    )
                branches: dict[str, dict[str, Any]] = {}
                if k0 is None:
                    k0 = run_generation(qa, prompt, gt, recent_chunks, [])
                branches["0"] = dict(k0)
                candidate_chunks = [item["chunk"] for item in queue[:3]]
                for k in (1, 2, 3):
                    branches[str(k)] = run_generation(qa, prompt, gt, recent_chunks, candidate_chunks[:k])
                minimum_k = None
                if not branches["0"]["correct"]:
                    for k in (1, 2, 3):
                        if branches[str(k)]["correct"]:
                            minimum_k = k
                            break
                else:
                    minimum_k = 0
                output = {
                    "question_id": qid,
                    "retrieval_variant": variant,
                    "source_path": source_path,
                    "video_id": task["video"],
                    "video_path": task["video_path"],
                    "timestamp": question.get("time_stamp"),
                    "timestamp_seconds": ts_sec,
                    "category": saved.get("task_type") or task.get("task_type"),
                    "question": question.get("question"),
                    "options": question.get("options"),
                    "ground_truth": gt,
                    "decode": {
                        "broad_backend": broad_backend,
                        "broad_video_start": broad_start,
                        "broad_video_end": video_end,
                        "broad_recent_frames_only": decode_recent_hint,
                        "recent_video_start": recent_video_start,
                        "recent_video_end": video_end,
                        "recent_start_time_seconds": recent_start,
                        "recent_end_time_seconds": recent_end,
                        "recent_chunk_ids": [chunk_id(chunk) for chunk in recent_chunks],
                        "available_historical_chunks": len(older_chunks),
                    },
                    "candidate_queue": [
                        {
                            key_: value
                            for key_, value in item.items()
                            if key_ != "chunk"
                        }
                        for item in queue
                    ],
                    "retrieval_quality": quality,
                    "ranking_ms": ranking_ms,
                    "branches": branches,
                    "minimum_k_to_correct": minimum_k,
                }
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                handle.flush()
                completed[key] = output
                all_rows.append(output)

    rows_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        rows_by_variant[str(row["retrieval_variant"])].append(row)
    summary = {
        "sample_source": source_path,
        "variants": {
            variant: summarize_variant(sorted(rows_by_variant.get(variant, []), key=lambda item: int(item["question_id"])))
            for variant in args.variants
        },
        "queue_overlap_statistics": queue_overlap_stats(rows_by_variant),
        "newly_rescuable_vs_current": newly_rescuable(rows_by_variant),
    }
    (out_dir / "retrieval_variant_oracle_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "retrieval_variant_oracle_results.json").write_text(
        json.dumps(sorted(all_rows, key=lambda item: (int(item["question_id"]), str(item["retrieval_variant"]))), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(out_dir / "retrieval_variant_oracle_results.csv", all_rows)

    print("\n================================================================================")
    print("StreamingBench Retrieval Variant Oracle")
    print("================================================================================")
    for variant in args.variants:
        row = summary["variants"][variant]
        print(
            f"{variant}: K0={100 * row['K0_accuracy']:.2f}% "
            f"K1={100 * row['K1_accuracy']:.2f}% "
            f"K2={100 * row['K2_accuracy']:.2f}% "
            f"K3={100 * row['K3_accuracy']:.2f}% "
            f"Oracle={100 * row['Oracle_K_accuracy']:.2f}% "
            f"rescues={row['K0_wrong_rescue_breakdown']}"
        )
    print("Queue overlap:")
    for name, row in summary["queue_overlap_statistics"].items():
        print(f"  {name}: overlap@1={row['overlap@1']} overlap@3={row['overlap@3']}")
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
