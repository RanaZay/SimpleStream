#!/usr/bin/env python3
"""Controlled StreamingBench memory-addition experiment for saved PSM rows.

This tool intentionally does not change the PSM retriever/gate.  It replays only
samples where a previous PSM run saved historical memory chunks, then compares:

  A. RECENT_ONLY_CONTROL: saved PSM recent chunks, no memory.
  B. RETRIEVED_MEMORY: saved PSM recent chunks plus saved PSM memory chunks.
  C. CONTROL_HISTORY: deterministic historical chunks matched approximately by
     temporal distance and count.

The final answer path is always RecentWindowQAModel.generate_from_frames with
deterministic decoding.  The tool also saves decoded-frame hashes so identical
chunk IDs can be audited at the pixel level.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

PSM_HISTORY_INSTRUCTION = (
    "When historical frames are present, they appear before the six recent frames.\n\n"
)

decode_video_to_chunks_qwen = None
extract_mcq_answer = None
load_jsonl_results = None
build_prompt = None
resolve_video_path = None
timestamp_to_seconds = None
RecentWindowQAModel = None
select_recent_window_frames = None
AdaptiveWindowConfig = None
_evaluate_sufficiency = None
_extract_mcq_options = None
_get_clip_scorer = None


TEMPORAL_LABELS = (
    "PRESENT_STATE",
    "RECENT_EVENT",
    "PROSPECTIVE",
    "CUMULATIVE_HISTORY",
    "HISTORICAL",
    "CAUSAL",
    "OTHER",
)


def default_qa_device() -> str:
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass
class ConditionResult:
    name: str
    prediction: str | None
    correct: bool
    raw_response: str
    recent_chunk_ids: list[int]
    memory_chunk_ids: list[int]
    final_chunk_ids: list[int]
    frame_hashes: list[str]
    final_prompt: str
    generation_kwargs: dict[str, Any]
    sufficiency_probe: dict[str, Any] | None
    generate_seconds: float
    error: str | None = None


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

    if not path.exists():
        parent = path.parent if path.parent.exists() else Path(".")
        nearby = find_result_candidates(parent, limit=20)
        hint = "\n".join(f"  {item}" for item in nearby) if nearby else "  (none found nearby)"
        raise FileNotFoundError(
            f"No such result path: {path}\n"
            f"Nearby StreamingBench result candidates under {parent}:\n{hint}"
        )

    merged = sorted(
        path.glob("streaming_bench_minicpmv46_results_*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not merged:
        merged = sorted(
            path.glob("**/streaming_bench_minicpmv46_results_*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
    if merged:
        return load_results(merged[0])

    rank_files = sorted(path.glob("rank_*/results_incremental.jsonl"))
    if not rank_files:
        rank_files = sorted(path.glob("**/rank_*/results_incremental.jsonl"))
    if not rank_files:
        nearby = find_result_candidates(path, limit=20)
        hint = "\n".join(f"  {item}" for item in nearby) if nearby else "  (none found)"
        raise FileNotFoundError(
            f"No StreamingBench results found under {path}\n"
            "Expected either streaming_bench_minicpmv46_results_*.json or "
            "rank_*/results_incremental.jsonl.\n"
            f"Candidate result paths:\n{hint}"
        )
    rows: list[dict[str, Any]] = []
    for rank_file in rank_files:
        part, _done = load_jsonl_results(str(rank_file))
        rows.extend(part)
    dedup: dict[Any, dict[str, Any]] = {}
    for row in rows:
        dedup[row.get("_index", row.get("_key"))] = row
    return "\n  ".join(str(item) for item in rank_files), sorted(
        dedup.values(), key=lambda item: int(item.get("_index", 0))
    )


def find_result_candidates(root: Path, limit: int = 20) -> list[Path]:
    if not root.exists():
        return []
    candidates: dict[Path, float] = {}
    for pattern in ("**/streaming_bench_minicpmv46_results_*.json", "**/rank_*/results_incremental.jsonl"):
        for item in root.glob(pattern):
            result_root = item.parent.parent if item.name == "results_incremental.jsonl" else item.parent
            try:
                mtime = item.stat().st_mtime
            except OSError:
                mtime = 0.0
            candidates[result_root] = max(candidates.get(result_root, 0.0), mtime)
    ranked = sorted(candidates.items(), key=lambda pair: pair[1], reverse=True)
    return [path for path, _mtime in ranked[:limit]]


def load_annotations(path: Path, video_dir: str) -> dict[int, dict[str, Any]]:
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
                    "video_path_raw": video_path_raw,
                    "video_path": video_path,
                    "video": os.path.basename(video_path),
                    "video_categories": entry.get("video_categories", ""),
                    "task_type": question.get("task_type", ""),
                    "question_obj": question,
                    "prompt": build_prompt(question),
                }
            )
    return {int(item["_index"]): item for item in tasks}


def int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            out.append(int(item))
    return out


def chunk_id(chunk: Any) -> int:
    return int(getattr(chunk, "chunk_index"))


def flatten_frames(chunks: list[Any]) -> list[Any]:
    return [frame for chunk in chunks for frame in chunk.frames]


def frame_hash(frame: Any) -> str:
    image = frame.convert("RGB")
    h = hashlib.sha1()
    h.update(str(image.size).encode("ascii"))
    h.update(image.tobytes())
    return h.hexdigest()


def answer_gt_from(row: dict[str, Any], question: dict[str, Any]) -> str:
    gt = extract_mcq_answer(row.get("answer_gt"))
    if gt:
        return gt
    return extract_mcq_answer(question.get("answer")) or str(question.get("answer", "")).strip().upper()


def semantic_temporal_intent(question: str) -> str:
    text = re.sub(r"\s+", " ", question.strip().lower())
    if re.search(r"\b(why|reason|cause|caused|because)\b", text):
        return "CAUSAL"
    if re.search(r"\b(next|will|likely|might|about to|going to|expect|after this)\b", text):
        return "PROSPECTIVE"
    if re.search(r"\b(so far|in total|total number|overall|throughout|how many times|count)\b", text):
        return "CUMULATIVE_HISTORY"
    if re.search(r"\b(right now|currently|at this moment|now visible|is .* doing now|are .* doing now)\b", text):
        return "PRESENT_STATE"
    if re.search(r"\b(just now|just happened|recently|immediately before|last action)\b", text):
        return "RECENT_EVENT"
    if re.search(r"\b(before|earlier|previously|at the beginning|initially|first|started)\b", text):
        return "HISTORICAL"
    return "OTHER"


def select_chunks_by_id(chunks_by_id: dict[int, Any], ids: list[int]) -> list[Any]:
    selected = []
    missing = []
    for value in ids:
        chunk = chunks_by_id.get(int(value))
        if chunk is None:
            missing.append(int(value))
        else:
            selected.append(chunk)
    if missing:
        raise KeyError(f"Decoded chunks do not contain saved chunk ids: {missing}")
    return selected


def build_control_history_ids(
    available_ids: list[int],
    recent_ids: list[int],
    retrieved_memory_ids: list[int],
) -> list[int]:
    if not retrieved_memory_ids:
        return []
    recent_set = set(recent_ids)
    retrieved_set = set(retrieved_memory_ids)
    candidates = [value for value in available_ids if value not in recent_set and value not in retrieved_set]
    if not candidates:
        candidates = [value for value in available_ids if value not in recent_set]
    chosen: list[int] = []
    recent_anchor = min(recent_ids) if recent_ids else (max(available_ids) + 1 if available_ids else 0)
    for memory_id in retrieved_memory_ids:
        target_distance = abs(recent_anchor - int(memory_id))
        pool = [value for value in candidates if value not in chosen]
        if not pool:
            break
        best = min(pool, key=lambda value: (abs(abs(recent_anchor - value) - target_distance), abs(value - memory_id), value))
        chosen.append(int(best))
    return sorted(chosen)


def generation_kwargs(qa: RecentWindowQAModel, context_time: float, downsample_mode: str | None) -> dict[str, Any]:
    return {
        "max_new_tokens": int(qa.max_new_tokens),
        "do_sample": False,
        "downsample_mode": downsample_mode or qa.downsample_mode,
        "max_slice_nums": qa.max_slice_nums,
        "context_time": context_time,
    }


def run_sufficiency_probe(
    qa: RecentWindowQAModel,
    chunks: list[Any],
    prompt: str,
    probe: bool,
) -> dict[str, Any] | None:
    if not probe:
        return None
    options = _extract_mcq_options(prompt)
    if not options:
        return {"skipped": True, "reason": "no_mcq_options"}
    scorer = _get_clip_scorer(qa)
    score, elapsed_ms = _evaluate_sufficiency(
        qa,
        chunks,
        prompt,
        options,
        scorer,
        margin_weight=float(os.environ.get("MINICPM_PSM_MARGIN_WEIGHT", "0.50")),
        entropy_weight=float(os.environ.get("MINICPM_PSM_ENTROPY_WEIGHT", "0.20")),
        visual_support_weight=float(os.environ.get("MINICPM_PSM_VISUAL_SUPPORT_WEIGHT", "0.30")),
    )
    return {**score, "probe_elapsed_ms": float(elapsed_ms)}


def run_condition(
    *,
    qa: RecentWindowQAModel,
    name: str,
    gt: str,
    recent_chunks: list[Any],
    memory_chunks: list[Any],
    base_prompt: str,
    context_time: float,
    downsample_mode: str | None,
    run_probe: bool,
) -> ConditionResult:
    final_chunks = [*sorted(memory_chunks, key=chunk_id), *recent_chunks]
    final_chunk_ids = [chunk_id(chunk) for chunk in final_chunks]
    recent_chunk_ids = [chunk_id(chunk) for chunk in recent_chunks]
    memory_chunk_ids = [chunk_id(chunk) for chunk in sorted(memory_chunks, key=chunk_id)]
    prompt = f"{PSM_HISTORY_INSTRUCTION}{base_prompt}" if memory_chunk_ids else base_prompt
    frames = flatten_frames(final_chunks)
    hashes = [frame_hash(frame) for frame in frames]
    sufficiency = run_sufficiency_probe(qa, final_chunks, prompt, run_probe)
    t0 = time.perf_counter()
    response = qa.generate_from_frames(frames, prompt, downsample_mode=downsample_mode)
    generate_seconds = time.perf_counter() - t0
    pred = extract_mcq_answer(response)
    return ConditionResult(
        name=name,
        prediction=pred,
        correct=bool(pred is not None and pred == gt),
        raw_response=response,
        recent_chunk_ids=recent_chunk_ids,
        memory_chunk_ids=memory_chunk_ids,
        final_chunk_ids=final_chunk_ids,
        frame_hashes=hashes,
        final_prompt=prompt,
        generation_kwargs=generation_kwargs(qa, context_time, downsample_mode),
        sufficiency_probe=sufficiency,
        generate_seconds=float(generate_seconds),
    )


def compare_pair(a: ConditionResult, b: ConditionResult) -> str:
    if a.correct and b.correct:
        return "both_correct"
    if (not a.correct) and b.correct:
        return "A_wrong_B_correct"
    if a.correct and (not b.correct):
        return "A_correct_B_wrong"
    return "both_wrong"


def summarize(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    counts = Counter(row[f"{left}_vs_{right}"] for row in rows if f"{left}_vs_{right}" in row)
    fixes = counts.get("A_wrong_B_correct", 0)
    harms = counts.get("A_correct_B_wrong", 0)
    return {
        "both_correct": counts.get("both_correct", 0),
        "left_wrong_right_correct": fixes,
        "left_correct_right_wrong": harms,
        "both_wrong": counts.get("both_wrong", 0),
        "net": fixes - harms,
        "left_accuracy": sum(bool(row["conditions"][left]["correct"]) for row in rows) / max(len(rows), 1),
        "right_accuracy": sum(bool(row["conditions"][right]["correct"]) for row in rows) / max(len(rows), 1),
    }


def grouped_summary(rows: list[dict[str, Any]], group_key: str, left: str, right: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_key) or "")].append(row)
    for group, items in sorted(groups.items()):
        summary = summarize(items, left, right)
        out.append({"group": group, "total": len(items), **summary})
    return out


def condition_to_json(condition: ConditionResult) -> dict[str, Any]:
    return {
        "name": condition.name,
        "prediction": condition.prediction,
        "correct": condition.correct,
        "raw_response": condition.raw_response,
        "recent_chunk_ids": condition.recent_chunk_ids,
        "memory_chunk_ids": condition.memory_chunk_ids,
        "final_chunk_ids": condition.final_chunk_ids,
        "frame_hashes": condition.frame_hashes,
        "final_prompt": condition.final_prompt,
        "generation_kwargs": condition.generation_kwargs,
        "sufficiency_probe": condition.sufficiency_probe,
        "generate_seconds": condition.generate_seconds,
        "error": condition.error,
    }


def import_runtime_modules() -> None:
    global AdaptiveWindowConfig
    global RecentWindowQAModel
    global build_prompt
    global decode_video_to_chunks_qwen
    global extract_mcq_answer
    global load_jsonl_results
    global resolve_video_path
    global select_recent_window_frames
    global timestamp_to_seconds
    global _evaluate_sufficiency
    global _extract_mcq_options
    global _get_clip_scorer

    from lib.minicpm.adaptive import AdaptiveWindowConfig as _AdaptiveWindowConfig
    from lib.minicpm.baseline import RecentWindowQAModel as _RecentWindowQAModel
    from lib.minicpm.baseline import select_recent_window_frames as _select_recent_window_frames
    from lib.minicpm.progressive_sufficiency import (
        _evaluate_sufficiency as __evaluate_sufficiency,
    )
    from lib.minicpm.progressive_sufficiency import (
        _extract_mcq_options as __extract_mcq_options,
    )
    from lib.minicpm.progressive_sufficiency import _get_clip_scorer as __get_clip_scorer
    from lib.shared.recent_window import (
        decode_video_to_chunks_qwen as _decode_video_to_chunks_qwen,
    )
    from lib.shared.recent_window import extract_mcq_answer as _extract_mcq_answer
    from lib.shared.recent_window import load_jsonl_results as _load_jsonl_results
    from main_experiments.minicpm_v46.streamingbench.eval_baseline import (
        build_prompt as _build_prompt,
    )
    from main_experiments.minicpm_v46.streamingbench.eval_baseline import (
        resolve_video_path as _resolve_video_path,
    )
    from main_experiments.minicpm_v46.streamingbench.eval_baseline import (
        timestamp_to_seconds as _timestamp_to_seconds,
    )

    AdaptiveWindowConfig = _AdaptiveWindowConfig
    RecentWindowQAModel = _RecentWindowQAModel
    select_recent_window_frames = _select_recent_window_frames
    _evaluate_sufficiency = __evaluate_sufficiency
    _extract_mcq_options = __extract_mcq_options
    _get_clip_scorer = __get_clip_scorer
    decode_video_to_chunks_qwen = _decode_video_to_chunks_qwen
    extract_mcq_answer = _extract_mcq_answer
    load_jsonl_results = _load_jsonl_results
    build_prompt = _build_prompt
    resolve_video_path = _resolve_video_path
    timestamp_to_seconds = _timestamp_to_seconds


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "question_id",
        "video_id",
        "timestamp",
        "category",
        "temporal_intent",
        "ground_truth",
        "recent_only_prediction",
        "retrieved_memory_prediction",
        "control_history_prediction",
        "recent_only_correct",
        "retrieved_memory_correct",
        "control_history_correct",
        "recent_only_vs_retrieved_memory",
        "retrieved_memory_vs_control_history",
        "recent_chunk_ids",
        "retrieved_memory_chunk_ids",
        "control_memory_chunk_ids",
        "retrieved_final_chunk_ids",
        "control_final_chunk_ids",
        "saved_psm_prediction",
        "saved_psm_correct",
        "saved_psm_stop_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            cond = row["conditions"]
            writer.writerow(
                {
                    "question_id": row["question_id"],
                    "video_id": row["video_id"],
                    "timestamp": row["timestamp"],
                    "category": row["category"],
                    "temporal_intent": row["temporal_intent"],
                    "ground_truth": row["ground_truth"],
                    "recent_only_prediction": cond["recent_only_control"]["prediction"],
                    "retrieved_memory_prediction": cond["retrieved_memory"]["prediction"],
                    "control_history_prediction": cond.get("control_history", {}).get("prediction"),
                    "recent_only_correct": cond["recent_only_control"]["correct"],
                    "retrieved_memory_correct": cond["retrieved_memory"]["correct"],
                    "control_history_correct": cond.get("control_history", {}).get("correct"),
                    "recent_only_vs_retrieved_memory": row.get("recent_only_control_vs_retrieved_memory"),
                    "retrieved_memory_vs_control_history": row.get("retrieved_memory_vs_control_history"),
                    "recent_chunk_ids": json.dumps(cond["recent_only_control"]["recent_chunk_ids"]),
                    "retrieved_memory_chunk_ids": json.dumps(cond["retrieved_memory"]["memory_chunk_ids"]),
                    "control_memory_chunk_ids": json.dumps(cond.get("control_history", {}).get("memory_chunk_ids", [])),
                    "retrieved_final_chunk_ids": json.dumps(cond["retrieved_memory"]["final_chunk_ids"]),
                    "control_final_chunk_ids": json.dumps(cond.get("control_history", {}).get("final_chunk_ids", [])),
                    "saved_psm_prediction": row["saved_psm_prediction"],
                    "saved_psm_correct": row["saved_psm_correct"],
                    "saved_psm_stop_reason": row["saved_psm_stop_reason"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled causal replay for StreamingBench PSM memory-added rows.")
    parser.add_argument("--ours", required=True, help="Saved PSM StreamingBench result directory or merged JSON.")
    parser.add_argument("--annotations", required=True, help="StreamingBench annotation JSON.")
    parser.add_argument("--video-dir", required=True, help="StreamingBench video root.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--qa-model", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--qa-device", default=None, help="Model device, default: cuda:0 when available, else cpu.")
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--recent-window", type=int, default=6)
    parser.add_argument("--context-time", type=float, default=70.0)
    parser.add_argument("--max-qa-tokens", type=int, default=256)
    parser.add_argument("--attn-implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"))
    parser.add_argument("--skip-control-history", action="store_true")
    parser.add_argument("--skip-sufficiency-probe", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0, help="Debug limit after selecting memory-added rows.")
    args = parser.parse_args()

    import_runtime_modules()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_path, saved_rows = load_results(Path(args.ours))
    annotations = load_annotations(Path(args.annotations), args.video_dir)

    memory_rows = []
    for row in saved_rows:
        adaptive = row.get("adaptive") or {}
        memory_ids = int_list(adaptive.get("memory_chunk_ids"))
        if memory_ids:
            memory_rows.append(row)
    memory_rows = sorted(memory_rows, key=lambda item: int(item.get("_index", 0)))
    if args.max_samples > 0:
        memory_rows = memory_rows[: args.max_samples]

    print(f"Saved PSM result source: {source_path}")
    print(f"Saved rows: {len(saved_rows)}")
    print(f"Memory-added rows selected: {len(memory_rows)}")
    print(f"Output directory: {out_dir}")

    qa_device = args.qa_device or default_qa_device()
    print(f"QA device: {qa_device}")
    qa = RecentWindowQAModel(
        model_name=args.qa_model,
        device=qa_device,
        max_new_tokens=args.max_qa_tokens,
        attn_implementation=args.attn_implementation,
    )
    config = AdaptiveWindowConfig.from_env()
    downsample_mode = None
    completed: dict[int, dict[str, Any]] = {}
    jsonl_path = out_dir / "controlled_memory_causal_results.jsonl"
    if jsonl_path.exists():
        for row in read_jsonl(jsonl_path):
            completed[int(row["question_id"])] = row
    run_probe = not bool(args.skip_sufficiency_probe)
    run_control = not bool(args.skip_control_history)

    with jsonl_path.open("a", encoding="utf-8") as handle:
        for ordinal, saved in enumerate(memory_rows, start=1):
            qid = int(saved.get("_index", ordinal - 1))
            if qid in completed:
                print(f"[{ordinal}/{len(memory_rows)}] skip qid={qid} (already done)", flush=True)
                continue
            task = annotations.get(qid)
            if task is None:
                raise KeyError(f"Question index {qid} is absent from annotations")
            question = task["question_obj"]
            adaptive = saved.get("adaptive") or {}
            recent_ids = int_list(adaptive.get("recent_chunk_ids"))
            memory_ids = int_list(adaptive.get("memory_chunk_ids"))
            if not recent_ids:
                recent_ids = int_list(saved.get("final_chunk_ids"))[-args.recent_window :]
            gt = answer_gt_from(saved, question)
            prompt = task["prompt"]
            ts_sec = float(timestamp_to_seconds(question["time_stamp"]))
            video_path = task["video_path"]
            video_end = ts_sec + 1e-4
            broad_start = max(0.0, ts_sec - max(float(args.context_time), float(args.chunk_duration)))
            decode_recent_hint = max(
                int(math.ceil(float(args.context_time) / max(float(args.chunk_duration), 1e-6))),
                int(args.recent_window) + max(int(getattr(config, "memory_search_chunks", 64)), len(memory_ids)),
            )

            print(f"[{ordinal}/{len(memory_rows)}] qid={qid} memory={memory_ids} recent={recent_ids}", flush=True)
            broad_chunks, broad_backend = decode_video_to_chunks_qwen(
                video_path=video_path,
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
                video_path=video_path,
                chunk_duration=args.chunk_duration,
                fps=args.fps,
                recent_frames_only=args.recent_window,
                video_start=recent_video_start,
                video_end=video_end,
                cdas_config=None,
            )
            recent_by_id = {chunk_id(chunk): chunk for chunk in recent_selection.selected_chunks}
            recent_chunks = []
            for cid in recent_ids:
                recent_chunks.append(recent_by_id.get(cid) or broad_by_id[cid])
            memory_chunks = select_chunks_by_id(broad_by_id, memory_ids)
            control_memory_ids = build_control_history_ids(sorted(broad_by_id), recent_ids, memory_ids) if run_control else []
            control_memory_chunks = select_chunks_by_id(broad_by_id, control_memory_ids) if control_memory_ids else []

            conditions: dict[str, ConditionResult] = {}
            try:
                conditions["recent_only_control"] = run_condition(
                    qa=qa,
                    name="recent_only_control",
                    gt=gt,
                    recent_chunks=recent_chunks,
                    memory_chunks=[],
                    base_prompt=prompt,
                    context_time=float(args.context_time),
                    downsample_mode=downsample_mode,
                    run_probe=run_probe,
                )
                conditions["retrieved_memory"] = run_condition(
                    qa=qa,
                    name="retrieved_memory",
                    gt=gt,
                    recent_chunks=recent_chunks,
                    memory_chunks=memory_chunks,
                    base_prompt=prompt,
                    context_time=float(args.context_time),
                    downsample_mode=downsample_mode,
                    run_probe=run_probe,
                )
                if run_control:
                    conditions["control_history"] = run_condition(
                        qa=qa,
                        name="control_history",
                        gt=gt,
                        recent_chunks=recent_chunks,
                        memory_chunks=control_memory_chunks,
                        base_prompt=prompt,
                        context_time=float(args.context_time),
                        downsample_mode=downsample_mode,
                        run_probe=run_probe,
                    )
            except Exception as exc:
                raise RuntimeError(f"Failed controlled replay for question_id={qid}") from exc

            json_conditions = {name: condition_to_json(condition) for name, condition in conditions.items()}
            output_row = {
                "question_id": qid,
                "key": saved.get("_key"),
                "video_id": task["video"],
                "video_path": video_path,
                "timestamp": question.get("time_stamp"),
                "timestamp_seconds": ts_sec,
                "category": saved.get("task_type") or task.get("task_type"),
                "temporal_intent": semantic_temporal_intent(str(question.get("question", ""))),
                "question": question.get("question"),
                "options": question.get("options"),
                "ground_truth": gt,
                "conditions": json_conditions,
                "recent_only_control_vs_retrieved_memory": compare_pair(
                    conditions["recent_only_control"], conditions["retrieved_memory"]
                ),
                "retrieved_memory_vs_control_history": (
                    compare_pair(conditions["retrieved_memory"], conditions["control_history"])
                    if "control_history" in conditions
                    else None
                ),
                "decode": {
                    "broad_backend": broad_backend,
                    "broad_video_start": broad_start,
                    "broad_video_end": video_end,
                    "broad_recent_frames_only": decode_recent_hint,
                    "broad_decoded_chunk_ids": sorted(broad_by_id),
                    "recent_backend": recent_selection.decode_backend,
                    "recent_video_start": recent_video_start,
                    "recent_video_end": video_end,
                    "recent_decoded_chunk_ids": [chunk_id(chunk) for chunk in recent_selection.selected_chunks],
                },
                "saved_psm_prediction": extract_mcq_answer(saved.get("response")),
                "saved_psm_correct": bool(saved.get("correct")),
                "saved_psm_response": saved.get("response"),
                "saved_psm_final_chunk_ids": int_list(saved.get("final_chunk_ids")),
                "saved_psm_adaptive": adaptive,
                "saved_psm_stop_reason": adaptive.get("stop_reason"),
            }
            handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            handle.flush()
            completed[qid] = output_row

    rows = [completed[key] for key in sorted(completed)]
    (out_dir / "controlled_memory_causal_results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(out_dir / "controlled_memory_causal_results.csv", rows)

    summary = {
        "source": source_path,
        "selected_memory_added_samples": len(memory_rows),
        "completed_samples": len(rows),
        "recent_only_control_vs_retrieved_memory": summarize(rows, "recent_only_control", "retrieved_memory"),
        "by_category_A_vs_B": grouped_summary(rows, "category", "recent_only_control", "retrieved_memory"),
        "by_temporal_intent_A_vs_B": grouped_summary(rows, "temporal_intent", "recent_only_control", "retrieved_memory"),
    }
    if run_control:
        summary["retrieved_memory_vs_control_history"] = summarize(rows, "retrieved_memory", "control_history")
        summary["by_category_B_vs_C"] = grouped_summary(rows, "category", "retrieved_memory", "control_history")
        summary["by_temporal_intent_B_vs_C"] = grouped_summary(rows, "temporal_intent", "retrieved_memory", "control_history")

    (out_dir / "controlled_memory_causal_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
