"""
Distributed StreamingBench evaluation for MiniCPM-V-4.6 with a conservative
hybrid input policy:

  - Recent-6 still-frame SimpleStream for most questions.
  - A short K-second video clip only for explicit immediate-motion questions.
  - Cumulative/counting/reference questions are guarded away from the clip path.

This keeps the experiment separate from the existing baseline/adaptive files.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

os.environ.setdefault("NCCL_TIMEOUT", "7200")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "0")

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from lib.cdas_sampler import CDASConfig
from lib.minicpm.baseline import query_recent_window
from lib.minicpm.hybrid_recent_clip import (
    attach_hybrid_metadata,
    build_clip_prompt,
    choose_hybrid_route,
)
from lib.minicpm.recent_clip import RecentClipQAModel, query_recent_clip
from lib.shared.recent_window import extract_mcq_answer, load_jsonl_results, save_json
from main_experiments.minicpm_v46.streamingbench.eval_baseline import (
    build_prompt,
    compute_summary,
    make_key,
    resolve_video_path,
    timestamp_to_seconds,
)
from main_experiments.minicpm_v46.streamingbench.eval_baseline_dist import (
    _load_tasks,
    _merge_rank_outputs,
    _profile_fields,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _result_record(
    *,
    task: dict[str, Any],
    response: str | None,
    correct: bool,
    answer_gt: str,
    decode_backend: str | None,
    result: Any | None,
    route: str | None = None,
    route_reason: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    question = task["question"]
    record: dict[str, Any] = {
        "_index": int(task["_index"]),
        "_key": make_key(task["video_basename"], question, question_limit=80),
        "video": task["video_basename"],
        "video_categories": task.get("video_categories", ""),
        "task_type": question.get("task_type", ""),
        "time_stamp": question["time_stamp"],
        "question": question["question"],
        "options": question.get("options", []),
        "answer_gt": answer_gt,
        "response": response,
        "correct": correct,
    }
    if route is not None:
        record["hybrid_recent_clip"] = {"route": route, "route_reason": route_reason}
    if error is not None:
        record["error"] = error
        return record

    record.update(
        {
            "decode_backend": decode_backend,
            "final_chunk_ids": result.final_chunk_ids,
            "generate_time": result.generate_time,
            "ttft_seconds": result.ttft_seconds,
            "num_vision_tokens": result.num_vision_tokens,
            "num_vision_tokens_before": result.num_vision_tokens_before,
            "num_vision_tokens_after": result.num_vision_tokens_after,
            "num_frames": result.num_frames,
        }
    )
    _profile_fields(record, getattr(result, "profile_metadata", None))
    hybrid_metadata = getattr(result, "hybrid_recent_clip_metadata", None)
    if hybrid_metadata is not None:
        record["hybrid_recent_clip"] = hybrid_metadata
    return record


def _print_summary(results: list[dict[str, Any]], clip_seconds: float) -> None:
    summary = compute_summary(results)
    print("\n" + "=" * 60)
    print(f"StreamingBench Recent-Window Results (MiniCPM-V-4.6 + HybridRecentClip(K={clip_seconds:g}s))")
    print("=" * 60)
    for row in summary["tasks"]:
        print(f"  {row['task_type']}: {row['accuracy']:.2f}% ({row['correct']}/{row['total']})")
    overall = summary["overall"]
    print(f"\n  Overall: {overall['accuracy']:.2f}% ({overall['correct']}/{overall['total']})")
    print(f"  Errors: {summary['error_count']}")

    route_counts: dict[str, dict[str, int]] = {}
    for row in results:
        route = (row.get("hybrid_recent_clip") or {}).get("route", "unknown")
        stats = route_counts.setdefault(route, {"total": 0, "correct": 0})
        stats["total"] += 1
        stats["correct"] += int(bool(row.get("correct")))
    print("\n  Route breakdown:")
    for route in sorted(route_counts):
        stats = route_counts[route]
        total = stats["total"]
        correct = stats["correct"]
        acc = 100.0 * correct / total if total else 0.0
        print(f"    {route}: {acc:.2f}% ({correct}/{total})")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed StreamingBench MiniCPM-V-4.6 HybridRecentClip")
    parser.add_argument("--anno-path", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--qa-model", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--qa-device", default=None)
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-qa-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--recent-frames-only", type=int, default=6)
    parser.add_argument("--context-time", type=int, default=-1)
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    args = parser.parse_args()

    if args.top_k != 0:
        raise ValueError("HybridRecentClip only supports --top-k 0.")

    dist_timeout_seconds = int(os.environ.get("MINICPM_DIST_TIMEOUT_SECONDS", "7200"))
    accelerator = Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=dist_timeout_seconds))]
    )
    cdas_config = CDASConfig(enabled=False)

    tasks, video_count = _load_tasks(args.anno_path, args.video_dir)
    if args.max_samples > 0:
        tasks = tasks[: args.max_samples]
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator.print("\n" + "=" * 60)
    accelerator.print("StreamingBench Distributed Evaluation (MiniCPM-V-4.6 + HybridRecentClip)")
    accelerator.print("=" * 60)
    accelerator.print(f"Videos: {video_count}, Questions: {len(tasks)}, Processes: {accelerator.num_processes}")
    accelerator.print(f"Recent frames: {args.recent_frames_only}, clip_seconds={args.clip_seconds:g}")
    accelerator.print(f"Results: {args.output_dir}")
    accelerator.print("=" * 60 + "\n")

    def build_evaluator() -> RecentClipQAModel:
        return RecentClipQAModel(
            model_name=args.qa_model,
            device=args.qa_device or accelerator.device,
            max_new_tokens=args.max_qa_tokens,
            attn_implementation=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
        )

    if os.environ.get("MINICPM_SERIALIZE_MODEL_LOAD", "").strip().lower() in {"1", "true", "yes", "on"}:
        load_job_id = os.environ.get("SLURM_JOB_ID", "local")
        load_marker_dir = os.path.join(
            args.output_dir,
            f".minicpm_model_load_{load_job_id}_{accelerator.num_processes}",
        )
        os.makedirs(load_marker_dir, exist_ok=True)
        previous_marker = os.path.join(load_marker_dir, f"rank_{accelerator.process_index - 1}.done")
        current_marker = os.path.join(load_marker_dir, f"rank_{accelerator.process_index}.done")
        timeout_seconds = float(os.environ.get("MINICPM_MODEL_LOAD_TIMEOUT", "7200"))
        if accelerator.process_index > 0:
            wait_start = time.perf_counter()
            while not os.path.exists(previous_marker):
                if time.perf_counter() - wait_start > timeout_seconds:
                    raise TimeoutError(f"Timed out waiting for MiniCPM load marker: {previous_marker}")
                time.sleep(2.0)
        print(f"[rank {accelerator.process_index}] Loading MiniCPM-V model", flush=True)
        qa = build_evaluator()
        with open(current_marker, "w") as marker:
            marker.write(datetime.now().isoformat() + "\n")
    else:
        qa = build_evaluator()

    with accelerator.split_between_processes(tasks) as local_tasks:
        local_tasks = list(local_tasks)

    rank_dir = os.path.join(args.output_dir, f"rank_{accelerator.process_index}")
    os.makedirs(rank_dir, exist_ok=True)
    ckpt_path = os.path.join(rank_dir, "results_incremental.jsonl")
    _existing_rows, done_keys = load_jsonl_results(ckpt_path)

    with open(ckpt_path, "a") as ckpt_file:
        for local_index, task in enumerate(local_tasks, start=1):
            question = task["question"]
            key = make_key(task["video_basename"], question, question_limit=80)
            if key in done_keys:
                logger.info("[rank %d] skip %s", accelerator.process_index, key)
                continue

            video_path = task["video_path"]
            answer_gt = extract_mcq_answer(str(question.get("answer", ""))) or str(question.get("answer", "")).strip().upper()
            route, route_reason = choose_hybrid_route(str(question.get("question", "")))
            if not os.path.exists(video_path):
                record = _result_record(
                    task=task,
                    response=None,
                    correct=False,
                    answer_gt=answer_gt,
                    decode_backend=None,
                    result=None,
                    route=route,
                    route_reason=route_reason,
                    error=f"Missing video: {video_path}",
                )
            else:
                ts_sec = float(timestamp_to_seconds(question["time_stamp"]))
                try:
                    if route == "recent_clip":
                        prompt = build_clip_prompt(str(question.get("question", "")), list(question.get("options", [])))
                        result, decode_backend = query_recent_clip(
                            qa,
                            video_path=video_path,
                            prompt=prompt,
                            question_time_seconds=ts_sec,
                            clip_seconds=float(args.clip_seconds),
                        )
                    else:
                        prompt = build_prompt(question)
                        window_seconds = (
                            float(args.context_time)
                            if args.context_time > 0
                            else float(args.recent_frames_only) * float(args.chunk_duration)
                        )
                        video_start = max(0.0, ts_sec - max(window_seconds, float(args.chunk_duration)))
                        effective_recent_chunks = max(
                            int(args.recent_frames_only),
                            int(math.ceil(window_seconds / max(float(args.chunk_duration), 1e-6))),
                        )
                        result, decode_backend = query_recent_window(
                            qa=qa,
                            video_path=video_path,
                            prompt=prompt,
                            chunk_duration=args.chunk_duration,
                            fps=args.fps,
                            recent_frames_only=effective_recent_chunks,
                            video_start=video_start,
                            video_end=ts_sec + 1e-4,
                            cdas_config=cdas_config,
                        )
                    attach_hybrid_metadata(
                        result,
                        route=route,
                        route_reason=route_reason,
                        clip_seconds=float(args.clip_seconds),
                    )
                    response = result.answer
                    pred = extract_mcq_answer(response)
                    correct = bool(pred is not None and pred == answer_gt)
                    record = _result_record(
                        task=task,
                        response=response,
                        correct=correct,
                        answer_gt=answer_gt,
                        decode_backend=decode_backend,
                        result=result,
                        route=route,
                        route_reason=route_reason,
                    )
                    logger.info(
                        "[rank %d %d/%d] %s %s route=%s -> %s (gt=%s)",
                        accelerator.process_index,
                        local_index,
                        len(local_tasks),
                        question["time_stamp"],
                        question.get("task_type", ""),
                        route,
                        response[:80] if response else "None",
                        answer_gt,
                    )
                except Exception as exc:
                    record = _result_record(
                        task=task,
                        response=None,
                        correct=False,
                        answer_gt=answer_gt,
                        decode_backend=None,
                        result=None,
                        route=route,
                        route_reason=route_reason,
                        error=str(exc),
                    )
                    logger.error(
                        "[rank %d %d/%d] %s route=%s failed: %s",
                        accelerator.process_index,
                        local_index,
                        len(local_tasks),
                        question["time_stamp"],
                        route,
                        exc,
                    )

            ckpt_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            ckpt_file.flush()
            done_keys.add(record["_key"])

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        merged = _merge_rank_outputs(args.output_dir)
        _print_summary(merged, clip_seconds=float(args.clip_seconds))
        summary = compute_summary(merged)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_json(
            os.path.join(args.output_dir, f"streaming_bench_minicpmv46_hybrid_recent_clip_results_{timestamp}.json"),
            {
                "config": {
                    "qa_model": args.qa_model,
                    "chunk_duration": args.chunk_duration,
                    "fps": args.fps,
                    "top_k": args.top_k,
                    "recent_frames_only": args.recent_frames_only,
                    "context_time": args.context_time,
                    "clip_seconds": args.clip_seconds,
                    "attn_implementation": os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
                    "downsample_mode": os.environ.get("MINICPM_DOWNSAMPLE_MODE", "16x"),
                    "max_slice_nums": os.environ.get("MINICPM_MAX_SLICE_NUMS", "1"),
                    "max_samples": int(args.max_samples),
                    "distributed": True,
                    "num_processes": accelerator.num_processes,
                },
                "summary": summary,
                "results": merged,
            },
        )
        save_json(os.path.join(args.output_dir, "scores_report.json"), summary)


if __name__ == "__main__":
    main()
