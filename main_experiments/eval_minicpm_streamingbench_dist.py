"""
Distributed StreamingBench evaluation for MiniCPM-V-4.6.

This keeps the same question protocol as eval_minicpm_streamingbench.py, but
uses Accelerate to split questions across processes. It is intended for the
8-GPU AMD runs where each rank loads MiniCPM on its own GPU, matching the OVO
evaluation launch pattern and avoiding the ROCm device_map=auto load path.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.cdas_sampler import CDASConfig
from lib.recent_window_eval import (
    extract_mcq_answer,
    load_jsonl_results,
    save_json,
)
from lib.recent_window_eval_minicpm import RecentWindowQAModel, query_all_frames, query_recent_window
from main_experiments.eval_minicpm_streamingbench import (
    build_prompt,
    compute_summary,
    format_options,
    make_key,
    print_summary,
    resolve_video_path,
    timestamp_to_seconds,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_tasks(anno_path: str, video_dir: str) -> tuple[list[dict[str, Any]], int]:
    with open(anno_path) as handle:
        all_data = json.load(handle)

    tasks: list[dict[str, Any]] = []
    for entry in all_data:
        video_path_raw = entry["video_path"]
        video_path = resolve_video_path(video_path_raw, video_dir)
        video_basename = os.path.basename(video_path)
        questions = list(entry["questions"])
        questions.sort(key=lambda item: timestamp_to_seconds(item["time_stamp"]))
        for question in questions:
            tasks.append(
                {
                    "_index": len(tasks),
                    "video_path_raw": video_path_raw,
                    "video_path": video_path,
                    "video_basename": video_basename,
                    "video_categories": entry.get("video_categories", ""),
                    "question": question,
                }
            )
    return tasks, len(all_data)


def _profile_fields(record: dict[str, Any], profile_metadata: dict[str, Any] | None) -> None:
    if profile_metadata is None:
        return
    record["profile"] = profile_metadata
    record["decode_time"] = profile_metadata.get("decode_time_seconds")
    record["end_to_end_time"] = profile_metadata.get("end_to_end_time_seconds")
    record["model_generate_time"] = profile_metadata.get("model_generate_time_seconds")
    record["preprocess_time"] = profile_metadata.get("preprocess_time_seconds")
    record["vision_preprocess_time_ms"] = profile_metadata.get("vision_preprocess_time_ms")
    record["vision_encoder_time_ms"] = profile_metadata.get("vision_encoder_time_ms")
    record["vision_resampler_time_ms"] = profile_metadata.get("vision_resampler_time_ms")
    record["vision_projector_time_ms"] = profile_metadata.get("vision_projector_time_ms")
    record["vision_hook_subtask_time_ms"] = profile_metadata.get("vision_hook_subtask_time_ms")
    record["vision_total_frontend_time_ms"] = profile_metadata.get("vision_total_frontend_time_ms")
    record["non_vision_generate_time_ms"] = profile_metadata.get("non_vision_generate_time_ms")
    record["prefill_forward_time_ms"] = profile_metadata.get("prefill_forward_time_ms")
    record["decode_forward_time_ms"] = profile_metadata.get("decode_forward_time_ms")
    record["prefill_kv_time_ms"] = profile_metadata.get("prefill_kv_time_ms")
    record["generate_first_token_time_ms"] = profile_metadata.get("generate_first_token_time_ms")
    record["generate_tokens_time_ms"] = profile_metadata.get("generate_tokens_time_ms")
    record["streamingtom_timeline_ms"] = profile_metadata.get("streamingtom_timeline_ms")
    record["st_vision_tower_ms"] = profile_metadata.get("st_vision_tower_ms")
    record["st_projector_ms"] = profile_metadata.get("st_projector_ms")
    record["st_compress_features_ms"] = profile_metadata.get("st_compress_features_ms")
    record["st_prefill_kv_ms"] = profile_metadata.get("st_prefill_kv_ms")
    record["st_store_kv_ms"] = profile_metadata.get("st_store_kv_ms")
    record["st_retrieval_forward_ms"] = profile_metadata.get("st_retrieval_forward_ms")
    record["st_reconstruct_kv_ms"] = profile_metadata.get("st_reconstruct_kv_ms")
    record["st_generate_first_token_ms"] = profile_metadata.get("st_generate_first_token_ms")
    record["st_generate_tokens_ms"] = profile_metadata.get("st_generate_tokens_ms")
    record["component_profile_enabled"] = profile_metadata.get("component_profile_enabled")
    record["gpu_peak_allocated_mb"] = profile_metadata.get("gpu_peak_allocated_mb")
    record["gpu_peak_reserved_mb"] = profile_metadata.get("gpu_peak_reserved_mb")
    record["gpu_peak_extra_allocated_mb"] = profile_metadata.get("gpu_peak_extra_allocated_mb")
    record["gpu_peak_extra_reserved_mb"] = profile_metadata.get("gpu_peak_extra_reserved_mb")


def _result_record(
    *,
    task: dict[str, Any],
    response: str | None,
    correct: bool,
    answer_gt: str,
    decode_backend: str | None,
    result: Any | None,
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
        "answer_gt": answer_gt,
        "response": response,
        "correct": correct,
    }
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
    full_frame_metadata = getattr(result, "full_frame_metadata", None)
    if full_frame_metadata is not None:
        record["full_frames"] = full_frame_metadata
    cdas_metadata = getattr(result, "cdas_metadata", None)
    if cdas_metadata is not None:
        record["cdas"] = cdas_metadata
    return record


def _merge_rank_outputs(output_dir: str) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for path in sorted(Path(output_dir).glob("rank_*/results_incremental.jsonl")):
        rows, _done = load_jsonl_results(str(path))
        merged.extend(rows)
    deduped: dict[str, dict[str, Any]] = {}
    for row in merged:
        key = row.get("_key")
        if isinstance(key, str):
            deduped[key] = row
    return sorted(deduped.values(), key=lambda item: int(item.get("_index", 0)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed StreamingBench evaluation for MiniCPM-V-4.6")
    parser.add_argument("--anno-path", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--qa-model", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--qa-device", default=None)
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-qa-tokens", type=int, default=256)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional smoke-test limit on total StreamingBench questions before distributed splitting.",
    )
    parser.add_argument("--recent-frames-only", "--recent-frames-buffer", dest="recent_frames_only", type=int, default=4)
    parser.add_argument("--context-time", type=int, default=-1)
    parser.add_argument("--frame-selection", choices=["recent", "all"], default="recent")
    parser.add_argument("--cdas-enable", action="store_true")
    parser.add_argument("--cdas-mode", choices=["binary", "three_level"], default="three_level")
    parser.add_argument("--cdas-skip-threshold", type=float, default=0.03)
    parser.add_argument("--cdas-high-threshold", type=float, default=0.12)
    parser.add_argument("--cdas-anchor-seconds", type=float, default=2.0)
    parser.add_argument("--cdas-min-accepted-fps", type=float, default=0.25)
    parser.add_argument("--cdas-gray-weight", type=float, default=0.50)
    parser.add_argument("--cdas-edge-weight", type=float, default=0.30)
    parser.add_argument("--cdas-hist-weight", type=float, default=0.20)
    parser.add_argument("--cdas-resize", type=int, default=96)
    parser.add_argument("--cdas-log-scores", action="store_true")
    args = parser.parse_args()

    if args.top_k != 0:
        raise ValueError("Distributed MiniCPM StreamingBench only supports --top-k 0.")

    dist_timeout_seconds = int(os.environ.get("MINICPM_DIST_TIMEOUT_SECONDS", "7200"))
    accelerator = Accelerator(
        kwargs_handlers=[
            InitProcessGroupKwargs(timeout=timedelta(seconds=dist_timeout_seconds)),
        ]
    )
    cdas_config = CDASConfig(
        enabled=bool(args.cdas_enable),
        mode=args.cdas_mode,
        skip_threshold=args.cdas_skip_threshold,
        high_threshold=args.cdas_high_threshold,
        anchor_seconds=args.cdas_anchor_seconds,
        min_accepted_fps=args.cdas_min_accepted_fps,
        gray_weight=args.cdas_gray_weight,
        edge_weight=args.cdas_edge_weight,
        hist_weight=args.cdas_hist_weight,
        resize=args.cdas_resize,
        log_scores=bool(args.cdas_log_scores),
    )
    cdas_config.validate()

    tasks, video_count = _load_tasks(args.anno_path, args.video_dir)
    if args.max_samples > 0:
        tasks = tasks[: args.max_samples]
    os.makedirs(args.output_dir, exist_ok=True)
    accelerator.print("\n" + "=" * 60)
    accelerator.print("StreamingBench Distributed Evaluation (MiniCPM-V-4.6)")
    accelerator.print("=" * 60)
    accelerator.print(f"Videos: {video_count}, Questions: {len(tasks)}, Processes: {accelerator.num_processes}")
    if args.max_samples > 0:
        accelerator.print(f"Smoke-test max samples: {args.max_samples}")
    accelerator.print(
        f"Frame selection: {args.frame_selection}, fps={args.fps}, "
        f"chunk_duration={args.chunk_duration}, context_time={args.context_time}"
    )
    accelerator.print(f"Results: {args.output_dir}")
    accelerator.print("=" * 60 + "\n")

    def build_evaluator() -> RecentWindowQAModel:
        return RecentWindowQAModel(
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
            if not os.path.exists(video_path):
                record = _result_record(
                    task=task,
                    response=None,
                    correct=False,
                    answer_gt=answer_gt,
                    decode_backend=None,
                    result=None,
                    error=f"Missing video: {video_path}",
                )
            else:
                ts_sec = float(timestamp_to_seconds(question["time_stamp"]))
                prompt = build_prompt(question)
                try:
                    if args.frame_selection == "all":
                        video_start = (
                            max(0.0, ts_sec - max(float(args.context_time), float(args.chunk_duration)))
                            if args.context_time > 0
                            else 0.0
                        )
                        result, decode_backend = query_all_frames(
                            qa=qa,
                            video_path=video_path,
                            prompt=prompt,
                            chunk_duration=args.chunk_duration,
                            fps=args.fps,
                            video_start=video_start,
                            video_end=ts_sec + 1e-4,
                        )
                    else:
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
                    )
                    logger.info(
                        "[rank %d %d/%d] %s %s -> %s (gt=%s)",
                        accelerator.process_index,
                        local_index,
                        len(local_tasks),
                        question["time_stamp"],
                        question.get("task_type", ""),
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
                        error=str(exc),
                    )
                    logger.error(
                        "[rank %d %d/%d] %s failed: %s",
                        accelerator.process_index,
                        local_index,
                        len(local_tasks),
                        question["time_stamp"],
                        exc,
                    )

            ckpt_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            ckpt_file.flush()
            done_keys.add(record["_key"])

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        merged = _merge_rank_outputs(args.output_dir)
        print_summary(merged, frame_selection=args.frame_selection)
        summary = compute_summary(merged)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_json(
            os.path.join(args.output_dir, f"streaming_bench_minicpmv46_results_{timestamp}.json"),
            {
                "config": {
                    "qa_model": args.qa_model,
                    "chunk_duration": args.chunk_duration,
                    "fps": args.fps,
                    "top_k": args.top_k,
                    "recent_frames_only": args.recent_frames_only,
                    "context_time": args.context_time,
                    "frame_selection": args.frame_selection,
                    "cache_enabled": False,
                    "attn_implementation": os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
                    "downsample_mode": os.environ.get("MINICPM_DOWNSAMPLE_MODE", "16x"),
                    "max_slice_nums": os.environ.get("MINICPM_MAX_SLICE_NUMS", "1"),
                    "max_samples": int(args.max_samples),
                    "cdas": cdas_config.__dict__,
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
