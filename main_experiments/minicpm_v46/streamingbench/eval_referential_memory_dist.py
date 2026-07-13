#!/usr/bin/env python3
"""Distributed StreamingBench run with lightweight previous-question memory.

This evaluator keeps all questions from the same video on one rank so a small
per-video memory can resolve prompts like "the person mentioned in the first
question" without storing all previous frames.
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
from lib.minicpm.baseline import RecentWindowQAModel
from lib.minicpm.referential_memory import (
    ReferentialMemoryEntry,
    make_memory_entry,
    query_referential_memory_window,
)
from lib.shared.recent_window import extract_mcq_answer, load_jsonl_results, save_json
from main_experiments.minicpm_v46.streamingbench.eval_baseline import (
    build_prompt,
    compute_summary,
    make_key,
    resolve_video_path,
    timestamp_to_seconds,
)
from main_experiments.minicpm_v46.streamingbench.eval_baseline_dist import (
    _merge_rank_outputs,
    _result_record,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_video_groups(
    anno_path: str,
    video_dir: str,
    *,
    max_samples: int = 0,
) -> tuple[list[dict[str, Any]], int, int]:
    with open(anno_path) as handle:
        all_data = json.load(handle)

    groups: list[dict[str, Any]] = []
    total_questions = 0
    for entry in all_data:
        if max_samples > 0 and total_questions >= max_samples:
            break

        video_path_raw = entry["video_path"]
        video_path = resolve_video_path(video_path_raw, video_dir)
        video_basename = os.path.basename(video_path)
        questions = list(entry["questions"])
        questions.sort(key=lambda item: timestamp_to_seconds(item["time_stamp"]))

        tasks: list[dict[str, Any]] = []
        for question_index, question in enumerate(questions):
            if max_samples > 0 and total_questions >= max_samples:
                break
            tasks.append(
                {
                    "_index": total_questions,
                    "_question_index": question_index,
                    "video_path_raw": video_path_raw,
                    "video_path": video_path,
                    "video_basename": video_basename,
                    "video_categories": entry.get("video_categories", ""),
                    "question": question,
                }
            )
            total_questions += 1

        if tasks:
            groups.append(
                {
                    "video_path_raw": video_path_raw,
                    "video_path": video_path,
                    "video_basename": video_basename,
                    "video_categories": entry.get("video_categories", ""),
                    "tasks": tasks,
                }
            )
    return groups, len(all_data), total_questions


def _memory_entry_from_record(row: dict[str, Any]) -> ReferentialMemoryEntry | None:
    raw = row.get("referential_memory_entry")
    if not isinstance(raw, dict):
        return None
    try:
        return ReferentialMemoryEntry(
            question_index=int(raw["question_index"]),
            question=str(raw["question"]),
            response=raw.get("response"),
            task_type=str(raw.get("task_type", "")),
            time_stamp=str(raw.get("time_stamp", "")),
            selected_chunk_ids=[int(value) for value in raw.get("selected_chunk_ids", [])],
            selected_timestamps=[float(value) for value in raw.get("selected_timestamps", [])],
        )
    except Exception:
        return None


def _print_referential_summary(results: list[dict[str, Any]], frame_selection: str = "recent") -> None:
    summary = compute_summary(results)
    label = "All-Frames" if frame_selection == "all" else "Recent-Window"
    print("\n" + "=" * 60)
    print(f"StreamingBench {label} Results (MiniCPM-V-4.6 + ReferentialMemory(recent6))")
    print("=" * 60)
    for row in summary["tasks"]:
        print(f"  {row['task_type']}: {row['accuracy']:.2f}% ({row['correct']}/{row['total']})")
    overall = summary["overall"]
    print(f"\n  Overall: {overall['accuracy']:.2f}% ({overall['correct']}/{overall['total']})")
    print(f"  Errors: {summary['error_count']}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed StreamingBench MiniCPM-V-4.6 with referential memory")
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
    parser.add_argument("--recent-frames-only", "--recent-frames-buffer", dest="recent_frames_only", type=int, default=6)
    parser.add_argument("--reference-frames", type=int, default=2)
    parser.add_argument("--context-time", type=int, default=-1)
    parser.add_argument("--frame-selection", choices=["recent"], default="recent")
    args = parser.parse_args()

    if args.top_k != 0:
        raise ValueError("Referential memory run only supports --top-k 0.")

    dist_timeout_seconds = int(os.environ.get("MINICPM_DIST_TIMEOUT_SECONDS", "7200"))
    accelerator = Accelerator(
        kwargs_handlers=[
            InitProcessGroupKwargs(timeout=timedelta(seconds=dist_timeout_seconds)),
        ]
    )
    cdas_config = CDASConfig(enabled=False)

    groups, video_count, total_questions = _load_video_groups(
        args.anno_path,
        args.video_dir,
        max_samples=int(args.max_samples),
    )
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator.print("\n" + "=" * 60)
    accelerator.print("StreamingBench Referential-Memory Evaluation (MiniCPM-V-4.6)")
    accelerator.print("=" * 60)
    accelerator.print(f"Videos: {video_count}, Evaluated video groups: {len(groups)}")
    accelerator.print(f"Questions: {total_questions}, Processes: {accelerator.num_processes}")
    accelerator.print(
        f"Current window={args.recent_frames_only}, reference_frames={args.reference_frames}, "
        f"fps={args.fps}, chunk_duration={args.chunk_duration}, context_time={args.context_time}"
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

    with accelerator.split_between_processes(groups) as local_groups:
        local_groups = list(local_groups)

    rank_dir = os.path.join(args.output_dir, f"rank_{accelerator.process_index}")
    os.makedirs(rank_dir, exist_ok=True)
    ckpt_path = os.path.join(rank_dir, "results_incremental.jsonl")
    existing_rows, done_keys = load_jsonl_results(ckpt_path)
    existing_by_key = {row.get("_key"): row for row in existing_rows if isinstance(row.get("_key"), str)}

    with open(ckpt_path, "a") as ckpt_file:
        for local_video_index, group in enumerate(local_groups, start=1):
            video_path = group["video_path"]
            video_basename = group["video_basename"]
            tasks = list(group["tasks"])
            memory: list[ReferentialMemoryEntry] = []
            logger.info(
                "[rank %d video %d/%d] %s (%d questions)",
                accelerator.process_index,
                local_video_index,
                len(local_groups),
                video_basename,
                len(tasks),
            )

            for local_index, task in enumerate(tasks, start=1):
                question = task["question"]
                key = make_key(video_basename, question, question_limit=80)
                if key in done_keys:
                    entry = _memory_entry_from_record(existing_by_key.get(key, {}))
                    if entry is not None:
                        memory.append(entry)
                    logger.info("[rank %d] skip %s", accelerator.process_index, key)
                    continue

                answer_gt = (
                    extract_mcq_answer(str(question.get("answer", "")))
                    or str(question.get("answer", "")).strip().upper()
                )
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
                        result, decode_backend, selection = query_referential_memory_window(
                            qa=qa,
                            video_path=video_path,
                            prompt=prompt,
                            question_text=str(question.get("question", "")),
                            memory=memory,
                            chunk_duration=args.chunk_duration,
                            fps=args.fps,
                            recent_frames_only=effective_recent_chunks,
                            reference_frames=args.reference_frames,
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
                        metadata = getattr(result, "referential_memory_metadata", None)
                        if metadata is not None:
                            record["referential_memory"] = metadata
                        memory_entry = make_memory_entry(
                            question_index=int(task["_question_index"]),
                            question_text=str(question.get("question", "")),
                            response=response,
                            task_type=str(question.get("task_type", "")),
                            time_stamp=str(question.get("time_stamp", "")),
                            selection=selection,
                        )
                        record["referential_memory_entry"] = {
                            "question_index": memory_entry.question_index,
                            "question": memory_entry.question,
                            "response": memory_entry.response,
                            "task_type": memory_entry.task_type,
                            "time_stamp": memory_entry.time_stamp,
                            "selected_chunk_ids": memory_entry.selected_chunk_ids,
                            "selected_timestamps": memory_entry.selected_timestamps,
                        }
                        memory.append(memory_entry)
                        logger.info(
                            "[rank %d video %d/%d q%d/%d] %s %s -> %s (gt=%s, ref=%s)",
                            accelerator.process_index,
                            local_video_index,
                            len(local_groups),
                            local_index,
                            len(tasks),
                            question["time_stamp"],
                            question.get("task_type", ""),
                            response[:80] if response else "None",
                            answer_gt,
                            metadata.get("memory_triggered") if isinstance(metadata, dict) else None,
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
                            "[rank %d video %d/%d q%d/%d] %s failed: %s",
                            accelerator.process_index,
                            local_video_index,
                            len(local_groups),
                            local_index,
                            len(tasks),
                            question["time_stamp"],
                            exc,
                        )

                ckpt_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                ckpt_file.flush()
                done_keys.add(record["_key"])

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        merged = _merge_rank_outputs(args.output_dir)
        _print_referential_summary(merged, frame_selection=args.frame_selection)
        summary = compute_summary(merged)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_json(
            os.path.join(args.output_dir, f"streaming_bench_minicpmv46_referential_results_{timestamp}.json"),
            {
                "config": {
                    "qa_model": args.qa_model,
                    "chunk_duration": args.chunk_duration,
                    "fps": args.fps,
                    "top_k": args.top_k,
                    "recent_frames_only": args.recent_frames_only,
                    "reference_frames": args.reference_frames,
                    "context_time": args.context_time,
                    "frame_selection": args.frame_selection,
                    "cache_enabled": False,
                    "referential_memory": True,
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
