#!/usr/bin/env python3
"""OVO-Bench MiniCPM-V-4.6 with lightweight answer-grounded referential memory.

This mirrors the StreamingBench referential-memory setup while preserving the
OVO result/checkpoint format. OVO is mostly independent QA, so the memory is
most useful when multiple samples from the same video/forward item are processed
on the same rank or when a forward item contains several sub-questions.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Any

os.environ.setdefault("NCCL_TIMEOUT", "7200")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "0")

from accelerate import Accelerator
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from ovo_constants import BACKWARD_TASKS, FORWARD_TASKS, REAL_TIME_TASKS
from lib.cdas_sampler import CDASConfig
from lib.minicpm.baseline import RecentWindowQAModel, _result_metadata
from lib.minicpm.referential_memory import (
    AnswerGroundedFrameScorer,
    ReferentialMemoryEntry,
    make_memory_entry,
    query_referential_memory_window,
)
from lib.shared.recent_window import build_ovo_prompt, print_ovo_results
from main_experiments.qwen.evals.eval_qwen3vl_ovo import (
    append_checkpoint_row,
    get_checkpoint_path,
    get_done_path,
    load_checkpoint_state,
    make_ovo_key,
    merge_shard_results,
    wait_for_done_markers,
    write_done_marker,
)

MODEL_LABEL = "MiniCPM-V-4.6 + AnswerGroundedReferentialMemory(recent6)"


def _memory_entry_to_record(entry: ReferentialMemoryEntry) -> dict[str, Any]:
    return {
        "question_index": entry.question_index,
        "question": entry.question,
        "response": entry.response,
        "task_type": entry.task_type,
        "time_stamp": entry.time_stamp,
        "selected_chunk_ids": entry.selected_chunk_ids,
        "selected_timestamps": entry.selected_timestamps,
        "anchor_chunk_ids": entry.anchor_chunk_ids or [],
        "anchor_timestamps": entry.anchor_timestamps or [],
        "anchor_scores": entry.anchor_scores or [],
        "anchor_candidate_chunk_ids": entry.anchor_candidate_chunk_ids or [],
        "anchor_candidate_timestamps": entry.anchor_candidate_timestamps or [],
        "anchor_candidate_scores": entry.anchor_candidate_scores or [],
        "memory_text": entry.memory_text,
        "anchor_scoring": entry.anchor_scoring,
        "anchor_scoring_error": entry.anchor_scoring_error,
    }


def _video_memory_key(anno: dict[str, Any]) -> str:
    return str(anno.get("video") or anno.get("id") or "")


def _question_text_for_forward(anno: dict[str, Any], index: int) -> str:
    task = str(anno.get("task", ""))
    if task == "REC":
        return f"How many times did they {anno.get('activity', '')}?"
    if task == "SSR":
        return str(anno.get("test_info", [{}])[index].get("step", ""))
    return str(anno.get("question", ""))


def evaluate_ovo_backward_realtime_referential(
    *,
    anno: dict[str, Any],
    chunked_dir: str,
    qa: RecentWindowQAModel,
    memory: list[ReferentialMemoryEntry],
    frame_scorer: AnswerGroundedFrameScorer | None,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    reference_frames: int,
    memory_anchor_frames: int,
    cdas_config: CDASConfig,
) -> dict[str, Any]:
    video_path = os.path.join(chunked_dir, f"{anno['id']}.mp4")
    response = None
    metadata: dict[str, Any] = {}
    if os.path.exists(video_path):
        prompt = build_ovo_prompt(anno["task"], anno)
        try:
            result, decode_backend, selection = query_referential_memory_window(
                qa=qa,
                video_path=video_path,
                prompt=prompt,
                question_text=str(anno.get("question", "")),
                memory=memory,
                chunk_duration=chunk_duration,
                fps=fps,
                recent_frames_only=recent_frames_only,
                reference_frames=reference_frames,
                cdas_config=cdas_config,
            )
            response = result.answer
            metadata = _result_metadata(result, decode_backend)
            ref_meta = getattr(result, "referential_memory_metadata", None)
            if ref_meta is not None:
                metadata["referential_memory"] = ref_meta
            entry = make_memory_entry(
                question_index=len(memory),
                question_text=str(anno.get("question", "")),
                response=response,
                task_type=str(anno.get("task", "")),
                time_stamp="",
                selection=selection,
                options=list(anno.get("options", []) or []),
                answer_grounded=True,
                frame_scorer=frame_scorer,
                anchor_frames=int(memory_anchor_frames),
                anchor_text_mode="answer",
            )
            metadata["referential_memory_entry"] = _memory_entry_to_record(entry)
            memory.append(entry)
        except Exception as exc:
            metadata = {
                "error": str(exc),
                "error_type": type(exc).__name__,
                "video_path": video_path,
            }
    else:
        metadata = {
            "error": f"Missing video: {video_path}",
            "error_type": "FileNotFoundError",
            "video_path": video_path,
        }
    return {
        "id": anno["id"],
        "video": anno["video"],
        "task": anno["task"],
        "question": anno["question"],
        "response": response,
        "ground_truth": chr(65 + anno["gt"]),
        **metadata,
    }


def evaluate_ovo_forward_referential(
    *,
    anno: dict[str, Any],
    chunked_dir: str,
    qa: RecentWindowQAModel,
    frame_scorer: AnswerGroundedFrameScorer | None,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    reference_frames: int,
    memory_anchor_frames: int,
    cdas_config: CDASConfig,
) -> dict[str, Any]:
    result_anno = copy.deepcopy(anno)
    memory: list[ReferentialMemoryEntry] = []
    for index, test_info in enumerate(result_anno["test_info"]):
        video_path = os.path.join(chunked_dir, f"{anno['id']}_{index}.mp4")
        if not os.path.exists(video_path):
            test_info["response"] = None
            test_info["error"] = f"Missing video: {video_path}"
            test_info["error_type"] = "FileNotFoundError"
            continue
        prompt = build_ovo_prompt(anno["task"], anno, index=index)
        question_text = _question_text_for_forward(anno, index)
        try:
            result, decode_backend, selection = query_referential_memory_window(
                qa=qa,
                video_path=video_path,
                prompt=prompt,
                question_text=question_text,
                memory=memory,
                chunk_duration=chunk_duration,
                fps=fps,
                recent_frames_only=recent_frames_only,
                reference_frames=reference_frames,
                cdas_config=cdas_config,
            )
            test_info["response"] = result.answer
            test_info.update(_result_metadata(result, decode_backend))
            ref_meta = getattr(result, "referential_memory_metadata", None)
            if ref_meta is not None:
                test_info["referential_memory"] = ref_meta
            entry = make_memory_entry(
                question_index=index,
                question_text=question_text,
                response=result.answer,
                task_type=str(anno.get("task", "")),
                time_stamp="",
                selection=selection,
                options=[],
                answer_grounded=True,
                frame_scorer=frame_scorer,
                anchor_frames=int(memory_anchor_frames),
                anchor_text_mode="answer",
            )
            test_info["referential_memory_entry"] = _memory_entry_to_record(entry)
            memory.append(entry)
        except Exception as exc:
            test_info["response"] = None
            test_info["error"] = str(exc)
            test_info["error_type"] = type(exc).__name__
            test_info["video_path"] = video_path
    return result_anno


def main() -> None:
    parser = argparse.ArgumentParser(description="OVO-Bench MiniCPM-V-4.6 with answer-grounded referential memory")
    parser.add_argument("--model_path", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--qa_device", default=None)
    parser.add_argument("--anno_path", default="data/ovo_bench/ovo_bench_new.json")
    parser.add_argument("--chunked_dir", default="data/ovo_bench/chunked_videos")
    parser.add_argument("--result_dir", default="results/ovo_bench_answer_grounded_referential_memory_minicpmv46")
    parser.add_argument("--recent_frames_only", type=int, default=6)
    parser.add_argument("--reference_frames", type=int, default=2)
    parser.add_argument("--memory_anchor_frames", type=int, default=1)
    parser.add_argument("--memory_clip_model", default=os.environ.get("MINICPM_REF_CLIP_MODEL", "openai/clip-vit-base-patch32"))
    parser.add_argument("--memory_clip_device", default=os.environ.get("MINICPM_REF_CLIP_DEVICE", ""))
    parser.add_argument("--chunk_duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_qa_tokens", type=int, default=256)
    parser.add_argument("--max_samples_per_split", type=int, default=None)
    args = parser.parse_args()

    cdas_config = CDASConfig(enabled=False)
    accelerator = Accelerator()

    with open(args.anno_path) as handle:
        annotations = json.load(handle)

    backward_anno = [anno for anno in annotations if anno["task"] in BACKWARD_TASKS]
    realtime_anno = [anno for anno in annotations if anno["task"] in REAL_TIME_TASKS]
    forward_anno = [anno for anno in annotations if anno["task"] in FORWARD_TASKS]

    random.seed(42)
    random.shuffle(backward_anno)
    random.shuffle(realtime_anno)
    random.shuffle(forward_anno)
    if args.max_samples_per_split is not None:
        if args.max_samples_per_split < 1:
            raise ValueError("--max_samples_per_split must be >= 1")
        backward_anno = backward_anno[: args.max_samples_per_split]
        realtime_anno = realtime_anno[: args.max_samples_per_split]
        forward_anno = forward_anno[: args.max_samples_per_split]

    accelerator.print(f"\n{'=' * 60}")
    accelerator.print(f"OVO-Bench Referential-Memory Evaluation ({MODEL_LABEL})")
    accelerator.print(f"{'=' * 60}")
    accelerator.print(f"Backward: {len(backward_anno)}, Realtime: {len(realtime_anno)}, Forward: {len(forward_anno)}")
    accelerator.print(f"Processes: {accelerator.num_processes}")
    accelerator.print(
        f"recent_frames_only={args.recent_frames_only}, reference_frames={args.reference_frames}, "
        f"anchor_frames={args.memory_anchor_frames}, fps={args.fps}, chunk_duration={args.chunk_duration}"
    )
    accelerator.print(f"CLIP scorer: {args.memory_clip_model}")
    accelerator.print(f"Results: {args.result_dir}")
    accelerator.print(f"{'=' * 60}\n")

    def build_evaluator() -> RecentWindowQAModel:
        return RecentWindowQAModel(
            model_name=args.model_path,
            device=args.qa_device or accelerator.device,
            max_new_tokens=args.max_qa_tokens,
            attn_implementation=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
        )

    if os.environ.get("MINICPM_SERIALIZE_MODEL_LOAD", "").strip().lower() in {"1", "true", "yes", "on"}:
        load_job_id = os.environ.get("SLURM_JOB_ID", "local")
        load_marker_dir = os.path.join(args.result_dir, f".minicpm_model_load_{load_job_id}_{accelerator.num_processes}")
        os.makedirs(load_marker_dir, exist_ok=True)
        previous_marker = os.path.join(load_marker_dir, f"rank_{accelerator.process_index - 1}.done")
        current_marker = os.path.join(load_marker_dir, f"rank_{accelerator.process_index}.done")
        timeout_seconds = float(os.environ.get("MINICPM_MODEL_LOAD_TIMEOUT", "7200"))
        if accelerator.process_index > 0:
            wait_start = time.perf_counter()
            while not os.path.exists(previous_marker):
                if time.perf_counter() - wait_start > timeout_seconds:
                    raise TimeoutError(f"Timed out waiting for previous MiniCPM load marker: {previous_marker}")
                time.sleep(2.0)
        print(f"[rank {accelerator.process_index}] Loading MiniCPM-V model", flush=True)
        evaluator = build_evaluator()
        with open(current_marker, "w") as marker:
            marker.write(datetime.now().isoformat() + "\n")
    else:
        evaluator = build_evaluator()

    clip_device = args.memory_clip_device.strip() or str(accelerator.device)
    print(f"[rank {accelerator.process_index}] Loading answer-grounded frame scorer on {clip_device}", flush=True)
    frame_scorer = AnswerGroundedFrameScorer(model_name=args.memory_clip_model, device=clip_device)

    with accelerator.split_between_processes(backward_anno) as local_backward:
        local_backward = list(local_backward)
    with accelerator.split_between_processes(realtime_anno) as local_realtime:
        local_realtime = list(local_realtime)
    with accelerator.split_between_processes(forward_anno) as local_forward:
        local_forward = list(local_forward)

    checkpoint_path = get_checkpoint_path(args.result_dir, accelerator.process_index, accelerator.num_processes)
    done_path = get_done_path(args.result_dir, accelerator.process_index, accelerator.num_processes)
    if os.path.exists(done_path):
        os.remove(done_path)
    backward_results, realtime_results, forward_results, done_keys = load_checkpoint_state(checkpoint_path)
    video_memory: dict[str, list[ReferentialMemoryEntry]] = {}

    with open(checkpoint_path, "a") as checkpoint_file:
        for anno in tqdm(local_backward, desc=f"[GPU{accelerator.process_index}] Backward", disable=not accelerator.is_local_main_process):
            key = make_ovo_key(anno)
            if key in done_keys:
                continue
            memory = video_memory.setdefault(_video_memory_key(anno), [])
            result = evaluate_ovo_backward_realtime_referential(
                anno=anno,
                chunked_dir=args.chunked_dir,
                qa=evaluator,
                memory=memory,
                frame_scorer=frame_scorer,
                chunk_duration=args.chunk_duration,
                fps=args.fps,
                recent_frames_only=args.recent_frames_only,
                reference_frames=args.reference_frames,
                memory_anchor_frames=args.memory_anchor_frames,
                cdas_config=cdas_config,
            )
            backward_results.append(result)
            done_keys.add(key)
            append_checkpoint_row(checkpoint_file, result)

        for anno in tqdm(local_realtime, desc=f"[GPU{accelerator.process_index}] Realtime", disable=not accelerator.is_local_main_process):
            key = make_ovo_key(anno)
            if key in done_keys:
                continue
            memory = video_memory.setdefault(_video_memory_key(anno), [])
            result = evaluate_ovo_backward_realtime_referential(
                anno=anno,
                chunked_dir=args.chunked_dir,
                qa=evaluator,
                memory=memory,
                frame_scorer=frame_scorer,
                chunk_duration=args.chunk_duration,
                fps=args.fps,
                recent_frames_only=args.recent_frames_only,
                reference_frames=args.reference_frames,
                memory_anchor_frames=args.memory_anchor_frames,
                cdas_config=cdas_config,
            )
            realtime_results.append(result)
            done_keys.add(key)
            append_checkpoint_row(checkpoint_file, result)

        for anno in tqdm(local_forward, desc=f"[GPU{accelerator.process_index}] Forward", disable=not accelerator.is_local_main_process):
            key = make_ovo_key(anno)
            if key in done_keys:
                continue
            result = evaluate_ovo_forward_referential(
                anno=anno,
                chunked_dir=args.chunked_dir,
                qa=evaluator,
                frame_scorer=frame_scorer,
                chunk_duration=args.chunk_duration,
                fps=args.fps,
                recent_frames_only=args.recent_frames_only,
                reference_frames=args.reference_frames,
                memory_anchor_frames=args.memory_anchor_frames,
                cdas_config=cdas_config,
            )
            forward_results.append(result)
            done_keys.add(key)
            append_checkpoint_row(checkpoint_file, result)

    write_done_marker(done_path)

    if accelerator.is_main_process:
        wait_for_done_markers(args.result_dir, accelerator.num_processes)
        all_backward, all_realtime, all_forward = merge_shard_results(args.result_dir, accelerator.num_processes)
        print_ovo_results(MODEL_LABEL, all_backward, all_realtime, all_forward)
        os.makedirs(args.result_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.result_dir, f"minicpmv46_answer_grounded_referential_results_{timestamp}.json")
        with open(output_path, "w") as handle:
            json.dump(
                {
                    "config": {
                        "model_path": args.model_path,
                        "qa_device": args.qa_device or str(accelerator.device),
                        "recent_frames_only": args.recent_frames_only,
                        "reference_frames": args.reference_frames,
                        "memory_anchor_frames": args.memory_anchor_frames,
                        "memory_clip_model": args.memory_clip_model,
                        "chunk_duration": args.chunk_duration,
                        "fps": args.fps,
                        "max_qa_tokens": args.max_qa_tokens,
                        "max_samples_per_split": args.max_samples_per_split,
                        "attn_implementation": os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
                        "downsample_mode": os.environ.get("MINICPM_DOWNSAMPLE_MODE", "16x"),
                        "max_slice_nums": os.environ.get("MINICPM_MAX_SLICE_NUMS", "1"),
                        "distributed": True,
                        "num_processes": accelerator.num_processes,
                    },
                    "backward": all_backward,
                    "realtime": all_realtime,
                    "forward": all_forward,
                },
                handle,
                indent=2,
            )
        print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
