"""
OVO-Bench recent-window evaluation for Qwen3-VL.

Aligned with internal eval_recent_frames_ovo.py:
decode video -> chunk by time -> keep the last N chunks -> generate_from_frames

High-level flow:
1. Load the OVO annotation JSON.
2. Split samples into Backward, Real-time, and Forward task groups.
3. Use Accelerate to divide each group across available GPU processes.
4. For each sample, decode the video window, keep the most recent frames,
   run Qwen3-VL, and append the result to a per-rank checkpoint file.
5. After all ranks finish, merge checkpoints and write the final JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Any

# Large video evaluations can have uneven per-rank runtime because video decode
# and filesystem I/O are not perfectly balanced. These settings reduce premature
# distributed timeouts while slower ranks are still working.
os.environ.setdefault("NCCL_TIMEOUT", "7200")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "0")

from accelerate import Accelerator
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from ovo_constants import BACKWARD_TASKS, FORWARD_TASKS, REAL_TIME_TASKS
from lib.shared.recent_window import load_jsonl_results
from lib.qwen.qwen3 import (
    RecentWindowQAModel,
    evaluate_ovo_backward_realtime,
    evaluate_ovo_forward,
    print_ovo_results,
)

MODEL_LABEL = "Qwen3-VL"


def make_ovo_key(item: dict[str, Any]) -> str:
    """Build a stable key used to skip already-finished samples on resume."""
    return f"{item.get('task', '')}:{item.get('id')}"


def get_checkpoint_path(result_dir: str, process_index: int, num_processes: int) -> str:
    """Return the JSONL checkpoint path for the current GPU process."""
    if num_processes == 1:
        os.makedirs(result_dir, exist_ok=True)
        return os.path.join(result_dir, "results_incremental.jsonl")
    shard_dir = os.path.join(result_dir, f"rank_{process_index}")
    os.makedirs(shard_dir, exist_ok=True)
    return os.path.join(shard_dir, "results_incremental.jsonl")


def get_done_path(result_dir: str, process_index: int, num_processes: int) -> str:
    """Return the small marker file path written when a rank fully finishes."""
    if num_processes == 1:
        os.makedirs(result_dir, exist_ok=True)
        return os.path.join(result_dir, "done")
    shard_dir = os.path.join(result_dir, f"rank_{process_index}")
    os.makedirs(shard_dir, exist_ok=True)
    return os.path.join(shard_dir, "done")


def strip_internal_fields(item: dict[str, Any]) -> dict[str, Any]:
    """Remove checkpoint-only fields before storing final public results."""
    return {key: value for key, value in item.items() if key != "_key"}


def load_checkpoint_state(path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    """Load this rank's checkpoint and rebuild split-specific result lists."""
    records, done_keys = load_jsonl_results(path)
    backward_results: list[dict[str, Any]] = []
    realtime_results: list[dict[str, Any]] = []
    forward_results: list[dict[str, Any]] = []

    for raw in records:
        item = strip_internal_fields(raw)
        key = raw.get("_key")
        if not isinstance(key, str) or not key:
            key = make_ovo_key(item)
        done_keys.add(key)
        task = item.get("task")
        if task in BACKWARD_TASKS:
            backward_results.append(item)
        elif task in REAL_TIME_TASKS:
            realtime_results.append(item)
        elif task in FORWARD_TASKS:
            forward_results.append(item)

    return backward_results, realtime_results, forward_results, done_keys


def append_checkpoint_row(handle: Any, item: dict[str, Any]) -> None:
    """Write one result immediately so interrupted runs can resume later."""
    record = dict(item)
    record["_key"] = make_ovo_key(item)
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def merge_shard_results(result_dir: str, num_processes: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge rank-wise checkpoints into one deduplicated result collection."""
    checkpoint_paths = (
        [os.path.join(result_dir, "results_incremental.jsonl")]
        if num_processes == 1
        else [os.path.join(result_dir, f"rank_{rank}", "results_incremental.jsonl") for rank in range(num_processes)]
    )

    backward_results: list[dict[str, Any]] = []
    realtime_results: list[dict[str, Any]] = []
    forward_results: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for path in checkpoint_paths:
        records, _ = load_jsonl_results(path)
        for raw in records:
            item = strip_internal_fields(raw)
            key = raw.get("_key")
            if not isinstance(key, str) or not key:
                key = make_ovo_key(item)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            task = item.get("task")
            if task in BACKWARD_TASKS:
                backward_results.append(item)
            elif task in REAL_TIME_TASKS:
                realtime_results.append(item)
            elif task in FORWARD_TASKS:
                forward_results.append(item)

    return backward_results, realtime_results, forward_results


def write_done_marker(path: str) -> None:
    """Signal that the current rank has finished writing all local results."""
    with open(path, "w") as handle:
        handle.write(datetime.now().isoformat() + "\n")


def wait_for_done_markers(result_dir: str, num_processes: int) -> None:
    """Wait until every distributed rank has written its done marker."""
    if num_processes <= 1:
        return

    timeout_seconds = float(os.environ.get("FILE_SYNC_TIMEOUT_SECONDS", "43200"))
    poll_interval = float(os.environ.get("FILE_SYNC_POLL_SECONDS", "10"))
    done_paths = [os.path.join(result_dir, f"rank_{rank}", "done") for rank in range(num_processes)]
    deadline = time.time() + timeout_seconds

    while True:
        missing = [path for path in done_paths if not os.path.exists(path)]
        if not missing:
            return
        if time.time() >= deadline:
            raise RuntimeError(f"Timed out waiting for rank completion markers: {missing}")
        time.sleep(poll_interval)


def main() -> None:
    # Command-line arguments are supplied by launchers such as
    # run_repro_qwen3vl_ovo_recent4.sh and run_repro_qwen3vl_2b_ovo_recent4.sh.
    # The default window settings match the SimpleStream recent-window setup.
    parser = argparse.ArgumentParser(description="OVO-Bench recent-window evaluation for Qwen3-VL")
    parser.add_argument("--model_path", required=True, help="Example: Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--anno_path", default="data/ovo_bench/ovo_bench_new.json")
    parser.add_argument("--chunked_dir", default="data/ovo_bench/chunked_videos")
    parser.add_argument("--result_dir", default="results/ovo_bench_recent_window_qwen3vl")
    parser.add_argument("--recent_frames_only", type=int, default=4)
    parser.add_argument("--chunk_duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_qa_tokens", type=int, default=256)
    parser.add_argument(
        "--max_samples_per_split",
        type=int,
        default=None,
        help="Optional sample cap applied independently to backward/realtime/forward after shuffle.",
    )
    args = parser.parse_args()

    # Accelerator starts one process per GPU when launched with
    # `accelerate launch --num_processes N`. We use it for data-parallel
    # evaluation, not for training.
    accelerator = Accelerator()

    # Load the OVO annotation list once in every process. Each process will later
    # keep only its assigned slice.
    with open(args.anno_path) as handle:
        annotations = json.load(handle)

    # OVO-Bench has three task families. Backward and Real-time are the main
    # Table-1 style categories; Forward is also evaluated when present.
    backward_anno = [anno for anno in annotations if anno["task"] in BACKWARD_TASKS]
    realtime_anno = [anno for anno in annotations if anno["task"] in REAL_TIME_TASKS]
    forward_anno = [anno for anno in annotations if anno["task"] in FORWARD_TASKS]

    # Shuffle before splitting so each GPU receives a mixed workload instead of
    # one rank getting only a specific task order or unusually long videos.
    random.seed(42)
    random.shuffle(backward_anno)
    random.shuffle(realtime_anno)
    random.shuffle(forward_anno)

    # Optional debugging shortcut: evaluate only a few samples per split.
    if args.max_samples_per_split is not None:
        if args.max_samples_per_split < 1:
            raise ValueError("--max_samples_per_split must be >= 1")
        backward_anno = backward_anno[: args.max_samples_per_split]
        realtime_anno = realtime_anno[: args.max_samples_per_split]
        forward_anno = forward_anno[: args.max_samples_per_split]

    accelerator.print(f"\n{'=' * 60}")
    accelerator.print(f"OVO-Bench Recent-Window Evaluation ({MODEL_LABEL})")
    accelerator.print(f"{'=' * 60}")
    accelerator.print(f"Backward: {len(backward_anno)}, Realtime: {len(realtime_anno)}, Forward: {len(forward_anno)}")
    accelerator.print(f"Processes: {accelerator.num_processes}")
    accelerator.print(
        f"Window: recent_frames_only={args.recent_frames_only}, "
        f"chunk_duration={args.chunk_duration}, fps={args.fps}"
    )
    if args.max_samples_per_split is not None:
        accelerator.print(f"Sample cap per split: {args.max_samples_per_split}")
    accelerator.print(f"{'=' * 60}\n")

    # Model wrapper: handles Qwen3-VL loading, video/image preprocessing, prompt
    # formatting, recent-frame generation, and output decoding.
    # On ROCm machines we can set ATTN_IMPLEMENTATION=sdpa to avoid incompatible
    # FlashAttention kernels.
    evaluator = RecentWindowQAModel(
        model_name=args.model_path,
        device=accelerator.device,
        max_new_tokens=args.max_qa_tokens,
        attn_implementation=os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2"),
    )

    # Split each task group across GPU processes. For example, with 4 GPUs each
    # rank receives roughly one quarter of Backward, Real-time, and Forward.
    with accelerator.split_between_processes(backward_anno) as local_backward:
        local_backward = list(local_backward)
    with accelerator.split_between_processes(realtime_anno) as local_realtime:
        local_realtime = list(local_realtime)
    with accelerator.split_between_processes(forward_anno) as local_forward:
        local_forward = list(local_forward)

    checkpoint_path = get_checkpoint_path(args.result_dir, accelerator.process_index, accelerator.num_processes)
    done_path = get_done_path(args.result_dir, accelerator.process_index, accelerator.num_processes)

    # If a previous run completed, remove this rank's old done marker so rank 0
    # does not merge results before the current run actually finishes.
    if os.path.exists(done_path):
        os.remove(done_path)

    # Resume support: load completed rows for this rank and skip them below.
    backward_results, realtime_results, forward_results, done_keys = load_checkpoint_state(checkpoint_path)

    with open(checkpoint_path, "a") as checkpoint_file:
        # Backward tasks are multiple-choice questions over a recent video window.
        for anno in tqdm(local_backward, desc=f"[GPU{accelerator.process_index}] Backward", disable=not accelerator.is_local_main_process):
            key = make_ovo_key(anno)
            if key in done_keys:
                continue
            result = evaluate_ovo_backward_realtime(
                anno=anno,
                chunked_dir=args.chunked_dir,
                qa=evaluator,
                chunk_duration=args.chunk_duration,
                fps=args.fps,
                recent_frames_only=args.recent_frames_only,
            )
            backward_results.append(result)
            done_keys.add(key)
            append_checkpoint_row(checkpoint_file, result)

        # Real-time tasks use the same evaluation function and video naming
        # convention as Backward tasks: {chunked_dir}/{sample_id}.mp4.
        for anno in tqdm(local_realtime, desc=f"[GPU{accelerator.process_index}] Realtime", disable=not accelerator.is_local_main_process):
            key = make_ovo_key(anno)
            if key in done_keys:
                continue
            result = evaluate_ovo_backward_realtime(
                anno=anno,
                chunked_dir=args.chunked_dir,
                qa=evaluator,
                chunk_duration=args.chunk_duration,
                fps=args.fps,
                recent_frames_only=args.recent_frames_only,
            )
            realtime_results.append(result)
            done_keys.add(key)
            append_checkpoint_row(checkpoint_file, result)

        # Forward tasks may contain multiple internal sub-questions/videos.
        # evaluate_ovo_forward handles that nested loop and updates test_info.
        for anno in tqdm(local_forward, desc=f"[GPU{accelerator.process_index}] Forward", disable=not accelerator.is_local_main_process):
            key = make_ovo_key(anno)
            if key in done_keys:
                continue
            result = evaluate_ovo_forward(
                anno=anno,
                chunked_dir=args.chunked_dir,
                qa=evaluator,
                chunk_duration=args.chunk_duration,
                fps=args.fps,
                recent_frames_only=args.recent_frames_only,
            )
            forward_results.append(result)
            done_keys.add(key)
            append_checkpoint_row(checkpoint_file, result)

    # Mark this rank as complete. Rank 0 waits for all markers before merging.
    write_done_marker(done_path)

    if accelerator.is_main_process:
        # Only rank 0 produces the final merged JSON and printed summary.
        wait_for_done_markers(args.result_dir, accelerator.num_processes)
        all_backward, all_realtime, all_forward = merge_shard_results(args.result_dir, accelerator.num_processes)
        print_ovo_results(MODEL_LABEL, all_backward, all_realtime, all_forward)
        os.makedirs(args.result_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.result_dir, f"qwen3vl_results_{timestamp}.json")
        with open(output_path, "w") as handle:
            json.dump(
                {
                    "config": {
                        "model_path": args.model_path,
                        "recent_frames_only": args.recent_frames_only,
                        "chunk_duration": args.chunk_duration,
                        "fps": args.fps,
                        "max_samples_per_split": args.max_samples_per_split,
                    },
                    "backward": all_backward,
                    "realtime": all_realtime,
                    "forward": all_forward,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
