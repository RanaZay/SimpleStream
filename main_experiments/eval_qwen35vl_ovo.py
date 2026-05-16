"""
OVO-Bench recent-window evaluation for Qwen3.5-VL.

This uses the same SimpleStream recent-window recipe as Qwen3-VL, with a small
Qwen3.5 compatibility wrapper for the updated Transformers vision-feature output.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime

os.environ.setdefault("NCCL_TIMEOUT", "7200")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "0")

from accelerate import Accelerator
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ovo_constants import BACKWARD_TASKS, FORWARD_TASKS, REAL_TIME_TASKS
from lib.recent_window_eval_qwen35 import (
    RecentWindowQAModel,
    evaluate_ovo_backward_realtime,
    evaluate_ovo_forward,
    print_ovo_results,
)
from main_experiments.eval_qwen3vl_ovo import (
    append_checkpoint_row,
    get_checkpoint_path,
    get_done_path,
    load_checkpoint_state,
    make_ovo_key,
    merge_shard_results,
    wait_for_done_markers,
    write_done_marker,
)

MODEL_LABEL = "Qwen3.5-VL"


def main() -> None:
    parser = argparse.ArgumentParser(description="OVO-Bench recent-window evaluation for Qwen3.5-VL")
    parser.add_argument("--model_path", required=True, help="Example: Qwen/Qwen3.5-0.8B")
    parser.add_argument("--anno_path", default="data/ovo_bench/ovo_bench_new.json")
    parser.add_argument("--chunked_dir", default="data/ovo_bench/chunked_videos")
    parser.add_argument("--result_dir", default="results/ovo_bench_recent_window_qwen35vl")
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

    evaluator = RecentWindowQAModel(
        model_name=args.model_path,
        device=accelerator.device,
        max_new_tokens=args.max_qa_tokens,
    )

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

    with open(checkpoint_path, "a") as checkpoint_file:
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

    write_done_marker(done_path)

    if accelerator.is_main_process:
        wait_for_done_markers(args.result_dir, accelerator.num_processes)
        all_backward, all_realtime, all_forward = merge_shard_results(args.result_dir, accelerator.num_processes)
        print_ovo_results(MODEL_LABEL, all_backward, all_realtime, all_forward)
        os.makedirs(args.result_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.result_dir, f"qwen35vl_results_{timestamp}.json")
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
