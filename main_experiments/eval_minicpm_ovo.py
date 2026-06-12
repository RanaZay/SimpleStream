"""
OVO-Bench evaluation for MiniCPM-V-4.6.

By default this keeps the same SimpleStream recent-window protocol used for
Qwen models. Use --frame_selection all to evaluate every decoded 1 FPS frame
from each OVO clip.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

os.environ.setdefault("NCCL_TIMEOUT", "7200")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "0")

from accelerate import Accelerator
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ovo_constants import BACKWARD_TASKS, FORWARD_TASKS, REAL_TIME_TASKS
from lib.recent_window_eval_minicpm import (
    RecentWindowQAModel,
    evaluate_ovo_backward_realtime,
    evaluate_ovo_forward,
    print_ovo_results,
)
from lib.cdas_sampler import CDASConfig
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

MODEL_LABEL = "MiniCPM-V-4.6"


def main() -> None:
    parser = argparse.ArgumentParser(description="OVO-Bench evaluation for MiniCPM-V-4.6")
    parser.add_argument("--model_path", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument(
        "--qa_device",
        default=None,
        help="Device map for MiniCPM. Use 'auto' to shard the model over visible GPUs.",
    )
    parser.add_argument("--anno_path", default="data/ovo_bench/ovo_bench_new.json")
    parser.add_argument("--chunked_dir", default="data/ovo_bench/chunked_videos")
    parser.add_argument("--result_dir", default="results/ovo_bench_recent_window_minicpmv46")
    parser.add_argument(
        "--frame_selection",
        choices=["recent", "all"],
        default="recent",
        help="recent = SimpleStream recent-window; all = use every decoded 1 FPS frame in each clip.",
    )
    parser.add_argument("--recent_frames_only", type=int, default=4)
    parser.add_argument("--chunk_duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_qa_tokens", type=int, default=256)
    parser.add_argument("--cdas_enable", action="store_true", help="Enable Content-Density Adaptive Sampling.")
    parser.add_argument("--cdas_mode", choices=["binary", "three_level"], default="three_level")
    parser.add_argument("--cdas_skip_threshold", type=float, default=0.03)
    parser.add_argument("--cdas_high_threshold", type=float, default=0.12)
    parser.add_argument("--cdas_anchor_seconds", type=float, default=2.0)
    parser.add_argument("--cdas_min_accepted_fps", type=float, default=0.25)
    parser.add_argument("--cdas_gray_weight", type=float, default=0.50)
    parser.add_argument("--cdas_edge_weight", type=float, default=0.30)
    parser.add_argument("--cdas_hist_weight", type=float, default=0.20)
    parser.add_argument("--cdas_resize", type=int, default=96)
    parser.add_argument("--cdas_log_scores", action="store_true")
    parser.add_argument(
        "--max_samples_per_split",
        type=int,
        default=None,
        help="Optional sample cap applied independently to backward/realtime/forward after shuffle.",
    )
    args = parser.parse_args()
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
    run_mode = "All-Frames" if args.frame_selection == "all" else "Recent-Window"
    accelerator.print(f"OVO-Bench {run_mode} Evaluation ({MODEL_LABEL})")
    accelerator.print(f"{'=' * 60}")
    accelerator.print(f"Backward: {len(backward_anno)}, Realtime: {len(realtime_anno)}, Forward: {len(forward_anno)}")
    accelerator.print(f"Processes: {accelerator.num_processes}")
    accelerator.print(
        f"Frame selection: {args.frame_selection}, "
        f"recent_frames_only={args.recent_frames_only}, "
        f"chunk_duration={args.chunk_duration}, fps={args.fps}"
    )
    if args.frame_selection == "all" and cdas_config.enabled:
        accelerator.print("CDAS is ignored when --frame_selection all is used.")
    if cdas_config.enabled:
        accelerator.print(
            "CDAS: "
            f"mode={cdas_config.mode}, "
            f"skip={cdas_config.skip_threshold}, "
            f"high={cdas_config.high_threshold}, "
            f"anchor={cdas_config.anchor_seconds}, "
            f"min_fps={cdas_config.min_accepted_fps}"
        )
    if args.max_samples_per_split is not None:
        accelerator.print(f"Sample cap per split: {args.max_samples_per_split}")
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
        load_marker_dir = os.path.join(
            args.result_dir,
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
                    raise TimeoutError(
                        f"Timed out waiting for previous MiniCPM load marker: {previous_marker}"
                    )
                time.sleep(2.0)

        print(f"[rank {accelerator.process_index}] Loading MiniCPM-V model", flush=True)
        evaluator = build_evaluator()
        with open(current_marker, "w") as marker:
            marker.write(datetime.now().isoformat() + "\n")
    else:
        evaluator = build_evaluator()

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
                cdas_config=cdas_config,
                frame_selection=args.frame_selection,
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
                cdas_config=cdas_config,
                frame_selection=args.frame_selection,
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
                cdas_config=cdas_config,
                frame_selection=args.frame_selection,
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
        output_path = os.path.join(args.result_dir, f"minicpmv46_results_{timestamp}.json")
        with open(output_path, "w") as handle:
            json.dump(
                {
                    "config": {
                        "model_path": args.model_path,
                        "qa_device": args.qa_device or str(accelerator.device),
                        "frame_selection": args.frame_selection,
                        "recent_frames_only": args.recent_frames_only,
                        "chunk_duration": args.chunk_duration,
                        "fps": args.fps,
                        "max_qa_tokens": args.max_qa_tokens,
                        "max_samples_per_split": args.max_samples_per_split,
                        "attn_implementation": os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
                        "downsample_mode": os.environ.get("MINICPM_DOWNSAMPLE_MODE", "16x"),
                        "max_slice_nums": os.environ.get("MINICPM_MAX_SLICE_NUMS", "1"),
                        "cdas": cdas_config.__dict__,
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
