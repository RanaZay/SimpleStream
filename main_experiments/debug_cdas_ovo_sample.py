"""
Debug CDAS decisions on one OVO-Bench sample without loading MiniCPM.

This is meant as a cheap pre-flight check before launching the full benchmark:
it decodes one OVO clip, applies Content-Density Adaptive Sampling, prints the
per-frame gate decisions, and can optionally save selected frames.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.cdas_sampler import CDASConfig, select_recent_frames_cdas
from ovo_constants import FORWARD_TASKS


def _load_annotations(path: str) -> list[dict[str, Any]]:
    with open(path) as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected OVO annotations to be a list, got {type(data).__name__}")
    return data


def _video_path_for(anno: dict[str, Any], chunked_dir: str, forward_index: int) -> str:
    sample_id = str(anno["id"])
    if anno.get("task") in FORWARD_TASKS:
        return os.path.join(chunked_dir, f"{sample_id}_{int(forward_index)}.mp4")
    return os.path.join(chunked_dir, f"{sample_id}.mp4")


def _select_annotation(args: argparse.Namespace, annotations: list[dict[str, Any]]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for anno in annotations:
        if args.id is not None and str(anno.get("id")) != str(args.id):
            continue
        if args.task is not None and str(anno.get("task")) != str(args.task):
            continue
        video_path = _video_path_for(anno, args.chunked_dir, args.forward_index)
        if not args.allow_missing_video and not os.path.exists(video_path):
            continue
        candidates.append(anno)

    if not candidates:
        hint = "Try --allow-missing-video to inspect annotation matching without requiring the mp4."
        raise ValueError(f"No matching OVO sample found. {hint}")
    if args.sample_index < 0 or args.sample_index >= len(candidates):
        raise IndexError(f"--sample-index {args.sample_index} is outside 0..{len(candidates) - 1}")
    return candidates[args.sample_index]


def _save_selected_frames(save_dir: str, selection_metadata: dict[str, Any], frames: list[Any]) -> None:
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamps = selection_metadata.get("selected_timestamps", [])
    chunk_ids = selection_metadata.get("selected_chunk_ids", [])
    actions = selection_metadata.get("selected_actions", [])
    scores = selection_metadata.get("selected_scores", [])
    for index, frame in enumerate(frames):
        ts = timestamps[index] if index < len(timestamps) else "na"
        chunk = chunk_ids[index] if index < len(chunk_ids) else "na"
        action = actions[index] if index < len(actions) else "na"
        score = scores[index] if index < len(scores) else "na"
        filename = f"selected_{index:02d}_chunk{chunk}_t{ts}_score{score}_{action}.jpg"
        frame.save(output_dir / filename)
    with open(output_dir / "cdas_metadata.json", "w") as handle:
        json.dump(selection_metadata, handle, indent=2, ensure_ascii=False)


def _format_prompt(task: str, anno: dict[str, Any], forward_index: int) -> str:
    from lib.recent_window_eval import build_ovo_prompt

    return build_ovo_prompt(task, anno, index=forward_index)


def _decode_chunks(args: argparse.Namespace, video_path: str):
    from lib.recent_window_eval import decode_video_to_chunks_qwen

    return decode_video_to_chunks_qwen(
        video_path=video_path,
        chunk_duration=args.chunk_duration,
        fps=args.fps,
        recent_frames_only=args.recent_frames_only,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug CDAS on one OVO-Bench sample without loading MiniCPM.")
    parser.add_argument("--anno-path", default="data/ovo_bench/ovo_bench_new.json")
    parser.add_argument("--chunked-dir", default="data/ovo_bench/chunked_videos")
    parser.add_argument("--id", default=None, help="Specific OVO sample id to debug.")
    parser.add_argument("--task", default=None, help="Optional OVO task filter, e.g. EPM, OCR, REC.")
    parser.add_argument("--sample-index", type=int, default=0, help="Index among matching samples with existing video.")
    parser.add_argument("--forward-index", type=int, default=0, help="Forward-task test_info/video index.")
    parser.add_argument("--allow-missing-video", action="store_true")
    parser.add_argument("--recent-frames-only", type=int, default=4)
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--default-downsample-mode", default=os.environ.get("MINICPM_DOWNSAMPLE_MODE", "16x"))
    parser.add_argument("--cdas-mode", choices=["binary", "three_level"], default="three_level")
    parser.add_argument("--cdas-skip-threshold", type=float, default=0.03)
    parser.add_argument("--cdas-high-threshold", type=float, default=0.12)
    parser.add_argument("--cdas-anchor-seconds", type=float, default=2.0)
    parser.add_argument("--cdas-min-accepted-fps", type=float, default=0.25)
    parser.add_argument("--cdas-gray-weight", type=float, default=0.50)
    parser.add_argument("--cdas-edge-weight", type=float, default=0.30)
    parser.add_argument("--cdas-hist-weight", type=float, default=0.20)
    parser.add_argument("--cdas-resize", type=int, default=96)
    parser.add_argument("--max-decisions", type=int, default=80)
    parser.add_argument("--save-dir", default=None, help="Optional directory for selected frames and metadata.")
    args = parser.parse_args()

    annotations = _load_annotations(args.anno_path)
    anno = _select_annotation(args, annotations)
    task = str(anno.get("task", ""))
    video_path = _video_path_for(anno, args.chunked_dir, args.forward_index)

    print("=" * 80)
    print("OVO CDAS DEBUG SAMPLE")
    print("=" * 80)
    print(f"id: {anno.get('id')}")
    print(f"task: {task}")
    print(f"question: {anno.get('question')}")
    if "gt" in anno:
        print(f"gt: {anno.get('gt')}")
    if task in FORWARD_TASKS:
        print(f"forward_index: {args.forward_index}")
        info = anno.get("test_info", [])
        if args.forward_index < len(info):
            print(f"forward_test_info: {info[args.forward_index]}")
    print(f"video_path: {video_path}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    prompt = _format_prompt(task, anno, args.forward_index)
    print("\nPrompt preview:")
    print(prompt[:600])

    chunks, decode_backend = _decode_chunks(args, video_path)
    decoded_frames = sum(len(chunk.frames) for chunk in chunks)
    print("\nDecode:")
    print(f"backend: {decode_backend}")
    print(f"chunks: {len(chunks)}")
    print(f"decoded_frames: {decoded_frames}")
    print(f"chunk_ids: {[chunk.chunk_index for chunk in chunks]}")

    window_size = max(1, int(args.recent_frames_only))
    baseline_chunks = chunks[-window_size:]
    baseline_frames = sum(len(chunk.frames) for chunk in baseline_chunks)
    print("\nSimpleStream fixed recent-window baseline:")
    print(f"recent_frames_only: {window_size}")
    print(f"final_chunk_ids: {[chunk.chunk_index for chunk in baseline_chunks]}")
    print(f"final_frames: {baseline_frames}")
    print(f"default_downsample_mode: {args.default_downsample_mode}")

    config = CDASConfig(
        enabled=True,
        mode=args.cdas_mode,
        skip_threshold=args.cdas_skip_threshold,
        high_threshold=args.cdas_high_threshold,
        anchor_seconds=args.cdas_anchor_seconds,
        min_accepted_fps=args.cdas_min_accepted_fps,
        gray_weight=args.cdas_gray_weight,
        edge_weight=args.cdas_edge_weight,
        hist_weight=args.cdas_hist_weight,
        resize=args.cdas_resize,
        log_scores=True,
    )
    config.validate()
    selection = select_recent_frames_cdas(
        chunks=chunks,
        window_size=window_size,
        config=config,
        default_downsample_mode=args.default_downsample_mode,
    )
    meta = selection.metadata

    print("\nCDAS summary:")
    for key in (
        "mode",
        "downsample_scope",
        "skip_threshold",
        "high_threshold",
        "anchor_seconds",
        "min_accepted_fps",
        "decoded_frames",
        "accepted_frames",
        "skipped_frames",
        "selected_frames",
        "frame_reduction",
        "selected_downsample_mode",
        "selected_chunk_ids",
        "selected_timestamps",
        "selected_scores",
        "selected_actions",
        "accepted_action_counts",
    ):
        print(f"{key}: {meta.get(key)}")

    decisions = list(meta.get("decisions", []))
    print("\nCDAS decisions:")
    print("idx  time      chunk  score    action  keep  reason")
    print("-" * 66)
    shown = decisions[: max(0, int(args.max_decisions))]
    for index, row in enumerate(shown):
        print(
            f"{index:03d}  "
            f"{float(row['timestamp']):8.3f}  "
            f"{int(row['chunk_index']):5d}  "
            f"{float(row['score']):7.5f}  "
            f"{str(row['action']):6s}  "
            f"{str(row['accepted']):4s}  "
            f"{row['reason']}"
        )
    if len(decisions) > len(shown):
        print(f"... omitted {len(decisions) - len(shown)} decisions; use --max-decisions {len(decisions)}")

    if args.save_dir:
        _save_selected_frames(args.save_dir, meta, selection.frames)
        print(f"\nSaved selected frames and metadata to: {args.save_dir}")


if __name__ == "__main__":
    main()
