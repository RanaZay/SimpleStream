#!/usr/bin/env python3
"""Distributed full StreamingBench eval for isolated exact-six recent samplers."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from lib.minicpm.baseline import (  # noqa: E402
    RecentWindowQAModel,
    _build_profile,
    _capture_gpu_memory,
    _reset_gpu_memory_peaks,
    _synchronize_gpu_devices,
)
from lib.shared.recent_window import RecentWindowResult, decode_video_to_chunks_qwen  # noqa: E402
from main_experiments.minicpm_v46.streamingbench import eval_baseline_dist as dist_sb  # noqa: E402


@dataclass(frozen=True)
class FrameRecord:
    image: Image.Image
    timestamp: float
    chunk_id: int
    frame_index: int

    @property
    def key(self) -> tuple[int, int, float]:
        return (self.chunk_id, self.frame_index, round(self.timestamp, 6))


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def _chunk_id(chunk: Any) -> int:
    return int(getattr(chunk, "chunk_index"))


def _chunk_bounds(chunk: Any) -> tuple[float, float]:
    timestamps = getattr(chunk, "frame_timestamps", None) or []
    numeric = [float(ts) for ts in timestamps if isinstance(ts, (int, float)) and math.isfinite(float(ts))]
    if numeric:
        return min(numeric), max(numeric)
    start = _number(getattr(chunk, "start_time", None))
    end = _number(getattr(chunk, "end_time", None))
    if start is None:
        start = float(_chunk_id(chunk))
    if end is None:
        end = start
    return (end, start) if end < start else (start, end)


def _frame_records_from_chunks(chunks: list[Any]) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for chunk in chunks:
        frames = list(getattr(chunk, "frames", []) or [])
        timestamps = getattr(chunk, "frame_timestamps", None) or []
        numeric = [float(ts) for ts in timestamps if isinstance(ts, (int, float)) and math.isfinite(float(ts))]
        start, end = _chunk_bounds(chunk)
        if len(numeric) != len(frames):
            if len(frames) == 1:
                numeric = [0.5 * (start + end)]
            elif end > start:
                step = (end - start) / max(1, len(frames) - 1)
                numeric = [start + index * step for index in range(len(frames))]
            else:
                numeric = [start + index * 1e-4 for index in range(len(frames))]
        for index, (frame, ts) in enumerate(zip(frames, numeric)):
            records.append(FrameRecord(frame, float(ts), _chunk_id(chunk), index))
    return sorted(records, key=lambda item: (item.timestamp, item.chunk_id, item.frame_index))


def _unique_chronological(records: list[FrameRecord], limit: int = 6) -> list[FrameRecord]:
    seen: set[tuple[int, int, float]] = set()
    out: list[FrameRecord] = []
    for record in sorted(records, key=lambda item: (item.timestamp, item.chunk_id, item.frame_index)):
        if record.key in seen:
            continue
        seen.add(record.key)
        out.append(record)
        if len(out) >= limit:
            break
    return out


def _nearest_to_targets(records: list[FrameRecord], targets: list[float], count: int = 6) -> list[FrameRecord]:
    selected: list[FrameRecord] = []
    used: set[tuple[int, int, float]] = set()
    for target in targets:
        candidates = [record for record in records if record.key not in used]
        if not candidates:
            break
        chosen = min(candidates, key=lambda item: (abs(item.timestamp - target), -item.timestamp))
        selected.append(chosen)
        used.add(chosen.key)
    if len(selected) < min(count, len(records)):
        for record in sorted(records, key=lambda item: item.timestamp, reverse=True):
            if record.key not in used:
                selected.append(record)
                used.add(record.key)
            if len(selected) >= min(count, len(records)):
                break
    return _unique_chronological(selected, count)


def _temporal_stats(timestamps: list[float]) -> tuple[float | None, float | None]:
    if not timestamps:
        return None, None
    ordered = sorted(timestamps)
    span = ordered[-1] - ordered[0]
    if len(ordered) < 2:
        return span, None
    return span, statistics.mean([b - a for a, b in zip(ordered, ordered[1:])])


def _image_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def _make_targets(sampler: str, ts_sec: float, count: int) -> list[float]:
    if sampler == "current_recent6":
        return [ts_sec - offset for offset in (5.0, 4.0, 3.0, 2.0, 1.0, 0.0)]
    if sampler == "uniform_dense6":
        return [ts_sec - 3.0 + i * (3.0 / max(1, count - 1)) for i in range(count)]
    raise ValueError(f"Unsupported recent sampler: {sampler}")


def query_recent_sampler_window(
    qa: RecentWindowQAModel,
    video_path: str,
    prompt: str,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    video_start: float | None = None,
    video_end: float | None = None,
    cdas_config: Any | None = None,
) -> tuple[RecentWindowResult, str]:
    del cdas_config
    sampler = os.environ.get("MINICPM_RECENT_SAMPLER", "current_recent6")
    count = max(1, int(recent_frames_only))
    if video_end is None:
        raise ValueError("recent sampler eval requires video_end")
    query_time = float(video_end) - 1e-4
    start = 0.0 if video_start is None else float(video_start)

    before_memory = _reset_gpu_memory_peaks()
    decode_t0 = time.perf_counter()
    saved_exact_recent = os.environ.pop("QWEN_EXACT_RECENT_DECODE", None)
    try:
        chunks, decode_backend = decode_video_to_chunks_qwen(
            video_path=video_path,
            chunk_duration=chunk_duration,
            fps=fps,
            recent_frames_only=max(count, int(math.ceil((float(video_end) - start) * max(float(fps), 1.0)))),
            video_start=start,
            video_end=video_end,
        )
    finally:
        if saved_exact_recent is not None:
            os.environ["QWEN_EXACT_RECENT_DECODE"] = saved_exact_recent
    decode_time = time.perf_counter() - decode_t0
    records = _frame_records_from_chunks(chunks)
    if not records:
        raise ValueError(f"No frames decoded from video: {video_path}")

    selection_t0 = time.perf_counter()
    targets = _make_targets(sampler, query_time, count)
    selected = _nearest_to_targets(records, targets, count)
    if len(records) >= count and len(selected) != count:
        raise AssertionError(f"{sampler} selected {len(selected)} frames, expected {count}")
    if any(b.timestamp < a.timestamp for a, b in zip(selected, selected[1:])):
        raise AssertionError(f"{sampler} did not preserve chronological ordering")
    frames = [record.image for record in selected]
    timestamps = [record.timestamp for record in selected]
    final_chunk_ids = [record.chunk_id for record in selected]
    frame_indices = [record.frame_index for record in selected]
    frame_hashes = [_image_sha256(frame) for frame in frames]
    frame_sizes = [[int(frame.width), int(frame.height)] for frame in frames]
    span, gap = _temporal_stats(timestamps)
    selection_time = time.perf_counter() - selection_t0

    t0 = time.perf_counter()
    answer = qa.generate_from_frames(frames, prompt)
    _synchronize_gpu_devices()
    generate_time = time.perf_counter() - t0
    ttft_seconds = getattr(qa, "_last_ttft_seconds", 0.0) or 0.0
    num_vision_tokens = getattr(qa, "_last_num_vision_tokens", 0) or 0
    num_frames = getattr(qa, "_last_num_vision_frames", 0) or len(frames)
    _synchronize_gpu_devices()
    after_memory = _capture_gpu_memory()
    profile_metadata = _build_profile(
        mode=f"recent_sampler_{sampler}",
        decode_time=decode_time,
        selection_time=selection_time,
        generate_time=generate_time,
        before_memory=before_memory,
        after_memory=after_memory,
        qa=qa,
    )

    result = RecentWindowResult(
        answer=answer,
        final_chunk_ids=final_chunk_ids,
        generate_time=generate_time,
        ttft_seconds=ttft_seconds,
        num_vision_tokens=num_vision_tokens,
        num_vision_tokens_before=num_vision_tokens,
        num_vision_tokens_after=num_vision_tokens,
        num_frames=num_frames,
    )
    result.profile_metadata = profile_metadata
    result.adaptive_metadata = {
        "mode": "recent_sampler_exact_six",
        "sampler": sampler,
        "target_timestamps": targets,
        "selected_timestamps": timestamps,
        "selected_chunk_ids": final_chunk_ids,
        "selected_frame_indices": frame_indices,
        "recent_frame_hashes": frame_hashes,
        "recent_frame_timestamps": timestamps,
        "recent_frame_indices": frame_indices,
        "recent_frame_sizes": frame_sizes,
        "selected_frame_count": len(selected),
        "candidate_frame_count": len(records),
        "candidate_chunk_count": len(chunks),
        "temporal_span_seconds": span,
        "mean_adjacent_spacing_seconds": gap,
        "query_time_seconds": query_time,
        "video_start_seconds": start,
        "video_end_seconds": float(video_end),
        "fps": float(fps),
        "exact_six_budget": bool(len(records) >= count and len(selected) == count),
        "short_video_less_than_six": bool(len(records) < count),
    }
    return result, decode_backend


def _print_recent_sampler_summary(results: list[dict[str, Any]], frame_selection: str = "recent") -> None:
    del frame_selection
    summary = dist_sb.compute_summary(results)
    sampler = os.environ.get("MINICPM_RECENT_SAMPLER", "current_recent6")
    print("\n" + "=" * 60)
    print(f"StreamingBench Recent-Sampler Results (MiniCPM-V-4.6 + {sampler})")
    print("=" * 60)
    for row in summary["tasks"]:
        print(f"  {row['task_type']}: {row['accuracy']:.2f}% ({row['correct']}/{row['total']})")
    overall = summary["overall"]
    print(f"\n  Overall: {overall['accuracy']:.2f}% ({overall['correct']}/{overall['total']})")
    print(f"  Errors: {summary['error_count']}")
    print("=" * 60)


def _consume_recent_sampler_args() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--recent-sampler",
        choices=["current_recent6", "uniform_dense6"],
        default=os.environ.get("MINICPM_RECENT_SAMPLER", "current_recent6"),
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    os.environ["MINICPM_RECENT_SAMPLER"] = args.recent_sampler


def main() -> None:
    _consume_recent_sampler_args()
    dist_sb.query_recent_window = query_recent_sampler_window
    dist_sb.print_summary = _print_recent_sampler_summary
    dist_sb.main()


if __name__ == "__main__":
    main()
