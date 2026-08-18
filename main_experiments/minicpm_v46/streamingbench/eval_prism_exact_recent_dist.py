#!/usr/bin/env python3
"""Distributed StreamingBench eval for PRISM with corrected exact-six Recent-6.

This is an isolated validation entry point. It does not change official PRISM;
it only replaces the Recent-6 selector that PRISM receives with the same
exact-six current_recent6 selector used by eval_recent_sampler_dist.py.
"""

from __future__ import annotations

import math
import os
import sys
import time
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from main_experiments.tools.determinism import configure_determinism

SEED = configure_determinism()

import lib.minicpm.adaptive as adaptive_mod  # noqa: E402
from lib.minicpm.baseline import RecentFrameSelection  # noqa: E402
from lib.shared.recent_window import decode_video_to_chunks_qwen  # noqa: E402
from main_experiments.minicpm_v46.streamingbench import eval_adaptive_dist as adaptive_eval  # noqa: E402
from main_experiments.minicpm_v46.streamingbench import eval_baseline_dist as dist_sb  # noqa: E402
from main_experiments.minicpm_v46.streamingbench.eval_recent_sampler_dist import (  # noqa: E402
    _frame_records_from_chunks,
    _make_targets,
    _nearest_to_targets,
    _temporal_stats,
)


def _one_frame_chunk(record: Any, sequence_index: int) -> Any:
    # Keep wrapper-local recent IDs unique and outside the absolute video chunk
    # namespace. PRISM history eligibility is timestamp-based; these IDs are
    # only metadata/control-set keys inside the isolated validation wrapper.
    chunk_index = -1_000_000 + int(sequence_index)
    return SimpleNamespace(
        chunk_index=chunk_index,
        frames=[record.image],
        frame_timestamps=[float(record.timestamp)],
        start_time=float(record.timestamp),
        end_time=float(record.timestamp),
        fps=None,
    )


def select_exact_current_recent_frames(
    qa: Any,
    video_path: str,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    video_start: float | None = None,
    video_end: float | None = None,
    cdas_config: Any | None = None,
) -> RecentFrameSelection:
    """Select exact-six frame-level current_recent6 context for PRISM baseline_recent."""

    del qa, cdas_config
    count = max(1, int(recent_frames_only))

    candidate_fps = float(os.environ.get("MINICPM_EXACT_RECENT_CANDIDATE_FPS", os.environ.get("RECENT_SAMPLER_FPS", "4.0")))
    start = None if video_start is None else float(video_start)
    end = None if video_end is None else float(video_end)
    window_seconds = float(count) * float(chunk_duration)
    decode_hint = max(count, int(math.ceil(window_seconds * max(candidate_fps, 1.0))))
    if start is not None and end is not None:
        decode_hint = max(count, int(math.ceil((end - start) * max(candidate_fps, 1.0))))

    decode_t0 = time.perf_counter()
    saved_exact_recent = os.environ.pop("QWEN_EXACT_RECENT_DECODE", None)
    try:
        chunks, decode_backend = decode_video_to_chunks_qwen(
            video_path=video_path,
            chunk_duration=chunk_duration,
            fps=candidate_fps,
            recent_frames_only=decode_hint,
            video_start=start,
            video_end=end,
        )
    finally:
        if saved_exact_recent is not None:
            os.environ["QWEN_EXACT_RECENT_DECODE"] = saved_exact_recent
    decode_time = time.perf_counter() - decode_t0
    records = _frame_records_from_chunks(chunks)
    if not records:
        raise ValueError(f"No frames decoded from video: {video_path}")
    query_time = (float(end) - 1e-4) if end is not None else max(record.timestamp for record in records)

    selection_t0 = time.perf_counter()
    targets = _make_targets("current_recent6", query_time, count)
    selected = _nearest_to_targets(records, targets, count)
    if len(records) >= count and len(selected) != count:
        raise AssertionError(f"current_recent6 selected {len(selected)} frames, expected {count}")
    if any(b.timestamp < a.timestamp for a, b in zip(selected, selected[1:])):
        raise AssertionError("current_recent6 did not preserve chronological ordering")

    frames = [record.image for record in selected]
    timestamps = [float(record.timestamp) for record in selected]
    selected_chunks = [_one_frame_chunk(record, index) for index, record in enumerate(selected)]
    final_chunk_ids = [int(chunk.chunk_index) for chunk in selected_chunks]
    span, gap = _temporal_stats(timestamps)
    selection_time = time.perf_counter() - selection_t0

    return RecentFrameSelection(
        frames=frames,
        final_chunk_ids=final_chunk_ids,
        selected_chunks=selected_chunks,
        downsample_mode=None,
        cdas_metadata={
            "exact_recent_selector": "current_recent6",
            "target_timestamps": targets,
            "selected_timestamps": timestamps,
            "selected_chunk_ids": final_chunk_ids,
            "selected_frame_indices": [int(record.frame_index) for record in selected],
            "selected_frame_count": len(selected),
            "candidate_frame_count": len(records),
            "candidate_chunk_count": len(chunks),
            "temporal_span_seconds": span,
            "mean_adjacent_spacing_seconds": gap,
            "query_time_seconds": query_time,
            "candidate_fps": candidate_fps,
            "exact_six_budget": bool(len(records) >= count and len(selected) == count),
            "short_video_less_than_six": bool(len(records) < count),
        },
        decode_time=decode_time,
        selection_time=selection_time,
        decode_backend=decode_backend,
        decoded_chunks=len(chunks),
        decoded_frames=len(records),
        video_start=start,
        video_end=end,
    )


def main() -> None:
    adaptive_eval._consume_adaptive_args()
    os.environ["MINICPM_SEED"] = str(SEED)
    adaptive_mod.select_recent_window_frames = select_exact_current_recent_frames
    dist_sb.query_recent_window = adaptive_mod.query_recent_window
    dist_sb.print_summary = adaptive_eval._print_adaptive_summary
    dist_sb.main()


if __name__ == "__main__":
    main()
