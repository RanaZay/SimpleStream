from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from PIL import Image, ImageChops, ImageFilter, ImageStat

if TYPE_CHECKING:
    from lib.shared.recent_window import EvalChunk


@dataclass(frozen=True)
class CDASConfig:
    """Content-Density Adaptive Sampling configuration.

    This v0 implementation is training-free and streaming-causal: each frame is
    compared only with the previous decoded frame. MiniCPM-V currently exposes
    downsample_mode per generation call through the HF path, so the 4x/16x
    decision is applied as a per-query global choice after frame admission.
    """

    enabled: bool = False
    mode: str = "three_level"
    skip_threshold: float = 0.03
    high_threshold: float = 0.12
    anchor_seconds: float = 2.0
    min_accepted_fps: float = 0.25
    gray_weight: float = 0.50
    edge_weight: float = 0.30
    hist_weight: float = 0.20
    resize: int = 96
    log_scores: bool = False

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.mode not in {"binary", "three_level"}:
            raise ValueError(f"Unsupported CDAS mode: {self.mode!r}")
        if self.skip_threshold < 0:
            raise ValueError("CDAS skip_threshold must be >= 0")
        if self.high_threshold < self.skip_threshold:
            raise ValueError("CDAS high_threshold must be >= skip_threshold")
        if self.anchor_seconds < 0:
            raise ValueError("CDAS anchor_seconds must be >= 0")
        if self.min_accepted_fps < 0:
            raise ValueError("CDAS min_accepted_fps must be >= 0")
        if self.resize < 8:
            raise ValueError("CDAS resize must be >= 8")


@dataclass
class CDASSelection:
    frames: list[Image.Image]
    final_chunk_ids: list[int]
    downsample_mode: str
    metadata: dict[str, Any]


@dataclass
class _FrameState:
    frame: Image.Image
    timestamp: float
    chunk_index: int
    score: float
    action: str
    reason: str


def _resample_filter() -> int:
    resampling = getattr(Image, "Resampling", None)
    return int(getattr(resampling, "BILINEAR", Image.BILINEAR))


def _prepare_gray(frame: Image.Image, size: int) -> Image.Image:
    return frame.convert("L").resize((size, size), _resample_filter())


def _mean_abs_diff(a: Image.Image, b: Image.Image) -> float:
    diff = ImageChops.difference(a, b)
    stat = ImageStat.Stat(diff)
    return float(stat.mean[0]) / 255.0


def _hist_l1(a: Image.Image, b: Image.Image) -> float:
    hist_a = a.histogram()
    hist_b = b.histogram()
    pixels = max(int(a.size[0]) * int(a.size[1]), 1)
    return sum(abs(x - y) for x, y in zip(hist_a, hist_b)) / float(2 * pixels)


def _novelty_score(current: Image.Image, previous: Image.Image, config: CDASConfig) -> float:
    current_gray = _prepare_gray(current, config.resize)
    previous_gray = _prepare_gray(previous, config.resize)
    gray_delta = _mean_abs_diff(current_gray, previous_gray)
    edge_delta = _mean_abs_diff(
        current_gray.filter(ImageFilter.FIND_EDGES),
        previous_gray.filter(ImageFilter.FIND_EDGES),
    )
    hist_delta = _hist_l1(current_gray, previous_gray)
    score = (
        config.gray_weight * gray_delta
        + config.edge_weight * edge_delta
        + config.hist_weight * hist_delta
    )
    return max(0.0, min(float(score), 1.0))


def _iter_frames(chunks: list["EvalChunk"]) -> list[tuple[Image.Image, float, int]]:
    rows: list[tuple[Image.Image, float, int]] = []
    for chunk in chunks:
        for index, frame in enumerate(chunk.frames):
            if index < len(chunk.frame_timestamps):
                timestamp = float(chunk.frame_timestamps[index])
            else:
                timestamp = float(chunk.start_time)
            rows.append((frame, timestamp, int(chunk.chunk_index)))
    rows.sort(key=lambda item: (item[1], item[2]))
    return rows


def _unique_in_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def select_recent_frames_cdas(
    chunks: list["EvalChunk"],
    window_size: int,
    config: CDASConfig,
    default_downsample_mode: str,
) -> CDASSelection:
    """Select recent frames with CDAS and return compact run metadata."""

    config.validate()
    frames = _iter_frames(chunks)
    if not frames:
        raise ValueError("CDAS received no decoded frames")

    accepted: list[_FrameState] = []
    decisions: list[dict[str, Any]] = []
    previous_frame: Image.Image | None = None
    last_accepted_ts: float | None = None

    for frame, timestamp, chunk_index in frames:
        if previous_frame is None:
            score = 1.0
            action = "16x" if config.mode == "three_level" else "encode"
            reason = "first_frame"
            keep = True
        else:
            score = _novelty_score(frame, previous_frame, config)
            if config.mode == "three_level" and score >= config.high_threshold:
                keep = True
                action = "4x"
                reason = "high_novelty"
            elif score >= config.skip_threshold:
                keep = True
                action = "16x" if config.mode == "three_level" else "encode"
                reason = "medium_novelty"
            else:
                keep = False
                action = "skip"
                reason = "low_novelty"

            if not keep and last_accepted_ts is not None:
                elapsed = max(0.0, float(timestamp) - float(last_accepted_ts))
                if config.anchor_seconds > 0 and elapsed >= config.anchor_seconds:
                    keep = True
                    action = "16x" if config.mode == "three_level" else "encode"
                    reason = "anchor"
                elif config.min_accepted_fps > 0 and elapsed >= 1.0 / max(config.min_accepted_fps, 1e-6):
                    keep = True
                    action = "16x" if config.mode == "three_level" else "encode"
                    reason = "min_fps"

        if keep:
            accepted.append(
                _FrameState(
                    frame=frame,
                    timestamp=float(timestamp),
                    chunk_index=int(chunk_index),
                    score=float(score),
                    action=action,
                    reason=reason,
                )
            )
            last_accepted_ts = float(timestamp)

        if config.log_scores or keep:
            decisions.append(
                {
                    "timestamp": round(float(timestamp), 4),
                    "chunk_index": int(chunk_index),
                    "score": round(float(score), 5),
                    "action": action,
                    "accepted": bool(keep),
                    "reason": reason,
                }
            )
        previous_frame = frame

    if not accepted:
        frame, timestamp, chunk_index = frames[-1]
        accepted.append(
            _FrameState(
                frame=frame,
                timestamp=float(timestamp),
                chunk_index=int(chunk_index),
                score=1.0,
                action="16x" if config.mode == "three_level" else "encode",
                reason="fallback_last_frame",
            )
        )

    final_states = accepted[-max(1, int(window_size)) :]
    final_frames = [item.frame for item in final_states]
    final_chunk_ids = _unique_in_order([item.chunk_index for item in final_states])
    if config.mode == "three_level" and any(item.action == "4x" for item in final_states):
        downsample_mode = "4x"
    else:
        downsample_mode = default_downsample_mode

    action_counts: dict[str, int] = {}
    for item in accepted:
        action_counts[item.action] = action_counts.get(item.action, 0) + 1

    decoded_count = len(frames)
    accepted_count = len(accepted)
    selected_count = len(final_states)
    metadata: dict[str, Any] = {
        "enabled": True,
        "mode": config.mode,
        "downsample_scope": "per_query_global",
        "skip_threshold": config.skip_threshold,
        "high_threshold": config.high_threshold,
        "anchor_seconds": config.anchor_seconds,
        "min_accepted_fps": config.min_accepted_fps,
        "decoded_frames": decoded_count,
        "accepted_frames": accepted_count,
        "skipped_frames": max(decoded_count - accepted_count, 0),
        "selected_frames": selected_count,
        "selected_timestamps": [round(item.timestamp, 4) for item in final_states],
        "selected_chunk_ids": final_chunk_ids,
        "selected_actions": [item.action for item in final_states],
        "selected_scores": [round(item.score, 5) for item in final_states],
        "selected_downsample_mode": downsample_mode,
        "frame_reduction": (
            1.0 - float(selected_count) / float(decoded_count)
            if decoded_count
            else 0.0
        ),
        "accepted_action_counts": action_counts,
        "decisions": decisions,
    }
    return CDASSelection(
        frames=final_frames,
        final_chunk_ids=final_chunk_ids,
        downsample_mode=downsample_mode,
        metadata=metadata,
    )
