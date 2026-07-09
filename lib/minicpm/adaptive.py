from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from lib.cdas_sampler import CDASConfig
from lib.minicpm.baseline import (
    RecentWindowQAModel,
    _build_profile,
    _capture_gpu_memory,
    _reset_gpu_memory_peaks,
    _synchronize_gpu_devices,
)
from lib.shared.recent_window import RecentWindowResult, decode_video_to_chunks_qwen


_HISTORY_RE = re.compile(
    r"\b("
    r"before|earlier|previous|previously|ago|past|history|throughout|"
    r"how many|how much time|count|times|total|after|next|then|later|"
    r"first|last|finally|event|causal|prospective|trace|backward|forward"
    r")\b",
    re.IGNORECASE,
)
_ACTION_RE = re.compile(
    r"\b("
    r"action|doing|do |does|perform|performing|move|moving|happen|happening|"
    r"activity|change|changes|start|stop|continue|inserted|appeared"
    r")\b",
    re.IGNORECASE,
)
_CURRENT_RE = re.compile(
    r"\b("
    r"right now|currently|current|color|wearing|holding|text|ocr|object|"
    r"where|what is|what are|who is"
    r")\b",
    re.IGNORECASE,
)
_LOCALIZATION_RE = re.compile(
    r"\b("
    r"text|ocr|word|words|sign|caption|logo|number|color|wearing|holding|"
    r"object|shape|where|traffic light|large structure|left side|right side|"
    r"person|man|woman|animal|vehicle"
    r")\b",
    re.IGNORECASE,
)
_TEXT_LOCALIZATION_RE = re.compile(
    r"\b(text|ocr|word|words|sign|caption|logo|number|letter|letters)\b",
    re.IGNORECASE,
)
_COUNT_MEMORY_RE = re.compile(r"\b(how many|count|times|total)\b", re.IGNORECASE)
_EARLY_MEMORY_RE = re.compile(r"\b(first|before|earlier|previous|previously|past|ago)\b", re.IGNORECASE)
_LATE_MEMORY_RE = re.compile(r"\b(last|finally|after|next|then|later|prospective|forward)\b", re.IGNORECASE)


@dataclass(frozen=True)
class AdaptiveWindowConfig:
    """Configuration for MiniCPM SimpleStream novelty variants.

    Modes:
      adaptive: choose a 4/6/8 recent window from the question.
      adaptive_dedup: adaptive window, then remove near-duplicate frames.
      adaptive_memory: adaptive window, plus older anchor frames for history questions.
      adaptive_dedup_memory: combine both additions.
      fixed_budget_memory: keep the chosen frame budget, replacing recent frames
        with older memory anchors for history questions.
      event_memory: adaptive memory with older anchors selected by visual change.
      fixed_event_memory: fixed-budget memory with visual-change anchors.
      episodic_memory: fixed-budget memory with one early context anchor and
        one high-change event anchor.
      first_anchor_memory: fixed-budget memory with first old anchor + recent frames.
      first_middle_anchor_memory: fixed-budget memory with first and middle old
        anchors + recent frames.
      foveated: adaptive recent window plus query-guided crop insets for
        localization-style questions.
      foveated_memory: recent-window memory plus query-guided crop insets for
        localization-style questions.
      online_memory: recent-6 backbone plus an online memory bank that scores
        older chunks by event change, text/detail signal, query type, recency,
        and temporal diversity.
    """

    mode: str = "adaptive"
    min_window: int = 4
    mid_window: int = 6
    max_window: int = 8
    dedup_threshold: float = 4.0
    dedup_min_frames: int = 4
    dedup_resize: int = 64
    memory_anchors: int = 2
    memory_search_chunks: int = 0
    foveation_grid: int = 4
    foveation_crop_fraction: float = 0.45
    foveation_inset_fraction: float = 0.46

    @classmethod
    def from_env(cls) -> "AdaptiveWindowConfig":
        return cls(
            mode=os.environ.get("MINICPM_ADAPTIVE_MODE", "adaptive"),
            min_window=int(os.environ.get("MINICPM_ADAPTIVE_MIN_WINDOW", "4")),
            mid_window=int(os.environ.get("MINICPM_ADAPTIVE_MID_WINDOW", "6")),
            max_window=int(os.environ.get("MINICPM_ADAPTIVE_MAX_WINDOW", "8")),
            dedup_threshold=float(os.environ.get("MINICPM_ADAPTIVE_DEDUP_THRESHOLD", "4.0")),
            dedup_min_frames=int(os.environ.get("MINICPM_ADAPTIVE_DEDUP_MIN_FRAMES", "4")),
            dedup_resize=int(os.environ.get("MINICPM_ADAPTIVE_DEDUP_RESIZE", "64")),
            memory_anchors=int(os.environ.get("MINICPM_ADAPTIVE_MEMORY_ANCHORS", "2")),
            memory_search_chunks=int(os.environ.get("MINICPM_ADAPTIVE_MEMORY_SEARCH_CHUNKS", "0")),
            foveation_grid=int(os.environ.get("MINICPM_ADAPTIVE_FOVEATION_GRID", "4")),
            foveation_crop_fraction=float(os.environ.get("MINICPM_ADAPTIVE_FOVEATION_CROP_FRACTION", "0.45")),
            foveation_inset_fraction=float(os.environ.get("MINICPM_ADAPTIVE_FOVEATION_INSET_FRACTION", "0.46")),
        )

    def validate(self) -> None:
        valid_modes = {
            "adaptive",
            "adaptive_dedup",
            "adaptive_memory",
            "adaptive_dedup_memory",
            "fixed_budget_memory",
            "event_memory",
            "fixed_event_memory",
            "episodic_memory",
            "first_anchor_memory",
            "first_middle_anchor_memory",
            "foveated",
            "foveated_memory",
            "online_memory",
        }
        if self.mode not in valid_modes:
            raise ValueError(f"Unknown adaptive mode {self.mode!r}; expected one of {sorted(valid_modes)}")
        if not (1 <= self.min_window <= self.mid_window <= self.max_window):
            raise ValueError("Adaptive windows must satisfy 1 <= min <= mid <= max")
        if self.dedup_min_frames < 1:
            raise ValueError("dedup_min_frames must be >= 1")
        if self.dedup_resize < 8:
            raise ValueError("dedup_resize must be >= 8")
        if self.memory_anchors < 0:
            raise ValueError("memory_anchors must be >= 0")
        if self.memory_search_chunks < 0:
            raise ValueError("memory_search_chunks must be >= 0")
        if self.foveation_grid < 1:
            raise ValueError("foveation_grid must be >= 1")
        if not (0.20 <= self.foveation_crop_fraction <= 0.90):
            raise ValueError("foveation_crop_fraction must be in [0.20, 0.90]")
        if not (0.20 <= self.foveation_inset_fraction <= 0.80):
            raise ValueError("foveation_inset_fraction must be in [0.20, 0.80]")

    @property
    def use_dedup(self) -> bool:
        return self.mode in {"adaptive_dedup", "adaptive_dedup_memory"}

    @property
    def use_memory(self) -> bool:
        return self.mode in {
            "adaptive_memory",
            "adaptive_dedup_memory",
            "fixed_budget_memory",
            "event_memory",
            "fixed_event_memory",
            "episodic_memory",
            "first_anchor_memory",
            "first_middle_anchor_memory",
            "foveated_memory",
            "online_memory",
        }

    @property
    def use_foveation(self) -> bool:
        return self.mode in {"foveated", "foveated_memory"}

    @property
    def online_memory(self) -> bool:
        return self.mode == "online_memory"

    @property
    def fixed_memory_budget(self) -> bool:
        return self.mode in {
            "fixed_budget_memory",
            "fixed_event_memory",
            "episodic_memory",
            "first_anchor_memory",
            "first_middle_anchor_memory",
        }

    @property
    def event_memory(self) -> bool:
        return self.mode in {"event_memory", "fixed_event_memory"}

    @property
    def episodic_memory(self) -> bool:
        return self.mode == "episodic_memory"

    @property
    def anchor_memory(self) -> bool:
        return self.mode in {"first_anchor_memory", "first_middle_anchor_memory"}


@dataclass
class AdaptiveSelection:
    frames: list[Image.Image]
    final_chunk_ids: list[int]
    metadata: dict[str, Any]


def classify_adaptive_window(prompt: str, config: AdaptiveWindowConfig) -> tuple[int, str]:
    """Choose the recent-window size from the user question/prompt text."""

    text = prompt.lower()
    if _HISTORY_RE.search(text):
        return config.max_window, "history_or_temporal"
    if _ACTION_RE.search(text):
        return config.mid_window, "action_or_event"
    if _CURRENT_RE.search(text):
        return config.min_window, "current_perception"
    return config.mid_window, "default_mid"


def _frame_signature(frame: Image.Image, resize: int) -> np.ndarray:
    gray = frame.convert("L").resize((resize, resize), Image.BILINEAR)
    return np.asarray(gray, dtype=np.float32)


def _mean_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left - right)))


def _evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count <= 0 or length <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    return sorted({round(i * (length - 1) / (count - 1)) for i in range(count)})


def _chunk_signature(chunk: Any, resize: int) -> np.ndarray:
    signatures = [_frame_signature(frame, resize) for frame in chunk.frames]
    if not signatures:
        return np.zeros((resize, resize), dtype=np.float32)
    return np.mean(np.stack(signatures, axis=0), axis=0)


def _should_foveate(prompt: str, reason: str, config: AdaptiveWindowConfig) -> bool:
    if not config.use_foveation:
        return False
    if reason == "history_or_temporal":
        return False
    return bool(_LOCALIZATION_RE.search(prompt))


def _score_crop(gray: np.ndarray, text_query: bool) -> float:
    if gray.size == 0:
        return 0.0
    contrast = float(np.std(gray))
    if gray.shape[0] > 1:
        grad_y = np.mean(np.abs(np.diff(gray, axis=0)))
    else:
        grad_y = 0.0
    if gray.shape[1] > 1:
        grad_x = np.mean(np.abs(np.diff(gray, axis=1)))
    else:
        grad_x = 0.0
    edge_score = float(grad_x + grad_y)
    if text_query:
        return 0.35 * contrast + 0.65 * edge_score
    return 0.55 * contrast + 0.45 * edge_score


def _select_foveal_box(
    frame: Image.Image,
    prompt: str,
    config: AdaptiveWindowConfig,
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    width, height = frame.size
    crop_w = max(1, min(width, int(round(width * config.foveation_crop_fraction))))
    crop_h = max(1, min(height, int(round(height * config.foveation_crop_fraction))))
    if crop_w >= width and crop_h >= height:
        return (0, 0, width, height), {
            "strategy": "full_frame",
            "score": 0.0,
            "text_query": bool(_TEXT_LOCALIZATION_RE.search(prompt)),
        }

    resize = max(32, int(config.dedup_resize))
    gray_small = _frame_signature(frame, resize)
    text_query = bool(_TEXT_LOCALIZATION_RE.search(prompt))
    grid = max(1, int(config.foveation_grid))
    best: tuple[float, float, tuple[int, int, int, int]] | None = None
    for row in range(grid):
        center_y = (row + 0.5) / grid
        for col in range(grid):
            center_x = (col + 0.5) / grid
            left = int(round(center_x * width - crop_w / 2))
            top = int(round(center_y * height - crop_h / 2))
            left = max(0, min(left, width - crop_w))
            top = max(0, min(top, height - crop_h))
            right = left + crop_w
            bottom = top + crop_h

            small_left = max(0, min(resize - 1, int(round(left / max(width, 1) * resize))))
            small_right = max(small_left + 1, min(resize, int(round(right / max(width, 1) * resize))))
            small_top = max(0, min(resize - 1, int(round(top / max(height, 1) * resize))))
            small_bottom = max(small_top + 1, min(resize, int(round(bottom / max(height, 1) * resize))))
            patch = gray_small[small_top:small_bottom, small_left:small_right]
            score = _score_crop(patch, text_query=text_query)
            center_prior = 1.0 - min(1.0, abs(center_x - 0.5) + abs(center_y - 0.5))
            total = score + (0.10 if not text_query else 0.03) * center_prior
            candidate = (total, center_prior, (left, top, right, bottom))
            if best is None or candidate[:2] > best[:2]:
                best = candidate

    assert best is not None
    return best[2], {
        "strategy": "edge_text" if text_query else "saliency_center",
        "score": float(best[0]),
        "center_prior": float(best[1]),
        "text_query": text_query,
    }


def _compose_foveated_frame(
    frame: Image.Image,
    box: tuple[int, int, int, int],
    config: AdaptiveWindowConfig,
) -> Image.Image:
    base = frame.convert("RGB").copy()
    width, height = base.size
    inset_w = max(1, int(round(width * config.foveation_inset_fraction)))
    inset_h = max(1, int(round(height * config.foveation_inset_fraction)))
    crop = base.crop(box).resize((inset_w, inset_h), Image.BICUBIC)

    box_center_x = (box[0] + box[2]) / 2
    box_center_y = (box[1] + box[3]) / 2
    margin = max(2, int(round(min(width, height) * 0.015)))
    if box_center_x > width / 2:
        inset_left = margin
    else:
        inset_left = width - inset_w - margin
    if box_center_y > height / 2:
        inset_top = margin
    else:
        inset_top = height - inset_h - margin
    inset_left = max(0, min(inset_left, width - inset_w))
    inset_top = max(0, min(inset_top, height - inset_h))

    draw = ImageDraw.Draw(base)
    draw.rectangle(box, outline=(255, 232, 96), width=max(2, margin // 2))
    base.paste(crop, (inset_left, inset_top))
    draw = ImageDraw.Draw(base)
    draw.rectangle(
        (inset_left, inset_top, inset_left + inset_w - 1, inset_top + inset_h - 1),
        outline=(255, 232, 96),
        width=max(2, margin // 2),
    )
    return base


def _apply_query_foveation(
    frames: list[Image.Image],
    prompt: str,
    reason: str,
    config: AdaptiveWindowConfig,
) -> tuple[list[Image.Image], list[dict[str, Any]]]:
    if not _should_foveate(prompt, reason, config):
        return frames, []
    foveated_frames: list[Image.Image] = []
    metadata: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        box, score_metadata = _select_foveal_box(frame, prompt, config)
        foveated_frames.append(_compose_foveated_frame(frame, box, config))
        metadata.append(
            {
                "frame_index": index,
                "box": [int(value) for value in box],
                **score_metadata,
            }
        )
    return foveated_frames, metadata


def _normalise(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _build_online_memory_bank(older_chunks: list[Any], config: AdaptiveWindowConfig) -> list[dict[str, Any]]:
    signatures = [_chunk_signature(chunk, config.dedup_resize) for chunk in older_chunks]
    change_scores: list[float] = []
    contrast_scores: list[float] = []
    text_detail_scores: list[float] = []
    previous_signature: np.ndarray | None = None
    for signature in signatures:
        if previous_signature is None:
            change_scores.append(0.0)
        else:
            change_scores.append(_mean_abs_diff(signature, previous_signature))
        contrast_scores.append(float(np.std(signature)))
        text_detail_scores.append(_score_crop(signature, text_query=True))
        previous_signature = signature

    change_norm = _normalise(change_scores)
    contrast_norm = _normalise(contrast_scores)
    text_norm = _normalise(text_detail_scores)
    denom = max(1, len(older_chunks) - 1)
    bank: list[dict[str, Any]] = []
    for index, chunk in enumerate(older_chunks):
        position = index / denom
        bank.append(
            {
                "index": index,
                "chunk": chunk,
                "chunk_id": int(chunk.chunk_index),
                "temporal_position": float(position),
                "event_change_score": float(change_scores[index]),
                "event_change_norm": float(change_norm[index]),
                "contrast_score": float(contrast_scores[index]),
                "contrast_norm": float(contrast_norm[index]),
                "text_detail_score": float(text_detail_scores[index]),
                "text_detail_norm": float(text_norm[index]),
            }
        )
    return bank


def _online_memory_base_scores(bank: list[dict[str, Any]], prompt: str) -> tuple[list[float], dict[str, bool]]:
    count_query = bool(_COUNT_MEMORY_RE.search(prompt))
    early_query = bool(_EARLY_MEMORY_RE.search(prompt))
    late_query = bool(_LATE_MEMORY_RE.search(prompt))
    text_query = bool(_TEXT_LOCALIZATION_RE.search(prompt))
    flags = {
        "count_query": count_query,
        "early_query": early_query,
        "late_query": late_query,
        "text_query": text_query,
    }
    scores: list[float] = []
    for entry in bank:
        position = float(entry["temporal_position"])
        event_score = float(entry["event_change_norm"])
        contrast_score = float(entry["contrast_norm"])
        text_score = float(entry["text_detail_norm"])
        recency_score = position
        early_score = 1.0 - position
        middle_score = 1.0 - abs(position - 0.5) * 2.0

        score = 0.45 * event_score + 0.20 * contrast_score + 0.15 * recency_score
        if count_query:
            score += 0.25 * event_score + 0.20 * middle_score
        if early_query:
            score += 0.35 * early_score
        if late_query:
            score += 0.25 * recency_score
        if text_query:
            score += 0.30 * text_score
        scores.append(float(score))
    return scores, flags


def _select_online_memory_chunks(
    older_chunks: list[Any],
    count: int,
    config: AdaptiveWindowConfig,
    prompt: str,
) -> tuple[list[Any], list[dict[str, Any]]]:
    if count <= 0 or not older_chunks:
        return [], []

    bank = _build_online_memory_bank(older_chunks, config)
    base_scores, query_flags = _online_memory_base_scores(bank, prompt)
    if count >= len(bank):
        selected_indices = set(range(len(bank)))
    else:
        selected: list[int] = []
        diversity_weight = 0.35 if query_flags["count_query"] else 0.22
        while len(selected) < count:
            best_index: int | None = None
            best_score: float | None = None
            for index, entry in enumerate(bank):
                if index in selected:
                    continue
                if selected:
                    denom = max(1, len(bank) - 1)
                    diversity = min(abs(index - chosen) / denom for chosen in selected)
                else:
                    diversity = 1.0
                score = base_scores[index] + diversity_weight * diversity
                # Prefer deterministic chronological tie-breaking.
                if best_score is None or score > best_score or (
                    score == best_score and best_index is not None and index < best_index
                ):
                    best_score = score
                    best_index = index
            if best_index is None:
                break
            selected.append(best_index)
        selected_indices = set(selected)

    selected_order = sorted(selected_indices)
    metadata = []
    for index, entry in enumerate(bank):
        metadata.append(
            {
                "chunk_id": int(entry["chunk_id"]),
                "selected": index in selected_indices,
                "online_memory_score": float(base_scores[index]),
                "event_change_score": float(entry["event_change_score"]),
                "event_change_norm": float(entry["event_change_norm"]),
                "contrast_norm": float(entry["contrast_norm"]),
                "text_detail_norm": float(entry["text_detail_norm"]),
                "temporal_position": float(entry["temporal_position"]),
                "query_flags": query_flags,
            }
        )
    return [bank[index]["chunk"] for index in selected_order], metadata


def _select_memory_chunks(
    older_chunks: list[Any],
    count: int,
    config: AdaptiveWindowConfig,
    prompt: str = "",
) -> tuple[list[Any], list[dict[str, Any]]]:
    if count <= 0 or not older_chunks:
        return [], []
    if count >= len(older_chunks):
        return older_chunks, [
            {"chunk_id": int(chunk.chunk_index), "event_change_score": None, "selected": True}
            for chunk in older_chunks
        ]

    if config.online_memory:
        return _select_online_memory_chunks(older_chunks, count, config, prompt)

    if config.episodic_memory:
        return _select_episodic_memory_chunks(older_chunks, count, config)

    if config.anchor_memory:
        return _select_simple_anchor_memory_chunks(older_chunks, count, config)

    if not config.event_memory:
        indices = _evenly_spaced_indices(len(older_chunks), count)
        return [older_chunks[index] for index in indices], [
            {
                "chunk_id": int(chunk.chunk_index),
                "event_change_score": None,
                "selected": index in indices,
            }
            for index, chunk in enumerate(older_chunks)
        ]

    signatures = [_chunk_signature(chunk, config.dedup_resize) for chunk in older_chunks]
    scores = [0.0]
    for index in range(1, len(signatures)):
        scores.append(_mean_abs_diff(signatures[index], signatures[index - 1]))

    selected_indices = sorted(
        sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:count]
    )
    selected_set = set(selected_indices)
    metadata = [
        {
            "chunk_id": int(chunk.chunk_index),
            "event_change_score": scores[index],
            "selected": index in selected_set,
        }
        for index, chunk in enumerate(older_chunks)
    ]
    return [older_chunks[index] for index in selected_indices], metadata


def _select_simple_anchor_memory_chunks(
    older_chunks: list[Any],
    count: int,
    config: AdaptiveWindowConfig,
) -> tuple[list[Any], list[dict[str, Any]]]:
    selected_indices = [0]
    if config.mode == "first_middle_anchor_memory" and count > 1:
        middle_index = len(older_chunks) // 2
        if middle_index not in selected_indices:
            selected_indices.append(middle_index)

    if len(selected_indices) < count:
        for index in _evenly_spaced_indices(len(older_chunks), count):
            if index not in selected_indices:
                selected_indices.append(index)
            if len(selected_indices) >= count:
                break

    selected_indices = sorted(selected_indices[:count])
    selected_set = set(selected_indices)
    metadata = [
        {
            "chunk_id": int(chunk.chunk_index),
            "event_change_score": None,
            "selected": index in selected_set,
            "anchor_role": (
                "first_anchor"
                if index == 0 and index in selected_set
                else "middle_anchor"
                if index in selected_set
                else None
            ),
        }
        for index, chunk in enumerate(older_chunks)
    ]
    return [older_chunks[index] for index in selected_indices], metadata


def _select_episodic_memory_chunks(
    older_chunks: list[Any],
    count: int,
    config: AdaptiveWindowConfig,
) -> tuple[list[Any], list[dict[str, Any]]]:
    signatures = [_chunk_signature(chunk, config.dedup_resize) for chunk in older_chunks]
    scores = [0.0]
    for index in range(1, len(signatures)):
        scores.append(_mean_abs_diff(signatures[index], signatures[index - 1]))

    selected_indices: list[int] = []

    # Episodic context: preserve a stable old reference point.
    selected_indices.append(0)

    # Episodic event: retrieve the strongest visual change not already selected.
    for index in sorted(range(len(scores)), key=lambda idx: (-scores[idx], idx)):
        if index not in selected_indices:
            selected_indices.append(index)
        if len(selected_indices) >= count:
            break

    # If more anchors are requested, fill the remaining slots with temporal coverage.
    if len(selected_indices) < count:
        for index in _evenly_spaced_indices(len(older_chunks), count):
            if index not in selected_indices:
                selected_indices.append(index)
            if len(selected_indices) >= count:
                break

    selected_indices = sorted(selected_indices[:count])
    selected_set = set(selected_indices)
    metadata = [
        {
            "chunk_id": int(chunk.chunk_index),
            "event_change_score": scores[index],
            "selected": index in selected_set,
            "episodic_role": (
                "context_anchor"
                if index == selected_indices[0]
                else "event_anchor"
                if index in selected_set
                else None
            ),
        }
        for index, chunk in enumerate(older_chunks)
    ]
    return [older_chunks[index] for index in selected_indices], metadata


def _memory_selector_label(config: AdaptiveWindowConfig) -> str:
    if config.online_memory:
        return "online_memory_bank"
    if config.mode == "first_middle_anchor_memory":
        return "first_middle_anchor"
    if config.mode == "first_anchor_memory":
        return "first_anchor"
    if config.episodic_memory:
        return "episodic_context_event"
    if config.event_memory:
        return "event_change"
    return "evenly_spaced"


def select_adaptive_frames(
    chunks: list[Any],
    prompt: str,
    config: AdaptiveWindowConfig | None = None,
) -> AdaptiveSelection:
    config = config or AdaptiveWindowConfig.from_env()
    config.validate()
    if not chunks:
        raise ValueError("No chunks available for adaptive selection.")

    window_size, reason = classify_adaptive_window(prompt, config)
    memory_triggered = bool(config.use_memory and reason == "history_or_temporal")
    recent_window_size = window_size
    if memory_triggered and config.fixed_memory_budget and config.memory_anchors > 0:
        recent_window_size = max(1, window_size - config.memory_anchors)

    recent_chunks = chunks[-recent_window_size:]
    memory_chunks: list[Any] = []
    memory_scores: list[dict[str, Any]] = []
    if memory_triggered and config.memory_anchors > 0:
        older_chunks = chunks[: max(0, len(chunks) - recent_window_size)]
        memory_chunks, memory_scores = _select_memory_chunks(
            older_chunks,
            config.memory_anchors,
            config,
            prompt=prompt,
        )

    selected_chunks = [*memory_chunks, *recent_chunks]
    candidate_frames = [frame for chunk in selected_chunks for frame in chunk.frames]
    candidate_chunk_ids = [
        int(chunk.chunk_index)
        for chunk in selected_chunks
        for _frame in chunk.frames
    ]
    candidate_timestamps = [
        float(ts)
        for chunk in selected_chunks
        for ts in chunk.frame_timestamps
    ]
    if not candidate_frames:
        raise ValueError("Adaptive selection produced no frames.")

    kept_indices = list(range(len(candidate_frames)))
    duplicate_filter_scores: list[dict[str, Any]] = []
    if config.use_dedup and len(candidate_frames) > 1:
        signatures = [_frame_signature(frame, config.dedup_resize) for frame in candidate_frames]
        kept_indices = [0]
        for index in range(1, len(candidate_frames)):
            diff = _mean_abs_diff(signatures[index], signatures[kept_indices[-1]])
            duplicate_filter_scores.append(
                {
                    "index": index,
                    "chunk_id": candidate_chunk_ids[index],
                    "mean_abs_diff_from_previous_kept": diff,
                    "kept": diff >= config.dedup_threshold,
                }
            )
            if diff >= config.dedup_threshold:
                kept_indices.append(index)

        last_index = len(candidate_frames) - 1
        if last_index not in kept_indices:
            kept_indices.append(last_index)

        min_frames = min(len(candidate_frames), max(1, int(config.dedup_min_frames)))
        if len(kept_indices) < min_frames:
            for index in reversed(range(len(candidate_frames))):
                if index not in kept_indices:
                    kept_indices.append(index)
                if len(kept_indices) >= min_frames:
                    break
        kept_indices = sorted(set(kept_indices))

    frames = [candidate_frames[index] for index in kept_indices]
    final_chunk_ids = [candidate_chunk_ids[index] for index in kept_indices]
    frames, foveation_boxes = _apply_query_foveation(frames, prompt, reason, config)
    metadata = {
        "mode": config.mode,
        "window_size": window_size,
        "window_reason": reason,
        "config": {
            "min_window": config.min_window,
            "mid_window": config.mid_window,
            "max_window": config.max_window,
            "dedup_threshold": config.dedup_threshold,
            "dedup_min_frames": config.dedup_min_frames,
            "dedup_resize": config.dedup_resize,
            "memory_anchors": config.memory_anchors,
            "memory_search_chunks": config.memory_search_chunks,
            "foveation_grid": config.foveation_grid,
            "foveation_crop_fraction": config.foveation_crop_fraction,
            "foveation_inset_fraction": config.foveation_inset_fraction,
        },
        "decoded_chunks": len(chunks),
        "recent_window_size": recent_window_size,
        "recent_chunk_ids": [int(chunk.chunk_index) for chunk in recent_chunks],
        "memory_triggered": memory_triggered,
        "memory_fixed_budget": bool(config.fixed_memory_budget),
        "memory_selector": _memory_selector_label(config),
        "memory_chunk_ids": [int(chunk.chunk_index) for chunk in memory_chunks],
        "memory_scores": memory_scores,
        "candidate_frames": len(candidate_frames),
        "selected_frames": len(frames),
        "candidate_chunk_ids": candidate_chunk_ids,
        "selected_chunk_ids": final_chunk_ids,
        "candidate_timestamps": candidate_timestamps,
        "selected_timestamps": [candidate_timestamps[index] for index in kept_indices],
        "dedup_applied": bool(config.use_dedup),
        "dedup_scores": duplicate_filter_scores,
        "foveation_applied": bool(foveation_boxes),
        "foveation_boxes": foveation_boxes,
    }
    return AdaptiveSelection(frames=frames, final_chunk_ids=final_chunk_ids, metadata=metadata)


def query_adaptive_window(
    qa: RecentWindowQAModel,
    video_path: str,
    prompt: str,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    video_start: float | None = None,
    video_end: float | None = None,
    cdas_config: CDASConfig | None = None,
) -> tuple[RecentWindowResult, str]:
    """Evaluate MiniCPM with adaptive SimpleStream-style frame selection.

    cdas_config is accepted for signature compatibility with the baseline
    evaluator, but adaptive runs do not apply CDAS.
    """

    del cdas_config
    config = AdaptiveWindowConfig.from_env()
    config.validate()
    before_memory = _reset_gpu_memory_peaks()

    _window_size, reason = classify_adaptive_window(prompt, config)
    memory_would_trigger = bool(config.use_memory and reason == "history_or_temporal")
    memory_search_chunks = max(config.memory_anchors, config.memory_search_chunks) if memory_would_trigger else 0
    decode_recent_hint = max(int(recent_frames_only), config.max_window + memory_search_chunks)
    decode_t0 = time.perf_counter()
    chunks, decode_backend = decode_video_to_chunks_qwen(
        video_path=video_path,
        chunk_duration=chunk_duration,
        fps=fps,
        recent_frames_only=decode_recent_hint,
        video_start=video_start,
        video_end=video_end,
    )
    decode_time = time.perf_counter() - decode_t0
    if not chunks:
        raise ValueError(f"No chunks decoded from video: {video_path}")

    selection_t0 = time.perf_counter()
    selection = select_adaptive_frames(chunks, prompt=prompt, config=config)
    selection_time = time.perf_counter() - selection_t0
    if not selection.frames:
        raise ValueError(f"No frames selected from video: {video_path}")

    t0 = time.perf_counter()
    answer = qa.generate_from_frames(selection.frames, prompt)
    _synchronize_gpu_devices()
    generate_time = time.perf_counter() - t0
    ttft_seconds = getattr(qa, "_last_ttft_seconds", 0.0) or 0.0
    num_vision_tokens = getattr(qa, "_last_num_vision_tokens", 0) or 0
    num_frames = getattr(qa, "_last_num_vision_frames", 0) or len(selection.frames)

    _synchronize_gpu_devices()
    after_memory = _capture_gpu_memory()
    profile_metadata = _build_profile(
        mode=config.mode,
        decode_time=decode_time,
        selection_time=selection_time,
        generate_time=generate_time,
        before_memory=before_memory,
        after_memory=after_memory,
        qa=qa,
    )
    profile_metadata["adaptive"] = selection.metadata
    profile_metadata["decoded_chunks"] = len(chunks)
    profile_metadata["decoded_frames"] = sum(len(chunk.frames) for chunk in chunks)
    profile_metadata["video_start"] = video_start
    profile_metadata["video_end"] = video_end

    result = RecentWindowResult(
        answer=answer,
        final_chunk_ids=selection.final_chunk_ids,
        generate_time=generate_time,
        ttft_seconds=ttft_seconds,
        num_vision_tokens=num_vision_tokens,
        num_vision_tokens_before=num_vision_tokens,
        num_vision_tokens_after=num_vision_tokens,
        num_frames=num_frames,
    )
    result.profile_metadata = profile_metadata
    result.adaptive_metadata = selection.metadata
    return result, decode_backend


# The existing evaluator imports this name, so expose a compatible override.
query_recent_window = query_adaptive_window
