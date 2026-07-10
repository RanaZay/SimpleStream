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
_GATED_STRONG_MEMORY_RE = re.compile(
    r"\b("
    r"before|earlier|previous|previously|past|ago|throughout|"
    r"first|initially|beginning|start|started|"
    r"after|then|later|finally|last|"
    r"how many times|times in total|in total|total number|"
    r"trace|backward|history"
    r")\b",
    re.IGNORECASE,
)
_GATED_CURRENT_GUARD_RE = re.compile(
    r"\b("
    r"right now|currently|current|just now|at this moment|"
    r"what is|what are|what color|wearing|holding|visible now|text appeared"
    r")\b",
    re.IGNORECASE,
)
_GATED_PROSPECTIVE_GUARD_RE = re.compile(
    r"\b("
    r"next|most likely|will|would|after this|prospective|forward"
    r")\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-z][a-z0-9_-]*", re.IGNORECASE)
_QUERY_STOPWORDS = {
    "about",
    "after",
    "again",
    "answer",
    "appeared",
    "are",
    "best",
    "can",
    "choice",
    "could",
    "directly",
    "does",
    "doing",
    "during",
    "enough",
    "from",
    "give",
    "happen",
    "have",
    "how",
    "information",
    "into",
    "letter",
    "many",
    "now",
    "only",
    "option",
    "options",
    "person",
    "provided",
    "question",
    "right",
    "should",
    "there",
    "they",
    "this",
    "time",
    "video",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
}
_BOUND_QUERY_STOPWORDS = _QUERY_STOPWORDS | {
    "advanced",
    "analyze",
    "assistant",
    "based",
    "best",
    "context",
    "describe",
    "explain",
    "frame",
    "frames",
    "given",
    "image",
    "images",
    "likely",
    "multiple",
    "please",
    "provide",
    "scene",
    "select",
    "speaker",
    "task",
    "using",
}
_COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
}
_TEXT_SEMANTIC_WORDS = {
    "caption",
    "letter",
    "letters",
    "logo",
    "number",
    "ocr",
    "sign",
    "text",
    "word",
    "words",
}
_TEXTURE_SEMANTIC_WORDS = {
    "basket",
    "baskets",
    "building",
    "clothes",
    "fence",
    "grass",
    "handle",
    "pattern",
    "road",
    "shirt",
    "structure",
    "table",
    "traffic",
    "vehicle",
    "woven",
    "wood",
    "wooden",
}


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
      semantic_memory: recent-6 backbone plus older anchors selected by
        question-grounded color/detail evidence and temporal diversity.
      semantic_episodic_memory: recent-6 backbone plus both semantic anchors
        and an episodic early/event anchor.
      bound_semantic_episodic_memory: recent-6 backbone plus older anchors
        selected where semantic query relevance and episodic event importance
        reinforce each other.
      gated_semantic_episodic_memory: recent-6 by default; activate bound
        semantic-episodic memory only when the question needs older evidence.
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
            "semantic_memory",
            "semantic_episodic_memory",
            "bound_semantic_episodic_memory",
            "gated_semantic_episodic_memory",
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
            "semantic_memory",
            "semantic_episodic_memory",
            "bound_semantic_episodic_memory",
            "gated_semantic_episodic_memory",
        }

    @property
    def use_foveation(self) -> bool:
        return self.mode in {"foveated", "foveated_memory"}

    @property
    def online_memory(self) -> bool:
        return self.mode == "online_memory"

    @property
    def semantic_memory(self) -> bool:
        return self.mode == "semantic_memory"

    @property
    def semantic_episodic_memory(self) -> bool:
        return self.mode == "semantic_episodic_memory"

    @property
    def bound_semantic_episodic_memory(self) -> bool:
        return self.mode == "bound_semantic_episodic_memory"

    @property
    def gated_semantic_episodic_memory(self) -> bool:
        return self.mode == "gated_semantic_episodic_memory"

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


def _singularise(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _query_text_only(prompt: str) -> str:
    text = prompt.lower()
    for marker in ("\noptions:", "\nonly give", "\nanswer yes", "\nis there enough"):
        index = text.find(marker)
        if index >= 0:
            text = text[:index]
            break
    return text


def _gated_memory_activation(prompt: str, reason: str) -> tuple[bool, str]:
    """Decide if an older memory retrieval is warranted for this question.

    This gate is intentionally conservative. The fixed recent-6 backbone already
    works well for current-state questions, so memory is activated only for
    strong history/count cues and guarded off for current/prospective prompts.
    """

    text = _query_text_only(prompt)
    strong_memory = bool(_GATED_STRONG_MEMORY_RE.search(text))
    count_total = bool(
        re.search(
            r"\b(how many times|times in total|in total|total number|count)\b",
            text,
            re.IGNORECASE,
        )
    )
    current_guard = bool(_GATED_CURRENT_GUARD_RE.search(text))
    prospective_guard = bool(_GATED_PROSPECTIVE_GUARD_RE.search(text))

    if strong_memory or count_total:
        if prospective_guard and not (_EARLY_MEMORY_RE.search(text) or count_total):
            return False, "prospective_recent_guard"
        return True, "strong_history_or_count_cue"
    if current_guard:
        return False, "current_state_guard"
    if prospective_guard:
        return False, "prospective_recent_guard"
    if reason != "history_or_temporal":
        return False, "no_history_reason"
    return False, "weak_history_cue"


def _memory_trigger_decision(
    prompt: str,
    reason: str,
    config: AdaptiveWindowConfig,
) -> tuple[bool, dict[str, Any]]:
    if not config.use_memory:
        return False, {
            "enabled": False,
            "activated": False,
            "reason": "memory_mode_disabled",
        }
    if config.gated_semantic_episodic_memory:
        activated, gate_reason = _gated_memory_activation(prompt, reason)
        return activated, {
            "enabled": True,
            "activated": bool(activated),
            "reason": gate_reason,
        }
    activated = reason == "history_or_temporal"
    return activated, {
        "enabled": False,
        "activated": bool(activated),
        "reason": "legacy_history_rule" if activated else "legacy_no_history_reason",
    }


def _extract_semantic_query(prompt: str) -> dict[str, Any]:
    text = _query_text_only(prompt)
    raw_tokens = [token.lower() for token in _WORD_RE.findall(text)]
    terms: list[str] = []
    seen: set[str] = set()
    for token in raw_tokens:
        if len(token) < 3 or token in _QUERY_STOPWORDS:
            continue
        token = _singularise(token)
        if token in _QUERY_STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    colors = sorted({("gray" if term == "grey" else term) for term in terms if term in _COLOR_WORDS})
    text_terms = sorted({term for term in terms if term in _TEXT_SEMANTIC_WORDS})
    texture_terms = sorted({term for term in terms if term in _TEXTURE_SEMANTIC_WORDS})
    object_terms = [
        term
        for term in terms
        if term not in colors and term not in text_terms and term not in texture_terms
    ]
    return {
        "terms": terms,
        "colors": colors,
        "text_terms": text_terms,
        "texture_terms": texture_terms,
        "object_terms": object_terms,
    }


def _extract_bound_semantic_query(prompt: str) -> dict[str, Any]:
    text = _query_text_only(prompt)
    raw_tokens = [token.lower() for token in _WORD_RE.findall(text)]
    terms: list[str] = []
    seen: set[str] = set()
    for token in raw_tokens:
        if len(token) < 3 or token in _BOUND_QUERY_STOPWORDS:
            continue
        token = _singularise(token)
        if token in _BOUND_QUERY_STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    colors = sorted({("gray" if term == "grey" else term) for term in terms if term in _COLOR_WORDS})
    text_terms = sorted({term for term in terms if term in _TEXT_SEMANTIC_WORDS})
    texture_terms = sorted({term for term in terms if term in _TEXTURE_SEMANTIC_WORDS})
    object_terms = [
        term
        for term in terms
        if term not in colors and term not in text_terms and term not in texture_terms
    ]
    return {
        "terms": terms,
        "colors": colors,
        "text_terms": text_terms,
        "texture_terms": texture_terms,
        "object_terms": object_terms,
    }


def _chunk_rgb_sample(chunk: Any, resize: int) -> np.ndarray:
    samples: list[np.ndarray] = []
    side = max(16, min(96, int(resize)))
    for frame in chunk.frames:
        rgb = frame.convert("RGB").resize((side, side), Image.BILINEAR)
        samples.append(np.asarray(rgb, dtype=np.float32) / 255.0)
    if not samples:
        return np.zeros((side, side, 3), dtype=np.float32)
    return np.mean(np.stack(samples, axis=0), axis=0)


def _rgb_color_features(rgb: np.ndarray) -> dict[str, float]:
    if rgb.size == 0:
        return {color: 0.0 for color in _COLOR_WORDS if color != "grey"}
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])
    sat = maxc - minc
    value = maxc
    brightness = (r + g + b) / 3.0
    features = {
        "black": np.mean(brightness < 0.20),
        "white": np.mean((brightness > 0.78) & (sat < 0.18)),
        "gray": np.mean((brightness > 0.22) & (brightness < 0.78) & (sat < 0.12)),
        "red": np.mean((r > 0.42) & (r > g * 1.25) & (r > b * 1.25)),
        "green": np.mean((g > 0.34) & (g > r * 1.15) & (g > b * 1.15)),
        "blue": np.mean((b > 0.34) & (b > r * 1.15) & (b > g * 1.15)),
        "yellow": np.mean((r > 0.52) & (g > 0.48) & (b < 0.36) & (abs(r - g) < 0.30)),
        "orange": np.mean((r > 0.50) & (g > 0.25) & (g < 0.68) & (b < 0.34) & (r > g)),
        "brown": np.mean((r > 0.24) & (g > 0.14) & (b < 0.36) & (r > g * 1.05) & (g > b * 1.05) & (value < 0.78)),
        "pink": np.mean((r > 0.55) & (b > 0.38) & (g < 0.52) & (r > g * 1.15)),
        "purple": np.mean((r > 0.30) & (b > 0.38) & (g < 0.38)),
    }
    return {key: float(value) for key, value in features.items()}


def _semantic_color_score(color_features: dict[str, float], colors: list[str]) -> float:
    if not colors:
        return 0.0
    return float(max(color_features.get(color, 0.0) for color in colors))


def _semantic_proxy_score(
    entry: dict[str, Any],
    semantic_query: dict[str, Any],
) -> float:
    colors = semantic_query["colors"]
    text_terms = semantic_query["text_terms"]
    texture_terms = semantic_query["texture_terms"]
    object_terms = semantic_query["object_terms"]

    color_score = _semantic_color_score(entry["color_features"], colors)
    text_score = float(entry["text_detail_norm"]) if text_terms else 0.0
    texture_score = 0.5 * float(entry["contrast_norm"]) + 0.5 * float(entry["text_detail_norm"])
    if not texture_terms:
        texture_score = 0.0

    # Generic object words do not have detectors here, so use detail and change
    # as a cheap proxy for visible object evidence.
    object_score = 0.0
    if object_terms:
        object_score = (
            0.35 * float(entry["contrast_norm"])
            + 0.35 * float(entry["text_detail_norm"])
            + 0.30 * float(entry["event_change_norm"])
        )

    active = int(bool(colors)) + int(bool(text_terms)) + int(bool(texture_terms)) + int(bool(object_terms))
    if active == 0:
        return 0.0
    weighted = (
        0.42 * color_score
        + 0.25 * text_score
        + 0.20 * texture_score
        + 0.28 * object_score
    )
    return float(min(1.0, weighted / max(0.42, active * 0.28)))


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
    color_features_by_chunk: list[dict[str, float]] = []
    previous_signature: np.ndarray | None = None
    for chunk, signature in zip(older_chunks, signatures):
        if previous_signature is None:
            change_scores.append(0.0)
        else:
            change_scores.append(_mean_abs_diff(signature, previous_signature))
        contrast_scores.append(float(np.std(signature)))
        text_detail_scores.append(_score_crop(signature, text_query=True))
        color_features_by_chunk.append(_rgb_color_features(_chunk_rgb_sample(chunk, config.dedup_resize)))
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
                "color_features": color_features_by_chunk[index],
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


def _select_semantic_memory_chunks(
    older_chunks: list[Any],
    count: int,
    config: AdaptiveWindowConfig,
    prompt: str,
) -> tuple[list[Any], list[dict[str, Any]]]:
    if count <= 0 or not older_chunks:
        return [], []

    bank = _build_online_memory_bank(older_chunks, config)
    online_scores, query_flags = _online_memory_base_scores(bank, prompt)
    semantic_query = _extract_semantic_query(prompt)
    semantic_scores = [
        _semantic_proxy_score(entry, semantic_query)
        for entry in bank
    ]
    has_semantic_signal = bool(
        semantic_query["colors"]
        or semantic_query["text_terms"]
        or semantic_query["texture_terms"]
        or semantic_query["object_terms"]
    )

    combined_scores: list[float] = []
    for index, entry in enumerate(bank):
        event_score = float(entry["event_change_norm"])
        contrast_score = float(entry["contrast_norm"])
        recency_score = float(entry["temporal_position"])
        if has_semantic_signal:
            score = (
                0.55 * semantic_scores[index]
                + 0.25 * event_score
                + 0.10 * contrast_score
                + 0.10 * recency_score
            )
        else:
            score = online_scores[index]
        combined_scores.append(float(score))

    if count >= len(bank):
        selected_indices = set(range(len(bank)))
    else:
        selected: list[int] = []
        diversity_weight = 0.30 if has_semantic_signal else 0.22
        while len(selected) < count:
            best_index: int | None = None
            best_score: float | None = None
            for index in range(len(bank)):
                if index in selected:
                    continue
                if selected:
                    denom = max(1, len(bank) - 1)
                    diversity = min(abs(index - chosen) / denom for chosen in selected)
                else:
                    diversity = 1.0
                score = combined_scores[index] + diversity_weight * diversity
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
        color_hits = {
            color: float(entry["color_features"].get(color, 0.0))
            for color in semantic_query["colors"]
        }
        metadata.append(
            {
                "chunk_id": int(entry["chunk_id"]),
                "selected": index in selected_indices,
                "semantic_memory_score": float(combined_scores[index]),
                "semantic_proxy_score": float(semantic_scores[index]),
                "online_memory_score": float(online_scores[index]),
                "event_change_score": float(entry["event_change_score"]),
                "event_change_norm": float(entry["event_change_norm"]),
                "contrast_norm": float(entry["contrast_norm"]),
                "text_detail_norm": float(entry["text_detail_norm"]),
                "temporal_position": float(entry["temporal_position"]),
                "semantic_query": semantic_query,
                "semantic_color_hits": color_hits,
                "query_flags": query_flags,
            }
        )
    return [bank[index]["chunk"] for index in selected_order], metadata


def _select_semantic_episodic_memory_chunks(
    older_chunks: list[Any],
    count: int,
    config: AdaptiveWindowConfig,
    prompt: str,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Retrieve from semantic and episodic memory without changing either mode.

    For the default two anchors, one anchor comes from semantic query matching
    and one preserves an early episodic reference. With three or more anchors,
    episodic memory also contributes a high-change event anchor.
    """

    if count <= 0 or not older_chunks:
        return [], []

    episodic_slots = 0 if count == 1 else min(2, count - 1)
    semantic_slots = max(1, count - episodic_slots)

    semantic_chunks, semantic_metadata = _select_semantic_memory_chunks(
        older_chunks,
        semantic_slots,
        config,
        prompt,
    )
    episodic_chunks, episodic_metadata = _select_episodic_memory_chunks(
        older_chunks,
        episodic_slots,
        config,
    )

    chunk_to_index = {int(chunk.chunk_index): index for index, chunk in enumerate(older_chunks)}
    selected_semantic = {
        int(chunk.chunk_index)
        for chunk in semantic_chunks
    }
    selected_episodic = {
        int(chunk.chunk_index)
        for chunk in episodic_chunks
    }
    selected_ids: list[int] = []
    for chunk in [*semantic_chunks, *episodic_chunks]:
        chunk_id = int(chunk.chunk_index)
        if chunk_id not in selected_ids:
            selected_ids.append(chunk_id)

    if len(selected_ids) < count:
        ranked_semantic = sorted(
            semantic_metadata,
            key=lambda item: (-float(item.get("semantic_memory_score", 0.0)), int(item["chunk_id"])),
        )
        for item in ranked_semantic:
            chunk_id = int(item["chunk_id"])
            if chunk_id not in selected_ids:
                selected_ids.append(chunk_id)
                selected_semantic.add(chunk_id)
            if len(selected_ids) >= count:
                break

    if len(selected_ids) < count:
        ranked_episodic = sorted(
            episodic_metadata,
            key=lambda item: (
                -float(item.get("event_change_score") or 0.0),
                int(item["chunk_id"]),
            ),
        )
        for item in ranked_episodic:
            chunk_id = int(item["chunk_id"])
            if chunk_id not in selected_ids:
                selected_ids.append(chunk_id)
                selected_episodic.add(chunk_id)
            if len(selected_ids) >= count:
                break

    selected_ids = sorted(selected_ids[:count], key=lambda chunk_id: chunk_to_index[chunk_id])
    selected_id_set = set(selected_ids)
    semantic_meta_by_id = {int(item["chunk_id"]): item for item in semantic_metadata}
    episodic_meta_by_id = {int(item["chunk_id"]): item for item in episodic_metadata}

    metadata: list[dict[str, Any]] = []
    for chunk in older_chunks:
        chunk_id = int(chunk.chunk_index)
        semantic_meta = semantic_meta_by_id.get(chunk_id, {})
        episodic_meta = episodic_meta_by_id.get(chunk_id, {})
        semantic_selected = chunk_id in selected_semantic
        episodic_selected = chunk_id in selected_episodic
        if semantic_selected and episodic_selected:
            role = "semantic_and_episodic_anchor"
        elif episodic_selected:
            role = "episodic_anchor"
        elif semantic_selected:
            role = "semantic_anchor"
        else:
            role = None

        item = {
            "chunk_id": chunk_id,
            "selected": chunk_id in selected_id_set,
            "dual_memory_role": role,
            "semantic_selected": bool(semantic_selected),
            "episodic_selected": bool(episodic_selected),
            "semantic_memory_score": semantic_meta.get("semantic_memory_score"),
            "semantic_proxy_score": semantic_meta.get("semantic_proxy_score"),
            "online_memory_score": semantic_meta.get("online_memory_score"),
            "event_change_score": (
                episodic_meta.get("event_change_score")
                if "event_change_score" in episodic_meta
                else semantic_meta.get("event_change_score")
            ),
            "event_change_norm": semantic_meta.get("event_change_norm"),
            "contrast_norm": semantic_meta.get("contrast_norm"),
            "text_detail_norm": semantic_meta.get("text_detail_norm"),
            "temporal_position": semantic_meta.get("temporal_position"),
            "semantic_query": semantic_meta.get("semantic_query"),
            "semantic_color_hits": semantic_meta.get("semantic_color_hits"),
            "query_flags": semantic_meta.get("query_flags"),
            "episodic_role": episodic_meta.get("episodic_role"),
        }
        metadata.append(item)

    selected_chunks = [older_chunks[chunk_to_index[chunk_id]] for chunk_id in selected_ids]
    return selected_chunks, metadata


def _temporal_relevance_score(entry: dict[str, Any], query_flags: dict[str, bool]) -> float:
    position = float(entry["temporal_position"])
    early_score = 1.0 - position
    recency_score = position
    middle_score = 1.0 - abs(position - 0.5) * 2.0
    if query_flags["early_query"]:
        return float(early_score)
    if query_flags["late_query"]:
        return float(recency_score)
    if query_flags["count_query"]:
        return float(max(middle_score, float(entry["event_change_norm"])))
    return float(0.50 * middle_score + 0.50 * recency_score)


def _select_bound_semantic_episodic_memory_chunks(
    older_chunks: list[Any],
    count: int,
    config: AdaptiveWindowConfig,
    prompt: str,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Select anchors where question meaning and episode importance agree.

    This is a separate experimental selector from semantic_episodic_memory.
    Instead of independently choosing semantic and episodic anchors, each older
    chunk receives a joint score:

        semantic schema + episodic event importance + semantic*episodic binding.
    """

    if count <= 0 or not older_chunks:
        return [], []

    bank = _build_online_memory_bank(older_chunks, config)
    online_scores, query_flags = _online_memory_base_scores(bank, prompt)
    online_scores_norm = _normalise(online_scores)
    semantic_query = _extract_bound_semantic_query(prompt)
    semantic_scores = [
        _semantic_proxy_score(entry, semantic_query)
        for entry in bank
    ]
    has_schema = bool(
        semantic_query["colors"]
        or semantic_query["text_terms"]
        or semantic_query["texture_terms"]
        or semantic_query["object_terms"]
    )

    joint_scores: list[float] = []
    episodic_scores: list[float] = []
    binding_scores: list[float] = []
    temporal_scores: list[float] = []
    for index, entry in enumerate(bank):
        semantic_score = float(semantic_scores[index])
        event_score = float(entry["event_change_norm"])
        contrast_score = float(entry["contrast_norm"])
        detail_score = float(entry["text_detail_norm"])
        temporal_score = _temporal_relevance_score(entry, query_flags)
        episodic_score = (
            0.55 * event_score
            + 0.20 * contrast_score
            + 0.15 * temporal_score
            + 0.10 * detail_score
        )
        binding_score = semantic_score * episodic_score
        if has_schema:
            score = (
                0.40 * semantic_score
                + 0.25 * episodic_score
                + 0.20 * binding_score
                + 0.10 * temporal_score
                + 0.05 * contrast_score
            )
        else:
            # If no meaningful semantic schema remains after prompt cleanup,
            # fall back to a conservative episodic/online-memory score.
            score = (
                0.55 * episodic_score
                + 0.25 * float(online_scores_norm[index] if online_scores_norm else 0.0)
                + 0.15 * temporal_score
                + 0.05 * contrast_score
            )
        episodic_scores.append(float(episodic_score))
        binding_scores.append(float(binding_score))
        temporal_scores.append(float(temporal_score))
        joint_scores.append(float(score))

    if count >= len(bank):
        selected_indices = set(range(len(bank)))
    else:
        selected: list[int] = []
        while len(selected) < count:
            best_index: int | None = None
            best_score: float | None = None
            for index in range(len(bank)):
                if index in selected:
                    continue
                if selected:
                    denom = max(1, len(bank) - 1)
                    diversity = min(abs(index - chosen) / denom for chosen in selected)
                else:
                    diversity = 1.0
                score = joint_scores[index] + 0.25 * diversity
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
        color_hits = {
            color: float(entry["color_features"].get(color, 0.0))
            for color in semantic_query["colors"]
        }
        metadata.append(
            {
                "chunk_id": int(entry["chunk_id"]),
                "selected": index in selected_indices,
                "bound_memory_score": float(joint_scores[index]),
                "semantic_proxy_score": float(semantic_scores[index]),
                "episodic_importance_score": float(episodic_scores[index]),
                "semantic_episodic_binding_score": float(binding_scores[index]),
                "temporal_relevance_score": float(temporal_scores[index]),
                "online_memory_score": float(online_scores[index]),
                "event_change_score": float(entry["event_change_score"]),
                "event_change_norm": float(entry["event_change_norm"]),
                "contrast_norm": float(entry["contrast_norm"]),
                "text_detail_norm": float(entry["text_detail_norm"]),
                "temporal_position": float(entry["temporal_position"]),
                "semantic_query": semantic_query,
                "semantic_color_hits": color_hits,
                "query_flags": query_flags,
                "has_semantic_schema": bool(has_schema),
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

    if config.gated_semantic_episodic_memory or config.bound_semantic_episodic_memory:
        return _select_bound_semantic_episodic_memory_chunks(older_chunks, count, config, prompt)

    if config.semantic_episodic_memory:
        return _select_semantic_episodic_memory_chunks(older_chunks, count, config, prompt)

    if config.semantic_memory:
        return _select_semantic_memory_chunks(older_chunks, count, config, prompt)

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
    if config.gated_semantic_episodic_memory:
        return "gated_bound_semantic_episodic_memory"
    if config.bound_semantic_episodic_memory:
        return "bound_semantic_episodic_memory"
    if config.semantic_episodic_memory:
        return "semantic_episodic_memory"
    if config.semantic_memory:
        return "semantic_query_memory"
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
    memory_triggered, memory_gate = _memory_trigger_decision(prompt, reason, config)
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
        "memory_gate": memory_gate,
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
    memory_would_trigger, _memory_gate = _memory_trigger_decision(prompt, reason, config)
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
