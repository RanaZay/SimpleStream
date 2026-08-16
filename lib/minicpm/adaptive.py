from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from lib.cdas_sampler import CDASConfig
from lib.minicpm.baseline import (
    RecentWindowQAModel,
    _build_profile,
    _capture_gpu_memory,
    _reset_gpu_memory_peaks,
    _synchronize_gpu_devices,
    select_recent_window_frames,
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
_STRICT_MEMORY_RE = re.compile(
    r"\b("
    r"before|earlier|previous|previously|past|ago|throughout|"
    r"first|initially|beginning|at the beginning|from the start|"
    r"how many times|times in total|in total|total number|"
    r"trace|backward|history"
    r")\b",
    re.IGNORECASE,
)
_STRICT_CURRENT_RE = re.compile(
    r"\b("
    r"right now|currently|current|just now|at this moment|now|"
    r"what is|what are|what color|wearing|holding|visible|shown|"
    r"text|ocr|word|sign|caption|object|where"
    r")\b",
    re.IGNORECASE,
)
_STRICT_RECENT_RE = re.compile(
    r"\b("
    r"next|most likely|will|would|after this|prospective|forward|"
    r"action|doing|performing|spatial|text-rich"
    r")\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-z][a-z0-9_-]*", re.IGNORECASE)
_MCQ_OPTION_RE = re.compile(r"^\s*([A-E])[\.\)]\s*(.+?)\s*$", re.MULTILINE)
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
_COLOR_FEATURE_NAMES = sorted(color for color in _COLOR_WORDS if color != "grey")
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
_QUESTION_AWARE_EXTRA_DIMS = [
    "text_detail",
    "texture_detail",
    "object_detail",
    "event_change",
    "early_context",
    "middle_context",
    "recent_context",
    "count_coverage",
]


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
      strict_gated_semantic_memory: recent-6 by default; activate semantic
        memory only for explicit older-evidence questions.
      question_aware_memory: recent-6 backbone plus older anchors selected by
        question-to-window similarity inspired by WindowQuant relevance scoring.
      event_summary_memory: recent-6 backbone plus at most one older frame
        retrieved from a tiny bounded/diverse event-bookmark memory when a
        history-style question confidently matches it.
      budgeted_counterfactual_memory: recent-6 backbone plus a utility gate
        that retrieves K=0/1/2 older frames only when predicted answer benefit
        exceeds the visual-token/latency cost.
      progressive_evidence_memory: recent-6 backbone plus demand-driven
        semantic memory acquisition. It starts with recent evidence, estimates
        whether the selected evidence is sufficient, and adds at most K older
        anchors one by one until the evidence score is high enough or the
        marginal gain becomes too small.
      full_progressive_evidence_memory: the full version of progressive
        evidence acquisition. It starts from recent-6, scores MiniCPM's MCQ
        option probabilities, combines answer confidence with evidence support,
        and retrieves semantic anchors one by one only while the answer is
        insufficient.
      progressive_sufficiency_memory: isolated PRISM-style mode using Recent-6,
        a diverse CLIP-ranked history queue, iterative option-logit/visual-support
        sufficiency, and at most three historical frames.
      progressive_sufficiency_memory_heg: same PRISM ranking and budget, with
        option-specific historical evidence gain allowed to trigger retrieval
        even when the current context passes the sufficiency threshold.
      progressive_sufficiency_memory_conservative_gate: isolated PRISM-style
        mode that keeps low-sufficiency retrieval, stops high-sufficiency
        cases, and retrieves in the ambiguous band only when the best unused
        historical candidate is strong and temporally separated.
      progressive_sufficiency_memory_microclip: isolated PRISM experimental
        mode that tries Recent-6, then one semantic historical anchor, then a
        local temporal micro-clip around that same anchor.
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
            "strict_gated_semantic_memory",
            "question_aware_memory",
            "event_summary_memory",
            "budgeted_counterfactual_memory",
            "progressive_evidence_memory",
            "full_progressive_evidence_memory",
            "progressive_sufficiency_memory",
            "progressive_sufficiency_memory_heg",
            "progressive_sufficiency_memory_conservative_gate",
            "progressive_sufficiency_memory_microclip",
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
            "strict_gated_semantic_memory",
            "question_aware_memory",
            "event_summary_memory",
            "budgeted_counterfactual_memory",
            "progressive_evidence_memory",
            "full_progressive_evidence_memory",
            "progressive_sufficiency_memory",
            "progressive_sufficiency_memory_heg",
            "progressive_sufficiency_memory_conservative_gate",
            "progressive_sufficiency_memory_microclip",
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
    def strict_gated_semantic_memory(self) -> bool:
        return self.mode == "strict_gated_semantic_memory"

    @property
    def question_aware_memory(self) -> bool:
        return self.mode == "question_aware_memory"

    @property
    def event_summary_memory(self) -> bool:
        return self.mode == "event_summary_memory"

    @property
    def budgeted_counterfactual_memory(self) -> bool:
        return self.mode == "budgeted_counterfactual_memory"

    @property
    def progressive_evidence_memory(self) -> bool:
        return self.mode == "progressive_evidence_memory"

    @property
    def full_progressive_evidence_memory(self) -> bool:
        return self.mode == "full_progressive_evidence_memory"

    @property
    def progressive_evidence_like(self) -> bool:
        return self.mode in {"progressive_evidence_memory", "full_progressive_evidence_memory"}

    @property
    def progressive_sufficiency_memory(self) -> bool:
        return self.mode == "progressive_sufficiency_memory"

    @property
    def progressive_sufficiency_memory_heg(self) -> bool:
        return self.mode == "progressive_sufficiency_memory_heg"

    @property
    def progressive_sufficiency_memory_conservative_gate(self) -> bool:
        return self.mode == "progressive_sufficiency_memory_conservative_gate"

    @property
    def progressive_sufficiency_memory_microclip(self) -> bool:
        return self.mode == "progressive_sufficiency_memory_microclip"

    @property
    def progressive_sufficiency_like(self) -> bool:
        return self.mode in {
            "progressive_sufficiency_memory",
            "progressive_sufficiency_memory_heg",
            "progressive_sufficiency_memory_conservative_gate",
            "progressive_sufficiency_memory_microclip",
        }

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
    downsample_mode: str | None = None


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


def _structural_change(left: np.ndarray, right: np.ndarray) -> float:
    """Return a cheap SSIM-style change score in [0, 1]."""
    left = left.astype(np.float32)
    right = right.astype(np.float32)
    c1 = 6.5025
    c2 = 58.5225
    mu_left = float(np.mean(left))
    mu_right = float(np.mean(right))
    var_left = float(np.var(left))
    var_right = float(np.var(right))
    cov = float(np.mean((left - mu_left) * (right - mu_right)))
    numerator = (2.0 * mu_left * mu_right + c1) * (2.0 * cov + c2)
    denominator = (mu_left * mu_left + mu_right * mu_right + c1) * (var_left + var_right + c2)
    if denominator <= 0:
        return 0.0
    ssim = max(0.0, min(1.0, numerator / denominator))
    return float(1.0 - ssim)


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


def _chunk_timestamp(chunk: Any) -> float:
    timestamps = getattr(chunk, "frame_timestamps", None) or []
    if timestamps:
        return float(timestamps[0])
    return float(getattr(chunk, "chunk_index", 0))


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


def _strict_gated_memory_activation(prompt: str, reason: str) -> tuple[bool, str]:
    """Stricter general gate for StreamingBench-style recent-state questions.

    The default path is pure Recent-6. Older semantic anchors are allowed only
    for explicit history/count prompts. Local/current/prospective questions stay
    on the recent visual state because the added memory frames tend to distract
    MiniCPM on StreamingBench.
    """

    text = _query_text_only(prompt)
    count_total = bool(
        re.search(
            r"\b(how many times|times in total|in total|total number)\b",
            text,
            re.IGNORECASE,
        )
    )
    explicit_history = bool(_STRICT_MEMORY_RE.search(text))
    current_or_local = bool(_STRICT_CURRENT_RE.search(text))
    recent_or_forward = bool(_STRICT_RECENT_RE.search(text))

    if count_total:
        return True, "strict_count_total_cue"
    if not explicit_history:
        return False, "strict_no_explicit_history_cue"
    if current_or_local:
        return False, "strict_current_or_local_guard"
    if recent_or_forward and not _EARLY_MEMORY_RE.search(text):
        return False, "strict_recent_or_forward_guard"
    if reason != "history_or_temporal":
        return False, "strict_classifier_guard"
    return True, "strict_explicit_history_cue"


def _event_summary_memory_activation(prompt: str, reason: str) -> tuple[bool, str]:
    """Conservative gate for Conditional Event Bookmark Memory.

    Current/local questions should remain pure Recent-6. Explicit history,
    count, or temporal-reference questions may scan the tiny bookmark memory,
    and weak temporal cases are allowed to scan but still need a strong
    bookmark relevance margin before any old frame is injected.
    """

    gate_enabled = os.environ.get("MINICPM_EVENT_SUMMARY_GATE", "1").strip().lower()
    if gate_enabled in {"0", "false", "no", "off"}:
        return True, "event_gate_disabled"

    text = _query_text_only(prompt)
    reference_terms = re.compile(
        r"\b(previous question|mentioned|same person|same object|that object|he|she|it|they)\b",
        re.IGNORECASE,
    )
    count_total = bool(_COUNT_MEMORY_RE.search(text))
    explicit_history = bool(_STRICT_MEMORY_RE.search(text))
    reference_query = bool(reference_terms.search(text))
    current_or_local = bool(_STRICT_CURRENT_RE.search(text))
    recent_or_forward = bool(_STRICT_RECENT_RE.search(text))

    if count_total:
        return True, "event_count_total_cue"
    if reference_query:
        return True, "event_reference_cue"
    if explicit_history and not current_or_local:
        return True, "event_explicit_history_cue"
    if explicit_history and current_or_local:
        return True, "event_history_with_current_terms"
    if current_or_local:
        return False, "event_current_state_guard"
    if recent_or_forward:
        return False, "event_recent_or_forward_guard"
    if reason == "history_or_temporal":
        return True, "event_uncertain_temporal_reason"
    return False, "event_no_history_cue"


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
    if config.strict_gated_semantic_memory:
        activated, gate_reason = _strict_gated_memory_activation(prompt, reason)
        return activated, {
            "enabled": True,
            "activated": bool(activated),
            "reason": gate_reason,
        }
    if config.question_aware_memory:
        activated, gate_reason = _gated_memory_activation(prompt, reason)
        return activated, {
            "enabled": True,
            "activated": bool(activated),
            "reason": f"question_aware_{gate_reason}",
        }
    if config.event_summary_memory:
        activated, gate_reason = _event_summary_memory_activation(prompt, reason)
        return activated, {
            "enabled": True,
            "activated": bool(activated),
            "reason": gate_reason,
        }
    if config.budgeted_counterfactual_memory:
        return True, {
            "enabled": True,
            "activated": True,
            "reason": "utility_gate_pending_until_recent_context",
        }
    if config.progressive_evidence_like:
        return True, {
            "enabled": True,
            "activated": True,
            "reason": "full_progressive_evidence_pending_until_model_score"
            if config.full_progressive_evidence_memory
            else "progressive_evidence_pending_until_recent_context",
        }
    if config.progressive_sufficiency_like:
        return True, {
            "enabled": True,
            "activated": True,
            "reason": (
                "progressive_sufficiency_heg_pending"
                if config.progressive_sufficiency_memory_heg
                else "progressive_sufficiency_microclip_pending"
                if config.progressive_sufficiency_memory_microclip
                else "progressive_sufficiency_pending"
            ),
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


def _l2_normalise(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return vector
    return vector / norm


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


def _question_aware_query_vector(
    semantic_query: dict[str, Any],
    query_flags: dict[str, bool],
    prompt: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a lightweight text-side vector for question-window matching.

    This is intentionally model-free: it mirrors WindowQuant's idea of
    text/window relevance, but keeps the experiment cheap enough to run inside
    the existing MiniCPM pipeline without loading a second encoder.
    """

    vector: list[float] = []
    color_set = set(semantic_query["colors"])
    for color in _COLOR_FEATURE_NAMES:
        vector.append(1.0 if color in color_set else 0.0)

    text_query = bool(semantic_query["text_terms"] or _TEXT_LOCALIZATION_RE.search(prompt))
    texture_query = bool(semantic_query["texture_terms"])
    object_query = bool(semantic_query["object_terms"] or _LOCALIZATION_RE.search(prompt))
    action_query = bool(_ACTION_RE.search(prompt) or query_flags["early_query"] or query_flags["late_query"])
    count_query = bool(query_flags["count_query"])

    early_weight = 1.0 if query_flags["early_query"] else 0.0
    recent_weight = 1.0 if query_flags["late_query"] else 0.0
    middle_weight = 1.0 if count_query else 0.0
    if not (early_weight or recent_weight or middle_weight):
        # A weak recent prior for underspecified prompts; recent-6 still carries
        # the main evidence, but this keeps older anchors from drifting too far.
        recent_weight = 0.35

    vector.extend(
        [
            1.0 if text_query else 0.0,
            1.0 if texture_query else 0.0,
            1.0 if object_query else 0.0,
            1.0 if action_query else 0.0,
            early_weight,
            middle_weight,
            recent_weight,
            1.0 if count_query else 0.0,
        ]
    )
    flags = {
        "text_query": bool(text_query),
        "texture_query": bool(texture_query),
        "object_query": bool(object_query),
        "action_query": bool(action_query),
        "count_query": bool(count_query),
        "early_query": bool(query_flags["early_query"]),
        "late_query": bool(query_flags["late_query"]),
    }
    return _l2_normalise(np.asarray(vector, dtype=np.float32)), flags


def _question_aware_visual_vector(entry: dict[str, Any]) -> np.ndarray:
    vector: list[float] = []
    color_features = entry["color_features"]
    for color in _COLOR_FEATURE_NAMES:
        vector.append(float(color_features.get(color, 0.0)))

    event_score = float(entry["event_change_norm"])
    contrast_score = float(entry["contrast_norm"])
    text_score = float(entry["text_detail_norm"])
    position = float(entry["temporal_position"])
    texture_score = 0.50 * contrast_score + 0.50 * text_score
    object_score = 0.35 * contrast_score + 0.35 * text_score + 0.30 * event_score
    early_score = 1.0 - position
    middle_score = max(0.0, 1.0 - abs(position - 0.5) * 2.0)
    recent_score = position
    count_coverage = max(event_score, middle_score)
    vector.extend(
        [
            text_score,
            texture_score,
            object_score,
            event_score,
            early_score,
            middle_score,
            recent_score,
            count_coverage,
        ]
    )
    return _l2_normalise(np.asarray(vector, dtype=np.float32))


def _cosine_score(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    return float(np.clip(np.dot(left, right), 0.0, 1.0))


def _select_question_aware_memory_chunks(
    older_chunks: list[Any],
    count: int,
    config: AdaptiveWindowConfig,
    prompt: str,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Select older anchors by question-to-window similarity.

    Inspired by WindowQuant's window relevance idea, but used here for frame
    retrieval rather than KV precision assignment. Recent-6 remains the main
    SimpleStream input; this selector only decides which older windows are
    worth adding when the gate asks for memory.
    """

    if count <= 0 or not older_chunks:
        return [], []

    bank = _build_online_memory_bank(older_chunks, config)
    online_scores, query_flags = _online_memory_base_scores(bank, prompt)
    online_scores_norm = _normalise(online_scores)
    semantic_query = _extract_bound_semantic_query(prompt)
    query_vector, question_flags = _question_aware_query_vector(semantic_query, query_flags, prompt)

    similarity_scores: list[float] = []
    temporal_scores: list[float] = []
    final_scores: list[float] = []
    visual_vectors: list[np.ndarray] = []
    for index, entry in enumerate(bank):
        visual_vector = _question_aware_visual_vector(entry)
        visual_vectors.append(visual_vector)
        similarity = _cosine_score(query_vector, visual_vector)
        temporal_score = _temporal_relevance_score(entry, query_flags)
        event_score = float(entry["event_change_norm"])
        online_score = float(online_scores_norm[index] if online_scores_norm else 0.0)

        if question_flags["count_query"]:
            score = 0.55 * similarity + 0.25 * temporal_score + 0.15 * event_score + 0.05 * online_score
        elif question_flags["action_query"]:
            score = 0.65 * similarity + 0.15 * event_score + 0.15 * temporal_score + 0.05 * online_score
        else:
            score = 0.75 * similarity + 0.15 * temporal_score + 0.10 * online_score

        similarity_scores.append(float(similarity))
        temporal_scores.append(float(temporal_score))
        final_scores.append(float(score))

    if count >= len(bank):
        selected_indices = set(range(len(bank)))
    else:
        selected: list[int] = []
        diversity_weight = 0.35 if question_flags["count_query"] else 0.25
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
                score = final_scores[index] + diversity_weight * diversity
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
                "question_aware_memory_score": float(final_scores[index]),
                "question_window_similarity": float(similarity_scores[index]),
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
                "question_aware_flags": question_flags,
            }
        )
    return [bank[index]["chunk"] for index in selected_order], metadata


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


def _event_summary_words(entry: dict[str, Any]) -> list[str]:
    words: list[str] = [str(entry.get("change_level", "change")), "visual", "event"]
    if float(entry["structural_change_norm"]) >= 0.75:
        words.append("large-change")
    elif float(entry["structural_change_norm"]) >= 0.45:
        words.append("medium-change")
    else:
        words.append("small-change")

    colors = entry.get("top_colors") or []
    words.extend(str(color) for color in colors[:2])
    if float(entry["text_detail_norm"]) >= 0.55:
        words.extend(["text", "detail"])
    if float(entry["contrast_norm"]) >= 0.55:
        words.append("object")
    motion_region = entry.get("motion_region")
    if motion_region:
        words.extend(["motion", str(motion_region)])
    return words


def _chunk_motion_region(current: np.ndarray, previous: np.ndarray | None) -> str:
    if previous is None or current.shape != previous.shape:
        return "unknown"
    diff = np.abs(current.astype(np.float32) - previous.astype(np.float32))
    if diff.size == 0:
        return "unknown"
    rows = np.array_split(diff, 3, axis=0)
    grid_scores: list[tuple[float, int, int]] = []
    for row_index, row in enumerate(rows):
        cols = np.array_split(row, 3, axis=1)
        for col_index, cell in enumerate(cols):
            grid_scores.append((float(np.mean(cell)), row_index, col_index))
    _score, row_index, col_index = max(grid_scores, key=lambda item: (item[0], -item[1], -item[2]))
    vertical = ["top", "center", "bottom"][row_index]
    horizontal = ["left", "center", "right"][col_index]
    if vertical == "center" and horizontal == "center":
        return "center"
    if vertical == "center":
        return horizontal
    if horizontal == "center":
        return vertical
    return f"{vertical}-{horizontal}"


def _event_change_level(score: float) -> str:
    if score >= 0.75:
        return "large"
    if score >= 0.45:
        return "medium"
    return "small"


def _color_feature_change(left: dict[str, float], right: dict[str, float] | None) -> float:
    if not right:
        return 0.0
    keys = sorted(set(left) | set(right))
    return float(sum(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys) / max(1, len(keys)))


def _build_event_summary_bank(older_chunks: list[Any], config: AdaptiveWindowConfig) -> list[dict[str, Any]]:
    if not older_chunks:
        return []
    signatures = [_chunk_signature(chunk, config.dedup_resize) for chunk in older_chunks]
    structural_changes = [0.0]
    abs_changes = [0.0]
    edge_changes = [0.0]
    color_changes = [0.0]
    contrast_scores: list[float] = []
    text_detail_scores: list[float] = []
    color_features: list[dict[str, float]] = []
    motion_regions: list[str] = []
    for index, (chunk, signature) in enumerate(zip(older_chunks, signatures)):
        colors = _rgb_color_features(_chunk_rgb_sample(chunk, config.dedup_resize))
        if index > 0:
            structural_changes.append(_structural_change(signature, signatures[index - 1]))
            abs_changes.append(_mean_abs_diff(signature, signatures[index - 1]))
            edge_changes.append(
                abs(
                    _score_crop(signature, text_query=True)
                    - _score_crop(signatures[index - 1], text_query=True)
                )
            )
            color_changes.append(_color_feature_change(colors, color_features[index - 1]))
            motion_regions.append(_chunk_motion_region(signature, signatures[index - 1]))
        else:
            motion_regions.append("unknown")
        contrast_scores.append(float(np.std(signature)))
        text_detail_scores.append(_score_crop(signature, text_query=True))
        color_features.append(colors)

    structural_norm = _normalise(structural_changes)
    edge_change_norm = _normalise(edge_changes)
    color_change_norm = _normalise(color_changes)
    contrast_norm = _normalise(contrast_scores)
    text_norm = _normalise(text_detail_scores)
    bank: list[dict[str, Any]] = []
    for index, chunk in enumerate(older_chunks):
        colors = sorted(color_features[index].items(), key=lambda item: (-float(item[1]), item[0]))
        top_colors = [name for name, value in colors if float(value) > 0.18][:3]
        temporal_position = index / max(1, len(older_chunks) - 1)
        importance = (
            0.45 * float(structural_norm[index])
            + 0.20 * float(edge_change_norm[index])
            + 0.15 * float(text_norm[index])
            + 0.10 * float(color_change_norm[index])
            + 0.10 * (1.0 - abs(temporal_position - 0.5) * 2.0)
        )
        entry = {
            "index": index,
            "chunk": chunk,
            "chunk_id": int(chunk.chunk_index),
            "timestamp": _chunk_timestamp(chunk),
            "temporal_position": float(temporal_position),
            "structural_change": float(structural_changes[index]),
            "structural_change_norm": float(structural_norm[index]),
            "abs_change_score": float(abs_changes[index]),
            "edge_change_norm": float(edge_change_norm[index]),
            "color_change_norm": float(color_change_norm[index]),
            "contrast_norm": float(contrast_norm[index]),
            "text_detail_norm": float(text_norm[index]),
            "motion_region": motion_regions[index],
            "top_colors": top_colors,
            "event_importance": float(importance),
            "change_level": _event_change_level(float(structural_norm[index])),
        }
        summary_words = _event_summary_words(entry)
        entry["summary_words"] = summary_words
        entry["summary"] = (
            f"t={entry['timestamp']:.1f}s | {entry['change_level']} visual change"
            f" | colors={','.join(top_colors) if top_colors else 'none'}"
            f" | motion={entry['motion_region']}"
            f" | text_likelihood={entry['text_detail_norm']:.2f}"
        )
        bank.append(entry)
    return bank


def _temporally_far_enough(entry: dict[str, Any], selected: list[dict[str, Any]], min_gap_seconds: float) -> bool:
    timestamp = float(entry["timestamp"])
    return all(abs(timestamp - float(other["timestamp"])) >= min_gap_seconds for other in selected)


def _select_diverse_event_bookmarks(bank: list[dict[str, Any]], max_items: int, threshold: float) -> list[dict[str, Any]]:
    if max_items <= 0:
        return []
    min_gap_seconds = float(os.environ.get("MINICPM_EVENT_SUMMARY_MIN_GAP_SECONDS", "3.0"))
    eligible = [entry for entry in bank if float(entry["event_importance"]) >= threshold]
    if not eligible:
        eligible = sorted(bank, key=lambda entry: (-float(entry["event_importance"]), int(entry["chunk_id"])))[:max_items]

    selected: list[dict[str, Any]] = []
    slots = [
        ("early", lambda e: float(e["temporal_position"]) <= 0.34),
        ("middle", lambda e: 0.34 < float(e["temporal_position"]) < 0.67),
        ("late", lambda e: float(e["temporal_position"]) >= 0.67),
    ]
    for _name, predicate in slots:
        candidates = [entry for entry in eligible if predicate(entry)]
        candidates.sort(key=lambda entry: (-float(entry["event_importance"]), int(entry["chunk_id"])))
        for candidate in candidates:
            if _temporally_far_enough(candidate, selected, min_gap_seconds):
                selected.append(candidate)
                break
        if len(selected) >= max_items:
            break

    for candidate in sorted(eligible, key=lambda entry: (-float(entry["event_importance"]), int(entry["chunk_id"]))):
        if len(selected) >= max_items:
            break
        if int(candidate["chunk_id"]) in {int(entry["chunk_id"]) for entry in selected}:
            continue
        if _temporally_far_enough(candidate, selected, min_gap_seconds):
            selected.append(candidate)

    return sorted(selected[:max_items], key=lambda entry: int(entry["chunk_id"]))


def _select_event_summary_memory_chunks(
    older_chunks: list[Any],
    count: int,
    config: AdaptiveWindowConfig,
    prompt: str,
) -> tuple[list[Any], list[dict[str, Any]]]:
    if count <= 0 or not older_chunks:
        return [], []

    start_time = time.perf_counter()
    bank = _build_event_summary_bank(older_chunks, config)
    event_scan_latency_ms = (time.perf_counter() - start_time) * 1000.0
    if not bank:
        return [], []

    max_items = int(os.environ.get("MINICPM_EVENT_SUMMARY_MAX_ITEMS", "5"))
    importance_threshold = float(os.environ.get("MINICPM_EVENT_SUMMARY_IMPORTANCE_THRESHOLD", "0.45"))
    query_threshold = float(os.environ.get("MINICPM_EVENT_SUMMARY_QUERY_THRESHOLD", "0.50"))
    query_margin = float(os.environ.get("MINICPM_EVENT_SUMMARY_QUERY_MARGIN", "0.08"))
    max_retrieved = max(1, min(int(os.environ.get("MINICPM_EVENT_SUMMARY_MAX_RETRIEVED", "1")), count))
    memory_entries = _select_diverse_event_bookmarks(bank, max_items, importance_threshold)

    semantic_query = _extract_semantic_query(prompt)
    query_terms = set(semantic_query["terms"])
    query_colors = set(semantic_query["colors"])
    query_text = _query_text_only(prompt)
    history_query = bool(_STRICT_MEMORY_RE.search(query_text) or _COUNT_MEMORY_RE.search(query_text))
    early_query = bool(_EARLY_MEMORY_RE.search(query_text))
    late_query = bool(_LATE_MEMORY_RE.search(query_text))
    text_query = bool(_TEXT_LOCALIZATION_RE.search(query_text))
    retrieval_start = time.perf_counter()
    metadata: list[dict[str, Any]] = []
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for entry in memory_entries:
        summary_terms = {_singularise(word.lower()) for word in entry["summary_words"]}
        summary_colors = set(entry.get("top_colors") or [])
        term_overlap = len(query_terms & summary_terms) / max(1, len(query_terms))
        color_overlap = 1.0 if query_colors and (query_colors & summary_colors) else 0.0
        cue_match = color_overlap
        if text_query:
            cue_match = max(cue_match, float(entry["text_detail_norm"]))
        temporal_match = 0.5
        if early_query:
            temporal_match = 1.0 - float(entry["temporal_position"])
        elif late_query:
            temporal_match = float(entry["temporal_position"])
        elif _COUNT_MEMORY_RE.search(query_text):
            temporal_match = 1.0 - abs(float(entry["temporal_position"]) - 0.5) * 2.0
        score = (
            0.35 * term_overlap
            + 0.20 * cue_match
            + 0.15 * (1.0 if history_query else 0.35)
            + 0.20 * float(entry["event_importance"])
            + 0.10 * temporal_match
        )
        scored.append((score, int(entry["chunk_id"]), entry))

    scored.sort(key=lambda item: (-item[0], item[1]))
    best_score = float(scored[0][0]) if scored else 0.0
    second_score = float(scored[1][0]) if len(scored) > 1 else 0.0
    margin = best_score - second_score
    selected = scored[:max_retrieved] if best_score >= query_threshold and margin >= query_margin else []
    selected_ids = {int(entry["chunk_id"]) for _score, _chunk_id, entry in selected}
    bookmark_retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

    for score, _chunk_id, entry in scored:
        metadata.append(
            {
                "chunk_id": int(entry["chunk_id"]),
                "selected": int(entry["chunk_id"]) in selected_ids,
                "event_summary_score": float(score),
                "event_summary_best_score": best_score,
                "event_summary_second_score": second_score,
                "event_summary_margin": margin,
                "summary": entry["summary"],
                "timestamp": float(entry["timestamp"]),
                "temporal_position": float(entry["temporal_position"]),
                "structural_change": float(entry["structural_change"]),
                "structural_change_norm": float(entry["structural_change_norm"]),
                "edge_change_norm": float(entry["edge_change_norm"]),
                "color_change_norm": float(entry["color_change_norm"]),
                "contrast_norm": float(entry["contrast_norm"]),
                "text_detail_norm": float(entry["text_detail_norm"]),
                "motion_region": entry.get("motion_region"),
                "top_colors": entry.get("top_colors") or [],
                "event_importance": float(entry["event_importance"]),
                "change_level": entry.get("change_level"),
                "memory_size": len(memory_entries),
                "event_scan_latency_ms": event_scan_latency_ms,
                "bookmark_retrieval_ms": bookmark_retrieval_ms,
                "query_terms": sorted(query_terms),
                "query_colors": sorted(query_colors),
                "query_threshold": query_threshold,
                "query_margin_threshold": query_margin,
                "history_query": history_query,
                "early_query": early_query,
                "late_query": late_query,
                "text_query": text_query,
            }
        )
    selected_chunks = [entry["chunk"] for _score, _chunk_id, entry in selected]
    return selected_chunks, metadata


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


def _recent_evidence_coverage_score(
    recent_chunks: list[Any],
    prompt: str,
    config: AdaptiveWindowConfig,
) -> tuple[float, dict[str, Any]]:
    if not recent_chunks:
        return 0.0, {"reason": "no_recent_chunks"}

    bank = _build_online_memory_bank(recent_chunks, config)
    query_flags = {
        "count_query": bool(_COUNT_MEMORY_RE.search(prompt)),
        "early_query": bool(_EARLY_MEMORY_RE.search(prompt)),
        "late_query": bool(_LATE_MEMORY_RE.search(prompt)),
        "text_query": bool(_TEXT_LOCALIZATION_RE.search(prompt)),
    }
    semantic_query = _extract_bound_semantic_query(prompt)
    query_vector, question_flags = _question_aware_query_vector(semantic_query, query_flags, prompt)
    semantic_query_simple = _extract_semantic_query(prompt)

    similarity_scores: list[float] = []
    semantic_scores: list[float] = []
    for entry in bank:
        similarity_scores.append(_cosine_score(query_vector, _question_aware_visual_vector(entry)))
        semantic_scores.append(_semantic_proxy_score(entry, semantic_query_simple))

    best_similarity = max(similarity_scores) if similarity_scores else 0.0
    best_semantic = max(semantic_scores) if semantic_scores else 0.0
    # Recent evidence is considered covered if either the question-window
    # vector or the direct semantic proxy finds strong support in Recent-6.
    coverage = max(best_similarity, best_semantic)
    return float(coverage), {
        "best_recent_similarity": float(best_similarity),
        "best_recent_semantic_score": float(best_semantic),
        "semantic_query": semantic_query_simple,
        "question_aware_flags": question_flags,
    }


def _temporal_nonlocality_score(prompt: str, reason: str) -> tuple[float, dict[str, Any]]:
    text = _query_text_only(prompt)
    explicit_history = bool(_STRICT_MEMORY_RE.search(text) or _EARLY_MEMORY_RE.search(text))
    count_query = bool(_COUNT_MEMORY_RE.search(text))
    reference_query = bool(
        re.search(
            r"\b(previous question|mentioned|same person|same object|that person|that object|earlier person)\b",
            text,
            re.IGNORECASE,
        )
    )
    current_guard = bool(_STRICT_CURRENT_RE.search(text))
    prospective_guard = bool(_GATED_PROSPECTIVE_GUARD_RE.search(text))

    score = 0.0
    if explicit_history:
        score += 0.55
    if count_query:
        score += 0.40
    if reference_query:
        score += 0.45
    if reason == "history_or_temporal":
        score += 0.20
    if prospective_guard and not explicit_history and not count_query:
        score -= 0.35
    if current_guard and not explicit_history and not count_query and not reference_query:
        score -= 0.45
    return float(max(0.0, min(1.0, score))), {
        "explicit_history": explicit_history,
        "count_query": count_query,
        "reference_query": reference_query,
        "current_guard": current_guard,
        "prospective_guard": prospective_guard,
        "window_reason": reason,
    }


def _select_budgeted_counterfactual_memory_chunks(
    older_chunks: list[Any],
    recent_chunks: list[Any],
    max_count: int,
    config: AdaptiveWindowConfig,
    prompt: str,
    reason: str,
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    """Utility-gated memory routing for mobile streaming QA.

    The selector estimates whether historical frames are likely to change the
    answer enough to justify their visual-token cost. It does not blindly turn
    on memory for every temporal-looking prompt; it uses recent coverage and
    best memory relevance before assigning K=0/1/2.
    """

    max_count = max(0, int(max_count))
    if max_count <= 0 or not older_chunks:
        return [], [], {
            "enabled": True,
            "activated": False,
            "reason": "no_budget_or_no_history",
            "selected_budget": 0,
        }

    search_chunks = int(config.memory_search_chunks)
    if search_chunks > 0:
        older_chunks = older_chunks[-search_chunks:]

    bank = _build_online_memory_bank(older_chunks, config)
    online_scores, query_flags = _online_memory_base_scores(bank, prompt)
    online_norm = _normalise(online_scores)
    semantic_query = _extract_semantic_query(prompt)
    temporal_need, temporal_meta = _temporal_nonlocality_score(prompt, reason)
    recent_coverage, recent_meta = _recent_evidence_coverage_score(recent_chunks, prompt, config)

    utility_threshold = float(os.environ.get("MINICPM_BCM_UTILITY_THRESHOLD", "0.58"))
    high_utility_threshold = float(os.environ.get("MINICPM_BCM_HIGH_UTILITY_THRESHOLD", "0.82"))
    memory_relevance_threshold = float(os.environ.get("MINICPM_BCM_MEMORY_RELEVANCE_THRESHOLD", "0.46"))
    uncertainty_weight = float(os.environ.get("MINICPM_BCM_UNCERTAINTY_WEIGHT", "0.45"))
    temporal_weight = float(os.environ.get("MINICPM_BCM_TEMPORAL_WEIGHT", "0.55"))
    memory_weight = float(os.environ.get("MINICPM_BCM_MEMORY_WEIGHT", "0.70"))
    cost_weight = float(os.environ.get("MINICPM_BCM_COST_WEIGHT", "0.25"))
    max_budget = min(max_count, int(os.environ.get("MINICPM_BCM_MAX_RETRIEVED", str(max_count))))
    max_budget = max(0, max_budget)
    retrieval_cost = max_budget / max(1.0, float(config.mid_window + max_budget))

    candidate_scores: list[float] = []
    temporal_scores: list[float] = []
    semantic_scores: list[float] = []
    for index, entry in enumerate(bank):
        semantic_score = _semantic_proxy_score(entry, semantic_query)
        temporal_score = _temporal_relevance_score(entry, query_flags)
        event_score = float(entry["event_change_norm"])
        detail_score = 0.5 * float(entry["contrast_norm"]) + 0.5 * float(entry["text_detail_norm"])
        online_score = float(online_norm[index] if online_norm else 0.0)
        score = (
            0.45 * semantic_score
            + 0.20 * temporal_score
            + 0.15 * event_score
            + 0.10 * detail_score
            + 0.10 * online_score
        )
        candidate_scores.append(float(score))
        temporal_scores.append(float(temporal_score))
        semantic_scores.append(float(semantic_score))

    best_memory_relevance = max(candidate_scores) if candidate_scores else 0.0
    recent_uncertainty_proxy = max(0.0, 1.0 - recent_coverage)
    utility_score = (
        temporal_weight * temporal_need
        + uncertainty_weight * recent_uncertainty_proxy
        + memory_weight * best_memory_relevance
        - cost_weight * retrieval_cost
    )

    if temporal_need <= 0.0:
        selected_budget = 0
        gate_reason = "no_temporal_need"
    elif best_memory_relevance < memory_relevance_threshold:
        selected_budget = 0
        gate_reason = "memory_relevance_below_threshold"
    elif utility_score < utility_threshold:
        selected_budget = 0
        gate_reason = "utility_below_threshold"
    elif utility_score >= high_utility_threshold and max_budget >= 2 and temporal_need >= 0.65:
        selected_budget = 2
        gate_reason = "high_utility_two_frames"
    else:
        selected_budget = min(1, max_budget)
        gate_reason = "utility_one_frame"

    selected_indices: list[int] = []
    if selected_budget > 0:
        diversity_weight = 0.30 if query_flags.get("count_query") else 0.18
        while len(selected_indices) < selected_budget:
            best_index: int | None = None
            best_score: float | None = None
            for index in range(len(bank)):
                if index in selected_indices:
                    continue
                if selected_indices:
                    denom = max(1, len(bank) - 1)
                    diversity = min(abs(index - chosen) / denom for chosen in selected_indices)
                else:
                    diversity = 1.0
                score = candidate_scores[index] + diversity_weight * diversity
                if best_score is None or score > best_score or (
                    score == best_score and best_index is not None and index < best_index
                ):
                    best_index = index
                    best_score = score
            if best_index is None:
                break
            selected_indices.append(best_index)

    selected_set = set(selected_indices)
    gate = {
        "enabled": True,
        "activated": bool(selected_indices),
        "reason": gate_reason,
        "selected_budget": len(selected_indices),
        "max_budget": max_budget,
        "temporal_need": float(temporal_need),
        "recent_coverage": float(recent_coverage),
        "recent_uncertainty_proxy": float(recent_uncertainty_proxy),
        "memory_relevance": float(best_memory_relevance),
        "retrieval_cost": float(retrieval_cost),
        "utility_score": float(utility_score),
        "utility_threshold": utility_threshold,
        "high_utility_threshold": high_utility_threshold,
        "memory_relevance_threshold": memory_relevance_threshold,
        "weights": {
            "temporal": temporal_weight,
            "uncertainty": uncertainty_weight,
            "memory": memory_weight,
            "cost": cost_weight,
        },
        "temporal_meta": temporal_meta,
        "recent_meta": recent_meta,
    }

    metadata = []
    for index, entry in enumerate(bank):
        metadata.append(
            {
                "chunk_id": int(entry["chunk_id"]),
                "selected": index in selected_set,
                "budgeted_counterfactual_score": float(candidate_scores[index]),
                "semantic_proxy_score": float(semantic_scores[index]),
                "temporal_relevance_score": float(temporal_scores[index]),
                "online_memory_score": float(online_scores[index]),
                "event_change_score": float(entry["event_change_score"]),
                "event_change_norm": float(entry["event_change_norm"]),
                "contrast_norm": float(entry["contrast_norm"]),
                "text_detail_norm": float(entry["text_detail_norm"]),
                "temporal_position": float(entry["temporal_position"]),
                "semantic_query": semantic_query,
                "query_flags": query_flags,
                "utility_gate": gate,
            }
        )

    selected_order = sorted(selected_indices)
    return [bank[index]["chunk"] for index in selected_order], metadata, gate


def _select_progressive_evidence_memory_chunks(
    older_chunks: list[Any],
    recent_chunks: list[Any],
    max_count: int,
    config: AdaptiveWindowConfig,
    prompt: str,
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    """Demand-driven semantic memory acquisition.

    This is intentionally not a hard-coded recent/history classifier. Recent-6
    is treated as the cheapest evidence context. Historical frames are admitted
    only when the recent evidence coverage is low and each retrieved candidate
    gives enough additional question-grounded support.
    """

    max_count = max(0, int(max_count))
    if max_count <= 0 or not older_chunks:
        return [], [], {
            "enabled": True,
            "activated": False,
            "reason": "no_budget_or_no_history",
            "selected_budget": 0,
        }

    search_chunks = int(config.memory_search_chunks)
    if search_chunks > 0:
        older_chunks = older_chunks[-search_chunks:]

    bank = _build_online_memory_bank(older_chunks, config)
    online_scores, query_flags = _online_memory_base_scores(bank, prompt)
    semantic_query = _extract_semantic_query(prompt)
    semantic_scores = [_semantic_proxy_score(entry, semantic_query) for entry in bank]
    has_semantic_signal = bool(
        semantic_query["colors"]
        or semantic_query["text_terms"]
        or semantic_query["texture_terms"]
        or semantic_query["object_terms"]
    )
    recent_coverage, recent_meta = _recent_evidence_coverage_score(recent_chunks, prompt, config)

    sufficiency_threshold = float(os.environ.get("MINICPM_PEM_SUFFICIENCY_THRESHOLD", "0.62"))
    low_evidence_threshold = float(os.environ.get("MINICPM_PEM_LOW_EVIDENCE_THRESHOLD", "0.42"))
    memory_relevance_threshold = float(os.environ.get("MINICPM_PEM_MEMORY_RELEVANCE_THRESHOLD", "0.46"))
    marginal_gain_threshold = float(os.environ.get("MINICPM_PEM_MARGINAL_GAIN_THRESHOLD", "0.06"))
    diversity_weight = float(os.environ.get("MINICPM_PEM_DIVERSITY_WEIGHT", "0.18"))
    max_budget = min(max_count, int(os.environ.get("MINICPM_PEM_MAX_RETRIEVED", str(max_count))))
    max_budget = max(0, max_budget)

    evidence_scores: list[float] = []
    for index, entry in enumerate(bank):
        event_score = float(entry["event_change_norm"])
        contrast_score = float(entry["contrast_norm"])
        recency_score = float(entry["temporal_position"])
        if has_semantic_signal:
            score = (
                0.62 * semantic_scores[index]
                + 0.16 * event_score
                + 0.12 * contrast_score
                + 0.10 * recency_score
            )
        else:
            score = online_scores[index]
        evidence_scores.append(float(score))

    selected_indices: list[int] = []
    current_support = float(recent_coverage)
    stop_reason = "recent_evidence_sufficient"
    acquisition_trace: list[dict[str, Any]] = [
        {
            "step": 0,
            "context": "recent",
            "evidence_support": current_support,
            "selected_chunk_id": None,
            "marginal_gain": 0.0,
            "decision": "stop" if current_support >= sufficiency_threshold else "expand",
        }
    ]

    if current_support < sufficiency_threshold and max_budget > 0:
        stop_reason = "budget_exhausted"
        while len(selected_indices) < max_budget:
            best_index: int | None = None
            best_score: float | None = None
            for index in range(len(bank)):
                if index in selected_indices:
                    continue
                if selected_indices:
                    denom = max(1, len(bank) - 1)
                    diversity = min(abs(index - chosen) / denom for chosen in selected_indices)
                else:
                    diversity = 1.0
                score = evidence_scores[index] + diversity_weight * diversity
                if best_score is None or score > best_score or (
                    score == best_score and best_index is not None and index < best_index
                ):
                    best_index = index
                    best_score = score

            if best_index is None:
                stop_reason = "no_candidate"
                break

            candidate_support = float(evidence_scores[best_index])
            marginal_gain = max(0.0, candidate_support - current_support)
            if candidate_support < memory_relevance_threshold:
                stop_reason = "candidate_relevance_below_threshold"
                acquisition_trace.append(
                    {
                        "step": len(selected_indices) + 1,
                        "context": "candidate_rejected",
                        "selected_chunk_id": int(bank[best_index]["chunk_id"]),
                        "candidate_support": candidate_support,
                        "evidence_support": current_support,
                        "marginal_gain": float(marginal_gain),
                        "decision": stop_reason,
                    }
                )
                break
            if current_support >= low_evidence_threshold and marginal_gain < marginal_gain_threshold:
                stop_reason = "marginal_gain_too_small"
                acquisition_trace.append(
                    {
                        "step": len(selected_indices) + 1,
                        "context": "candidate_rejected",
                        "selected_chunk_id": int(bank[best_index]["chunk_id"]),
                        "candidate_support": candidate_support,
                        "evidence_support": current_support,
                        "marginal_gain": float(marginal_gain),
                        "decision": stop_reason,
                    }
                )
                break

            selected_indices.append(best_index)
            current_support = max(current_support, candidate_support)
            if current_support >= sufficiency_threshold:
                stop_reason = "evidence_sufficient_after_retrieval"
            acquisition_trace.append(
                {
                    "step": len(selected_indices),
                    "context": "recent_plus_memory",
                    "selected_chunk_id": int(bank[best_index]["chunk_id"]),
                    "candidate_support": candidate_support,
                    "evidence_support": float(current_support),
                    "marginal_gain": float(marginal_gain),
                    "decision": "stop" if current_support >= sufficiency_threshold else "expand",
                }
            )
            if current_support >= sufficiency_threshold:
                break

    selected_set = set(selected_indices)
    gate = {
        "enabled": True,
        "activated": bool(selected_indices),
        "reason": stop_reason,
        "selected_budget": len(selected_indices),
        "max_budget": max_budget,
        "recent_coverage": float(recent_coverage),
        "final_evidence_support": float(current_support),
        "sufficiency_threshold": sufficiency_threshold,
        "low_evidence_threshold": low_evidence_threshold,
        "memory_relevance_threshold": memory_relevance_threshold,
        "marginal_gain_threshold": marginal_gain_threshold,
        "has_semantic_signal": has_semantic_signal,
        "semantic_query": semantic_query,
        "recent_meta": recent_meta,
        "acquisition_trace": acquisition_trace,
    }

    metadata = []
    for index, entry in enumerate(bank):
        metadata.append(
            {
                "chunk_id": int(entry["chunk_id"]),
                "selected": index in selected_set,
                "progressive_evidence_score": float(evidence_scores[index]),
                "semantic_proxy_score": float(semantic_scores[index]),
                "online_memory_score": float(online_scores[index]),
                "event_change_score": float(entry["event_change_score"]),
                "event_change_norm": float(entry["event_change_norm"]),
                "contrast_norm": float(entry["contrast_norm"]),
                "text_detail_norm": float(entry["text_detail_norm"]),
                "temporal_position": float(entry["temporal_position"]),
                "semantic_query": semantic_query,
                "query_flags": query_flags,
                "evidence_gate": gate,
            }
        )

    selected_order = sorted(selected_indices)
    return [bank[index]["chunk"] for index in selected_order], metadata, gate


def _extract_mcq_options(prompt: str) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in _MCQ_OPTION_RE.finditer(prompt):
        letter = match.group(1).upper()
        if letter in seen:
            continue
        text = match.group(2).strip()
        if text:
            options.append({"letter": letter, "text": text})
            seen.add(letter)
    return options


def _option_token_ids(tokenizer: Any, letter: str) -> list[int]:
    token_ids: list[int] = []
    for variant in (letter, f" {letter}", f"{letter}.", f" {letter}.", f"({letter}", f" ({letter}"):
        try:
            ids = tokenizer.encode(variant, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(variant)
        if ids:
            token_id = int(ids[0])
            if token_id not in token_ids:
                token_ids.append(token_id)
    return token_ids


@torch.inference_mode()
def _score_mcq_options_from_frames(
    qa: RecentWindowQAModel,
    frames: list[Image.Image],
    prompt: str,
    options: list[dict[str, str]],
) -> dict[str, Any]:
    """Cheap MiniCPM confidence pass for A/B/C/D without generating a full answer."""

    if not options:
        return {
            "available": False,
            "reason": "no_mcq_options",
            "predicted_letter": None,
            "predicted_text": "",
            "confidence": 0.0,
            "margin": 0.0,
            "entropy": 1.0,
            "option_probs": {},
        }

    tokenizer = getattr(qa.processor, "tokenizer", None)
    if tokenizer is None:
        return {
            "available": False,
            "reason": "missing_tokenizer",
            "predicted_letter": None,
            "predicted_text": "",
            "confidence": 0.0,
            "margin": 0.0,
            "entropy": 1.0,
            "option_probs": {},
        }

    content = [{"type": "image", "image": frame} for frame in frames]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    template_kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    processor_kwargs: dict[str, Any] = {
        "downsample_mode": getattr(qa, "downsample_mode", None),
        "max_slice_nums": getattr(qa, "max_slice_nums", 1),
        "use_image_id": False,
    }
    processor_kwargs = {key: value for key, value in processor_kwargs.items() if value is not None}
    score_t0 = time.perf_counter()
    try:
        inputs = qa.processor.apply_chat_template(
            messages,
            **template_kwargs,
            processor_kwargs=processor_kwargs,
        )
    except TypeError:
        inputs = qa.processor.apply_chat_template(
            messages,
            **template_kwargs,
            **processor_kwargs,
        )
    inputs = inputs.to(qa.model.device)
    preprocess_seconds = time.perf_counter() - score_t0

    forward_t0 = time.perf_counter()
    outputs = qa.model(**inputs)
    _synchronize_gpu_devices()
    forward_seconds = time.perf_counter() - forward_t0

    logits = outputs.logits[:, -1, :].float().squeeze(0)
    option_logits: list[torch.Tensor] = []
    option_letters: list[str] = []
    option_text_by_letter = {item["letter"]: item["text"] for item in options}
    token_ids_by_letter: dict[str, list[int]] = {}
    for item in options:
        letter = item["letter"]
        ids = _option_token_ids(tokenizer, letter)
        token_ids_by_letter[letter] = ids
        if ids:
            ids_tensor = torch.tensor(ids, device=logits.device, dtype=torch.long)
            option_logits.append(torch.max(logits.index_select(0, ids_tensor)))
            option_letters.append(letter)

    if not option_logits:
        return {
            "available": False,
            "reason": "no_option_token_ids",
            "predicted_letter": None,
            "predicted_text": "",
            "confidence": 0.0,
            "margin": 0.0,
            "entropy": 1.0,
            "option_probs": {},
            "option_token_ids": token_ids_by_letter,
            "score_preprocess_ms": preprocess_seconds * 1000.0,
            "score_forward_ms": forward_seconds * 1000.0,
        }

    stacked = torch.stack(option_logits)
    probs_tensor = torch.softmax(stacked, dim=0)
    probs = {letter: float(probs_tensor[index].item()) for index, letter in enumerate(option_letters)}
    sorted_probs = sorted(probs.items(), key=lambda item: (-item[1], item[0]))
    predicted_letter = sorted_probs[0][0]
    top_prob = sorted_probs[0][1]
    second_prob = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
    margin = max(0.0, top_prob - second_prob)
    entropy = 0.0
    for prob in probs.values():
        if prob > 0.0:
            entropy -= prob * float(np.log(prob))
    entropy_norm = entropy / max(float(np.log(max(2, len(probs)))), 1e-6)
    confidence = 0.65 * margin + 0.35 * (1.0 - entropy_norm)
    return {
        "available": True,
        "predicted_letter": predicted_letter,
        "predicted_text": option_text_by_letter.get(predicted_letter, ""),
        "confidence": float(confidence),
        "margin": float(margin),
        "entropy": float(entropy_norm),
        "top_prob": float(top_prob),
        "second_prob": float(second_prob),
        "option_probs": probs,
        "option_token_ids": token_ids_by_letter,
        "score_preprocess_ms": preprocess_seconds * 1000.0,
        "score_forward_ms": forward_seconds * 1000.0,
    }


def _rank_full_progressive_candidates(
    older_chunks: list[Any],
    recent_chunks: list[Any],
    config: AdaptiveWindowConfig,
    prompt: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    search_chunks = int(config.memory_search_chunks)
    if search_chunks > 0:
        older_chunks = older_chunks[-search_chunks:]
    if not older_chunks:
        return [], {"semantic_query": _extract_semantic_query(prompt), "recent_meta": {}}

    bank = _build_online_memory_bank(older_chunks, config)
    online_scores, query_flags = _online_memory_base_scores(bank, prompt)
    semantic_query = _extract_semantic_query(prompt)
    semantic_scores = [_semantic_proxy_score(entry, semantic_query) for entry in bank]
    recent_support, recent_meta = _recent_evidence_coverage_score(recent_chunks, prompt, config)
    candidates: list[dict[str, Any]] = []
    for index, entry in enumerate(bank):
        score = (
            0.60 * float(semantic_scores[index])
            + 0.15 * float(entry["event_change_norm"])
            + 0.15 * float(entry["contrast_norm"])
            + 0.10 * float(entry["temporal_position"])
        )
        candidates.append(
            {
                "bank_index": index,
                "chunk": entry["chunk"],
                "chunk_id": int(entry["chunk_id"]),
                "full_progressive_candidate_score": float(score),
                "semantic_proxy_score": float(semantic_scores[index]),
                "online_memory_score": float(online_scores[index]),
                "event_change_score": float(entry["event_change_score"]),
                "event_change_norm": float(entry["event_change_norm"]),
                "contrast_norm": float(entry["contrast_norm"]),
                "text_detail_norm": float(entry["text_detail_norm"]),
                "temporal_position": float(entry["temporal_position"]),
                "semantic_query": semantic_query,
                "query_flags": query_flags,
            }
        )
    candidates.sort(key=lambda item: (-float(item["full_progressive_candidate_score"]), int(item["chunk_id"])))
    return candidates, {
        "semantic_query": semantic_query,
        "query_flags": query_flags,
        "recent_proxy_support": float(recent_support),
        "recent_meta": recent_meta,
    }


def _full_progressive_evidence_selection(
    qa: RecentWindowQAModel,
    chunks: list[Any],
    prompt: str,
    config: AdaptiveWindowConfig,
) -> AdaptiveSelection:
    recent_window_size = config.mid_window
    recent_chunks = chunks[-recent_window_size:]
    older_chunks = chunks[: max(0, len(chunks) - recent_window_size)]
    options = _extract_mcq_options(prompt)
    candidates, rank_meta = _rank_full_progressive_candidates(older_chunks, recent_chunks, config, prompt)

    max_budget = min(
        max(0, int(config.memory_anchors)),
        max(0, int(os.environ.get("MINICPM_FPEM_MAX_RETRIEVED", str(config.memory_anchors)))),
    )
    sufficiency_threshold = float(os.environ.get("MINICPM_FPEM_SUFFICIENCY_THRESHOLD", "0.55"))
    marginal_gain_threshold = float(os.environ.get("MINICPM_FPEM_MARGINAL_GAIN_THRESHOLD", "0.025"))
    confidence_weight = float(os.environ.get("MINICPM_FPEM_CONFIDENCE_WEIGHT", "0.55"))
    evidence_weight = float(os.environ.get("MINICPM_FPEM_EVIDENCE_WEIGHT", "0.45"))
    relevance_threshold = float(os.environ.get("MINICPM_FPEM_MEMORY_RELEVANCE_THRESHOLD", "0.35"))

    selected_memory_chunks: list[Any] = []
    selected_candidate_ids: set[int] = set()
    trace: list[dict[str, Any]] = []
    previous_sufficiency: float | None = None
    stop_reason = "budget_exhausted"

    for step in range(max_budget + 1):
        context_chunks = [*sorted(selected_memory_chunks, key=lambda chunk: int(chunk.chunk_index)), *recent_chunks]
        context_frames = [frame for chunk in context_chunks for frame in chunk.frames]
        option_score = _score_mcq_options_from_frames(qa, context_frames, prompt, options)
        support_prompt = prompt
        predicted_text = str(option_score.get("predicted_text") or "")
        if predicted_text:
            support_prompt = f"{_query_text_only(prompt)} {predicted_text}"
        evidence_support, evidence_meta = _recent_evidence_coverage_score(context_chunks, support_prompt, config)
        answer_confidence = float(option_score.get("confidence", 0.0))
        sufficiency = (
            confidence_weight * answer_confidence
            + evidence_weight * float(evidence_support)
        )
        marginal_gain = 0.0 if previous_sufficiency is None else sufficiency - previous_sufficiency
        trace_entry = {
            "step": step,
            "context": "recent" if not selected_memory_chunks else "recent_plus_memory",
            "selected_memory_chunk_ids": [int(chunk.chunk_index) for chunk in selected_memory_chunks],
            "num_frames": len(context_frames),
            "predicted_letter": option_score.get("predicted_letter"),
            "predicted_text": option_score.get("predicted_text"),
            "answer_confidence": answer_confidence,
            "answer_margin": float(option_score.get("margin", 0.0)),
            "answer_entropy": float(option_score.get("entropy", 1.0)),
            "option_probs": option_score.get("option_probs", {}),
            "evidence_support": float(evidence_support),
            "sufficiency": float(sufficiency),
            "marginal_gain": float(marginal_gain),
            "score_available": bool(option_score.get("available")),
            "score_reason": option_score.get("reason"),
            "score_preprocess_ms": option_score.get("score_preprocess_ms"),
            "score_forward_ms": option_score.get("score_forward_ms"),
            "evidence_meta": evidence_meta,
        }
        if sufficiency >= sufficiency_threshold:
            stop_reason = "answer_sufficient"
            trace_entry["decision"] = "stop"
            trace.append(trace_entry)
            break
        if step > 0 and marginal_gain < marginal_gain_threshold:
            stop_reason = "marginal_gain_too_small"
            trace_entry["decision"] = "stop"
            trace.append(trace_entry)
            break
        if step >= max_budget:
            trace_entry["decision"] = "stop"
            trace.append(trace_entry)
            break

        next_candidate: dict[str, Any] | None = None
        for candidate in candidates:
            if int(candidate["chunk_id"]) in selected_candidate_ids:
                continue
            if float(candidate["full_progressive_candidate_score"]) < relevance_threshold:
                continue
            next_candidate = candidate
            break
        if next_candidate is None:
            stop_reason = "no_relevant_candidate"
            trace_entry["decision"] = "stop_no_candidate"
            trace.append(trace_entry)
            break
        trace_entry["decision"] = "retrieve"
        trace_entry["retrieved_next_chunk_id"] = int(next_candidate["chunk_id"])
        trace_entry["retrieved_next_score"] = float(next_candidate["full_progressive_candidate_score"])
        trace.append(trace_entry)
        selected_candidate_ids.add(int(next_candidate["chunk_id"]))
        selected_memory_chunks.append(next_candidate["chunk"])
        previous_sufficiency = sufficiency

    selected_memory_chunks = sorted(selected_memory_chunks, key=lambda chunk: int(chunk.chunk_index))
    selected_chunks = [*selected_memory_chunks, *recent_chunks]
    frames = [frame for chunk in selected_chunks for frame in chunk.frames]
    chunk_ids = [int(chunk.chunk_index) for chunk in selected_chunks for _frame in chunk.frames]
    timestamps = [float(ts) for chunk in selected_chunks for ts in chunk.frame_timestamps]
    memory_scores = []
    selected_ids = {int(chunk.chunk_index) for chunk in selected_memory_chunks}
    for candidate in candidates:
        item = {key: value for key, value in candidate.items() if key != "chunk"}
        item["selected"] = int(item["chunk_id"]) in selected_ids
        memory_scores.append(item)

    metadata = {
        "mode": config.mode,
        "window_size": config.mid_window,
        "window_reason": f"full_progressive_evidence_recent{config.mid_window}_backbone",
        "config": {
            "min_window": config.min_window,
            "mid_window": config.mid_window,
            "max_window": config.max_window,
            "memory_anchors": config.memory_anchors,
            "memory_search_chunks": config.memory_search_chunks,
            "full_progressive_sufficiency_threshold": sufficiency_threshold,
            "full_progressive_marginal_gain_threshold": marginal_gain_threshold,
            "full_progressive_confidence_weight": confidence_weight,
            "full_progressive_evidence_weight": evidence_weight,
            "full_progressive_relevance_threshold": relevance_threshold,
        },
        "decoded_chunks": len(chunks),
        "recent_window_size": recent_window_size,
        "recent_chunk_ids": [int(chunk.chunk_index) for chunk in recent_chunks],
        "memory_triggered": bool(selected_memory_chunks),
        "memory_gate": {
            "enabled": True,
            "activated": bool(selected_memory_chunks),
            "reason": stop_reason,
            "selected_budget": len(selected_memory_chunks),
            "max_budget": max_budget,
            "mcq_options": options,
            "rank_meta": rank_meta,
            "sufficiency_trace": trace,
        },
        "memory_fixed_budget": False,
        "memory_selector": _memory_selector_label(config),
        "memory_chunk_ids": [int(chunk.chunk_index) for chunk in selected_memory_chunks],
        "memory_scores": memory_scores,
        "candidate_frames": len(frames),
        "selected_frames": len(frames),
        "candidate_chunk_ids": chunk_ids,
        "selected_chunk_ids": chunk_ids,
        "candidate_timestamps": timestamps,
        "selected_timestamps": timestamps,
        "dedup_applied": False,
        "dedup_scores": [],
        "foveation_applied": False,
        "foveation_boxes": [],
    }
    return AdaptiveSelection(frames=frames, final_chunk_ids=chunk_ids, metadata=metadata)


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

    if config.strict_gated_semantic_memory:
        return _select_semantic_memory_chunks(older_chunks, count, config, prompt)

    if config.question_aware_memory:
        return _select_question_aware_memory_chunks(older_chunks, count, config, prompt)

    if config.event_summary_memory:
        return _select_event_summary_memory_chunks(older_chunks, count, config, prompt)

    if config.budgeted_counterfactual_memory:
        return _select_semantic_memory_chunks(older_chunks, count, config, prompt)

    if config.progressive_evidence_memory:
        return _select_semantic_memory_chunks(older_chunks, count, config, prompt)

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
    if config.strict_gated_semantic_memory:
        return "strict_gated_semantic_memory"
    if config.question_aware_memory:
        return "question_aware_window_similarity"
    if config.event_summary_memory:
        return "conditional_event_bookmark_memory"
    if config.budgeted_counterfactual_memory:
        return "budgeted_counterfactual_utility_gate"
    if config.progressive_evidence_memory:
        return "progressive_evidence_acquisition"
    if config.full_progressive_evidence_memory:
        return "full_progressive_evidence_acquisition"
    if config.progressive_sufficiency_memory:
        return "progressive_sufficiency_memory"
    if config.progressive_sufficiency_memory_heg:
        return "progressive_sufficiency_memory_heg"
    if config.progressive_sufficiency_memory_conservative_gate:
        return "progressive_sufficiency_memory_conservative_gate"
    if config.progressive_sufficiency_memory_microclip:
        return "progressive_sufficiency_memory_microclip"
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
    if (
        config.event_summary_memory
        or config.budgeted_counterfactual_memory
        or config.progressive_evidence_like
    ):
        window_size = config.mid_window
        if config.event_summary_memory:
            reason = f"event_summary_recent{config.mid_window}_backbone"
        elif config.budgeted_counterfactual_memory:
            reason = f"budgeted_counterfactual_recent{config.mid_window}_backbone"
        else:
            reason = f"progressive_evidence_recent{config.mid_window}_backbone"
    memory_triggered, memory_gate = _memory_trigger_decision(prompt, reason, config)
    recent_window_size = window_size
    if memory_triggered and config.fixed_memory_budget and config.memory_anchors > 0:
        recent_window_size = max(1, window_size - config.memory_anchors)

    recent_chunks = chunks[-recent_window_size:]
    memory_chunks: list[Any] = []
    memory_scores: list[dict[str, Any]] = []
    if memory_triggered and config.memory_anchors > 0:
        older_chunks = chunks[: max(0, len(chunks) - recent_window_size)]
        if config.budgeted_counterfactual_memory:
            memory_chunks, memory_scores, memory_gate = _select_budgeted_counterfactual_memory_chunks(
                older_chunks,
                recent_chunks,
                config.memory_anchors,
                config,
                prompt=prompt,
                reason=reason,
            )
            memory_triggered = bool(memory_chunks)
        elif config.progressive_evidence_like:
            memory_chunks, memory_scores, memory_gate = _select_progressive_evidence_memory_chunks(
                older_chunks,
                recent_chunks,
                config.memory_anchors,
                config,
                prompt=prompt,
            )
            memory_triggered = bool(memory_chunks)
        else:
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
    answer_prompt = prompt
    if config.progressive_sufficiency_like:
        from lib.minicpm.progressive_sufficiency import (
            select_progressive_sufficiency_memory,
            select_progressive_sufficiency_memory_microclip,
        )

        recent_window = max(1, int(config.max_window))
        recent_video_start = video_start
        if video_end is not None:
            recent_video_start = max(0.0, float(video_end) - float(recent_window) * float(chunk_duration))
        baseline_recent = select_recent_window_frames(
            qa=qa,
            video_path=video_path,
            chunk_duration=chunk_duration,
            fps=fps,
            recent_frames_only=recent_window,
            video_start=recent_video_start,
            video_end=video_end,
            cdas_config=cdas_config,
        )
        progressive_kwargs = {
            "prompt": prompt,
            "config": config,
            "recent_chunks": baseline_recent.selected_chunks,
            "recent_frames": baseline_recent.frames,
            "recent_chunk_ids": baseline_recent.final_chunk_ids,
            "recent_downsample_mode": baseline_recent.downsample_mode,
            "baseline_recent_metadata": {
                "decode_backend": baseline_recent.decode_backend,
                "decode_time": baseline_recent.decode_time,
                "selection_time": baseline_recent.selection_time,
                "decoded_chunks": baseline_recent.decoded_chunks,
                "decoded_frames": baseline_recent.decoded_frames,
                "video_start": baseline_recent.video_start,
                "video_end": baseline_recent.video_end,
                "cdas": baseline_recent.cdas_metadata,
            },
        }
        if config.progressive_sufficiency_memory_microclip:
            progressive_selection = select_progressive_sufficiency_memory_microclip(
                qa,
                chunks,
                **progressive_kwargs,
            )
        else:
            progressive_selection = select_progressive_sufficiency_memory(
                qa,
                chunks,
                **progressive_kwargs,
                enable_heg=config.progressive_sufficiency_memory_heg,
                enable_conservative_gate=config.progressive_sufficiency_memory_conservative_gate,
            )
        selection = AdaptiveSelection(
            frames=progressive_selection.frames,
            final_chunk_ids=progressive_selection.final_chunk_ids,
            metadata=progressive_selection.metadata,
            downsample_mode=progressive_selection.downsample_mode,
        )
        answer_prompt = progressive_selection.answer_prompt
    elif config.full_progressive_evidence_memory:
        selection = _full_progressive_evidence_selection(qa, chunks, prompt=prompt, config=config)
    else:
        selection = select_adaptive_frames(chunks, prompt=prompt, config=config)
    selection_time = time.perf_counter() - selection_t0
    if not selection.frames:
        raise ValueError(f"No frames selected from video: {video_path}")

    t0 = time.perf_counter()
    answer = qa.generate_from_frames(selection.frames, answer_prompt, downsample_mode=selection.downsample_mode)
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
    if config.progressive_sufficiency_like:
        selection.metadata["final_generation_ms"] = float(generate_time * 1000.0)
        selection.metadata["ttft_seconds"] = float(ttft_seconds)
        selection.metadata["num_vision_tokens"] = int(num_vision_tokens)
        selection.metadata["num_vision_frames"] = int(num_frames)
        selection.metadata["end_to_end_time_seconds"] = float(decode_time + selection_time + generate_time)
        selection.metadata["peak_allocated_gpu_mb"] = float(profile_metadata.get("gpu_peak_allocated_mb", 0.0))
        selection.metadata["peak_reserved_gpu_mb"] = float(profile_metadata.get("gpu_peak_reserved_mb", 0.0))
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
