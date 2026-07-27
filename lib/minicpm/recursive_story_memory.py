from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any

from PIL import Image

from lib.cdas_sampler import CDASConfig
from lib.minicpm.baseline import (
    RecentWindowQAModel,
    _build_profile,
    _capture_gpu_memory,
    _reset_gpu_memory_peaks,
    _synchronize_gpu_devices,
)
from lib.shared.recent_window import RecentWindowResult, decode_video_to_chunks_qwen

# ---------------------------------------------------------------------------
# Recursive Story Memory (RSM)
#
# This is the "textual story memory" proposal exactly as pitched on the
# slides: a *single* bounded natural-language narrative S_t that is
# recursively rewritten by the model itself as the stream progresses, kept
# alongside a small raw recent-frame window. It is intentionally a separate
# module from lib/minicpm/story_memory.py, which instead keeps a growing list
# of discrete per-chunk notes bounded by truncation/deduplication rather than
# LLM-driven re-summarization. Use this module when you want the mechanism
# described in "Proposal 1: Textual Story Memory" (single rewritten story,
# hard token budget, LLM-driven compression on overflow).
# ---------------------------------------------------------------------------


@dataclass
class RecursiveStoryMemoryConfig:
    recent_frames: int = 6
    """Size of the raw recent window, in decoded chunks (not raw frames)."""

    update_batch_chunks: int = 4
    """How many evicted chunks are folded into the story per rewrite call.
    Set to 1 for the strictest reading of the proposal ("fires whenever a
    frame leaves the recent window"); higher values trade update fidelity
    for fewer, cheaper LLM calls."""

    max_story_tokens: int = 256
    """L_max: hard token budget for the story. Enforced by asking the model
    to comply, then one compression retry, then a hard truncation fallback."""

    rewrite_max_new_tokens: int = 384
    """Generation budget for the rewrite call. Must leave headroom above
    max_story_tokens since the model does not always land exactly on budget."""

    compress_max_new_tokens: int = 320
    """Generation budget for the compression-retry call."""

    max_compression_attempts: int = 1
    """Number of extra compression calls allowed before hard truncation."""

    full_context: bool = True
    """If true, always decode the video from t=0 (cache avoids recomputation
    across questions on the same video). Mirrors lib/minicpm/story_memory.py."""

    prompt_version: str = "v1_recursive"

    @classmethod
    def from_env(cls) -> "RecursiveStoryMemoryConfig":
        return cls(
            recent_frames=max(1, int(os.environ.get("MINICPM_RSM_RECENT_FRAMES", "6"))),
            update_batch_chunks=max(1, int(os.environ.get("MINICPM_RSM_UPDATE_BATCH", "4"))),
            max_story_tokens=max(16, int(os.environ.get("MINICPM_RSM_MAX_STORY_TOKENS", "256"))),
            rewrite_max_new_tokens=max(
                32, int(os.environ.get("MINICPM_RSM_REWRITE_MAX_NEW_TOKENS", "384"))
            ),
            compress_max_new_tokens=max(
                32, int(os.environ.get("MINICPM_RSM_COMPRESS_MAX_NEW_TOKENS", "320"))
            ),
            max_compression_attempts=max(
                0, int(os.environ.get("MINICPM_RSM_MAX_COMPRESSION_ATTEMPTS", "1"))
            ),
            full_context=os.environ.get("MINICPM_RSM_FULL_CONTEXT", "1").strip().lower()
            in {"1", "true", "yes", "on"},
            prompt_version=os.environ.get("MINICPM_RSM_PROMPT_VERSION", "v1_recursive").strip()
            or "v1_recursive",
        )


@dataclass
class _StoryState:
    story_text: str = ""
    story_tokens: int = 0
    folded_chunk_index: int = -1
    rewrite_calls: int = 0
    compression_calls: int = 0
    hard_truncations: int = 0


def _strip_wrapper_text(text: str) -> str:
    """Best-effort cleanup if the model adds a preamble despite instructions."""
    text = text.strip()
    text = re.sub(
        r"^(updated story|compressed story|story)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.strip(" \t\r\n\"'")
    return text


def _format_new_observations(timestamps: list[float]) -> str:
    return "\n".join(f"- image {i + 1}: [t={ts:.1f}s]" for i, ts in enumerate(timestamps))


def _build_rewrite_prompt(prior_story: str, timestamps: list[float], max_story_tokens: int) -> str:
    story_block = prior_story if prior_story.strip() else "(empty, nothing observed yet)"
    return (
        "You are maintaining a single running text memory (a \"story\") for a "
        "streaming-video QA system.\n\n"
        "CURRENT STORY SO FAR:\n"
        f"{story_block}\n\n"
        "NEW OBSERVATIONS (attached images, in order):\n"
        f"{_format_new_observations(timestamps)}\n\n"
        "Task: rewrite the story into a single updated version that:\n"
        "- incorporates the new observations above\n"
        "- preserves important earlier facts, named entities, and any still-unresolved action or thread\n"
        "- drops resolved, purely descriptive detail that is no longer useful\n"
        "- keeps inline [t=Xs] timestamp markers for events\n"
        f"- is at most {max_story_tokens} tokens long\n\n"
        "Do not answer any question. Do not describe these instructions. "
        "Output ONLY the updated story text, nothing else."
    )


def _build_compression_prompt(draft_text: str, prev_token_count: int, max_story_tokens: int) -> str:
    return (
        f"Your previous rewrite was too long ({prev_token_count} tokens; the limit is "
        f"{max_story_tokens} tokens).\n\n"
        "CURRENT DRAFT:\n"
        f"{draft_text}\n\n"
        f"Compress this draft to at most {max_story_tokens} tokens. Keep named entities, "
        "event order, and unresolved threads. Drop resolved or redundant descriptive detail. "
        "Keep inline [t=Xs] markers. Output ONLY the compressed story text, nothing else."
    )


def _build_query_prompt(story_text: str, original_prompt: str) -> str:
    story_block = story_text if story_text.strip() else "(empty)"
    return (
        "You are answering a streaming-video question.\n\n"
        "STORY MEMORY (earlier context, text-only, may be incomplete):\n"
        f"{story_block}\n\n"
        "The attached images are the most recent frames and are your primary visual "
        "evidence for the current moment.\n\n"
        "Decision policy:\n"
        "- For questions about the current/right-now moment, trust the attached recent frames.\n"
        "- For questions about earlier events, counts, or ordering, use the STORY MEMORY.\n"
        "- If they conflict on a current fact, trust the recent frames.\n\n"
        "QUESTION:\n"
        f"{original_prompt}"
    )


class RecursiveStoryMemoryQAModel(RecentWindowQAModel):
    """MiniCPM wrapper implementing a single, recursively-rewritten text memory.

    Unlike lib/minicpm/story_memory.py (a list of discrete per-chunk notes
    bounded by deduplication/truncation), this keeps exactly one story string
    S_t per video. Every time chunks leave the recent window, the model is
    asked to *rewrite* S_t to fold in the new evidence while staying under a
    hard token budget, with a compression retry and a hard-truncation
    fallback to guarantee the budget is never exceeded.
    """

    def __init__(
        self,
        *args: Any,
        rsm_config: RecursiveStoryMemoryConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.rsm_config = rsm_config or RecursiveStoryMemoryConfig.from_env()
        self._rsm_cache: dict[str, _StoryState] = {}

    def _rsm_key(self, video_path: str, fps: float, chunk_duration: float) -> str:
        return f"{os.path.abspath(video_path)}|fps={float(fps):.4f}|chunk={float(chunk_duration):.4f}"

    def _count_tokens(self, text: str) -> int:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return len(text.split())
        try:
            return len(tokenizer(text, add_special_tokens=False)["input_ids"])
        except Exception:
            return len(text.split())

    def _truncate_to_budget(self, text: str, max_tokens: int) -> str:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            words = text.split()
            return " ".join(words[:max_tokens])
        try:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) <= max_tokens:
                return text
            keep_head = max(1, max_tokens // 5)
            keep_tail = max_tokens - keep_head
            head_ids = ids[:keep_head]
            tail_ids = ids[-keep_tail:] if keep_tail > 0 else []
            head_text = tokenizer.decode(head_ids, skip_special_tokens=True)
            tail_text = tokenizer.decode(tail_ids, skip_special_tokens=True)
            return f"{head_text} ... {tail_text}".strip()
        except Exception:
            words = text.split()
            return " ".join(words[:max_tokens])

    def _generate_text(self, frames: list[Image.Image], prompt: str, max_new_tokens: int) -> tuple[str, float]:
        old_max_new_tokens = self.max_new_tokens
        self.max_new_tokens = int(max_new_tokens)
        t0 = time.perf_counter()
        try:
            raw = super().generate_from_frames(frames, prompt)
        finally:
            self.max_new_tokens = old_max_new_tokens
        elapsed = time.perf_counter() - t0
        return _strip_wrapper_text(raw), elapsed

    def _rewrite_story(
        self,
        state: _StoryState,
        frames: list[Image.Image],
        timestamps: list[float],
    ) -> dict[str, Any]:
        config = self.rsm_config
        stats = {"rewrite_time_seconds": 0.0, "compress_time_seconds": 0.0, "rewrite_calls": 0, "compression_calls": 0}

        prompt = _build_rewrite_prompt(state.story_text, timestamps, config.max_story_tokens)
        draft, elapsed = self._generate_text(frames, prompt, config.rewrite_max_new_tokens)
        stats["rewrite_time_seconds"] += elapsed
        stats["rewrite_calls"] += 1
        state.rewrite_calls += 1

        token_count = self._count_tokens(draft)
        attempts = 0
        while token_count > config.max_story_tokens and attempts < config.max_compression_attempts:
            compress_prompt = _build_compression_prompt(draft, token_count, config.max_story_tokens)
            draft, elapsed = self._generate_text(frames, compress_prompt, config.compress_max_new_tokens)
            stats["compress_time_seconds"] += elapsed
            stats["compression_calls"] += 1
            state.compression_calls += 1
            token_count = self._count_tokens(draft)
            attempts += 1

        if token_count > config.max_story_tokens:
            draft = self._truncate_to_budget(draft, config.max_story_tokens)
            token_count = self._count_tokens(draft)
            state.hard_truncations += 1

        state.story_text = draft
        state.story_tokens = token_count
        return stats

    def _advance_story(
        self,
        *,
        video_key: str,
        chunks: list[Any],
        window_chunk_ids: set[int],
    ) -> dict[str, Any]:
        config = self.rsm_config
        state = self._rsm_cache.setdefault(video_key, _StoryState())

        foldable = sorted(
            (
                chunk
                for chunk in chunks
                if chunk.chunk_index > state.folded_chunk_index
                and chunk.chunk_index not in window_chunk_ids
                and chunk.frames
            ),
            key=lambda chunk: chunk.chunk_index,
        )

        total_stats = {
            "rewrite_time_seconds": 0.0,
            "compress_time_seconds": 0.0,
            "rewrite_calls": 0,
            "compression_calls": 0,
            "folded_chunks": 0,
        }
        if not foldable:
            return total_stats

        batch_size = config.update_batch_chunks
        for start in range(0, len(foldable), batch_size):
            batch = foldable[start : start + batch_size]
            frames = [chunk.frames[-1] for chunk in batch]
            timestamps = [
                float(chunk.frame_timestamps[-1] if chunk.frame_timestamps else chunk.end_time)
                for chunk in batch
            ]
            stats = self._rewrite_story(state, frames, timestamps)
            total_stats["rewrite_time_seconds"] += stats["rewrite_time_seconds"]
            total_stats["compress_time_seconds"] += stats["compress_time_seconds"]
            total_stats["rewrite_calls"] += stats["rewrite_calls"]
            total_stats["compression_calls"] += stats["compression_calls"]
            total_stats["folded_chunks"] += len(batch)
            state.folded_chunk_index = max(state.folded_chunk_index, batch[-1].chunk_index)

        return total_stats

    def build_query_prompt(self, *, original_prompt: str, story_text: str) -> str:
        return _build_query_prompt(story_text, original_prompt)


def query_recent_window(
    qa: RecursiveStoryMemoryQAModel,
    video_path: str,
    prompt: str,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    video_start: float | None = None,
    video_end: float | None = None,
    cdas_config: CDASConfig | None = None,
) -> tuple[RecentWindowResult, str]:
    if not isinstance(qa, RecursiveStoryMemoryQAModel):
        raise TypeError("Recursive story memory evaluation requires RecursiveStoryMemoryQAModel.")

    before_memory = _reset_gpu_memory_peaks()
    decode_t0 = time.perf_counter()
    saved_exact_recent = os.environ.pop("QWEN_EXACT_RECENT_DECODE", None)
    try:
        chunks, decode_backend = decode_video_to_chunks_qwen(
            video_path=video_path,
            chunk_duration=chunk_duration,
            fps=fps,
            recent_frames_only=None,
            video_start=0.0 if qa.rsm_config.full_context else video_start,
            video_end=video_end,
        )
    finally:
        if saved_exact_recent is not None:
            os.environ["QWEN_EXACT_RECENT_DECODE"] = saved_exact_recent
    decode_time = time.perf_counter() - decode_t0
    if not chunks:
        raise ValueError(f"No chunks decoded from video: {video_path}")

    selection_t0 = time.perf_counter()
    window_size = max(1, int(os.environ.get("MINICPM_RSM_RECENT_FRAMES", recent_frames_only)))
    recent_chunks = list(chunks[-window_size:])
    recent_frames = [frame for chunk in recent_chunks for frame in chunk.frames]
    final_chunk_ids = [chunk.chunk_index for chunk in recent_chunks]
    if not recent_frames:
        raise ValueError(f"No frames decoded from video: {video_path}")

    video_key = qa._rsm_key(video_path, fps=fps, chunk_duration=chunk_duration)
    update_stats = qa._advance_story(
        video_key=video_key,
        chunks=chunks,
        window_chunk_ids=set(final_chunk_ids),
    )
    state = qa._rsm_cache[video_key]
    query_prompt = qa.build_query_prompt(original_prompt=prompt, story_text=state.story_text)
    # update_stats["rewrite_time_seconds"] / ["compress_time_seconds"] are already
    # inside this elapsed span (the LLM calls happened inside _advance_story above);
    # do not add them again.
    selection_time = time.perf_counter() - selection_t0

    t0 = time.perf_counter()
    answer = qa.generate_from_frames(recent_frames, query_prompt)
    _synchronize_gpu_devices()
    generate_time = time.perf_counter() - t0
    ttft_seconds = getattr(qa, "_last_ttft_seconds", 0.0) or 0.0
    num_vision_tokens = getattr(qa, "_last_num_vision_tokens", 0) or 0
    num_frames = getattr(qa, "_last_num_vision_frames", 0) or len(recent_frames)
    _synchronize_gpu_devices()
    after_memory = _capture_gpu_memory()

    profile_metadata = _build_profile(
        mode="recursive_story_memory_recent_window",
        decode_time=decode_time,
        selection_time=selection_time,
        generate_time=generate_time,
        before_memory=before_memory,
        after_memory=after_memory,
        qa=qa,
    )
    profile_metadata["decoded_chunks"] = len(chunks)
    profile_metadata["decoded_frames"] = sum(len(chunk.frames) for chunk in chunks)
    profile_metadata["video_start"] = 0.0 if qa.rsm_config.full_context else video_start
    profile_metadata["video_end"] = video_end

    peak_allocated_mb = profile_metadata.get("gpu_peak_allocated_mb", 0.0)
    profile_metadata["recursive_story_memory"] = {
        "recent_frames": window_size,
        "update_batch_chunks": qa.rsm_config.update_batch_chunks,
        "max_story_tokens": qa.rsm_config.max_story_tokens,
        "full_context": qa.rsm_config.full_context,
        "prompt_version": qa.rsm_config.prompt_version,
        "folded_chunk_index": state.folded_chunk_index,
        "folded_chunks_this_call": update_stats["folded_chunks"],
        "rewrite_calls_this_call": update_stats["rewrite_calls"],
        "compression_calls_this_call": update_stats["compression_calls"],
        "rewrite_calls_total": state.rewrite_calls,
        "compression_calls_total": state.compression_calls,
        "hard_truncations_total": state.hard_truncations,
        "selected_chunk_ids": final_chunk_ids,
        "story_tokens": state.story_tokens,
        "story_text": state.story_text,
        # Field names matched to the slide's "Timing Components We Will Log
        # for TSM" table so slides and logs stay in sync.
        "tsm_story_update_ms": update_stats["rewrite_time_seconds"] * 1000.0,
        "tsm_story_compress_ms": update_stats["compress_time_seconds"] * 1000.0,
        "tsm_prefill_ms": profile_metadata.get("prefill_forward_time_ms"),
        "tsm_first_token_ms": profile_metadata.get("generate_first_token_time_ms"),
        "tsm_generate_ms": profile_metadata.get("generate_tokens_time_ms"),
        "tsm_peak_mem_mb": peak_allocated_mb,
        "tsm_story_len_tokens": state.story_tokens,
    }

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
    result.recursive_story_memory_metadata = profile_metadata["recursive_story_memory"]
    return result, decode_backend
