from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any

from PIL import Image

from lib.cdas_sampler import CDASConfig, select_recent_frames_cdas
from lib.minicpm.baseline import (
    RecentWindowQAModel,
    _build_profile,
    _capture_gpu_memory,
    _reset_gpu_memory_peaks,
    _synchronize_gpu_devices,
)
from lib.shared.recent_window import RecentWindowResult, decode_video_to_chunks_qwen


@dataclass
class RecentDescriptionConfig:
    recent_frames: int = 6
    description_max_new_tokens: int = 256
    max_note_words: int = 45
    max_prompt_chars: int = 6000

    @classmethod
    def from_env(cls) -> "RecentDescriptionConfig":
        return cls(
            recent_frames=max(1, int(os.environ.get("MINICPM_RECENT_DESC_FRAMES", "6"))),
            description_max_new_tokens=max(
                64,
                int(os.environ.get("MINICPM_RECENT_DESC_MAX_TOKENS", "256")),
            ),
            max_note_words=max(12, int(os.environ.get("MINICPM_RECENT_DESC_MAX_WORDS", "45"))),
            max_prompt_chars=max(1000, int(os.environ.get("MINICPM_RECENT_DESC_MAX_PROMPT_CHARS", "6000"))),
        )


def _clean_note(text: str, max_words: int) -> str:
    text = re.sub(r"^\s*(?:[-*]\s*)?[0-9]+[\).\:-]\s*", "", text.strip())
    text = re.sub(r"\s+", " ", text).strip(" -:\t\r\n")
    if not text:
        return "No reliable visible details."
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    return text


def _parse_numbered_notes(raw: str, expected: int, max_words: int) -> list[str]:
    notes: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^(?:[-*]\s*)?[0-9]+[\).\:-]\s+", line):
            notes.append(_clean_note(line, max_words))
        elif len(notes) < expected and len(raw.splitlines()) <= expected + 4:
            notes.append(_clean_note(line, max_words))
    if len(notes) < expected:
        rough = [part.strip() for part in re.split(r";|\n", raw) if part.strip()]
        for item in rough:
            if len(notes) >= expected:
                break
            notes.append(_clean_note(item, max_words))
    while len(notes) < expected:
        notes.append("No reliable visible details.")
    return notes[:expected]


class RecentDescriptionQAModel(RecentWindowQAModel):
    """MiniCPM wrapper for Proposal 2: recent visual frames + recent text notes."""

    def __init__(
        self,
        *args: Any,
        recent_description_config: RecentDescriptionConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.recent_description_config = recent_description_config or RecentDescriptionConfig.from_env()

    def describe_recent_frames(self, frames: list[Image.Image]) -> tuple[list[str], float]:
        if not frames:
            return [], 0.0

        prompt = (
            "You are preparing visual evidence notes for video question answering.\n"
            "Describe each image independently but preserve details that help answer later questions.\n"
            "For each frame, include:\n"
            "- people with position (left/center/right/front/back) and distinguishing clothing or identity cues;\n"
            "- actions, interactions, and object handling;\n"
            "- visible objects, text, numbers, colors, counts, and spatial relations;\n"
            "- anything that changed from the previous frame if visible.\n"
            "Do not answer the question. Do not speculate beyond visible evidence.\n"
            "Use concise but complete factual notes.\n"
            f"Return exactly {len(frames)} numbered lines, one line per image."
        )

        old_max_new_tokens = self.max_new_tokens
        self.max_new_tokens = self.recent_description_config.description_max_new_tokens
        t0 = time.perf_counter()
        try:
            raw = super().generate_from_frames(frames, prompt)
        finally:
            self.max_new_tokens = old_max_new_tokens
        elapsed = time.perf_counter() - t0
        notes = _parse_numbered_notes(
            raw,
            len(frames),
            self.recent_description_config.max_note_words,
        )
        return notes, elapsed

    def build_recent_description_prompt(
        self,
        *,
        original_prompt: str,
        notes: list[str],
        chunk_ids: list[int],
    ) -> str:
        lines = []
        for index, note in enumerate(notes, start=1):
            chunk_text = f" chunk={chunk_ids[index - 1]}" if index - 1 < len(chunk_ids) else ""
            lines.append(f"Frame {index}{chunk_text}: {note}")
        notes_text = "\n".join(lines) if lines else "(not available)"
        prompt = (
            "You are answering a streaming-video multiple-choice question.\n"
            "You receive two synchronized sources for the same recent frames:\n"
            "1. RECENT FRAMES as images.\n"
            "2. RECENT FRAME DESCRIPTIONS as text notes generated from those images.\n\n"
            "Use the images as the primary evidence. Use the descriptions to double-check fine details "
            "such as identity, clothing, numbers, objects, colors, counts, actions, and spatial relations.\n"
            "If the text descriptions conflict with the images, trust the images.\n\n"
            "RECENT FRAME DESCRIPTIONS:\n"
            f"{notes_text}\n\n"
            "QUESTION:\n"
            f"{original_prompt}"
        )
        if len(prompt) <= self.recent_description_config.max_prompt_chars:
            return prompt
        return prompt[-self.recent_description_config.max_prompt_chars :]


def query_recent_window(
    qa: RecentDescriptionQAModel,
    video_path: str,
    prompt: str,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    video_start: float | None = None,
    video_end: float | None = None,
    cdas_config: CDASConfig | None = None,
) -> tuple[RecentWindowResult, str]:
    if not isinstance(qa, RecentDescriptionQAModel):
        raise TypeError("Recent description evaluation requires RecentDescriptionQAModel.")

    before_memory = _reset_gpu_memory_peaks()
    decode_t0 = time.perf_counter()
    chunks, decode_backend = decode_video_to_chunks_qwen(
        video_path=video_path,
        chunk_duration=chunk_duration,
        fps=fps,
        recent_frames_only=recent_frames_only,
        video_start=video_start,
        video_end=video_end,
    )
    decode_time = time.perf_counter() - decode_t0
    if not chunks:
        raise ValueError(f"No chunks decoded from video: {video_path}")

    selection_t0 = time.perf_counter()
    window_size = max(1, int(os.environ.get("MINICPM_RECENT_DESC_FRAMES", recent_frames_only)))
    cdas_metadata: dict[str, Any] | None = None
    selected_downsample_mode: str | None = None
    if cdas_config is not None and cdas_config.enabled:
        selection = select_recent_frames_cdas(
            chunks=chunks,
            window_size=window_size,
            config=cdas_config,
            default_downsample_mode=qa.downsample_mode,
        )
        recent_frames = selection.frames
        final_chunk_ids = selection.final_chunk_ids
        selected_downsample_mode = selection.downsample_mode
        cdas_metadata = selection.metadata
    else:
        recent_chunks = list(chunks[-window_size:])
        recent_frames = [frame for chunk in recent_chunks for frame in chunk.frames]
        final_chunk_ids = [chunk.chunk_index for chunk in recent_chunks]
    if not recent_frames:
        raise ValueError(f"No frames decoded from video: {video_path}")

    notes, description_time = qa.describe_recent_frames(recent_frames)
    enriched_prompt = qa.build_recent_description_prompt(
        original_prompt=prompt,
        notes=notes,
        chunk_ids=final_chunk_ids,
    )
    selection_time = time.perf_counter() - selection_t0

    t0 = time.perf_counter()
    answer = qa.generate_from_frames(
        recent_frames,
        enriched_prompt,
        downsample_mode=selected_downsample_mode,
    )
    _synchronize_gpu_devices()
    generate_time = time.perf_counter() - t0
    ttft_seconds = getattr(qa, "_last_ttft_seconds", 0.0) or 0.0
    num_vision_tokens = getattr(qa, "_last_num_vision_tokens", 0) or 0
    num_frames = getattr(qa, "_last_num_vision_frames", 0) or len(recent_frames)
    _synchronize_gpu_devices()
    after_memory = _capture_gpu_memory()

    profile_metadata = _build_profile(
        mode="recent_description_window",
        decode_time=decode_time,
        selection_time=selection_time,
        generate_time=generate_time,
        before_memory=before_memory,
        after_memory=after_memory,
        qa=qa,
    )
    profile_metadata["decoded_chunks"] = len(chunks)
    profile_metadata["decoded_frames"] = sum(len(chunk.frames) for chunk in chunks)
    profile_metadata["recent_description"] = {
        "recent_frames": window_size,
        "description_time_seconds": description_time,
        "description_max_new_tokens": qa.recent_description_config.description_max_new_tokens,
        "description_max_words": qa.recent_description_config.max_note_words,
        "prompt_chars": len(enriched_prompt),
        "selected_chunk_ids": final_chunk_ids,
        "notes": notes,
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
    result.recent_description_metadata = profile_metadata["recent_description"]
    if cdas_metadata is not None:
        cdas_metadata["actual_vision_tokens"] = num_vision_tokens
        cdas_metadata["actual_vision_frames"] = num_frames
        cdas_metadata["actual_downsample_mode"] = getattr(qa, "_last_downsample_mode", qa.downsample_mode)
        result.cdas_metadata = cdas_metadata
    return result, decode_backend
