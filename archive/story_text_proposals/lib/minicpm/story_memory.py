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


@dataclass
class StoryMemoryConfig:
    recent_frames: int = 6
    batch_size: int = 8
    max_items: int = 96
    max_prompt_chars: int = 9000
    description_max_new_tokens: int = 192
    duplicate_jaccard_threshold: float = 0.82
    describe_stride: int = 1
    full_context: bool = True
    prompt_version: str = "v2_evidence"

    @classmethod
    def from_env(cls) -> "StoryMemoryConfig":
        return cls(
            recent_frames=max(1, int(os.environ.get("MINICPM_STORY_RECENT_FRAMES", "6"))),
            batch_size=max(1, int(os.environ.get("MINICPM_STORY_BATCH_SIZE", "8"))),
            max_items=max(1, int(os.environ.get("MINICPM_STORY_MAX_ITEMS", "96"))),
            max_prompt_chars=max(1000, int(os.environ.get("MINICPM_STORY_MAX_PROMPT_CHARS", "9000"))),
            description_max_new_tokens=max(
                32,
                int(os.environ.get("MINICPM_STORY_DESC_MAX_TOKENS", "192")),
            ),
            duplicate_jaccard_threshold=float(
                os.environ.get("MINICPM_STORY_DUPLICATE_JACCARD", "0.82")
            ),
            describe_stride=max(1, int(os.environ.get("MINICPM_STORY_DESCRIBE_STRIDE", "1"))),
            full_context=os.environ.get("MINICPM_STORY_FULL_CONTEXT", "1").strip().lower()
            in {"1", "true", "yes", "on"},
            prompt_version=os.environ.get("MINICPM_STORY_PROMPT_VERSION", "v2_evidence").strip()
            or "v2_evidence",
        )


@dataclass
class StoryMemoryEntry:
    video_key: str
    chunk_id: int
    timestamp: float
    description: str


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip().lower())
    text = re.sub(r"^[0-9]+[\).\:-]\s*", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()


def _jaccard(a: str, b: str) -> float:
    a_words = set(_normalize_text(a).split())
    b_words = set(_normalize_text(b).split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / max(1, len(a_words | b_words))


def _clean_description(text: str) -> str:
    text = re.sub(r"^\s*[0-9]+[\).\:-]\s*", "", text.strip())
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" -:\t\r\n")
    if not text:
        return "no salient change"
    words = text.split()
    max_words = max(12, int(os.environ.get("MINICPM_STORY_NOTE_MAX_WORDS", "36")))
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    return text


def _parse_numbered_notes(raw: str, expected: int) -> list[str]:
    notes: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^(?:[-*]\s*)?[0-9]+[\).\:-]\s+", line):
            notes.append(_clean_description(line))
        elif len(notes) < expected and len(raw.splitlines()) <= expected + 3:
            notes.append(_clean_description(line))
    if len(notes) < expected:
        rough = [part.strip() for part in re.split(r";|\n", raw) if part.strip()]
        for item in rough:
            if len(notes) >= expected:
                break
            notes.append(_clean_description(item))
    while len(notes) < expected:
        notes.append("no salient change")
    return notes[:expected]


def _format_story_lines(entries: list[StoryMemoryEntry]) -> list[str]:
    lines = []
    for entry in entries:
        lines.append(f"[t={entry.timestamp:.1f}s] {entry.description}")
    return lines


class StoryMemoryQAModel(RecentWindowQAModel):
    """MiniCPM wrapper with compact text memory over older streaming frames.

    The old visual stream is converted into short chronological notes. At query
    time, only recent frames are passed visually; older context is passed as text.
    """

    def __init__(self, *args: Any, story_config: StoryMemoryConfig | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.story_config = story_config or StoryMemoryConfig.from_env()
        self._story_cache: dict[str, dict[int, StoryMemoryEntry]] = {}

    def _story_key(self, video_path: str, fps: float, chunk_duration: float) -> str:
        return f"{os.path.abspath(video_path)}|fps={float(fps):.4f}|chunk={float(chunk_duration):.4f}"

    def _describe_batch(
        self,
        frames: list[Image.Image],
        previous_note: str | None = None,
    ) -> tuple[list[str], float]:
        if not frames:
            return [], 0.0

        prompt = (
            "You are writing compact visual memory for a streaming-video QA system.\n"
            "For each image, write exactly one evidence note that preserves facts useful for future questions.\n"
            "Each note should be one concise sentence, about 20-35 words.\n\n"
            "Include visible evidence when present:\n"
            "- people: position, clothing, jersey/ID number, role, and who they interact with\n"
            "- actions/events: what is happening now and what changed from the previous note\n"
            "- objects/text: object names, colors, counts, written text, numbers, signs, screens\n"
            "- spatial relations: left/right/center, near/far, behind/in front, held by, passed to\n\n"
            "Rules:\n"
            "- Do not answer any benchmark question.\n"
            "- Do not invent hidden causes, identities, or text. Use 'unclear' if uncertain.\n"
            "- If the scene repeats, restate the stable state plus any small change.\n"
            "- Keep names generic unless a visible label/number identifies them.\n"
            f"Return exactly {len(frames)} numbered lines, one line per image."
        )
        if previous_note:
            prompt += (
                "\n\nPrevious memory state, for continuity only:\n"
                f"{previous_note}\n"
                "Carry forward only persistent visible entities or actions; do not add unseen details."
            )

        old_max_new_tokens = self.max_new_tokens
        self.max_new_tokens = self.story_config.description_max_new_tokens
        t0 = time.perf_counter()
        try:
            raw = super().generate_from_frames(frames, prompt)
        finally:
            self.max_new_tokens = old_max_new_tokens
        elapsed = time.perf_counter() - t0
        return _parse_numbered_notes(raw, len(frames)), elapsed

    def _ensure_story_entries(
        self,
        *,
        video_key: str,
        chunks: list[Any],
    ) -> dict[str, Any]:
        cache = self._story_cache.setdefault(video_key, {})
        config = self.story_config
        missing = [chunk for chunk in chunks if chunk.chunk_index not in cache]
        if not missing:
            return {
                "generated_descriptions": 0,
                "description_batches": 0,
                "description_time_seconds": 0.0,
            }

        description_time = 0.0
        batches = 0
        generated = 0
        pending_frames: list[Image.Image] = []
        pending_chunks: list[Any] = []
        previous_note = None
        if cache:
            previous_note = cache[max(cache)].description

        for chunk in missing:
            if chunk.chunk_index % config.describe_stride != 0 and chunk is not missing[-1]:
                continue
            if not chunk.frames:
                continue
            pending_frames.append(chunk.frames[-1])
            pending_chunks.append(chunk)
            if len(pending_frames) >= config.batch_size:
                notes, elapsed = self._describe_batch(pending_frames, previous_note=previous_note)
                description_time += elapsed
                batches += 1
                for item, note in zip(pending_chunks, notes):
                    note = _clean_description(note)
                    cache[item.chunk_index] = StoryMemoryEntry(
                        video_key=video_key,
                        chunk_id=item.chunk_index,
                        timestamp=float(item.frame_timestamps[-1] if item.frame_timestamps else item.end_time),
                        description=note,
                    )
                    previous_note = note
                    generated += 1
                pending_frames = []
                pending_chunks = []

        if pending_frames:
            notes, elapsed = self._describe_batch(pending_frames, previous_note=previous_note)
            description_time += elapsed
            batches += 1
            for item, note in zip(pending_chunks, notes):
                note = _clean_description(note)
                cache[item.chunk_index] = StoryMemoryEntry(
                    video_key=video_key,
                    chunk_id=item.chunk_index,
                    timestamp=float(item.frame_timestamps[-1] if item.frame_timestamps else item.end_time),
                    description=note,
                )
                generated += 1

        return {
            "generated_descriptions": generated,
            "description_batches": batches,
            "description_time_seconds": description_time,
        }

    def _select_story_entries(
        self,
        *,
        video_key: str,
        recent_chunk_ids: set[int],
        max_chunk_id: int,
    ) -> tuple[list[StoryMemoryEntry], list[StoryMemoryEntry]]:
        config = self.story_config
        cache = self._story_cache.get(video_key, {})
        ordered = [
            cache[idx]
            for idx in sorted(cache)
            if idx <= max_chunk_id and _normalize_text(cache[idx].description)
        ]

        compact: list[StoryMemoryEntry] = []
        previous = ""
        for entry in ordered:
            normalized = _normalize_text(entry.description)
            if not normalized:
                continue
            if previous and _jaccard(previous, entry.description) >= config.duplicate_jaccard_threshold:
                continue
            compact.append(entry)
            previous = entry.description

        recent_entries = [entry for entry in compact if entry.chunk_id in recent_chunk_ids]
        older_entries = [entry for entry in compact if entry.chunk_id not in recent_chunk_ids]

        if len(older_entries) > config.max_items:
            keep_first = min(8, config.max_items // 4)
            keep_last = config.max_items - keep_first
            older_entries = older_entries[:keep_first] + older_entries[-keep_last:]
        return older_entries, recent_entries

    def build_story_prompt(
        self,
        *,
        original_prompt: str,
        older_entries: list[StoryMemoryEntry],
        recent_entries: list[StoryMemoryEntry],
    ) -> str:
        older_lines = _format_story_lines(older_entries)
        recent_lines = _format_story_lines(recent_entries)

        older_text = "\n".join(older_lines) if older_lines else "(empty)"
        recent_text = "\n".join(recent_lines) if recent_lines else "(not available)"
        prompt = (
            "You are answering a streaming-video multiple-choice question.\n"
            "Inputs:\n"
            "1. STORY MEMORY is text-only evidence from earlier frames. It may contain useful history but can be incomplete.\n"
            "2. RECENT FRAMES are the primary visual evidence for the current moment.\n"
            "3. RECENT FRAME NOTES are auxiliary summaries of those recent frames.\n\n"
            "Decision policy:\n"
            "- For current/right-now/present visual questions, trust RECENT FRAMES first.\n"
            "- For previous/earlier/first/before/after/count/reference questions, use STORY MEMORY to recover older context.\n"
            "- Use RECENT FRAME NOTES only to clarify the recent images, not to replace the images.\n"
            "- If text memory conflicts with the visible recent frames for a current fact, prefer the recent frames.\n"
            "- Answer using the requested option/format from the question. Do not explain unless asked.\n\n"
            "STORY MEMORY BEFORE RECENT FRAMES:\n"
            f"{older_text}\n\n"
            "RECENT FRAME NOTES:\n"
            f"{recent_text}\n\n"
            "QUESTION:\n"
            f"{original_prompt}"
        )
        if len(prompt) <= self.story_config.max_prompt_chars:
            return prompt

        budget = self.story_config.max_prompt_chars
        fixed = prompt.replace(older_text, "")
        older_budget = max(1000, budget - len(fixed))
        trimmed_lines: list[str] = []
        total = 0
        for line in reversed(older_lines):
            if total + len(line) + 1 > older_budget:
                break
            trimmed_lines.append(line)
            total += len(line) + 1
        older_text = "\n".join(reversed(trimmed_lines)) if trimmed_lines else "(trimmed)"
        return (
            "You are answering a streaming-video multiple-choice question.\n"
            "Use STORY MEMORY for older events and RECENT FRAMES for present visual details.\n"
            "If they conflict on a current fact, trust RECENT FRAMES. Answer in the requested format.\n\n"
            "STORY MEMORY BEFORE RECENT FRAMES:\n"
            f"{older_text}\n\n"
            "RECENT FRAME NOTES:\n"
            f"{recent_text}\n\n"
            "QUESTION:\n"
            f"{original_prompt}"
        )


def query_recent_window(
    qa: StoryMemoryQAModel,
    video_path: str,
    prompt: str,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    video_start: float | None = None,
    video_end: float | None = None,
    cdas_config: CDASConfig | None = None,
) -> tuple[RecentWindowResult, str]:
    if not isinstance(qa, StoryMemoryQAModel):
        raise TypeError("Story memory evaluation requires StoryMemoryQAModel.")

    before_memory = _reset_gpu_memory_peaks()
    decode_t0 = time.perf_counter()
    saved_exact_recent = os.environ.pop("QWEN_EXACT_RECENT_DECODE", None)
    try:
        chunks, decode_backend = decode_video_to_chunks_qwen(
            video_path=video_path,
            chunk_duration=chunk_duration,
            fps=fps,
            recent_frames_only=None,
            video_start=0.0 if qa.story_config.full_context else video_start,
            video_end=video_end,
        )
    finally:
        if saved_exact_recent is not None:
            os.environ["QWEN_EXACT_RECENT_DECODE"] = saved_exact_recent
    decode_time = time.perf_counter() - decode_t0
    if not chunks:
        raise ValueError(f"No chunks decoded from video: {video_path}")

    selection_t0 = time.perf_counter()
    window_size = max(1, int(os.environ.get("MINICPM_STORY_RECENT_FRAMES", recent_frames_only)))
    recent_chunks = list(chunks[-window_size:])
    recent_frames = [frame for chunk in recent_chunks for frame in chunk.frames]
    final_chunk_ids = [chunk.chunk_index for chunk in recent_chunks]
    if not recent_frames:
        raise ValueError(f"No frames decoded from video: {video_path}")

    video_key = qa._story_key(video_path, fps=fps, chunk_duration=chunk_duration)
    story_stats = qa._ensure_story_entries(video_key=video_key, chunks=chunks)
    older_entries, recent_entries = qa._select_story_entries(
        video_key=video_key,
        recent_chunk_ids=set(final_chunk_ids),
        max_chunk_id=max(final_chunk_ids),
    )
    story_prompt = qa.build_story_prompt(
        original_prompt=prompt,
        older_entries=older_entries,
        recent_entries=recent_entries,
    )
    selection_time = time.perf_counter() - selection_t0

    t0 = time.perf_counter()
    answer = qa.generate_from_frames(recent_frames, story_prompt)
    _synchronize_gpu_devices()
    generate_time = time.perf_counter() - t0
    ttft_seconds = getattr(qa, "_last_ttft_seconds", 0.0) or 0.0
    num_vision_tokens = getattr(qa, "_last_num_vision_tokens", 0) or 0
    num_frames = getattr(qa, "_last_num_vision_frames", 0) or len(recent_frames)
    _synchronize_gpu_devices()
    after_memory = _capture_gpu_memory()

    profile_metadata = _build_profile(
        mode="story_memory_recent_window",
        decode_time=decode_time,
        selection_time=selection_time,
        generate_time=generate_time,
        before_memory=before_memory,
        after_memory=after_memory,
        qa=qa,
    )
    profile_metadata["decoded_chunks"] = len(chunks)
    profile_metadata["decoded_frames"] = sum(len(chunk.frames) for chunk in chunks)
    profile_metadata["video_start"] = 0.0 if qa.story_config.full_context else video_start
    profile_metadata["video_end"] = video_end
    profile_metadata["story_memory"] = {
        "recent_frames": window_size,
        "batch_size": qa.story_config.batch_size,
        "max_items": qa.story_config.max_items,
        "describe_stride": qa.story_config.describe_stride,
        "full_context": qa.story_config.full_context,
        "prompt_version": qa.story_config.prompt_version,
        "older_items_used": len(older_entries),
        "recent_items_used": len(recent_entries),
        "prompt_chars": len(story_prompt),
        "selected_chunk_ids": final_chunk_ids,
        "older_chunk_ids": [entry.chunk_id for entry in older_entries],
        "recent_chunk_ids": [entry.chunk_id for entry in recent_entries],
        **story_stats,
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
    result.story_memory_metadata = profile_metadata["story_memory"]
    return result, decode_backend
