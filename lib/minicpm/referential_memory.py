from __future__ import annotations

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


_ORDINAL_INDEX = {
    "first": 0,
    "1st": 0,
    "second": 1,
    "2nd": 1,
    "third": 2,
    "3rd": 2,
    "fourth": 3,
    "4th": 3,
    "fifth": 4,
    "5th": 4,
}
_ORDINAL_RE = re.compile(
    r"\b(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th)\s+question\b",
    re.IGNORECASE,
)
_PREVIOUS_RE = re.compile(r"\b(previous|last)\s+question\b", re.IGNORECASE)
_MENTION_RE = re.compile(
    r"\b("
    r"mentioned|same person|same man|same woman|same object|same item|"
    r"that person|that man|that woman|that object|that item|"
    r"he\b|she\b|him\b|her\b|it\b|they\b|them\b"
    r")",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z][a-z0-9_-]*", re.IGNORECASE)
_REFERENCE_STOPWORDS = {
    "about",
    "after",
    "answer",
    "are",
    "before",
    "best",
    "current",
    "currently",
    "does",
    "doing",
    "first",
    "from",
    "have",
    "how",
    "last",
    "mentioned",
    "now",
    "option",
    "previous",
    "question",
    "right",
    "same",
    "second",
    "that",
    "the",
    "third",
    "this",
    "video",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
}


@dataclass
class ReferentialMemoryEntry:
    question_index: int
    question: str
    response: str | None
    task_type: str
    time_stamp: str
    selected_chunk_ids: list[int]
    selected_timestamps: list[float]


@dataclass
class ReferentialSelection:
    frames: list[Image.Image]
    current_frames: list[Image.Image]
    reference_frames: list[Image.Image]
    final_chunk_ids: list[int]
    metadata: dict[str, Any]


def _clean_question_text(prompt_or_question: str) -> str:
    text = str(prompt_or_question)
    match = re.search(r"Question:\s*(.*?)\n\s*Options:", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _token_set(text: str) -> set[str]:
    tokens = set()
    for token in _TOKEN_RE.findall(text.lower()):
        if len(token) < 3 or token in _REFERENCE_STOPWORDS:
            continue
        tokens.add(token[:-1] if len(token) > 3 and token.endswith("s") else token)
    return tokens


def _reference_gate(question_text: str, memory: list[ReferentialMemoryEntry]) -> tuple[bool, dict[str, Any]]:
    text = question_text.lower()
    if not memory:
        return False, {"activated": False, "reason": "no_previous_question_memory"}

    ordinal = _ORDINAL_RE.search(text)
    if ordinal:
        target = _ORDINAL_INDEX.get(ordinal.group(1).lower())
        if target is not None and target < len(memory):
            return True, {
                "activated": True,
                "reason": "explicit_ordinal_question_reference",
                "target_question_index": int(target),
            }
        return False, {
            "activated": False,
            "reason": "ordinal_reference_out_of_range",
            "requested": ordinal.group(1).lower(),
        }

    if _PREVIOUS_RE.search(text):
        return True, {
            "activated": True,
            "reason": "explicit_previous_question_reference",
            "target_question_index": len(memory) - 1,
        }

    if _MENTION_RE.search(text):
        return True, {
            "activated": True,
            "reason": "implicit_referential_phrase",
            "target_question_index": None,
        }

    return False, {"activated": False, "reason": "no_referential_phrase"}


def _rank_reference_candidates(
    question_text: str,
    memory: list[ReferentialMemoryEntry],
    target_question_index: int | None,
) -> tuple[ReferentialMemoryEntry | None, list[dict[str, Any]]]:
    if not memory:
        return None, []
    if target_question_index is not None:
        if 0 <= target_question_index < len(memory):
            entry = memory[target_question_index]
            return entry, [
                {
                    "question_index": entry.question_index,
                    "score": 1.0,
                    "selected": True,
                    "reason": "explicit_index",
                    "question": entry.question,
                    "response": entry.response,
                }
            ]
        return None, []

    query_tokens = _token_set(question_text)
    candidates: list[dict[str, Any]] = []
    for age, entry in enumerate(reversed(memory), start=1):
        entry_tokens = _token_set(f"{entry.question} {entry.response or ''}")
        overlap = len(query_tokens & entry_tokens)
        union = len(query_tokens | entry_tokens) or 1
        lexical_score = overlap / union
        recency_score = 1.0 / age
        score = 0.70 * lexical_score + 0.30 * recency_score
        candidates.append(
            {
                "question_index": entry.question_index,
                "score": float(score),
                "lexical_overlap": int(overlap),
                "recency_rank": int(age),
                "question": entry.question,
                "response": entry.response,
                "selected": False,
            }
        )
    candidates.sort(key=lambda item: (-float(item["score"]), -int(item["question_index"])))
    if not candidates:
        return None, []
    candidates[0]["selected"] = True
    selected_index = int(candidates[0]["question_index"])
    selected = next((entry for entry in memory if entry.question_index == selected_index), None)
    return selected, candidates[:5]


def _evenly_pick(values: list[float], count: int) -> list[float]:
    if count <= 0 or not values:
        return []
    unique = sorted({float(value) for value in values})
    if count >= len(unique):
        return unique
    if count == 1:
        return [unique[len(unique) // 2]]
    indices = sorted({round(i * (len(unique) - 1) / (count - 1)) for i in range(count)})
    return [unique[index] for index in indices]


def _decode_reference_frames(
    video_path: str,
    timestamps: list[float],
    *,
    chunk_duration: float,
    fps: float,
) -> tuple[list[Image.Image], list[int], list[float], list[str]]:
    frames: list[Image.Image] = []
    chunk_ids: list[int] = []
    selected_timestamps: list[float] = []
    backends: list[str] = []
    for ts in timestamps:
        start = max(0.0, float(ts) - max(0.50, 0.5 * float(chunk_duration)))
        end = float(ts) + max(0.50, 0.5 * float(chunk_duration))
        chunks, backend = decode_video_to_chunks_qwen(
            video_path=video_path,
            chunk_duration=chunk_duration,
            fps=fps,
            recent_frames_only=1,
            video_start=start,
            video_end=end,
        )
        if not chunks:
            continue
        chunk = chunks[-1]
        if not chunk.frames:
            continue
        frames.append(chunk.frames[-1])
        chunk_ids.append(int(chunk.chunk_index))
        selected_timestamps.append(float(chunk.frame_timestamps[-1] if chunk.frame_timestamps else ts))
        backends.append(str(backend))
    return frames, chunk_ids, selected_timestamps, backends


def _build_referential_prompt(
    original_prompt: str,
    *,
    reference_entry: ReferentialMemoryEntry | None,
    reference_count: int,
    current_count: int,
) -> str:
    if reference_entry is None or reference_count <= 0:
        return original_prompt
    return (
        "You are given video frames in two groups. "
        f"The first {reference_count} frame(s) are REFERENCE frames from an earlier question. "
        f"The last {current_count} frame(s) are CURRENT recent frames for the current timestamp. "
        "Use the reference frames only to resolve phrases like 'the person/object mentioned in the previous question'. "
        "When the question asks about the current state, answer using the current frames.\n\n"
        f"Referenced previous question: {reference_entry.question}\n"
        f"Previous model answer: {reference_entry.response or 'N/A'}\n\n"
        f"{original_prompt}"
    )


def select_referential_frames(
    *,
    video_path: str,
    prompt: str,
    question_text: str,
    memory: list[ReferentialMemoryEntry],
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    reference_frames: int,
    video_start: float | None,
    video_end: float | None,
) -> tuple[ReferentialSelection, str]:
    chunks, decode_backend = decode_video_to_chunks_qwen(
        video_path=video_path,
        chunk_duration=chunk_duration,
        fps=fps,
        recent_frames_only=recent_frames_only,
        video_start=video_start,
        video_end=video_end,
    )
    if not chunks:
        raise ValueError(f"No chunks decoded from video: {video_path}")

    current_chunks = chunks[-max(1, int(recent_frames_only)) :]
    current_frames = [frame for chunk in current_chunks for frame in chunk.frames]
    current_chunk_ids = [int(chunk.chunk_index) for chunk in current_chunks for _frame in chunk.frames]
    current_timestamps = [float(ts) for chunk in current_chunks for ts in chunk.frame_timestamps]
    if not current_frames:
        raise ValueError(f"No current frames decoded from video: {video_path}")

    gate_active, gate = _reference_gate(question_text, memory)
    reference_entry: ReferentialMemoryEntry | None = None
    reference_candidate_scores: list[dict[str, Any]] = []
    decoded_reference_frames: list[Image.Image] = []
    reference_chunk_ids: list[int] = []
    reference_timestamps: list[float] = []
    reference_backends: list[str] = []
    if gate_active:
        reference_entry, reference_candidate_scores = _rank_reference_candidates(
            question_text,
            memory,
            gate.get("target_question_index"),
        )
        if reference_entry is not None:
            target_timestamps = _evenly_pick(reference_entry.selected_timestamps, int(reference_frames))
            (
                decoded_reference_frames,
                reference_chunk_ids,
                reference_timestamps,
                reference_backends,
            ) = _decode_reference_frames(
                video_path,
                target_timestamps,
                chunk_duration=chunk_duration,
                fps=fps,
            )

    all_frames = [*decoded_reference_frames, *current_frames]
    final_chunk_ids = [*reference_chunk_ids, *current_chunk_ids]
    metadata = {
        "mode": "referential_memory",
        "memory_triggered": bool(gate_active and decoded_reference_frames),
        "memory_gate": gate,
        "memory_selector": "previous_question_pointer",
        "reference_question": (
            {
                "question_index": reference_entry.question_index,
                "question": reference_entry.question,
                "response": reference_entry.response,
                "task_type": reference_entry.task_type,
                "time_stamp": reference_entry.time_stamp,
                "stored_selected_chunk_ids": reference_entry.selected_chunk_ids,
                "stored_selected_timestamps": reference_entry.selected_timestamps,
            }
            if reference_entry is not None
            else None
        ),
        "reference_candidate_scores": reference_candidate_scores,
        "reference_frames": len(decoded_reference_frames),
        "reference_chunk_ids": reference_chunk_ids,
        "reference_timestamps": reference_timestamps,
        "reference_decode_backends": reference_backends,
        "current_frames": len(current_frames),
        "current_chunk_ids": current_chunk_ids,
        "current_timestamps": current_timestamps,
        "selected_frames": len(all_frames),
        "selected_chunk_ids": final_chunk_ids,
        "selected_timestamps": [*reference_timestamps, *current_timestamps],
        "memory_size_before": len(memory),
        "prompt_rewritten": bool(reference_entry is not None and decoded_reference_frames),
    }
    return (
        ReferentialSelection(
            frames=all_frames,
            current_frames=current_frames,
            reference_frames=decoded_reference_frames,
            final_chunk_ids=final_chunk_ids,
            metadata=metadata,
        ),
        decode_backend,
    )


def query_referential_memory_window(
    *,
    qa: RecentWindowQAModel,
    video_path: str,
    prompt: str,
    question_text: str,
    memory: list[ReferentialMemoryEntry],
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    reference_frames: int,
    video_start: float | None = None,
    video_end: float | None = None,
    cdas_config: CDASConfig | None = None,
) -> tuple[RecentWindowResult, str, ReferentialSelection]:
    del cdas_config
    before_memory = _reset_gpu_memory_peaks()

    decode_t0 = time.perf_counter()
    selection, decode_backend = select_referential_frames(
        video_path=video_path,
        prompt=prompt,
        question_text=question_text,
        memory=memory,
        chunk_duration=chunk_duration,
        fps=fps,
        recent_frames_only=recent_frames_only,
        reference_frames=reference_frames,
        video_start=video_start,
        video_end=video_end,
    )
    decode_time = time.perf_counter() - decode_t0

    selection_t0 = time.perf_counter()
    reference_info = selection.metadata.get("reference_question")
    if isinstance(reference_info, dict) and selection.reference_frames:
        entry = ReferentialMemoryEntry(
            question_index=int(reference_info["question_index"]),
            question=str(reference_info["question"]),
            response=reference_info.get("response"),
            task_type=str(reference_info.get("task_type", "")),
            time_stamp=str(reference_info.get("time_stamp", "")),
            selected_chunk_ids=[int(value) for value in reference_info.get("stored_selected_chunk_ids", [])],
            selected_timestamps=[float(value) for value in reference_info.get("stored_selected_timestamps", [])],
        )
        rewritten_prompt = _build_referential_prompt(
            prompt,
            reference_entry=entry,
            reference_count=len(selection.reference_frames),
            current_count=len(selection.current_frames),
        )
    else:
        rewritten_prompt = prompt
    selection_time = time.perf_counter() - selection_t0

    t0 = time.perf_counter()
    answer = qa.generate_from_frames(selection.frames, rewritten_prompt)
    _synchronize_gpu_devices()
    generate_time = time.perf_counter() - t0
    ttft_seconds = getattr(qa, "_last_ttft_seconds", 0.0) or 0.0
    num_vision_tokens = getattr(qa, "_last_num_vision_tokens", 0) or 0
    num_frames = getattr(qa, "_last_num_vision_frames", 0) or len(selection.frames)

    _synchronize_gpu_devices()
    after_memory = _capture_gpu_memory()
    profile_metadata = _build_profile(
        mode="referential_memory",
        decode_time=decode_time,
        selection_time=selection_time,
        generate_time=generate_time,
        before_memory=before_memory,
        after_memory=after_memory,
        qa=qa,
    )
    profile_metadata["referential_memory"] = selection.metadata
    profile_metadata["decoded_frames"] = len(selection.frames)
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
    result.referential_memory_metadata = selection.metadata
    return result, decode_backend, selection


def make_memory_entry(
    *,
    question_index: int,
    question_text: str,
    response: str | None,
    task_type: str,
    time_stamp: str,
    selection: ReferentialSelection,
) -> ReferentialMemoryEntry:
    return ReferentialMemoryEntry(
        question_index=int(question_index),
        question=str(question_text),
        response=response,
        task_type=str(task_type),
        time_stamp=str(time_stamp),
        selected_chunk_ids=[int(value) for value in selection.metadata.get("current_chunk_ids", [])],
        selected_timestamps=[float(value) for value in selection.metadata.get("current_timestamps", [])],
    )
