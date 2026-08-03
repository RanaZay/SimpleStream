from __future__ import annotations

from typing import Any


CURRENT_CLIP_TERMS = (
    "last stroke",
    "last strokes",
    "latest",
    "just now",
    "actions just now",
    "action just now",
    "during the last",
    "last action",
    "latest action",
    "currently doing",
    "doing right now",
)

FRAME_PREFERRED_TERMS = (
    "what color",
    "which color",
    "what colors",
    "what is written",
    "what text",
    "which street",
    "where is",
    "where are",
    "which way",
    "what is mounted",
    "what is the food",
    "what is the person grabbing",
)

CUMULATIVE_TERMS = (
    "how many",
    "in total",
    "so far",
    "previous question",
    "mentioned in the previous",
    "mentioned before",
)


def choose_hybrid_route(question: str) -> tuple[str, str]:
    """Route a StreamingBench question to Recent-6 frames or a short video clip.

    This is intentionally conservative:
    - visual detail/current object questions stay on still frames;
    - cumulative/reference questions stay off the clip path for now;
    - only explicit immediate-motion/final-moment questions use the short clip.
    """

    text = " ".join(str(question).lower().split())
    if any(term in text for term in CUMULATIVE_TERMS):
        return "recent6_frames", "cumulative_or_referential_guard"
    if any(term in text for term in FRAME_PREFERRED_TERMS):
        return "recent6_frames", "visual_detail_guard"
    if any(term in text for term in CURRENT_CLIP_TERMS):
        return "recent_clip", "immediate_temporal_motion"
    return "recent6_frames", "default_recent6"


def format_options(options: list[str]) -> str:
    formatted: list[str] = []
    for index, option in enumerate(options):
        text = str(option).strip()
        if not text.startswith(("A.", "B.", "C.", "D.")):
            text = f"{chr(65 + index)}. {text}"
        formatted.append(text)
    return "\n".join(formatted)


def build_clip_prompt(question: str, options: list[str]) -> str:
    return (
        "You are an advanced video question-answering AI assistant.\n"
        "You are given a short video clip immediately before the question timestamp and a multiple-choice question.\n"
        "Use the temporal order inside the clip. If the question says \"right now\", \"just now\", "
        "\"currently\", \"last\", or \"latest\", focus most on the final moments of the clip.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{format_options(options)}\n\n"
        "Only give the best option's letter (A, B, C, or D) directly."
    )


def attach_hybrid_metadata(
    result: Any,
    *,
    route: str,
    route_reason: str,
    clip_seconds: float,
) -> None:
    final_chunk_ids = list(getattr(result, "final_chunk_ids", []) or [])
    result.hybrid_recent_clip_metadata = {
        "route": route,
        "route_reason": route_reason,
        "clip_seconds": float(clip_seconds) if route == "recent_clip" else 0.0,
        "recent_chunk_ids": final_chunk_ids if route == "recent6_frames" else [],
        "clip_chunk_ids": final_chunk_ids if route == "recent_clip" else [],
        "selected_chunk_ids": final_chunk_ids,
    }
