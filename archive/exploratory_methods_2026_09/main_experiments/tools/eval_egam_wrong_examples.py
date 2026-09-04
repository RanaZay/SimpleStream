#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from lib.minicpm.adaptive import query_recent_window as query_semantic_window  # noqa: E402
from lib.minicpm.hybrid_recent_clip import attach_hybrid_metadata, build_clip_prompt  # noqa: E402
from lib.minicpm.recent_clip import RecentClipQAModel, query_recent_clip  # noqa: E402
from lib.shared.recent_window import extract_mcq_answer  # noqa: E402
from main_experiments.minicpm_v46.streamingbench.eval_baseline import (  # noqa: E402
    build_prompt,
    timestamp_to_seconds,
)
from main_experiments.tools.eval_recent_clip_wrong_examples import (  # noqa: E402
    _parse_options,
    _profile_fields,
    _resolve_video_path,
)
from main_experiments.tools.eval_semantic_hybrid_clip_wrong_examples import (  # noqa: E402
    _load_records,
    _query_recent6_direct,
)


PRESENT_STRONG = (
    "right now",
    "currently",
    "at this moment",
    "visible now",
    "now wearing",
    "now holding",
    "current outfit",
)
PRESENT_WEAK = (
    "what color",
    "what colors",
    "which color",
    "where is",
    "where are",
    "what is mounted",
    "what is written",
    "what text",
    "which street",
    "visible",
)
LOCAL_TEMPORAL_TERMS = (
    "just now",
    "last stroke",
    "last strokes",
    "last action",
    "latest action",
    "most recent",
    "immediately before",
    "immediately after",
    "during the last",
)
ACTION_TERMS = (
    "doing",
    "performing",
    "action",
    "move",
    "moving",
    "turning",
    "pass",
    "hit",
    "stroke",
)
HISTORY_STRONG = (
    "earlier",
    "previously",
    "at the beginning",
    "before this",
    "throughout the video",
    "so far",
)
HISTORY_WEAK = (
    "before",
    "after",
    "next",
    "later",
    "already",
)
REFERENTIAL_TERMS = (
    "previous question",
    "mentioned in the previous",
    "mentioned before",
    "same person",
    "same object",
    "that person",
    "that object",
)
COUNT_TERMS = (
    "how many",
    "number of times",
    "in total",
    "how often",
    "count",
)


def _text(question: str) -> str:
    return " ".join(str(question).lower().split())


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def score_evidence_routes(question: str) -> dict[str, float]:
    text = _text(question)
    return {
        "referential": 1.4 if _has_any(text, REFERENTIAL_TERMS) else 0.0,
        "counting": 1.2 if _has_any(text, COUNT_TERMS) else 0.0,
        "present": (
            (1.0 if _has_any(text, PRESENT_STRONG) else 0.0)
            + (0.4 if _has_any(text, PRESENT_WEAK) else 0.0)
        ),
        "motion": (
            (1.0 if _has_any(text, LOCAL_TEMPORAL_TERMS) else 0.0)
            + (0.7 if _has_any(text, ACTION_TERMS) else 0.0)
            + (0.5 if "just" in text or "latest" in text or "last" in text else 0.0)
        ),
        "history": (
            (1.0 if _has_any(text, HISTORY_STRONG) else 0.0)
            + (0.6 if _has_any(text, HISTORY_WEAK) else 0.0)
            + (0.4 if "what might" in text or "most likely" in text else 0.0)
        ),
    }


def choose_egam_route(
    question: str,
    *,
    route_threshold: float,
    history_threshold: float,
    motion_threshold: float,
) -> tuple[str, str, dict[str, float]]:
    scores = score_evidence_routes(question)
    text = _text(question)

    if scores["referential"] >= route_threshold:
        return "recent6_frames", "referential_unresolved_fallback", scores
    if scores["counting"] >= route_threshold:
        return "recent6_frames", "counting_counter_untrusted_fallback", scores
    if scores["present"] >= route_threshold and scores["history"] < history_threshold:
        return "recent6_frames", "explicit_present_gate", scores
    if scores["motion"] >= motion_threshold and scores["present"] < 1.0:
        return "recent_clip_k2", "local_motion_gate", scores
    if scores["motion"] >= motion_threshold and _has_any(text, LOCAL_TEMPORAL_TERMS):
        return "recent_clip_k2", "local_motion_gate", scores
    if scores["history"] >= history_threshold:
        return "semantic_memory_recent6_m3", "history_semantic_gate", scores
    if max(scores.values()) < route_threshold:
        return "recent6_frames", "low_confidence_recent6_fallback", scores
    return "recent6_frames", "conservative_recent6_fallback", scores


def _write_record(
    *,
    out,
    index: int,
    source: dict[str, Any],
    video_path: Path,
    route: str,
    route_reason: str,
    route_scores: dict[str, float],
    result,
    backend: str,
    answer_gt: str,
    response: str,
    pred: str | None,
    correct: bool,
) -> dict[str, Any]:
    final_chunk_ids = list(getattr(result, "final_chunk_ids", []) or [])
    record: dict[str, Any] = {
        "_index": int(source.get("_index", index)),
        "_key": source.get("_key", f"{video_path.name}_{index}"),
        "video": video_path.name,
        "video_path": str(video_path),
        "task_type": source.get("task_type", ""),
        "time_stamp": source.get("time_stamp", ""),
        "question": source.get("question", ""),
        "options": _parse_options(source.get("options")),
        "answer_gt": answer_gt,
        "source_response": source.get("response"),
        "source_correct": source.get("correct"),
        "response": response,
        "pred": pred,
        "correct": correct,
        "decode_backend": backend,
        "final_chunk_ids": final_chunk_ids,
        "generate_time": getattr(result, "generate_time", None),
        "ttft_seconds": getattr(result, "ttft_seconds", None),
        "num_vision_tokens": getattr(result, "num_vision_tokens", None),
        "num_vision_tokens_before": getattr(result, "num_vision_tokens_before", None),
        "num_vision_tokens_after": getattr(result, "num_vision_tokens_after", None),
        "num_frames": getattr(result, "num_frames", None),
        "egam": {
            "route": route,
            "route_reason": route_reason,
            "route_scores": route_scores,
        },
        "hybrid_recent_clip": {
            "route": route,
            "route_reason": route_reason,
            "route_scores": route_scores,
            "mode": "egam_wrong_examples",
        },
        "adaptive": {
            "mode": "egam_semantic_clip",
            "memory_triggered": route == "semantic_memory_recent6_m3",
            "route": route,
            "route_reason": route_reason,
            "route_scores": route_scores,
            "recent_chunk_ids": final_chunk_ids,
            "memory_chunk_ids": [],
            "selected_chunk_ids": final_chunk_ids,
            "selected_timestamps": [float(x) for x in final_chunk_ids],
        },
    }
    _profile_fields(record, getattr(result, "profile_metadata", None))
    adaptive_meta = getattr(result, "adaptive_metadata", None)
    if isinstance(adaptive_meta, dict):
        memory_ids = list(adaptive_meta.get("memory_chunk_ids", []) or [])
        recent_ids = list(adaptive_meta.get("recent_chunk_ids", []) or final_chunk_ids)
        selected_ids = list(adaptive_meta.get("selected_chunk_ids", []) or final_chunk_ids)
        record["adaptive"].update(
            {
                "memory_triggered": bool(memory_ids),
                "memory_selector": adaptive_meta.get("memory_selector"),
                "memory_chunk_ids": memory_ids,
                "recent_chunk_ids": recent_ids,
                "selected_chunk_ids": selected_ids,
                "memory_scores": adaptive_meta.get("memory_scores", []),
            }
        )
    out.write(json.dumps(record, ensure_ascii=False) + "\n")
    out.flush()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Evidence-Gated Adaptive Memory on the 30 StreamingBench wrong examples."
    )
    parser.add_argument(
        "--examples-jsonl",
        type=Path,
        default=Path("reports/semantic_memory_streamingbench_wrong30_pasted/results_incremental.jsonl"),
    )
    parser.add_argument("--video-dir", type=Path, default=Path("data/streamingbench/videos"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/egam_wrong30_results"))
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    parser.add_argument("--context-seconds", type=float, default=60.0)
    parser.add_argument("--recent-frames", type=int, default=6)
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--route-threshold", type=float, default=1.0)
    parser.add_argument("--history-threshold", type=float, default=1.0)
    parser.add_argument("--motion-threshold", type=float, default=1.2)
    parser.add_argument("--qa-model", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--qa-device", default="auto")
    parser.add_argument("--max-qa-tokens", type=int, default=256)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--keep-clips", action="store_true")
    args = parser.parse_args()

    os.environ["MINICPM_ADAPTIVE_MODE"] = "semantic_memory"
    os.environ["MINICPM_ADAPTIVE_MIN_WINDOW"] = str(args.recent_frames)
    os.environ["MINICPM_ADAPTIVE_MID_WINDOW"] = str(args.recent_frames)
    os.environ["MINICPM_ADAPTIVE_MAX_WINDOW"] = str(args.recent_frames)
    os.environ["MINICPM_ADAPTIVE_MEMORY_ANCHORS"] = os.environ.get("MINICPM_ADAPTIVE_MEMORY_ANCHORS", "3")
    os.environ["MINICPM_ADAPTIVE_MEMORY_SEARCH_CHUNKS"] = os.environ.get(
        "MINICPM_ADAPTIVE_MEMORY_SEARCH_CHUNKS",
        "32",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clip_dir = args.output_dir / "clips"
    if args.keep_clips:
        clip_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(args.examples_jsonl, args.max_examples)
    qa = RecentClipQAModel(
        model_name=args.qa_model,
        device=args.qa_device,
        max_new_tokens=args.max_qa_tokens,
        attn_implementation=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
    )

    out_path = args.output_dir / "results_incremental.jsonl"
    correct = 0
    route_counts: dict[str, int] = {}
    route_correct: dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as out:
        for index, source in enumerate(records, start=1):
            video_path = _resolve_video_path(source, args.video_dir)
            options = _parse_options(source.get("options"))
            question = str(source.get("question", ""))
            answer_gt = str(source.get("answer_gt") or source.get("answer") or "").strip()
            time_seconds = float(timestamp_to_seconds(str(source.get("time_stamp", "0:00:00"))))
            route, route_reason, route_scores = choose_egam_route(
                question,
                route_threshold=args.route_threshold,
                history_threshold=args.history_threshold,
                motion_threshold=args.motion_threshold,
            )

            try:
                if route == "recent_clip_k2":
                    keep_clip_path = None
                    if args.keep_clips:
                        keep_clip_path = clip_dir / f"{index:02d}_{video_path.stem}_k{args.clip_seconds:g}.mp4"
                    prompt = build_clip_prompt(question, options)
                    result, backend = query_recent_clip(
                        qa,
                        video_path=str(video_path),
                        prompt=prompt,
                        question_time_seconds=time_seconds,
                        clip_seconds=args.clip_seconds,
                        keep_clip_path=keep_clip_path,
                    )
                    attach_hybrid_metadata(
                        result,
                        route="recent_clip",
                        route_reason=route_reason,
                        clip_seconds=args.clip_seconds,
                    )
                elif route == "semantic_memory_recent6_m3":
                    prompt = build_prompt({"question": question, "options": options})
                    result, backend = query_semantic_window(
                        qa,
                        video_path=str(video_path),
                        prompt=prompt,
                        chunk_duration=args.chunk_duration,
                        fps=args.fps,
                        recent_frames_only=args.recent_frames,
                        video_start=max(0.0, time_seconds - args.context_seconds),
                        video_end=time_seconds,
                    )
                    attach_hybrid_metadata(
                        result,
                        route="semantic_memory",
                        route_reason=route_reason,
                        clip_seconds=args.clip_seconds,
                    )
                else:
                    prompt = build_prompt({"question": question, "options": options})
                    result, backend = _query_recent6_direct(
                        qa,
                        video_path=video_path,
                        prompt=prompt,
                        question_time_seconds=time_seconds,
                        recent_frames=args.recent_frames,
                        output_dir=args.output_dir,
                    )
                    attach_hybrid_metadata(
                        result,
                        route="recent6_frames",
                        route_reason=route_reason,
                        clip_seconds=args.clip_seconds,
                    )

                response = result.answer
                pred = extract_mcq_answer(response)
                is_correct = bool(pred and answer_gt and pred == answer_gt)
                record = _write_record(
                    out=out,
                    index=index,
                    source=source,
                    video_path=video_path,
                    route=route,
                    route_reason=route_reason,
                    route_scores=route_scores,
                    result=result,
                    backend=backend,
                    answer_gt=answer_gt,
                    response=response,
                    pred=pred,
                    correct=is_correct,
                )
            except Exception as exc:
                is_correct = False
                record = {
                    "_index": int(source.get("_index", index)),
                    "_key": source.get("_key", f"{video_path.name}_{index}"),
                    "video": video_path.name,
                    "video_path": str(video_path),
                    "task_type": source.get("task_type", ""),
                    "time_stamp": source.get("time_stamp", ""),
                    "question": question,
                    "options": options,
                    "answer_gt": answer_gt,
                    "source_response": source.get("response"),
                    "source_correct": source.get("correct"),
                    "response": None,
                    "pred": None,
                    "correct": False,
                    "error": repr(exc),
                    "egam": {
                        "route": route,
                        "route_reason": route_reason,
                        "route_scores": route_scores,
                    },
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()

            correct += int(is_correct)
            route_counts[route] = route_counts.get(route, 0) + 1
            route_correct[route] = route_correct.get(route, 0) + int(is_correct)
            print(
                f"[{index}/{len(records)}] {route} ({route_reason}) "
                f"{record.get('task_type')} -> {record.get('response')} "
                f"(pred={record.get('pred')} gt={answer_gt} correct={record.get('correct')}) "
                f"scores={route_scores}",
                flush=True,
            )

    total = len(records)
    print("=" * 80)
    print(f"EGAM wrong-set accuracy: {100.0 * correct / total if total else 0.0:.2f}% ({correct}/{total})")
    for route in sorted(route_counts):
        n = route_counts[route]
        c = route_correct.get(route, 0)
        print(f"  {route}: {100.0 * c / n if n else 0.0:.2f}% ({c}/{n})")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
