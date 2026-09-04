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
from main_experiments.tools.eval_egam_wrong_examples import (  # noqa: E402
    _has_any,
    _text,
    _write_record,
    score_evidence_routes,
)
from main_experiments.tools.eval_recent_clip_wrong_examples import (  # noqa: E402
    _parse_options,
    _resolve_video_path,
)
from main_experiments.tools.eval_semantic_hybrid_clip_wrong_examples import (  # noqa: E402
    _load_records,
    _query_recent6_direct,
)


STATIC_PRESENT_TERMS = (
    "right now",
    "currently",
    "at this moment",
    "visible now",
    "what color",
    "which color",
    "where is",
    "where are",
    "wearing",
    "holding",
    "mounted",
    "written",
    "text",
    "sign",
)
ACTION_OR_TRANSITION_TERMS = (
    "doing",
    "performing",
    "action",
    "moving",
    "move",
    "turn",
    "turning",
    "pass",
    "passed",
    "hit",
    "stroke",
    "serve",
    "throw",
    "grab",
    "put",
    "pick",
)
IMMEDIATE_TERMS = (
    "just now",
    "last",
    "latest",
    "most recent",
    "immediately",
)
HISTORY_TERMS = (
    "earlier",
    "previously",
    "previous",
    "before",
    "after",
    "so far",
    "in total",
    "throughout",
    "at the beginning",
    "mentioned",
)
COUNT_TERMS = (
    "how many",
    "number of times",
    "in total",
    "count",
    "how often",
)


def choose_egam_v2_route(question: str) -> tuple[str, str, dict[str, float]]:
    """Less conservative EGAM routing.

    V1 asked "does the question explicitly need history?" and otherwise kept
    Recent-6. V2 asks "is Recent-6 clearly enough?" If not, it escalates to an
    evidence branch. This is a deterministic proxy for the confidence-gated
    policy until option-logit confidence is available.
    """
    text = _text(question)
    scores = score_evidence_routes(question)
    is_static_present = _has_any(text, STATIC_PRESENT_TERMS)
    is_motion = _has_any(text, ACTION_OR_TRANSITION_TERMS) or _has_any(text, IMMEDIATE_TERMS)
    is_immediate_motion = is_motion and _has_any(text, IMMEDIATE_TERMS)
    is_history = _has_any(text, HISTORY_TERMS)
    is_count = _has_any(text, COUNT_TERMS)

    scores.update(
        {
            "static_present_proxy": 1.0 if is_static_present else 0.0,
            "immediate_motion_proxy": 1.0 if is_immediate_motion else 0.0,
            "history_proxy": 1.0 if is_history else 0.0,
            "count_proxy": 1.0 if is_count else 0.0,
        }
    )

    if is_count:
        return "semantic_memory_recent6_m3", "counting_escalates_to_semantic", scores
    if is_history:
        return "semantic_memory_recent6_m3", "history_or_reference_escalates_to_semantic", scores
    if is_immediate_motion:
        return "recent_clip_k2", "immediate_motion_clip", scores
    if is_motion and not is_static_present:
        return "recent_clip_k2", "action_without_static_present_clip", scores
    if is_static_present and not is_motion:
        return "recent6_frames", "clear_static_present_recent6", scores
    return "semantic_memory_recent6_m3", "ambiguous_escalates_to_semantic", scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EGAM-v2 confidence-triggered routing on StreamingBench wrong examples."
    )
    parser.add_argument(
        "--examples-jsonl",
        type=Path,
        default=Path("reports/semantic_memory_streamingbench_wrong30_pasted/results_incremental.jsonl"),
    )
    parser.add_argument("--video-dir", type=Path, default=Path("data/streamingbench/videos"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/egam_v2_wrong30_results"))
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    parser.add_argument("--context-seconds", type=float, default=60.0)
    parser.add_argument("--recent-frames", type=int, default=6)
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
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
            route, route_reason, route_scores = choose_egam_v2_route(question)

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
    print(f"EGAM-v2 wrong-set accuracy: {100.0 * correct / total if total else 0.0:.2f}% ({correct}/{total})")
    for route in sorted(route_counts):
        n = route_counts[route]
        c = route_correct.get(route, 0)
        print(f"  {route}: {100.0 * c / n if n else 0.0:.2f}% ({c}/{n})")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
