#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from lib.minicpm.adaptive import query_recent_window as query_semantic_window  # noqa: E402
from lib.minicpm.hybrid_recent_clip import (  # noqa: E402
    attach_hybrid_metadata,
    build_clip_prompt,
    choose_hybrid_route,
)
from lib.minicpm.event_count_memory import query_event_count_memory  # noqa: E402
from lib.minicpm.recent_clip import RecentClipQAModel, query_recent_clip  # noqa: E402
from lib.shared.recent_window import RecentWindowResult  # noqa: E402
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

COUNT_MEMORY_TERMS = (
    "how many",
    "how many times",
    "in total",
    "so far",
    "number of times",
)


def _load_records(path: Path, max_examples: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if max_examples > 0:
        records = records[:max_examples]
    return records


def _extract_recent_frames_direct(
    video_path: Path,
    *,
    question_time_seconds: float,
    recent_frames: int,
    output_dir: Path,
) -> tuple[list[Image.Image], list[int], float]:
    start = max(0, int(round(question_time_seconds)) - recent_frames + 1)
    timestamps = list(range(start, start + recent_frames))
    frames: list[Image.Image] = []
    decode_start = time.perf_counter()
    with tempfile.TemporaryDirectory(dir=str(output_dir)) as tmp:
        tmp_dir = Path(tmp)
        for idx, ts in enumerate(timestamps):
            frame_path = tmp_dir / f"frame_{idx:02d}.jpg"
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                str(max(0, ts)),
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                "-y",
                str(frame_path),
            ]
            subprocess.run(cmd, check=True)
            frames.append(Image.open(frame_path).convert("RGB"))
    decode_time = time.perf_counter() - decode_start
    return frames, timestamps, decode_time


def _query_recent6_direct(
    qa: RecentClipQAModel,
    *,
    video_path: Path,
    prompt: str,
    question_time_seconds: float,
    recent_frames: int,
    output_dir: Path,
) -> tuple[RecentWindowResult, str]:
    frames, timestamps, decode_time = _extract_recent_frames_direct(
        video_path,
        question_time_seconds=question_time_seconds,
        recent_frames=recent_frames,
        output_dir=output_dir,
    )
    generate_start = time.perf_counter()
    answer = qa.generate_from_frames(frames, prompt)
    generate_time = time.perf_counter() - generate_start
    profile_metadata = {
        "mode": "recent6_direct_ffmpeg",
        "decode_time_seconds": decode_time,
        "selection_time_seconds": 0.0,
        "generate_time_seconds": generate_time,
        "end_to_end_time_seconds": decode_time + generate_time,
        "model_generate_time_seconds": getattr(qa, "_last_model_generate_seconds", generate_time),
        "preprocess_time_seconds": getattr(qa, "_last_preprocess_seconds", 0.0),
        "ttft_seconds": getattr(qa, "_last_ttft_seconds", 0.0) or 0.0,
        "generate_first_token_time_ms": (getattr(qa, "_last_ttft_seconds", 0.0) or 0.0) * 1000.0,
        "model_generate_time_ms": getattr(qa, "_last_model_generate_seconds", generate_time) * 1000.0,
    }
    num_vision_tokens = getattr(qa, "_last_num_vision_tokens", 0) or len(frames) * 66
    result = RecentWindowResult(
        answer=answer,
        final_chunk_ids=timestamps,
        generate_time=generate_time,
        ttft_seconds=getattr(qa, "_last_ttft_seconds", 0.0) or 0.0,
        num_vision_tokens=num_vision_tokens,
        num_vision_tokens_before=num_vision_tokens,
        num_vision_tokens_after=num_vision_tokens,
        num_frames=len(frames),
    )
    result.profile_metadata = profile_metadata
    return result, "ffmpeg_direct"


def _write_record(
    *,
    out,
    index: int,
    source: dict[str, Any],
    video_path: Path,
    route: str,
    route_reason: str,
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
        "hybrid_recent_clip": {
            "route": route,
            "route_reason": route_reason,
        },
        "adaptive": {
            "mode": "semantic_hybrid_clip_event_count",
            "memory_triggered": False,
            "route": route,
            "route_reason": route_reason,
            "recent_chunk_ids": final_chunk_ids,
            "memory_chunk_ids": [],
            "selected_chunk_ids": final_chunk_ids,
            "selected_timestamps": [float(x) for x in final_chunk_ids],
        },
    }
    _profile_fields(record, getattr(result, "profile_metadata", None))
    event_count_memory = getattr(result, "event_count_memory_metadata", None)
    if event_count_memory:
        event_ids = list(event_count_memory.get("event_chunk_ids", []) or [])
        recent_ids = list(event_count_memory.get("recent_chunk_ids", []) or [])
        record["event_count_memory"] = event_count_memory
        record["adaptive"]["memory_triggered"] = True
        record["adaptive"]["memory_selector"] = "event_count_memory"
        record["adaptive"]["memory_chunk_ids"] = event_ids
        record["adaptive"]["recent_chunk_ids"] = recent_ids
        record["adaptive"]["selected_chunk_ids"] = list(event_count_memory.get("selected_chunk_ids", final_chunk_ids))
        record["adaptive"]["memory_scores"] = event_count_memory.get("event_scores", [])
    out.write(json.dumps(record, ensure_ascii=False) + "\n")
    out.flush()
    return record


def _is_count_memory_question(question: str) -> bool:
    text = " ".join(str(question).lower().split())
    return any(term in text for term in COUNT_MEMORY_TERMS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a hybrid semantic-memory / last-K-second clip diagnostic on "
            "StreamingBench wrong examples."
        )
    )
    parser.add_argument(
        "--examples-jsonl",
        type=Path,
        default=Path("reports/semantic_memory_streamingbench_wrong30_pasted/results_incremental.jsonl"),
    )
    parser.add_argument("--video-dir", type=Path, default=Path("data/streamingbench/videos"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/semantic_hybrid_clip_k2_wrong30_results"),
    )
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    parser.add_argument(
        "--context-seconds",
        type=float,
        default=60.0,
        help="Seconds of pre-question video context decoded for semantic-memory routing.",
    )
    parser.add_argument("--recent-frames", type=int, default=6)
    parser.add_argument("--event-memory-max-events", type=int, default=5)
    parser.add_argument("--event-memory-min-gap", type=int, default=3)
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
            if _is_count_memory_question(question):
                route_raw, route_reason = "event_count_memory", "counting_event_state"
            else:
                route_raw, route_reason = choose_hybrid_route(question)
            use_clip = route_raw == "recent_clip"
            route = (
                "recent_clip_k2"
                if use_clip
                else "event_count_memory"
                if route_raw == "event_count_memory"
                else "semantic_memory_recent6_m3"
            )

            try:
                if use_clip:
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
                elif route_raw == "event_count_memory":
                    result, backend = query_event_count_memory(
                        qa,
                        video_path=str(video_path),
                        question=question,
                        options=options,
                        chunk_duration=args.chunk_duration,
                        fps=args.fps,
                        recent_frames_only=args.recent_frames,
                        video_start=max(0.0, time_seconds - args.context_seconds),
                        video_end=time_seconds,
                        max_events=args.event_memory_max_events,
                        min_gap_chunks=args.event_memory_min_gap,
                    )
                    attach_hybrid_metadata(
                        result,
                        route="event_count_memory",
                        route_reason=route_reason,
                        clip_seconds=args.clip_seconds,
                    )
                else:
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
                    "hybrid_recent_clip": {
                        "route": route,
                        "route_reason": route_reason,
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
                f"(pred={record.get('pred')} gt={answer_gt} correct={record.get('correct')})",
                flush=True,
            )

    total = len(records)
    print("=" * 80)
    print(f"Hybrid Recent-6/K={args.clip_seconds:g}s wrong-set accuracy: {100.0 * correct / total if total else 0.0:.2f}% ({correct}/{total})")
    for route in sorted(route_counts):
        n = route_counts[route]
        c = route_correct.get(route, 0)
        print(f"  {route}: {100.0 * c / n if n else 0.0:.2f}% ({c}/{n})")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
