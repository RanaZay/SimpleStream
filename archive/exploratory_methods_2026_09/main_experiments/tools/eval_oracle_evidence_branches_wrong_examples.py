#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from lib.minicpm.adaptive import query_recent_window as query_semantic_window  # noqa: E402
from lib.minicpm.hybrid_recent_clip import build_clip_prompt  # noqa: E402
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


def _branch_record(
    *,
    branch: str,
    result,
    backend: str,
    answer_gt: str,
) -> dict[str, Any]:
    response = result.answer
    pred = extract_mcq_answer(response)
    correct = bool(pred and answer_gt and pred == answer_gt)
    record = {
        "branch": branch,
        "response": response,
        "pred": pred,
        "correct": correct,
        "decode_backend": backend,
        "final_chunk_ids": list(getattr(result, "final_chunk_ids", []) or []),
        "generate_time": getattr(result, "generate_time", None),
        "ttft_seconds": getattr(result, "ttft_seconds", None),
        "num_vision_tokens": getattr(result, "num_vision_tokens", None),
        "num_frames": getattr(result, "num_frames", None),
    }
    _profile_fields(record, getattr(result, "profile_metadata", None))
    adaptive_meta = getattr(result, "adaptive_metadata", None)
    if isinstance(adaptive_meta, dict):
        record["adaptive"] = adaptive_meta
    recent_clip = getattr(result, "recent_clip_metadata", None)
    if isinstance(recent_clip, dict):
        record["recent_clip"] = recent_clip
    return record


def _run_recent6(
    qa: RecentClipQAModel,
    *,
    source: dict[str, Any],
    video_path: Path,
    options: list[str],
    time_seconds: float,
    recent_frames: int,
    output_dir: Path,
    answer_gt: str,
) -> dict[str, Any]:
    prompt = build_prompt({"question": str(source.get("question", "")), "options": options})
    result, backend = _query_recent6_direct(
        qa,
        video_path=video_path,
        prompt=prompt,
        question_time_seconds=time_seconds,
        recent_frames=recent_frames,
        output_dir=output_dir,
    )
    return _branch_record(branch="recent6", result=result, backend=backend, answer_gt=answer_gt)


def _run_clip(
    qa: RecentClipQAModel,
    *,
    source: dict[str, Any],
    video_path: Path,
    options: list[str],
    time_seconds: float,
    clip_seconds: float,
    keep_clip_path: Path | None,
    answer_gt: str,
) -> dict[str, Any]:
    prompt = build_clip_prompt(str(source.get("question", "")), options)
    result, backend = query_recent_clip(
        qa,
        video_path=str(video_path),
        prompt=prompt,
        question_time_seconds=time_seconds,
        clip_seconds=clip_seconds,
        keep_clip_path=keep_clip_path,
    )
    return _branch_record(branch="clip_k2", result=result, backend=backend, answer_gt=answer_gt)


def _run_semantic(
    qa: RecentClipQAModel,
    *,
    source: dict[str, Any],
    video_path: Path,
    options: list[str],
    time_seconds: float,
    context_seconds: float,
    recent_frames: int,
    chunk_duration: float,
    fps: float,
    answer_gt: str,
) -> dict[str, Any]:
    prompt = build_prompt({"question": str(source.get("question", "")), "options": options})
    result, backend = query_semantic_window(
        qa,
        video_path=str(video_path),
        prompt=prompt,
        chunk_duration=chunk_duration,
        fps=fps,
        recent_frames_only=recent_frames,
        video_start=max(0.0, time_seconds - context_seconds),
        video_end=time_seconds,
    )
    return _branch_record(branch="semantic_m3", result=result, backend=backend, answer_gt=answer_gt)


def _solved_by(branches: dict[str, dict[str, Any]]) -> list[str]:
    return [name for name, record in branches.items() if record.get("correct")]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Recent-6, K=2 clip, and Semantic M=3 on the same wrong examples for oracle routing analysis."
    )
    parser.add_argument(
        "--examples-jsonl",
        type=Path,
        default=Path("reports/semantic_memory_streamingbench_wrong30_pasted/results_incremental.jsonl"),
    )
    parser.add_argument("--video-dir", type=Path, default=Path("data/streamingbench/videos"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/evidence_branch_oracle_wrong30"))
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

    branch_correct = Counter()
    branch_counts = Counter()
    unique_correct = Counter()
    oracle_correct = 0
    out_path = args.output_dir / "results_incremental.jsonl"
    with out_path.open("w", encoding="utf-8") as out:
        for index, source in enumerate(records, start=1):
            video_path = _resolve_video_path(source, args.video_dir)
            options = _parse_options(source.get("options"))
            answer_gt = str(source.get("answer_gt") or source.get("answer") or "").strip()
            time_seconds = float(timestamp_to_seconds(str(source.get("time_stamp", "0:00:00"))))
            keep_clip_path = None
            if args.keep_clips:
                keep_clip_path = clip_dir / f"{index:02d}_{video_path.stem}_k{args.clip_seconds:g}.mp4"

            branches = {
                "recent6": _run_recent6(
                    qa,
                    source=source,
                    video_path=video_path,
                    options=options,
                    time_seconds=time_seconds,
                    recent_frames=args.recent_frames,
                    output_dir=args.output_dir,
                    answer_gt=answer_gt,
                ),
                "clip_k2": _run_clip(
                    qa,
                    source=source,
                    video_path=video_path,
                    options=options,
                    time_seconds=time_seconds,
                    clip_seconds=args.clip_seconds,
                    keep_clip_path=keep_clip_path,
                    answer_gt=answer_gt,
                ),
                "semantic_m3": _run_semantic(
                    qa,
                    source=source,
                    video_path=video_path,
                    options=options,
                    time_seconds=time_seconds,
                    context_seconds=args.context_seconds,
                    recent_frames=args.recent_frames,
                    chunk_duration=args.chunk_duration,
                    fps=args.fps,
                    answer_gt=answer_gt,
                ),
            }
            solved = _solved_by(branches)
            oracle = bool(solved)
            oracle_correct += int(oracle)
            for name, branch in branches.items():
                branch_counts[name] += 1
                branch_correct[name] += int(bool(branch.get("correct")))
            if len(solved) == 1:
                unique_correct[solved[0]] += 1

            record = {
                "_index": int(source.get("_index", index)),
                "_key": source.get("_key", f"{video_path.name}_{index}"),
                "video": video_path.name,
                "video_path": str(video_path),
                "task_type": source.get("task_type", ""),
                "time_stamp": source.get("time_stamp", ""),
                "question": source.get("question", ""),
                "options": options,
                "answer_gt": answer_gt,
                "source_response": source.get("response"),
                "source_correct": source.get("correct"),
                "branches": branches,
                "solved_by": solved,
                "oracle_correct": oracle,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"[{index}/{len(records)}] oracle={oracle} solved_by={solved or '-'} "
                f"recent={branches['recent6']['pred']}:{branches['recent6']['correct']} "
                f"clip={branches['clip_k2']['pred']}:{branches['clip_k2']['correct']} "
                f"semantic={branches['semantic_m3']['pred']}:{branches['semantic_m3']['correct']}",
                flush=True,
            )

    total = len(records)
    print("=" * 80)
    print(f"Oracle branch accuracy: {100.0 * oracle_correct / total if total else 0.0:.2f}% ({oracle_correct}/{total})")
    for name in ("recent6", "clip_k2", "semantic_m3"):
        n = branch_counts[name]
        c = branch_correct[name]
        print(f"  {name}: {100.0 * c / n if n else 0.0:.2f}% ({c}/{n}); unique={unique_correct[name]}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
