#!/usr/bin/env python3
"""Smoke-test Recursive Story Memory (Proposal 1) on a single video before a full cluster run.

Runs a handful of real questions through RecursiveStoryMemoryQAModel and dumps,
for each question: the exact story text S_t at that point, its token count,
the model's answer, and whether it matched the ground truth. Intended as a
fast sanity check (a few minutes on one GPU) of story quality before launching
the full OVO-Bench / StreamingBench distributed evaluation.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from lib.minicpm.recursive_story_memory import RecursiveStoryMemoryQAModel, query_recent_window  # noqa: E402
from lib.shared.recent_window import extract_mcq_answer  # noqa: E402
from main_experiments.minicpm_v46.streamingbench.eval_baseline import (  # noqa: E402
    build_prompt,
    timestamp_to_seconds,
)


def _normalise_options(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except Exception:
            pass
        return [part.strip() for part in re.split(r"\s*[ABCD]\.\s*", text) if part.strip()]
    return []


def _load_questions(path: Path, max_questions: int, question_id_prefix: str | None) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    questions: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = dict(row)
        item.setdefault("question_id", item.get("id", f"debug_{index}"))
        if question_id_prefix and not str(item.get("question_id", "")).startswith(question_id_prefix):
            continue
        item.setdefault("task_type", item.get("task", "Debug"))
        item.setdefault("time_stamp", item.get("timestamp", "00:00:00"))
        item["options"] = _normalise_options(item.get("options", []))
        questions.append(item)
    questions.sort(key=lambda item: timestamp_to_seconds(str(item.get("time_stamp", "00:00:00"))))
    return questions[:max_questions] if max_questions > 0 else questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-path", type=Path, default=Path("reports/streamingbench_real_sqa_sample_1/sample_1_sqa.mp4"))
    parser.add_argument("--questions-path", type=Path, default=Path("reports/streamingbench_real_sqa_sample_1/Sequential_Question_Answering.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/recursive_story_memory_debug"))
    parser.add_argument("--model", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"))
    parser.add_argument("--max-questions", type=int, default=3)
    parser.add_argument(
        "--question-id-prefix",
        default="",
        help="Only run questions whose question_id starts with this prefix. Empty disables the filter.",
    )
    parser.add_argument("--recent-frames", type=int, default=6)
    parser.add_argument("--update-batch", type=int, default=4)
    parser.add_argument("--max-story-tokens", type=int, default=256)
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    args = parser.parse_args()

    if not args.video_path.exists():
        raise FileNotFoundError(args.video_path)
    if not args.questions_path.exists():
        raise FileNotFoundError(args.questions_path)

    os.environ["MINICPM_RSM_RECENT_FRAMES"] = str(args.recent_frames)
    os.environ["MINICPM_RSM_UPDATE_BATCH"] = str(args.update_batch)
    os.environ["MINICPM_RSM_MAX_STORY_TOKENS"] = str(args.max_story_tokens)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    qa = RecursiveStoryMemoryQAModel(
        model_name=args.model,
        device=args.device,
        max_new_tokens=256,
        attn_implementation=args.attn_implementation,
    )

    question_id_prefix = args.question_id_prefix or None
    questions = _load_questions(args.questions_path, args.max_questions, question_id_prefix)
    if not questions:
        raise ValueError(f"No questions matched {args.questions_path} with prefix={question_id_prefix!r}")

    jsonl_path = args.out_dir / "recursive_story_memory_debug_records.jsonl"
    correct_count = 0
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for q_index, question in enumerate(questions, start=1):
            ts_sec = float(timestamp_to_seconds(str(question.get("time_stamp", "00:00:00"))))
            prompt = build_prompt(question)
            result, decode_backend = query_recent_window(
                qa=qa,
                video_path=str(args.video_path),
                prompt=prompt,
                chunk_duration=args.chunk_duration,
                fps=args.fps,
                recent_frames_only=args.recent_frames,
                video_start=0.0,
                video_end=ts_sec + 1e-4,
            )
            pred = extract_mcq_answer(result.answer)
            answer_gt = extract_mcq_answer(str(question.get("answer", ""))) or str(question.get("answer", "")).strip().upper()
            correct = bool(pred is not None and pred == answer_gt)
            correct_count += int(correct)
            rsm_meta = result.recursive_story_memory_metadata

            print("=" * 72)
            print(f"Q{q_index} | t={question.get('time_stamp')} | task={question.get('task_type')}")
            print(f"Question: {question.get('question')}")
            print(f"Answer: {result.answer!r}  (pred={pred}, gt={answer_gt}, correct={correct})")
            print(
                f"Story tokens: {rsm_meta['story_tokens']} / {args.max_story_tokens}  "
                f"(rewrite calls so far: {rsm_meta['rewrite_calls_total']}, "
                f"compression calls: {rsm_meta['compression_calls_total']}, "
                f"hard truncations: {qa._rsm_cache[qa._rsm_key(str(args.video_path), args.fps, args.chunk_duration)].hard_truncations})"
            )
            print(f"Story S_t:\n{rsm_meta['story_text']}")

            record = {
                "index": q_index,
                "time_stamp": question.get("time_stamp"),
                "task_type": question.get("task_type"),
                "question": question.get("question"),
                "response": result.answer,
                "pred": pred,
                "answer_gt": answer_gt,
                "correct": correct,
                "recent_chunk_ids": result.final_chunk_ids,
                "decode_backend": decode_backend,
                "recursive_story_memory": rsm_meta,
                "ttft_seconds": result.ttft_seconds,
                "generate_time": result.generate_time,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("=" * 72)
    print(f"Done: {correct_count}/{len(questions)} correct. Records: {jsonl_path}")


if __name__ == "__main__":
    main()
