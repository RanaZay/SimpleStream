#!/usr/bin/env python3
"""Distributed LongVideoBench validation evaluation for MiniCPM-V-4.6."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

os.environ.setdefault("NCCL_TIMEOUT", "7200")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "0")

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from lib.minicpm.baseline import RecentWindowQAModel
from lib.minicpm.baseline import query_recent_window as baseline_query_recent_window
from lib.minicpm.adaptive import query_recent_window as adaptive_query_recent_window
import lib.minicpm.adaptive as adaptive_mod
import lib.minicpm.baseline as baseline_mod
from lib.shared.recent_window import load_jsonl_results, save_json
from main_experiments.minicpm_v46.streamingbench.eval_prism_exact_recent_dist import (
    select_exact_current_recent_frames,
)
from main_experiments.tools.determinism import configure_determinism


SEED = configure_determinism()
LETTERS = ["A", "B", "C", "D", "E"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                return parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                return [text]
        return [text]
    return [value]


def _answer_letter(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, int):
        if 0 <= raw < len(LETTERS):
            return LETTERS[raw]
        if 1 <= raw <= len(LETTERS):
            return LETTERS[raw - 1]
    text = str(raw).strip().upper()
    if text in LETTERS:
        return text
    if text.isdigit():
        value = int(text)
        if 0 <= value < len(LETTERS):
            return LETTERS[value]
        if 1 <= value <= len(LETTERS):
            return LETTERS[value - 1]
    match = re.search(r"\b([A-E])\b", text)
    return match.group(1) if match else None


def extract_answer(response: str | None) -> str | None:
    if response is None or not str(response).strip():
        return None
    text = str(response).strip().upper()
    match = re.search(r"\b([A-E])\b", text)
    if match:
        return match.group(1)
    match = re.search(r"\b([1-5])\b", text)
    if match:
        return LETTERS[int(match.group(1)) - 1]
    return None


def _subtitle_text(root: Path, subtitle_path: str | None, max_chars: int) -> str:
    if not subtitle_path or max_chars <= 0:
        return ""
    path = root / "subtitles" / subtitle_path
    if not path.exists():
        path = root / subtitle_path
    if not path.exists():
        return ""
    try:
        data = json.load(path.open(encoding="utf-8"))
    except Exception:
        return ""
    lines: list[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                text = item.get("text") or item.get("line") or item.get("content") or item.get("sentence")
                if text:
                    lines.append(str(text).strip())
            elif item:
                lines.append(str(item).strip())
    elif isinstance(data, dict):
        for value in data.values():
            if isinstance(value, str):
                lines.append(value.strip())
            elif isinstance(value, dict):
                text = value.get("text") or value.get("line") or value.get("content") or value.get("sentence")
                if text:
                    lines.append(str(text).strip())
    text = " ".join(line for line in lines if line)
    return text[:max_chars]


def build_prompt(question: str, candidates: list[str], subtitle: str = "") -> str:
    option_lines = "\n".join(
        f"{letter}. {text}" for letter, text in zip(LETTERS, candidates)
    )
    subtitle_block = ""
    if subtitle:
        subtitle_block = f"\nSubtitle/context:\n{subtitle}\n"
    return (
        "You are an advanced video question-answering AI assistant.\n"
        "Answer the multiple-choice question using the provided video frames.\n"
        f"{subtitle_block}\n"
        f"Question: {question.strip()}\n"
        "Options:\n"
        f"{option_lines}\n\n"
        "Answer with only the option letter."
    )


def _resolve_video_path(root: Path, raw_path: str) -> str:
    path = Path(str(raw_path))
    candidates = [
        root / "videos" / path,
        root / "videos" / path.name,
        root / path,
    ]
    if path.suffix.lower() != ".mp4":
        candidates.extend(
            [
                root / "videos" / f"{path}.mp4",
                root / "videos" / f"{path.name}.mp4",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def _load_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = Path(args.data_root)
    rows = json.load(Path(args.annotation_json).open(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("LongVideoBench annotation file must contain a list.")

    tasks: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if "correct_choice" not in row and args.require_gt:
            continue
        candidates = [str(item).strip() for item in _as_list(row.get("candidates")) if str(item).strip()]
        if len(candidates) < 2:
            raise ValueError(f"LongVideoBench row {index} has no usable candidates.")
        answer_gt = _answer_letter(row.get("correct_choice"))
        video_path = _resolve_video_path(root, str(row.get("video_path", "")))
        subtitle = _subtitle_text(root, row.get("subtitle_path"), args.max_subtitle_chars)
        tasks.append(
            {
                "_index": index,
                "_key": f"longvideobench::{row.get('id', index)}",
                "id": str(row.get("id", index)),
                "video_path": video_path,
                "video": os.path.basename(video_path),
                "question": str(row.get("question", "")).strip(),
                "candidates": candidates,
                "answer_gt": answer_gt,
                "duration": row.get("duration"),
                "duration_group": row.get("duration_group", "unknown"),
                "question_category": row.get("question_category", "unknown"),
                "subtitle_path": row.get("subtitle_path"),
                "subtitle": subtitle,
            }
        )
    if args.max_samples > 0:
        tasks = tasks[: args.max_samples]
    return tasks


def _profile_fields(record: dict[str, Any], profile_metadata: dict[str, Any] | None) -> None:
    if profile_metadata is None:
        return
    record["profile"] = profile_metadata
    record["decode_time"] = profile_metadata.get("decode_time_seconds")
    record["end_to_end_time"] = profile_metadata.get("end_to_end_time_seconds")
    record["model_generate_time"] = profile_metadata.get("model_generate_time_seconds")
    record["preprocess_time"] = profile_metadata.get("preprocess_time_seconds")
    record["ttft_seconds"] = profile_metadata.get("generate_first_token_time_ms")
    record["gpu_peak_allocated_mb"] = profile_metadata.get("gpu_peak_allocated_mb")
    record["gpu_peak_reserved_mb"] = profile_metadata.get("gpu_peak_reserved_mb")


def _result_record(
    task: dict[str, Any],
    prompt: str,
    response: str | None,
    result: Any | None,
    decode_backend: str | None,
    mode: str,
    error: str | None = None,
) -> dict[str, Any]:
    prediction = extract_answer(response)
    record: dict[str, Any] = {
        "_index": int(task["_index"]),
        "_key": task["_key"],
        "dataset": "longvideobench/LongVideoBench",
        "mode": mode,
        "id": task["id"],
        "video": task["video"],
        "video_path": task["video_path"],
        "duration": task["duration"],
        "duration_group": task["duration_group"],
        "question_category": task["question_category"],
        "question": task["question"],
        "candidates": task["candidates"],
        "answer_gt": task["answer_gt"],
        "prediction": prediction,
        "response": response,
        "correct": bool(prediction == task["answer_gt"]) if task["answer_gt"] is not None else None,
        "subtitle_path": task["subtitle_path"],
        "used_subtitle_chars": len(task.get("subtitle") or ""),
        "prompt": prompt,
    }
    if error is not None:
        record["error"] = error
        return record
    record.update(
        {
            "decode_backend": decode_backend,
            "final_chunk_ids": result.final_chunk_ids,
            "generate_time": result.generate_time,
            "ttft_seconds": result.ttft_seconds,
            "num_vision_tokens": result.num_vision_tokens,
            "num_vision_tokens_before": result.num_vision_tokens_before,
            "num_vision_tokens_after": result.num_vision_tokens_after,
            "num_frames": result.num_frames,
        }
    )
    _profile_fields(record, getattr(result, "profile_metadata", None))
    return record


def _merge_rank_outputs(output_dir: str) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for path in sorted(Path(output_dir).glob("rank_*/results_incremental.jsonl")):
        rows, _done = load_jsonl_results(str(path))
        merged.extend(rows)
    deduped = {str(row.get("_key")): row for row in merged if row.get("_key")}
    return sorted(deduped.values(), key=lambda item: int(item.get("_index", 0)))


def _breakdown(records: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        groups.setdefault(str(row.get(field, "unknown")), []).append(row)
    summary: dict[str, dict[str, Any]] = {}
    for key, rows in sorted(groups.items()):
        total = len(rows)
        correct = sum(1 for row in rows if row.get("correct") is True)
        summary[key] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
        }
    return summary


def print_summary(records: list[dict[str, Any]], label: str) -> None:
    total = len(records)
    correct = sum(1 for row in records if row.get("correct") is True)
    errors = sum(1 for row in records if row.get("error"))
    acc = 100.0 * correct / total if total else 0.0
    print("=" * 78)
    print(f"LongVideoBench Results (MiniCPM-V-4.6 + {label})")
    print("=" * 78)
    print(f"Overall: {acc:.2f}% ({correct}/{total})")
    print(f"Errors: {errors}")
    for field, title in [
        ("duration_group", "By Duration Group"),
        ("question_category", "By Question Category"),
    ]:
        print()
        print(title + ":")
        for key, item in _breakdown(records, field).items():
            print(f"  {key}: {100.0 * item['accuracy']:.2f}% ({item['correct']}/{item['total']})")
    print("=" * 78)


def _build_model(args: argparse.Namespace, device: Any) -> RecentWindowQAModel:
    return RecentWindowQAModel(
        model_name=args.qa_model,
        device=args.qa_device or device,
        max_new_tokens=args.max_qa_tokens,
        attn_implementation=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed LongVideoBench validation evaluation for MiniCPM-V-4.6")
    parser.add_argument("--data-root", default="data/longvideobench")
    parser.add_argument("--annotation-json", default="data/longvideobench/lvb_val.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=[
            "recent6",
            "progressive_sufficiency_memory_clip_mmr_candidate_override_guarded_rollback_exact_recent",
        ],
        default="recent6",
    )
    parser.add_argument("--qa-model", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--qa-device", default=None)
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--recent-frames-only", type=int, default=6)
    parser.add_argument("--decode-context-chunks", type=int, default=192)
    parser.add_argument("--max-qa-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-subtitle-chars", type=int, default=0)
    parser.add_argument("--require-gt", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    baseline_mod.select_recent_window_frames = select_exact_current_recent_frames
    adaptive_mod.select_recent_window_frames = select_exact_current_recent_frames
    os.environ["MINICPM_SEED"] = str(SEED)
    os.environ["MINICPM_ADAPTIVE_MODE"] = args.mode

    accelerator = Accelerator(
        kwargs_handlers=[
            InitProcessGroupKwargs(
                timeout=timedelta(seconds=int(os.environ.get("MINICPM_DIST_TIMEOUT_SECONDS", "7200")))
            )
        ]
    )
    rank_dir = Path(args.output_dir) / f"rank_{accelerator.process_index}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    incremental_path = rank_dir / "results_incremental.jsonl"
    existing_rows, done_keys = load_jsonl_results(str(incremental_path))
    del existing_rows

    tasks = _load_tasks(args)
    with accelerator.split_between_processes(tasks) as split_tasks:
        local_tasks = list(split_tasks)
    logger.info(
        "[rank %s] LongVideoBench tasks local=%s total=%s mode=%s",
        accelerator.process_index,
        len(local_tasks),
        len(tasks),
        args.mode,
    )
    qa = _build_model(args, accelerator.device)

    with incremental_path.open("a", encoding="utf-8") as handle:
        for local_index, task in enumerate(local_tasks, start=1):
            if task["_key"] in done_keys:
                continue
            prompt = build_prompt(task["question"], task["candidates"], task.get("subtitle") or "")
            if not os.path.exists(task["video_path"]):
                record = _result_record(
                    task,
                    prompt=prompt,
                    response=None,
                    result=None,
                    decode_backend=None,
                    mode=args.mode,
                    error=f"Missing video file: {task['video_path']}",
                )
            else:
                try:
                    if args.mode == "recent6":
                        result, decode_backend = baseline_query_recent_window(
                            qa,
                            task["video_path"],
                            prompt,
                            chunk_duration=args.chunk_duration,
                            fps=args.fps,
                            recent_frames_only=args.recent_frames_only,
                            video_start=0.0,
                            video_end=None,
                        )
                    else:
                        result, decode_backend = adaptive_query_recent_window(
                            qa,
                            task["video_path"],
                            prompt,
                            chunk_duration=args.chunk_duration,
                            fps=args.fps,
                            recent_frames_only=args.decode_context_chunks,
                            video_start=0.0,
                            video_end=None,
                        )
                    record = _result_record(
                        task,
                        prompt=prompt,
                        response=result.answer,
                        result=result,
                        decode_backend=decode_backend,
                        mode=args.mode,
                    )
                    logger.info(
                        "[rank %s %s/%s] %s -> %s (gt=%s, correct=%s)",
                        accelerator.process_index,
                        local_index,
                        len(local_tasks),
                        task["id"],
                        record.get("prediction"),
                        task["answer_gt"],
                        record.get("correct"),
                    )
                except Exception as exc:
                    logger.exception("[rank %s] Failed %s", accelerator.process_index, task["_key"])
                    record = _result_record(
                        task,
                        prompt=prompt,
                        response=None,
                        result=None,
                        decode_backend=None,
                        mode=args.mode,
                        error=repr(exc),
                    )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        records = _merge_rank_outputs(args.output_dir)
        correct = sum(1 for row in records if row.get("correct") is True)
        total = len(records)
        summary = {
            "dataset": "longvideobench/LongVideoBench",
            "mode": args.mode,
            "records": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "errors": sum(1 for row in records if row.get("error")),
            "duration_group": _breakdown(records, "duration_group"),
            "question_category": _breakdown(records, "question_category"),
        }
        save_json(str(Path(args.output_dir) / "summary.json"), summary)
        save_json(str(Path(args.output_dir) / "longvideobench_minicpmv46_results.json"), {"results": records, "summary": summary})
        print_summary(records, args.mode)
        print(f"Saved: {args.output_dir}")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
