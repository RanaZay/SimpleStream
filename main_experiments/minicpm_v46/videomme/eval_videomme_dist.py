#!/usr/bin/env python3
"""Distributed Video-MME evaluation for MiniCPM-V-4.6.

This runner is intentionally isolated from the existing OVO/StreamingBench
entry points. It evaluates two causal recent-window style methods on the
official lmms-lab/Video-MME mirror:

  * recent6: corrected exact-six Recent-6 context at the end of the video.
  * progressive_sufficiency_memory_clip_mmr_evidence_contract: final PRISM.
"""

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

try:
    import pandas as pd
except ImportError:  # pragma: no cover - handled at runtime with a clear error.
    pd = None

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


def _first_present(row: dict[str, Any], names: list[str], default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    if isinstance(value, dict):
        items = []
        for key in sorted(value):
            items.append(value[key])
        return items
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                return _as_list(parsed)
            except Exception:
                pass
        parts = re.split(r"\n\s*|(?=\b[A-Ea-e][\.\)])", text)
        cleaned = [part.strip() for part in parts if part.strip()]
        return cleaned if len(cleaned) > 1 else [text]
    return [value]


def _option_texts(row: dict[str, Any]) -> list[str]:
    raw_options = _first_present(row, ["options", "option", "choices", "candidates"])
    options = _as_list(raw_options)
    if len(options) < 5:
        options = [_first_present(row, [f"option_{i}", f"option{i}", chr(ord("A") + i)], "") for i in range(5)]
    normalized: list[str] = []
    for option in options[:5]:
        text = str(option).strip()
        text = re.sub(r"^\s*[A-Ea-e][\.\)]\s*", "", text).strip()
        normalized.append(text)
    if len(normalized) < 5 or any(not item for item in normalized):
        raise ValueError(f"Video-MME row has no five-option field: {sorted(row.keys())}")
    return normalized


def _answer_letter(raw: Any) -> str:
    if isinstance(raw, int):
        if 0 <= raw < 5:
            return LETTERS[raw]
        if 1 <= raw <= 5:
            return LETTERS[raw - 1]
    text = str(raw).strip()
    upper = text.upper()
    if upper in LETTERS:
        return upper
    if upper.isdigit():
        value = int(upper)
        if 0 <= value < 5:
            return LETTERS[value]
        if 1 <= value <= 5:
            return LETTERS[value - 1]
    match = re.search(r"\b([A-E])\b", upper)
    if match:
        return match.group(1)
    raise ValueError(f"Could not normalize Video-MME answer: {raw!r}")


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


def build_prompt(question: str, options: list[str], subtitle: str | None = None) -> str:
    option_lines = "\n".join(f"{letter}. {text}" for letter, text in zip(LETTERS, options))
    subtitle_block = ""
    if subtitle:
        subtitle_block = f"\nSubtitle/context:\n{subtitle.strip()}\n"
    return (
        "You are an advanced video question-answering AI assistant.\n"
        "Answer the multiple-choice question using the provided video frames.\n"
        f"{subtitle_block}\n"
        f"Question: {question.strip()}\n"
        "Options:\n"
        f"{option_lines}\n\n"
        "Answer with only the option letter."
    )


def _normalize_video_id(raw: Any) -> str:
    text = str(raw).strip()
    if not text:
        return text
    text = os.path.basename(text)
    if text.lower().endswith(".mp4"):
        text = os.path.splitext(text)[0]
    return text


def _video_map(video_dir: str) -> dict[str, str]:
    root = Path(video_dir)
    mapping: dict[str, str] = {}
    if not root.exists():
        return mapping
    for path in root.rglob("*.mp4"):
        mapping.setdefault(path.stem, str(path))
        mapping.setdefault(path.name, str(path))
    return mapping


def resolve_video_path(video_id: str, video_dir: str, mapping: dict[str, str]) -> str:
    candidates = [
        video_id,
        f"{video_id}.mp4",
        os.path.basename(video_id),
        f"{os.path.basename(video_id)}.mp4",
    ]
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    direct = Path(video_dir) / f"{video_id}.mp4"
    return str(direct)


def _row_to_dicts(parquet_path: str) -> list[dict[str, Any]]:
    if pd is None:
        raise ImportError(
            "pandas/pyarrow are required for Video-MME parquet loading. "
            "Install with: python -m pip install pandas pyarrow"
        )
    frame = pd.read_parquet(parquet_path)
    return frame.to_dict(orient="records")


def _load_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = _row_to_dicts(args.annotation_parquet)
    videos = _video_map(args.video_dir)
    tasks: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        row = dict(row)
        raw_video = _first_present(
            row,
            ["videoID", "video_id", "video", "video_name", "video_path", "youtube_id", "url"],
            index,
        )
        video_id = _normalize_video_id(raw_video)
        question_id = str(_first_present(row, ["question_id", "qid", "id"], index))
        question = str(_first_present(row, ["question", "Question"], "")).strip()
        if not question:
            raise ValueError(f"Video-MME row {index} has no question. Columns: {sorted(row.keys())}")
        options = _option_texts(row)
        answer_gt = _answer_letter(_first_present(row, ["answer", "gt", "label", "Answer"]))
        duration = str(_first_present(row, ["duration", "duration_category"], "unknown"))
        domain = str(_first_present(row, ["domain", "category"], "unknown"))
        sub_category = str(_first_present(row, ["sub_category", "subcategory", "subtask"], "unknown"))
        task_type = str(_first_present(row, ["task_type", "task", "question_type"], "unknown"))
        tasks.append(
            {
                "_index": index,
                "_key": f"videomme::{question_id}",
                "question_id": question_id,
                "video_id": video_id,
                "video_path": resolve_video_path(video_id, args.video_dir, videos),
                "question": question,
                "options": options,
                "answer_gt": answer_gt,
                "duration": duration,
                "domain": domain,
                "sub_category": sub_category,
                "task_type": task_type,
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
    record["vision_preprocess_time_ms"] = profile_metadata.get("vision_preprocess_time_ms")
    record["vision_encoder_time_ms"] = profile_metadata.get("vision_encoder_time_ms")
    record["vision_resampler_time_ms"] = profile_metadata.get("vision_resampler_time_ms")
    record["vision_projector_time_ms"] = profile_metadata.get("vision_projector_time_ms")
    record["vision_hook_subtask_time_ms"] = profile_metadata.get("vision_hook_subtask_time_ms")
    record["non_vision_generate_time_ms"] = profile_metadata.get("non_vision_generate_time_ms")
    record["prefill_forward_time_ms"] = profile_metadata.get("prefill_forward_time_ms")
    record["decode_forward_time_ms"] = profile_metadata.get("decode_forward_time_ms")
    record["generate_first_token_time_ms"] = profile_metadata.get("generate_first_token_time_ms")
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
        "dataset": "lmms-lab/Video-MME",
        "mode": mode,
        "question_id": task["question_id"],
        "video_id": task["video_id"],
        "video": os.path.basename(task["video_path"]),
        "video_path": task["video_path"],
        "duration": task["duration"],
        "domain": task["domain"],
        "sub_category": task["sub_category"],
        "task_type": task["task_type"],
        "question": task["question"],
        "options": task["options"],
        "prompt": prompt,
        "answer_gt": task["answer_gt"],
        "prediction": prediction,
        "response": response,
        "correct": bool(prediction == task["answer_gt"]),
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
    adaptive_metadata = getattr(result, "adaptive_metadata", None)
    if adaptive_metadata is not None:
        record["adaptive"] = adaptive_metadata
    cdas_metadata = getattr(result, "cdas_metadata", None)
    if cdas_metadata is not None:
        record["cdas_metadata"] = cdas_metadata
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
        correct = sum(1 for row in rows if row.get("correct") is True)
        total = len(rows)
        summary[key] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
        }
    return summary


def print_summary(records: list[dict[str, Any]], label: str) -> None:
    correct = sum(1 for row in records if row.get("correct") is True)
    total = len(records)
    errors = sum(1 for row in records if row.get("error"))
    acc = 100.0 * correct / total if total else 0.0
    print("=" * 78)
    print(f"Video-MME Results (MiniCPM-V-4.6 + {label})")
    print("=" * 78)
    print(f"Overall: {acc:.2f}% ({correct}/{total})")
    print(f"Errors: {errors}")
    for field, title in [
        ("duration", "By Duration"),
        ("domain", "By Domain"),
        ("sub_category", "By Sub-Category"),
        ("task_type", "By Task Type"),
    ]:
        print()
        print(title + ":")
        for key, item in _breakdown(records, field).items():
            print(f"  {key}: {100.0 * item['accuracy']:.2f}% ({item['correct']}/{item['total']})")
    if records and any("adaptive" in row for row in records):
        triggered = [
            row
            for row in records
            if isinstance(row.get("adaptive"), dict)
            and row["adaptive"].get("memory_chunk_ids")
        ]
        print()
        print(f"Retrieved memory: {len(triggered)}/{total}")
    print("=" * 78)


def _build_model(args: argparse.Namespace, device: Any) -> RecentWindowQAModel:
    return RecentWindowQAModel(
        model_name=args.qa_model,
        device=args.qa_device or device,
        max_new_tokens=args.max_qa_tokens,
        attn_implementation=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed Video-MME evaluation for MiniCPM-V-4.6")
    parser.add_argument("--annotation-parquet", default="data/video_mme/videomme/test-00000-of-00001.parquet")
    parser.add_argument("--video-dir", default="data/video_mme/videos")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=[
            "recent6",
            "progressive_sufficiency_memory_clip_mmr_evidence_contract",
        ],
        required=True,
    )
    parser.add_argument("--qa-model", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--qa-device", default=None)
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--recent-frames-only", type=int, default=6)
    parser.add_argument("--decode-context-chunks", type=int, default=192)
    parser.add_argument("--max-qa-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    baseline_mod.select_recent_window_frames = select_exact_current_recent_frames
    adaptive_mod.select_recent_window_frames = select_exact_current_recent_frames
    os.environ["MINICPM_SEED"] = str(SEED)

    dist_timeout_seconds = int(os.environ.get("MINICPM_DIST_TIMEOUT_SECONDS", "7200"))
    accelerator = Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=dist_timeout_seconds))]
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
        "[rank %s] Video-MME tasks local=%s total=%s mode=%s",
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
            prompt = build_prompt(task["question"], task["options"])
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
                        "[rank %s %s/%s] %s/%s -> %s (gt=%s, correct=%s)",
                        accelerator.process_index,
                        local_index,
                        len(local_tasks),
                        task["video_id"],
                        task["question_id"],
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
            "dataset": "lmms-lab/Video-MME",
            "mode": args.mode,
            "records": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "errors": sum(1 for row in records if row.get("error")),
            "duration": _breakdown(records, "duration"),
            "domain": _breakdown(records, "domain"),
            "sub_category": _breakdown(records, "sub_category"),
            "task_type": _breakdown(records, "task_type"),
        }
        save_json(str(Path(args.output_dir) / "summary.json"), summary)
        print_summary(records, args.mode)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
