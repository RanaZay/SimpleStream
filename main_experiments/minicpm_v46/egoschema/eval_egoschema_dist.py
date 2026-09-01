"""
Distributed EgoSchema-Subset evaluation for MiniCPM-V-4.6.

This evaluator is intentionally separate from OVO/StreamingBench.  It uses the
official Hugging Face EgoSchema Subset split, resolves only the 500 subset video
IDs, and evaluates either:

  * recent6: SimpleStream-style recent 6 frames at 1 FPS.
  * progressive_sufficiency_memory: PRISM over a decoded 1 FPS history.
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
    from datasets import load_dataset
except ImportError:  # Local annotation JSON mode does not require datasets.
    load_dataset = None

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from lib.minicpm.baseline import RecentWindowQAModel
from lib.minicpm.baseline import query_recent_window as baseline_query_recent_window
from lib.minicpm.adaptive import query_recent_window as adaptive_query_recent_window
from lib.shared.recent_window import load_jsonl_results, save_json


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LETTERS = ["A", "B", "C", "D", "E"]


def _video_cache_dir() -> str:
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_dir = hf_home if os.path.basename(os.path.normpath(hf_home)) == "hub" else os.path.join(hf_home, "hub")
    return os.path.join(hub_dir, "egoschema", "videos")


def _normalize_video_id(raw: Any) -> str:
    text = str(raw).strip()
    if text.lower().endswith((".mp4", ".mp4")):
        text = os.path.splitext(os.path.basename(text))[0]
    return text


def resolve_video_path(video_id: str, video_dir: str) -> str:
    candidates = [
        Path(video_dir) / f"{video_id}.mp4",
        Path(video_dir) / f"{video_id}.MP4",
        Path(video_dir) / video_id,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def _option_texts(row: dict[str, Any]) -> list[str]:
    options = row.get("option")
    if options is None:
        options = row.get("options")
    if options is None:
        options = [row.get(f"option_{i}") for i in range(5)]
    if not isinstance(options, (list, tuple)) or len(options) < 5:
        raise ValueError(f"EgoSchema row has no five-option field: {row.keys()}")

    normalized: list[str] = []
    for idx, option in enumerate(list(options)[:5]):
        text = str(option).strip()
        text = re.sub(r"^\s*[A-Ea-e][\.\)]\s*", "", text).strip()
        normalized.append(text)
    return normalized


def _answer_letter(raw: Any) -> str:
    if isinstance(raw, int):
        if 0 <= raw < 5:
            return LETTERS[raw]
        if 1 <= raw <= 5:
            return LETTERS[raw - 1]
    text = str(raw).strip().upper()
    if text in LETTERS:
        return text
    if text.isdigit():
        value = int(text)
        if 0 <= value < 5:
            return LETTERS[value]
        if 1 <= value <= 5:
            return LETTERS[value - 1]
    match = re.search(r"\b([A-E])\b", text)
    if match:
        return match.group(1)
    raise ValueError(f"Could not normalize EgoSchema answer: {raw!r}")


def extract_egoschema_answer(response: str | None) -> str | None:
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


def build_prompt(question: str, options: list[str]) -> str:
    option_lines = "\n".join(f"{letter}. {text}" for letter, text in zip(LETTERS, options))
    return (
        "You are an advanced video question-answering AI assistant.\n"
        "The video is processed causally up to the end of the clip. "
        "Answer the multiple-choice question using the provided visual evidence.\n\n"
        f"Question: {question.strip()}\n"
        "Options:\n"
        f"{option_lines}\n\n"
        "Answer with only the option letter."
    )


def make_key(video_id: str) -> str:
    return f"egoschema::{video_id}"


def _load_annotation_rows(annotation_json: str) -> list[dict[str, Any]]:
    with open(annotation_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        rows = payload.get("records", payload.get("rows", payload.get("data")))
        if rows is None:
            raise ValueError(
                f"Annotation JSON {annotation_json} must contain records/rows/data or be a list."
            )
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ValueError(f"Annotation JSON {annotation_json} did not contain a list of rows.")
    return [dict(row) for row in rows]


def _load_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.annotation_json:
        dataset = _load_annotation_rows(args.annotation_json)
    else:
        if load_dataset is None:
            raise ImportError(
                "datasets is required unless --annotation-json is provided."
            )
        dataset = load_dataset(args.dataset_path, args.dataset_name, split=args.split)
    video_dir = args.video_dir or _video_cache_dir()
    tasks: list[dict[str, Any]] = []
    for index, item in enumerate(dataset):
        row = dict(item)
        raw_video_id = row.get("video_idx", row.get("video_id", row.get("video", index)))
        if isinstance(raw_video_id, dict) and "path" in raw_video_id:
            raw_video_id = raw_video_id["path"]
        video_id = _normalize_video_id(raw_video_id)
        options = _option_texts(row)
        answer_gt = _answer_letter(row.get("answer"))
        question = str(row.get("question", "")).strip()
        if not question:
            raise ValueError(f"EgoSchema row {index} has no question.")
        tasks.append(
            {
                "_index": index,
                "_key": make_key(video_id),
                "video_id": video_id,
                "video_path": resolve_video_path(video_id, video_dir),
                "question": question,
                "options": options,
                "answer_gt": answer_gt,
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
    record["vision_total_frontend_time_ms"] = profile_metadata.get("vision_total_frontend_time_ms")
    record["non_vision_generate_time_ms"] = profile_metadata.get("non_vision_generate_time_ms")
    record["prefill_forward_time_ms"] = profile_metadata.get("prefill_forward_time_ms")
    record["decode_forward_time_ms"] = profile_metadata.get("decode_forward_time_ms")
    record["prefill_kv_time_ms"] = profile_metadata.get("prefill_kv_time_ms")
    record["generate_first_token_time_ms"] = profile_metadata.get("generate_first_token_time_ms")
    record["generate_tokens_time_ms"] = profile_metadata.get("generate_tokens_time_ms")
    record["streamingtom_timeline_ms"] = profile_metadata.get("streamingtom_timeline_ms")
    record["st_vision_tower_ms"] = profile_metadata.get("st_vision_tower_ms")
    record["st_projector_ms"] = profile_metadata.get("st_projector_ms")
    record["st_compress_features_ms"] = profile_metadata.get("st_compress_features_ms")
    record["st_prefill_kv_ms"] = profile_metadata.get("st_prefill_kv_ms")
    record["st_store_kv_ms"] = profile_metadata.get("st_store_kv_ms")
    record["st_retrieval_forward_ms"] = profile_metadata.get("st_retrieval_forward_ms")
    record["st_reconstruct_kv_ms"] = profile_metadata.get("st_reconstruct_kv_ms")
    record["st_generate_first_token_ms"] = profile_metadata.get("st_generate_first_token_ms")
    record["st_generate_tokens_ms"] = profile_metadata.get("st_generate_tokens_ms")
    record["component_profile_enabled"] = profile_metadata.get("component_profile_enabled")
    record["gpu_peak_allocated_mb"] = profile_metadata.get("gpu_peak_allocated_mb")
    record["gpu_peak_reserved_mb"] = profile_metadata.get("gpu_peak_reserved_mb")
    record["gpu_peak_extra_allocated_mb"] = profile_metadata.get("gpu_peak_extra_allocated_mb")
    record["gpu_peak_extra_reserved_mb"] = profile_metadata.get("gpu_peak_extra_reserved_mb")


def _result_record(
    task: dict[str, Any],
    prompt: str,
    response: str | None,
    result: Any | None,
    decode_backend: str | None,
    mode: str,
    error: str | None = None,
) -> dict[str, Any]:
    prediction = extract_egoschema_answer(response)
    correct = prediction == task["answer_gt"]
    record: dict[str, Any] = {
        "_index": int(task["_index"]),
        "_key": task["_key"],
        "dataset": "lmms-lab/egoschema/Subset",
        "mode": mode,
        "video_idx": task["video_id"],
        "video": os.path.basename(task["video_path"]),
        "video_path": task["video_path"],
        "question": task["question"],
        "options": task["options"],
        "prompt": prompt,
        "answer_gt": task["answer_gt"],
        "prediction": prediction,
        "response": response,
        "correct": bool(correct),
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
    return record


def _merge_rank_outputs(output_dir: str) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for path in sorted(Path(output_dir).glob("rank_*/results_incremental.jsonl")):
        rows, _done = load_jsonl_results(str(path))
        merged.extend(rows)
    deduped = {str(row.get("_key")): row for row in merged if row.get("_key")}
    return sorted(deduped.values(), key=lambda item: int(item.get("_index", 0)))


def print_summary(records: list[dict[str, Any]], label: str) -> None:
    correct = sum(1 for row in records if row.get("correct") is True)
    total = len(records)
    errors = sum(1 for row in records if row.get("error"))
    acc = 100.0 * correct / total if total else 0.0
    print("=" * 78)
    print(f"EgoSchema-Subset Results ({label})")
    print("=" * 78)
    print(f"Overall: {acc:.2f}% ({correct}/{total})")
    print(f"Errors: {errors}")
    if records and any("adaptive" in row for row in records):
        triggered = [
            row
            for row in records
            if isinstance(row.get("adaptive"), dict)
            and row["adaptive"].get("memory_chunk_ids")
        ]
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
    parser = argparse.ArgumentParser(description="Distributed EgoSchema Subset evaluation for MiniCPM-V-4.6")
    parser.add_argument("--dataset-path", default="lmms-lab/egoschema")
    parser.add_argument("--dataset-name", default="Subset")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--annotation-json",
        default="",
        help="Optional local EgoSchema annotation JSON. Avoids Hugging Face dataset loading.",
    )
    parser.add_argument("--video-dir", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=[
            "recent6",
            "progressive_sufficiency_memory",
            "progressive_sufficiency_memory_clip_mmr_candidate_override_guarded_rollback_exact_recent",
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
        "[rank %s] EgoSchema tasks local=%s total=%s mode=%s",
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
                        "[rank %s %s/%s] %s -> %s (gt=%s, correct=%s)",
                        accelerator.process_index,
                        local_index,
                        len(local_tasks),
                        task["video_id"],
                        record.get("prediction"),
                        task["answer_gt"],
                        record.get("correct"),
                    )
                except Exception as exc:
                    logger.exception("[rank %s] Failed %s", accelerator.process_index, task["video_id"])
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
            "dataset": "lmms-lab/egoschema/Subset",
            "mode": args.mode,
            "records": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "errors": sum(1 for row in records if row.get("error")),
        }
        save_json(str(Path(args.output_dir) / "summary.json"), summary)
        print_summary(records, args.mode)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
