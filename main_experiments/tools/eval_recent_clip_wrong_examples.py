#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from lib.minicpm.recent_clip import RecentClipQAModel, query_recent_clip  # noqa: E402
from lib.shared.recent_window import extract_mcq_answer  # noqa: E402
from main_experiments.minicpm_v46.streamingbench.eval_baseline import timestamp_to_seconds  # noqa: E402


CLIP_PROMPT_TEMPLATE = """You are an advanced video question-answering AI assistant.
You are given a short video clip immediately before the question timestamp and a multiple-choice question.
Use the temporal order inside the clip. If the question says "right now", "just now", "currently", "last", or "latest", focus most on the final moments of the clip.

Question: {question}
Options:
{options}

Only give the best option's letter (A, B, C, or D) directly."""


def _parse_options(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except Exception:
            pass
    return []


def _resolve_video_path(record: dict[str, Any], video_dir: Path) -> Path:
    for key in ("video_path", "video_path_raw", "video"):
        value = record.get(key)
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        candidates = [path] if path.is_absolute() else []
        candidates.extend([video_dir / path.name, video_dir / value])
        for candidate in candidates:
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Could not resolve video for {record.get('video') or record.get('_key')}")


def _build_clip_prompt(question: str, options: list[str]) -> str:
    return CLIP_PROMPT_TEMPLATE.format(
        question=question,
        options="\n".join(options),
    )


def _profile_fields(record: dict[str, Any], profile_metadata: dict[str, Any] | None) -> None:
    if not profile_metadata:
        return
    record["profile"] = profile_metadata
    record["decode_time"] = profile_metadata.get("decode_time_seconds")
    record["end_to_end_time"] = profile_metadata.get("end_to_end_time_seconds")
    record["model_generate_time"] = profile_metadata.get("model_generate_time_seconds")
    record["preprocess_time"] = profile_metadata.get("preprocess_time_seconds")
    record["prefill_kv_time_ms"] = profile_metadata.get("prefill_kv_time_ms")
    record["generate_first_token_time_ms"] = profile_metadata.get("generate_first_token_time_ms")
    record["generate_tokens_time_ms"] = profile_metadata.get("generate_tokens_time_ms")
    record["gpu_peak_allocated_mb"] = profile_metadata.get("gpu_peak_allocated_mb")
    record["gpu_peak_reserved_mb"] = profile_metadata.get("gpu_peak_reserved_mb")
    record["component_profile_enabled"] = profile_metadata.get("component_profile_enabled")
    record["recent_clip"] = profile_metadata.get("recent_clip")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MiniCPM recent-video-clip QA on a small JSONL of StreamingBench examples."
    )
    parser.add_argument(
        "--examples-jsonl",
        type=Path,
        default=Path("reports/semantic_memory_streamingbench_wrong30_pasted/results_incremental.jsonl"),
    )
    parser.add_argument("--video-dir", type=Path, default=Path("data/streamingbench/videos"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/recent_clip_k2_wrong30_results"))
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    parser.add_argument("--qa-model", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--qa-device", default="auto")
    parser.add_argument("--max-qa-tokens", type=int, default=256)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument(
        "--keep-clips",
        action="store_true",
        help="Save extracted 2s clips under output-dir/clips for inspection.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clip_dir = args.output_dir / "clips"
    if args.keep_clips:
        clip_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with args.examples_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.max_examples > 0:
        records = records[: args.max_examples]

    qa = RecentClipQAModel(
        model_name=args.qa_model,
        device=args.qa_device,
        max_new_tokens=args.max_qa_tokens,
        attn_implementation=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
    )

    out_path = args.output_dir / "results_incremental.jsonl"
    correct = 0
    total = 0
    with out_path.open("w", encoding="utf-8") as out:
        for index, source in enumerate(records, start=1):
            video_path = _resolve_video_path(source, args.video_dir)
            options = _parse_options(source.get("options"))
            q = {
                "question": source.get("question", ""),
                "options": options,
                "time_stamp": source.get("time_stamp", ""),
            }
            prompt = _build_clip_prompt(str(q["question"]), options)
            answer_gt = str(source.get("answer_gt") or source.get("answer") or "").strip()
            time_seconds = float(timestamp_to_seconds(str(source.get("time_stamp", "0:00:00"))))
            keep_clip_path = None
            if args.keep_clips:
                keep_clip_path = clip_dir / f"{index:02d}_{Path(str(source.get('video', video_path.name))).stem}_k{args.clip_seconds:g}.mp4"
            try:
                result, backend = query_recent_clip(
                    qa,
                    video_path=str(video_path),
                    prompt=prompt,
                    question_time_seconds=time_seconds,
                    clip_seconds=args.clip_seconds,
                    keep_clip_path=keep_clip_path,
                )
                response = result.answer
                pred = extract_mcq_answer(response)
                is_correct = bool(pred and answer_gt and pred == answer_gt)
                record: dict[str, Any] = {
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
                    "response": response,
                    "pred": pred,
                    "correct": is_correct,
                    "decode_backend": backend,
                    "final_chunk_ids": result.final_chunk_ids,
                    "generate_time": result.generate_time,
                    "ttft_seconds": result.ttft_seconds,
                    "num_vision_tokens": result.num_vision_tokens,
                    "num_vision_tokens_before": result.num_vision_tokens_before,
                    "num_vision_tokens_after": result.num_vision_tokens_after,
                    "num_frames": result.num_frames,
                    "adaptive": {
                        "mode": "recent_clip",
                        "memory_triggered": False,
                        "recent_chunk_ids": result.final_chunk_ids,
                        "memory_chunk_ids": [],
                        "selected_chunk_ids": result.final_chunk_ids,
                        "selected_timestamps": [float(x) for x in result.final_chunk_ids],
                    },
                }
                _profile_fields(record, getattr(result, "profile_metadata", None))
            except Exception as exc:
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
                    "response": None,
                    "correct": False,
                    "error": repr(exc),
                }
            total += 1
            if record.get("correct"):
                correct += 1
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"[{index}/{len(records)}] {record.get('task_type')} -> "
                f"{record.get('response')} (pred={record.get('pred')} gt={answer_gt} correct={record.get('correct')})",
                flush=True,
            )

    print("=" * 80)
    print(f"Recent-clip K={args.clip_seconds:g}s wrong-set accuracy: {100.0 * correct / total if total else 0.0:.2f}% ({correct}/{total})")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
