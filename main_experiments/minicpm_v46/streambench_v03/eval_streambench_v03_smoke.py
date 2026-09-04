#!/usr/bin/env python3
"""Small local StreamBench-v0.3 smoke evaluator for MiniCPM-V-4.6.

StreamBench-v0.3 is open-ended QA, not MCQ. This script is intended for quick
engineering diagnostics only: it compares exact-six Recent-6 with the latest
PRISM visual path on the same samples, then reports lightweight text metrics by
StreamBench subtask class.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from main_experiments.tools.determinism import configure_determinism

SEED = configure_determinism()

import lib.minicpm.adaptive as adaptive_mod  # noqa: E402
from lib.minicpm.baseline import RecentWindowQAModel  # noqa: E402
from main_experiments.minicpm_v46.streamingbench.eval_prism_exact_recent_dist import (  # noqa: E402
    select_exact_current_recent_frames,
)
from main_experiments.minicpm_v46.streamingbench.eval_recent_sampler_dist import (  # noqa: E402
    query_recent_sampler_window,
)

SUBTASK_ORDER = ["OS", "LM", "SM", "CI", "KG", "SF"]


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize_text(prediction).split()
    ref_tokens = _normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts: dict[str, int] = defaultdict(int)
    for token in pred_tokens:
        pred_counts[token] += 1
    overlap = 0
    for token in ref_tokens:
        if pred_counts[token] > 0:
            overlap += 1
            pred_counts[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _contains_score(prediction: str, reference: str) -> bool:
    pred = _normalize_text(prediction)
    ref = _normalize_text(reference)
    return bool(pred and ref and (pred in ref or ref in pred))


def _resolve_video_path(root: Path, class_name: str, video_path: str) -> Path:
    candidates = [
        root / class_name / class_name / video_path,
        root / class_name / video_path,
        root / video_path,
    ]
    for candidate in candidates:
        if candidate.exists() and not candidate.name.startswith("._"):
            return candidate
    return candidates[0]


def _load_tasks(
    path: Path,
    data_root: Path,
    max_videos: int,
    max_questions: int,
    class_1: str,
    start_video: int,
) -> list[dict[str, Any]]:
    data = json.load(path.open(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Expected StreamBench annotation list, got {type(data).__name__}")

    tasks: list[dict[str, Any]] = []
    matched_videos = 0
    for video_index, entry in enumerate(data):
        if video_index < start_video:
            continue
        info = entry.get("info") or {}
        if class_1 and str(info.get("class_1") or "") != class_1:
            continue
        if max_videos > 0 and matched_videos >= max_videos:
            break
        matched_videos += 1
        class_name = str(info.get("class_1") or "")
        video_path_raw = str(info.get("video_path") or "")
        video_path = _resolve_video_path(data_root, class_name, video_path_raw)
        breakpoints = list(entry.get("breakpoint") or [])
        breakpoints.sort(key=lambda item: float(item.get("time", 0.0)))
        for bp_index, bp in enumerate(breakpoints):
            tasks.append(
                {
                    "_index": len(tasks),
                    "video_index": video_index,
                    "breakpoint_index": bp_index,
                    "video_path": str(video_path),
                    "video_path_raw": video_path_raw,
                    "video_name": info.get("video_name"),
                    "class_1": info.get("class_1"),
                    "class_2": info.get("class_2"),
                    "subtask": bp.get("class", ""),
                    "time": float(bp.get("time", 0.0)),
                    "question": str(bp.get("question", "")),
                    "answer_gt": str(bp.get("answer", "")),
                }
            )
            if max_questions > 0 and len(tasks) >= max_questions:
                return tasks
    return tasks


def _build_prompt(question: str) -> str:
    return (
        "Answer the question based on the video frames. "
        "Give a concise direct answer.\n\n"
        f"Question: {question}"
    )


def _profile_result(result: Any) -> dict[str, Any]:
    profile = getattr(result, "profile_metadata", None) or {}
    adaptive = getattr(result, "adaptive_metadata", None)
    return {
        "final_chunk_ids": getattr(result, "final_chunk_ids", None),
        "num_frames": getattr(result, "num_frames", None),
        "num_vision_tokens": getattr(result, "num_vision_tokens", None),
        "generate_time": getattr(result, "generate_time", None),
        "ttft_seconds": getattr(result, "ttft_seconds", None),
        "gpu_peak_allocated_mb": profile.get("gpu_peak_allocated_mb"),
        "adaptive": adaptive,
    }


def _run_method(
    *,
    method: str,
    qa: RecentWindowQAModel,
    task: dict[str, Any],
    chunk_duration: float,
    fps: float,
    recent_window: int,
    context_time: float,
) -> dict[str, Any]:
    ts_sec = float(task["time"])
    video_start = max(0.0, ts_sec - max(float(context_time), float(chunk_duration)))
    video_end = ts_sec + 1e-4
    prompt = _build_prompt(task["question"])

    t0 = time.perf_counter()
    if method == "recent6":
        os.environ["MINICPM_RECENT_SAMPLER"] = "current_recent6"
        result, decode_backend = query_recent_sampler_window(
            qa=qa,
            video_path=task["video_path"],
            prompt=prompt,
            chunk_duration=chunk_duration,
            fps=fps,
            recent_frames_only=recent_window,
            video_start=video_start,
            video_end=video_end,
        )
    elif method == "prism":
        os.environ["MINICPM_ADAPTIVE_MODE"] = "progressive_sufficiency_memory_clip_mmr_evidence_contract"
        os.environ["PRISM_CLIP_MODE"] = "evidence_contract"
        adaptive_mod.select_recent_window_frames = select_exact_current_recent_frames
        result, decode_backend = adaptive_mod.query_adaptive_window(
            qa=qa,
            video_path=task["video_path"],
            prompt=prompt,
            chunk_duration=chunk_duration,
            fps=fps,
            recent_frames_only=recent_window,
            video_start=video_start,
            video_end=video_end,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

    elapsed = time.perf_counter() - t0
    prediction = str(result.answer or "").strip()
    reference = str(task["answer_gt"])
    return {
        "method": method,
        "prediction": prediction,
        "reference": reference,
        "token_f1": _token_f1(prediction, reference),
        "contains_score": _contains_score(prediction, reference),
        "exact_match": _normalize_text(prediction) == _normalize_text(reference),
        "decode_backend": decode_backend,
        "elapsed_seconds": elapsed,
        **_profile_result(result),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_method_subtask: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    errors: dict[str, int] = defaultdict(int)
    for row in rows:
        for method_row in row["methods"]:
            if "error" in method_row:
                errors[str(method_row["method"])] += 1
                continue
            item = {**row, **method_row}
            by_method[method_row["method"]].append(item)
            by_method_subtask[(method_row["method"], str(row.get("subtask", "")))].append(item)

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"n": 0}
        return {
            "n": len(items),
            "mean_token_f1": sum(float(item["token_f1"]) for item in items) / len(items),
            "contains_rate": sum(bool(item["contains_score"]) for item in items) / len(items),
            "exact_match_rate": sum(bool(item["exact_match"]) for item in items) / len(items),
            "mean_latency_seconds": sum(float(item["elapsed_seconds"]) for item in items) / len(items),
        }

    return {
        "overall": {method: aggregate(items) for method, items in sorted(by_method.items())},
        "subtasks": {
            f"{method}:{subtask}": aggregate(items)
            for (method, subtask), items in sorted(by_method_subtask.items())
        },
        "errors": dict(sorted(errors.items())),
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("StreamBench-v0.3 Smoke Results (open-ended lexical diagnostics)")
    print("=" * 80)
    print("Token-F1 by StreamBench subtask")
    print("method | OS | LM | SM | CI | KG | SF | Avg")
    for method, overall in summary["overall"].items():
        values = []
        for subtask in SUBTASK_ORDER:
            row = summary["subtasks"].get(f"{method}:{subtask}", {"n": 0})
            values.append(f"{row['mean_token_f1']:.3f}" if row.get("n", 0) else "-")
        print(f"{method} | " + " | ".join(values) + f" | {overall['mean_token_f1']:.3f}")

    print("\nContains-rate by StreamBench subtask")
    print("method | OS | LM | SM | CI | KG | SF | Avg")
    for method, overall in summary["overall"].items():
        values = []
        for subtask in SUBTASK_ORDER:
            row = summary["subtasks"].get(f"{method}:{subtask}", {"n": 0})
            values.append(f"{row['contains_rate']:.3f}" if row.get("n", 0) else "-")
        print(f"{method} | " + " | ".join(values) + f" | {overall['contains_rate']:.3f}")

    print()
    print("method | n | token_f1 | contains | exact | latency")
    for method, row in summary["overall"].items():
        print(
            f"{method} | {row['n']} | {row['mean_token_f1']:.3f} | "
            f"{row['contains_rate']:.3f} | {row['exact_match_rate']:.3f} | "
            f"{row['mean_latency_seconds']:.3f}s"
        )
    if summary.get("errors"):
        print(f"errors: {summary['errors']}")
    print("\nPer-subtask:")
    print("method | subtask | n | token_f1 | contains | exact")
    for key, row in summary["subtasks"].items():
        method, subtask = key.split(":", 1)
        print(
            f"{method} | {subtask} | {row['n']} | {row['mean_token_f1']:.3f} | "
            f"{row['contains_rate']:.3f} | {row['exact_match_rate']:.3f}"
        )
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anno-path", default="data/streambench_v0_3/streaming_bench_v0.3.json")
    parser.add_argument("--data-root", default="data/streambench_v0_3")
    parser.add_argument("--output-dir", default="reports/streambench_v0_3_smoke")
    parser.add_argument("--qa-model", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--qa-device", default=None)
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--recent-window", type=int, default=6)
    parser.add_argument("--context-time", type=float, default=6.0)
    parser.add_argument("--max-videos", type=int, default=1)
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--class-1", default="", help="Optional StreamBench class_1 filter, e.g. Ego, Movie, WebVideo.")
    parser.add_argument("--start-video", type=int, default=0, help="Skip videos before this raw annotation index.")
    parser.add_argument("--methods", nargs="+", choices=["recent6", "prism"], default=["recent6", "prism"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    os.environ.setdefault("MINICPM_SEED", str(SEED))
    os.environ.setdefault("MINICPM_PSM_SUFFICIENCY_THRESHOLD", "0.62")
    os.environ.setdefault("MINICPM_PSM_CLIP_OVERRIDE_THRESHOLD", "0.2995")
    os.environ.setdefault("MINICPM_PSM_ARBITRATION_MIN_MARGIN", "0.60")
    os.environ.setdefault("MINICPM_PSM_ARBITRATION_MAX_SUFFICIENCY_DROP", "0.08")
    os.environ.setdefault("MINICPM_PSM_TEMPORAL_BAND_MIN_SECONDS", "3")
    os.environ.setdefault("MINICPM_PSM_TEMPORAL_BAND_MAX_SECONDS", "30")
    os.environ.setdefault("MINICPM_PSM_CANDIDATE_K1_DISAGREE_MAX_DISTANCE_SECONDS", "10")
    os.environ.setdefault("MINICPM_PSM_MAX_MEMORY_FRAMES", "3")
    os.environ.setdefault("MINICPM_PSM_HISTORY_SEARCH_CHUNKS", "64")
    os.environ.setdefault("MINICPM_PSM_HISTORY_CANDIDATE_POOL", "12")
    os.environ.setdefault("MINICPM_PSM_MMR_LAMBDA", "0.80")
    os.environ.setdefault("MINICPM_PSM_EXACT_RECENT_PRESERVE_SOURCE_IDS", "0")

    anno_path = Path(args.anno_path)
    data_root = Path(args.data_root)
    if not anno_path.exists():
        raise FileNotFoundError(f"Missing StreamBench annotations: {anno_path}")
    if not data_root.exists():
        raise FileNotFoundError(f"Missing StreamBench data root: {data_root}")

    tasks = _load_tasks(
        anno_path,
        data_root,
        args.max_videos,
        args.max_questions,
        args.class_1,
        args.start_video,
    )
    if not tasks:
        raise ValueError("No StreamBench tasks loaded")

    missing = [task["video_path"] for task in tasks if not Path(task["video_path"]).exists()]
    if missing:
        raise FileNotFoundError(f"Missing video for first task: {missing[0]}")

    qa = RecentWindowQAModel(
        model_name=args.qa_model,
        device=args.qa_device,
        max_new_tokens=args.max_new_tokens,
        attn_implementation=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks, start=1):
        print(
            f"[{task_index}/{len(tasks)}] video={task['video_name']} "
            f"time={task['time']:.1f}s subtask={task['subtask']}",
            flush=True,
        )
        row = {**task, "methods": []}
        for method in args.methods:
            try:
                result = _run_method(
                    method=method,
                    qa=qa,
                    task=task,
                    chunk_duration=args.chunk_duration,
                    fps=args.fps,
                    recent_window=args.recent_window,
                    context_time=args.context_time,
                )
                print(
                    f"  {method}: f1={result['token_f1']:.3f} "
                    f"answer={result['prediction'][:120]!r}",
                    flush=True,
                )
            except Exception as exc:
                result = {"method": method, "error": repr(exc)}
                print(f"  {method}: ERROR {exc!r}", flush=True)
            row["methods"].append(result)
        rows.append(row)

    summary = _summarize(rows)
    payload = {
        "config": vars(args),
        "note": "Open-ended diagnostic only; StreamBench official scoring may require a separate judge.",
        "summary": summary,
        "results": rows,
    }
    out_json = output_dir / "streambench_v0_3_smoke_results.json"
    out_summary = output_dir / "streambench_v0_3_smoke_summary.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_summary(summary)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
