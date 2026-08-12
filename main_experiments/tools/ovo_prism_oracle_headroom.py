#!/usr/bin/env python3
"""OVO-only PRISM oracle/headroom diagnostics.

This script is analysis-only. It does not change PRISM retrieval, ranking,
thresholds, or sufficiency behavior. It consumes completed Recent-6 and PRISM
result directories, then evaluates Recent-6 plus prefixes of the existing PRISM
candidate queue with the same final MiniCPM QA generation used by the benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from main_experiments.tools.determinism import configure_determinism  # noqa: E402

SEED = configure_determinism()

from accelerate import Accelerator  # noqa: E402

from ovo_constants import BACKWARD_TASKS, FORWARD_TASKS, REAL_TIME_TASKS  # noqa: E402
from lib.minicpm.baseline import RecentWindowQAModel, build_ovo_prompt  # noqa: E402
from lib.minicpm.progressive_sufficiency import _PSM_HISTORY_INSTRUCTION  # noqa: E402
from lib.shared.recent_window import decode_video_to_chunks_qwen  # noqa: E402


OVO_GROUPS = {
    "EPM": "backward",
    "HLD": "backward",
    "ASI": "backward",
    "OCR": "real_time",
    "OJR": "real_time",
    "ACR": "real_time",
    "STU": "real_time",
    "ATR": "real_time",
    "FPD": "real_time",
    "REC": "forward",
    "SSR": "forward",
    "CRR": "forward",
}


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get("results"), list):
        return value["results"]
    if isinstance(value, dict) and all(key in value for key in ("backward", "realtime", "forward")):
        rows: list[dict[str, Any]] = []
        for group in ("backward", "realtime", "forward"):
            for row in value[group]:
                if isinstance(row, dict):
                    rows.append({**row, "_source_group": group})
        return rows
    return []


def _flatten(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        row = dict(row)
        row.setdefault("_record_index", row_index)
        if isinstance(row.get("test_info"), list):
            for index, item in enumerate(row["test_info"]):
                if isinstance(item, dict):
                    flat.append({**row, **item, "_question_index": index})
        else:
            flat.append(row)
    return flat


def _load_records(result_dir: Path) -> list[dict[str, Any]]:
    if result_dir.is_file():
        rows = _flatten(_read_json_rows(result_dir))
        for index, row in enumerate(rows):
            row["_file"] = str(result_dir)
            row["_load_index"] = index
        return rows

    rows: list[dict[str, Any]] = []
    for path in sorted(result_dir.glob("rank_*/results_incremental.jsonl")):
        for row in _read_json_rows(path):
            row["_file"] = str(path)
            rows.append(row)
    if rows:
        rows = _flatten(rows)
        for index, row in enumerate(rows):
            row["_load_index"] = index
        return rows

    merged = result_dir / "merged_results.json"
    if merged.exists():
        rows = _flatten(_read_json_rows(merged))
        for index, row in enumerate(rows):
            row["_file"] = str(merged)
            row["_load_index"] = index
        return rows

    ovo_jsons = sorted(result_dir.glob("minicpmv46_results_*.json"))
    if ovo_jsons:
        path = ovo_jsons[-1]
        rows = _flatten(_read_json_rows(path))
        for index, row in enumerate(rows):
            row["_file"] = str(path)
            row["_load_index"] = index
        return rows

    raise FileNotFoundError(f"No supported result records found under {result_dir}")


def _extract_letter(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"A", "B", "C", "D", "E"}:
        return text
    match = re.search(r"\b([A-E])\b", text)
    return match.group(1) if match else ""


def _score_rec(response: Any, gt_count: Any) -> bool | None:
    if gt_count is None:
        return None
    nums = re.findall(r"\d+", str(response or ""))
    return bool(nums and "".join(nums) == str(gt_count))


def _score_yesno(response: Any, gt_type: Any) -> bool | None:
    if gt_type is None:
        return None
    text = str(response or "").strip().upper()
    try:
        gt_int = int(gt_type)
    except (TypeError, ValueError):
        return None
    if (text == "N" or "NO" in text) and gt_int == 0:
        return True
    if (text == "Y" or "YES" in text) and gt_int == 1:
        return True
    return False


def _score_row_response(task: str, row: dict[str, Any], response: Any) -> bool | None:
    if task in {"REC"}:
        return _score_rec(response, row.get("count"))
    if task in {"SSR", "CRR"}:
        return _score_yesno(response, row.get("type"))
    gt = _extract_letter(row.get("ground_truth") or row.get("answer_gt") or row.get("answer"))
    pred = _extract_letter(response)
    if not gt or not pred:
        return None
    return pred == gt


def _prediction_for_task(task: str, response: Any) -> str:
    if task == "REC":
        nums = re.findall(r"\d+", str(response or ""))
        return "".join(nums) if nums else ""
    if task in {"SSR", "CRR"}:
        text = str(response or "").strip().upper()
        if text == "Y" or "YES" in text:
            return "YES"
        if text == "N" or "NO" in text:
            return "NO"
        return ""
    return _extract_letter(response)


def _ground_truth_for_task(task: str, row: dict[str, Any]) -> str:
    if task == "REC":
        return str(row.get("count", ""))
    if task in {"SSR", "CRR"}:
        return "YES" if int(row.get("type", 0)) == 1 else "NO"
    return _extract_letter(row.get("ground_truth") or row.get("answer_gt") or row.get("answer"))


def _adaptive(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("adaptive"), dict):
        return row["adaptive"]
    profile = row.get("profile")
    if isinstance(profile, dict) and isinstance(profile.get("adaptive"), dict):
        return profile["adaptive"]
    return {}


def _sample_key(row: dict[str, Any]) -> str:
    primary = str(row.get("_key") or row.get("id") or "")
    qidx = row.get("_question_index")
    if qidx is None:
        qidx = row.get("question_index")
    parts = [
        primary,
        str(row.get("task") or row.get("task_type") or ""),
        str(qidx if qidx is not None else ""),
        str(row.get("question") or ""),
    ]
    return "|".join(parts)


def _group(task: str) -> str:
    return OVO_GROUPS.get(task, "unknown")


def _load_annotations(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        annotations = json.load(handle)
    return {str(item["id"]): item for item in annotations}


def _video_path(row: dict[str, Any], chunked_dir: Path) -> Path:
    task = str(row.get("task") or "")
    sample_id = str(row.get("id"))
    if task in FORWARD_TASKS:
        qidx = int(row.get("_question_index", row.get("question_index", 0)))
        return chunked_dir / f"{sample_id}_{qidx}.mp4"
    return chunked_dir / f"{sample_id}.mp4"


def _annotation_for_row(row: dict[str, Any], annotations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sample_id = str(row.get("id"))
    if sample_id not in annotations:
        raise KeyError(f"Missing annotation id={sample_id}")
    return annotations[sample_id]


def _prompt_for_row(row: dict[str, Any], annotation: dict[str, Any]) -> str:
    task = str(row.get("task") or annotation.get("task"))
    if task in FORWARD_TASKS:
        qidx = int(row.get("_question_index", row.get("question_index", 0)))
        return build_ovo_prompt(task, annotation, index=qidx)
    return build_ovo_prompt(task, annotation)


def _candidate_queue(row: dict[str, Any]) -> list[dict[str, Any]]:
    queue = _adaptive(row).get("candidate_queue")
    return [item for item in queue if isinstance(item, dict)] if isinstance(queue, list) else []


def _iteration(row: dict[str, Any], index: int) -> dict[str, Any]:
    iterations = _adaptive(row).get("iterations")
    if isinstance(iterations, list) and index < len(iterations) and isinstance(iterations[index], dict):
        return iterations[index]
    return {}


def _safe_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _top_candidate_value(queue: list[dict[str, Any]], key: str) -> float | None:
    if not queue:
        return None
    value = queue[0].get(key)
    return _safe_float(value)


def _decode_context(
    video_path: Path,
    candidate_ids: list[int],
    chunk_duration: float,
    fps: float,
    recent_window: int,
) -> tuple[list[Any], list[Any], list[int]]:
    chunks, _backend = decode_video_to_chunks_qwen(
        video_path=str(video_path),
        chunk_duration=chunk_duration,
        fps=fps,
        recent_frames_only=None,
    )
    by_id = {int(chunk.chunk_index): chunk for chunk in chunks}
    recent = list(chunks[-recent_window:])
    memory = [by_id[chunk_id] for chunk_id in candidate_ids if chunk_id in by_id]
    return chunks, recent, [int(chunk.chunk_index) for chunk in memory]


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    values = sorted(values)
    def pct(p: float) -> float:
        index = min(len(values) - 1, max(0, int(round((len(values) - 1) * p))))
        return values[index]
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "p10": pct(0.10),
        "p90": pct(0.90),
    }


def _roc_auc(labels: list[bool], scores: list[float]) -> float | None:
    pairs = [(score, label) for label, score in zip(labels, scores) if isinstance(score, (int, float))]
    positives = sum(1 for _score, label in pairs if label)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None
    pairs.sort(key=lambda item: item[0])
    rank_sum = 0.0
    idx = 0
    while idx < len(pairs):
        j = idx + 1
        while j < len(pairs) and pairs[j][0] == pairs[idx][0]:
            j += 1
        avg_rank = (idx + 1 + j) / 2.0
        for k in range(idx, j):
            if pairs[k][1]:
                rank_sum += avg_rank
        idx = j
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _counter_table(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key)) for row in rows))


def _breakdown(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for name, items in _group_by(rows, field).items():
        out[name] = dict(Counter(str(item.get("failure_mode")) for item in items))
    return out


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    return dict(grouped)


def _oracle_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    recent_correct = sum(1 for row in rows if row["k0_correct"])
    wrong_rows = [row for row in rows if not row["k0_correct"]]
    min_counts = Counter(str(row["minimum_correct_k"]) for row in rows)
    damage = {}
    for k in (1, 2, 3):
        damage[f"k{k}"] = sum(1 for row in rows if row["k0_correct"] and row.get(f"k{k}_correct") is False)
    oracle_correct = sum(1 for row in rows if row["minimum_correct_k"] != "none")
    return {
        "samples": total,
        "recent6_correct": recent_correct,
        "recent6_accuracy": 100.0 * recent_correct / total if total else None,
        "recent6_wrong": len(wrong_rows),
        "rescued_by_k1": sum(1 for row in rows if row["minimum_correct_k"] == 1),
        "additional_rescued_by_k2": sum(1 for row in rows if row["minimum_correct_k"] == 2),
        "additional_rescued_by_k3": sum(1 for row in rows if row["minimum_correct_k"] == 3),
        "not_rescuable": sum(1 for row in rows if row["minimum_correct_k"] == "none"),
        "minimum_k_distribution": dict(min_counts),
        "oracle_accuracy": 100.0 * oracle_correct / total if total else None,
        "damage_recent_correct": damage,
    }


def _classify_prism_vs_oracle(row: dict[str, Any]) -> str:
    prism_memory = bool(row["prism_memory_chunk_ids"])
    k0 = bool(row["k0_correct"])
    prism = bool(row["prism_correct"])
    minimum = row["minimum_correct_k"]
    memory_can_help = minimum in {1, 2, 3}
    if k0 and prism and not prism_memory:
        return "A_correct_stop_k0"
    if (not k0) and prism and prism_memory:
        return "B_retrieved_and_rescued"
    if (not k0) and (not prism_memory) and memory_can_help:
        return "C_false_stop_oracle_memory_could_rescue"
    if (not k0) and prism_memory and (not prism) and memory_can_help:
        return "D_retrieved_but_prefix_failed_later_oracle_could_rescue"
    if k0 and (not prism) and prism_memory:
        return "E_retrieved_and_damaged"
    if (not k0) and minimum == "none":
        return "F_no_useful_historical_evidence"
    if k0 and prism and prism_memory:
        return "retrieved_but_no_damage"
    if (not k0) and prism and not prism_memory:
        return "rescued_without_memory_generation_difference"
    if k0 and (not prism) and not prism_memory:
        return "damaged_without_memory_generation_difference"
    return "other"


def _default_run_id() -> str:
    for key in ("SLURM_JOB_ID", "TORCHELASTIC_RUN_ID"):
        value = os.environ.get(key)
        if value and value.lower() != "none":
            return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    port = os.environ.get("MASTER_PORT")
    if port:
        return f"port_{port}"
    return "manual"


def _wait_for_files(paths: list[Path], timeout_seconds: float, description: str) -> None:
    deadline = time.time() + timeout_seconds
    while True:
        missing = [path for path in paths if not path.exists()]
        if not missing:
            return
        if time.time() > deadline:
            names = ", ".join(str(path) for path in missing[:8])
            extra = "" if len(missing) <= 8 else f", ... ({len(missing)} missing)"
            raise TimeoutError(f"Timed out waiting for {description}: {names}{extra}")
        time.sleep(10.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--prism", type=Path, required=True)
    parser.add_argument("--anno-path", type=Path, default=Path("data/ovo_bench/ovo_bench_new.json"))
    parser.add_argument("--chunked-dir", type=Path, default=Path("data/ovo_bench/chunked_videos"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-path", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--qa-device", default=None)
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--recent-window", type=int, default=6)
    parser.add_argument("--max-qa-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--run-id", default=None, help="Unique id for this distributed run's shard files.")
    parser.add_argument("--file-sync-timeout", type=float, default=86400.0)
    args = parser.parse_args()

    accelerator = Accelerator()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.run_id or _default_run_id())
    ready_path = args.out_dir / f"oracle_run_{run_id}.ready"
    merge_done_path = args.out_dir / f"oracle_run_{run_id}.merge_done"
    done_paths = [args.out_dir / f"oracle_rank_{run_id}_{index}.done" for index in range(accelerator.num_processes)]
    if accelerator.is_main_process:
        for pattern in (
            f"oracle_rank_{run_id}_*.jsonl",
            f"oracle_rank_{run_id}_*.done",
            f"oracle_run_{run_id}.ready",
            f"oracle_run_{run_id}.merge_done",
        ):
            for path in args.out_dir.glob(pattern):
                path.unlink()
        ready_path.write_text(str(time.time()) + "\n", encoding="utf-8")
    else:
        _wait_for_files([ready_path], args.file_sync_timeout, "main-process run setup")

    baseline_rows = _load_records(args.baseline)
    prism_rows = _load_records(args.prism)
    annotations = _load_annotations(args.anno_path)

    baseline_by_key = {_sample_key(row): row for row in baseline_rows}
    prism_by_key = {_sample_key(row): row for row in prism_rows}
    keys = sorted(set(baseline_by_key) & set(prism_by_key))
    paired: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for key in keys:
        base = baseline_by_key[key]
        prism = prism_by_key[key]
        task = str(base.get("task") or prism.get("task") or "")
        base_correct = _score_row_response(task, base, base.get("response"))
        prism_correct = _score_row_response(task, base, prism.get("response"))
        if base_correct is None or prism_correct is None:
            continue
        paired.append((key, base, prism))
    if args.max_samples > 0:
        random.seed(SEED)
        random.shuffle(paired)
        paired = paired[: args.max_samples]

    with accelerator.split_between_processes(paired) as local_pairs:
        local_pairs = list(local_pairs)

    qa = RecentWindowQAModel(
        model_name=args.model_path,
        device=args.qa_device or accelerator.device,
        max_new_tokens=args.max_qa_tokens,
        attn_implementation=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
    )

    rank_path = args.out_dir / f"oracle_rank_{run_id}_{accelerator.process_index}.jsonl"
    with rank_path.open("w", encoding="utf-8") as handle:
        for index, (key, base, prism) in enumerate(local_pairs, start=1):
            task = str(base.get("task") or prism.get("task") or "")
            annotation = _annotation_for_row(base, annotations)
            prompt = _prompt_for_row(base, annotation)
            video_path = _video_path(base, args.chunked_dir)
            queue = _candidate_queue(prism)
            candidate_ids = [int(item["chunk_id"]) for item in queue[:3] if isinstance(item.get("chunk_id"), int)]
            chunks, recent, available_memory_ids = _decode_context(
                video_path,
                candidate_ids,
                args.chunk_duration,
                args.fps,
                args.recent_window,
            )
            by_id = {int(chunk.chunk_index): chunk for chunk in chunks}
            branches: list[dict[str, Any]] = []
            minimum_correct_k: int | str = 0 if _score_row_response(task, base, base.get("response")) else "none"
            for k in range(4):
                memory_ids = [chunk_id for chunk_id in candidate_ids[:k] if chunk_id in by_id]
                memory_chunks = sorted([by_id[chunk_id] for chunk_id in memory_ids], key=lambda chunk: int(chunk.chunk_index))
                context = [*memory_chunks, *recent]
                answer_prompt = f"{_PSM_HISTORY_INSTRUCTION}{prompt}" if memory_chunks else prompt
                if k == 0:
                    response = base.get("response")
                else:
                    frames = [frame for chunk in context for frame in chunk.frames]
                    response = qa.generate_from_frames(frames, answer_prompt)
                correct = _score_row_response(task, base, response)
                if correct and minimum_correct_k == "none":
                    minimum_correct_k = k
                branches.append(
                    {
                        "k": k,
                        "memory_chunk_ids": memory_ids,
                        "context_chunk_ids": [int(chunk.chunk_index) for chunk in context],
                        "response": response,
                        "prediction": _prediction_for_task(task, response),
                        "correct": bool(correct),
                    }
                )

            adaptive = _adaptive(prism)
            iter0 = _iteration(prism, 0)
            best_total = _top_candidate_value(queue, "total_score")
            best_semantic = _top_candidate_value(queue, "semantic_score")
            current_support = _safe_float(iter0.get("visual_support_norm"))
            historical_advantage = (
                None if best_total is None or current_support is None else float(best_total - current_support)
            )
            historical_ratio = (
                None if best_total is None or current_support is None else float(best_total / (current_support + 1e-6))
            )
            row = {
                "key": key,
                "id": base.get("id"),
                "task": task,
                "group": _group(task),
                "question": base.get("question"),
                "ground_truth": _ground_truth_for_task(task, base),
                "baseline_response": base.get("response"),
                "baseline_prediction": _prediction_for_task(task, base.get("response")),
                "baseline_correct": bool(branches[0]["correct"]),
                "prism_response": prism.get("response"),
                "prism_prediction": _prediction_for_task(task, prism.get("response")),
                "prism_correct": bool(_score_row_response(task, base, prism.get("response"))),
                "prism_memory_chunk_ids": adaptive.get("memory_chunk_ids") or [],
                "prism_stop_reason": adaptive.get("stop_reason"),
                "prism_final_sufficiency": adaptive.get("final_sufficiency"),
                "candidate_queue": queue,
                "candidate_top3_chunk_ids": candidate_ids,
                "available_memory_ids": available_memory_ids,
                "branches": branches,
                "minimum_correct_k": minimum_correct_k,
                "k0_correct": bool(branches[0]["correct"]),
                "k1_correct": bool(branches[1]["correct"]),
                "k2_correct": bool(branches[2]["correct"]),
                "k3_correct": bool(branches[3]["correct"]),
                "iter0_answer_margin": iter0.get("answer_margin"),
                "iter0_entropy_confidence": iter0.get("entropy_confidence"),
                "iter0_visual_support": iter0.get("visual_support_norm"),
                "iter0_sufficiency": iter0.get("sufficiency"),
                "iter0_predicted_option": iter0.get("predicted_option"),
                "best_historical_candidate_score": best_total,
                "best_historical_semantic_score": best_semantic,
                "historical_advantage": historical_advantage,
                "historical_ratio": historical_ratio,
                "iterations": adaptive.get("iterations") or [],
            }
            row["failure_mode"] = _classify_prism_vs_oracle(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[rank {accelerator.process_index} {index}/{len(local_pairs)}] "
                f"{task} min_k={minimum_correct_k} mode={row['failure_mode']}",
                flush=True,
            )

    done_paths[accelerator.process_index].write_text(str(time.time()) + "\n", encoding="utf-8")
    if not accelerator.is_main_process:
        _wait_for_files([merge_done_path], args.file_sync_timeout, "main-process merge")
        return

    _wait_for_files(done_paths, args.file_sync_timeout, "rank shards")
    rows: list[dict[str, Any]] = []
    for path in sorted(args.out_dir.glob(f"oracle_rank_{run_id}_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())

    for row in rows:
        row["failure_mode"] = _classify_prism_vs_oracle(row)

    summary: dict[str, Any] = {
        "record_matching": {
            "baseline_records": len(baseline_rows),
            "prism_records": len(prism_rows),
            "scoreable_pairs": len(rows),
        },
        "oracle_overall": _oracle_summary(rows),
        "oracle_by_group": {name: _oracle_summary(items) for name, items in sorted(_group_by(rows, "group").items())},
        "oracle_by_task": {name: _oracle_summary(items) for name, items in sorted(_group_by(rows, "task").items())},
        "failure_modes_overall": _counter_table(rows, "failure_mode"),
        "failure_modes_by_group": _breakdown(rows, "group"),
        "failure_modes_by_task": _breakdown(rows, "task"),
    }

    false_stops = [row for row in rows if row["failure_mode"] == "C_false_stop_oracle_memory_could_rescue"]
    damaged = [row for row in rows if row["failure_mode"] == "E_retrieved_and_damaged"]
    successful_retrievals = [row for row in rows if row["failure_mode"] == "B_retrieved_and_rescued"]
    correct_stops = [row for row in rows if row["failure_mode"] == "A_correct_stop_k0"]
    oracle_helpful = [row for row in rows if row["minimum_correct_k"] in {1, 2, 3}]
    oracle_not_helpful = [row for row in rows if row["minimum_correct_k"] == "none"]

    def signal_block(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "samples": len(items),
            "iter0_answer_margin": _numeric_summary([x for row in items if (x := _safe_float(row.get("iter0_answer_margin"))) is not None]),
            "iter0_entropy_confidence": _numeric_summary([x for row in items if (x := _safe_float(row.get("iter0_entropy_confidence"))) is not None]),
            "iter0_visual_support": _numeric_summary([x for row in items if (x := _safe_float(row.get("iter0_visual_support"))) is not None]),
            "iter0_sufficiency": _numeric_summary([x for row in items if (x := _safe_float(row.get("iter0_sufficiency"))) is not None]),
            "best_historical_candidate_score": _numeric_summary([x for row in items if (x := _safe_float(row.get("best_historical_candidate_score"))) is not None]),
            "best_historical_semantic_score": _numeric_summary([x for row in items if (x := _safe_float(row.get("best_historical_semantic_score"))) is not None]),
            "historical_advantage": _numeric_summary([x for row in items if (x := _safe_float(row.get("historical_advantage"))) is not None]),
            "historical_ratio": _numeric_summary([x for row in items if (x := _safe_float(row.get("historical_ratio"))) is not None]),
        }

    summary["false_stop_analysis"] = {
        "false_stops": signal_block(false_stops),
        "correct_stops": signal_block(correct_stops),
        "successful_retrievals": signal_block(successful_retrievals),
    }
    summary["damaged_retrieval_analysis"] = {
        "samples": len(damaged),
        "first_retrieved_frame_damages": sum(1 for row in damaged if row.get("k1_correct") is False),
        "later_frames_damage": sum(
            1 for row in damaged if row.get("k1_correct") is True and (row.get("k2_correct") is False or row.get("k3_correct") is False)
        ),
        "sufficiency_rises_while_correctness_falls": sum(
            1
            for row in damaged
            if any(
                isinstance(item, dict)
                and isinstance(item.get("gain_vs_previous"), (int, float))
                and float(item["gain_vs_previous"]) > 0
                for item in row.get("iterations", [])
            )
        ),
        "details_jsonl": "damaged_retrievals.jsonl",
    }
    summary["historical_advantage_analysis"] = {
        "oracle_memory_helpful": signal_block(oracle_helpful),
        "oracle_memory_not_helpful": signal_block(oracle_not_helpful),
        "rescued": signal_block(successful_retrievals),
        "damaged": signal_block(damaged),
        "false_stops": signal_block(false_stops),
    }

    labels = [row["minimum_correct_k"] in {1, 2, 3} for row in rows if not row["k0_correct"]]
    signal_rows = [row for row in rows if not row["k0_correct"]]
    aucs: dict[str, float | None] = {}
    for key in (
        "iter0_answer_margin",
        "iter0_entropy_confidence",
        "iter0_visual_support",
        "iter0_sufficiency",
        "historical_advantage",
        "historical_ratio",
        "best_historical_candidate_score",
        "best_historical_semantic_score",
    ):
        scores = [_safe_float(row.get(key)) for row in signal_rows]
        valid_labels = [label for label, score in zip(labels, scores) if score is not None]
        valid_scores = [float(score) for score in scores if score is not None]
        aucs[key] = _roc_auc(valid_labels, valid_scores)
    summary["oracle_memory_helpful_signal_auc_on_recent6_wrong"] = aucs
    best_auc_key = max((k for k, v in aucs.items() if v is not None), key=lambda k: float(aucs[k]), default=None)
    summary["recommended_next_prism_change"] = {
        "best_scalar_signal": best_auc_key,
        "note": (
            "Do not tune yet. If historical_advantage or historical_ratio separates helpful memory better "
            "than margin/entropy, the next PRISM change should gate retrieval by historical advantage rather "
            "than by confidence alone."
        ),
    }

    csv_path = args.out_dir / "oracle_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "key", "id", "task", "group", "question", "ground_truth",
            "baseline_prediction", "baseline_correct", "prism_prediction", "prism_correct",
            "minimum_correct_k", "failure_mode", "prism_memory_chunk_ids",
            "candidate_top3_chunk_ids", "iter0_answer_margin", "iter0_entropy_confidence",
            "iter0_visual_support", "iter0_sufficiency", "best_historical_candidate_score",
            "best_historical_semantic_score", "historical_advantage", "historical_ratio",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), (list, dict)) else row.get(key) for key in fieldnames})

    with (args.out_dir / "damaged_retrievals.jsonl").open("w", encoding="utf-8") as handle:
        for row in damaged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    (args.out_dir / "oracle_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (args.out_dir / "oracle_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    merge_done_path.write_text(str(time.time()) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved: {args.out_dir}")


if __name__ == "__main__":
    main()
