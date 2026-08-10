#!/usr/bin/env python3
"""Sample-level Recent-6 vs PRISM rescue/damage analysis.

This script is intentionally read-only with respect to model code. It consumes
completed result directories and writes CSV/JSON diagnostics for sufficiency
controller tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


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


def _load_results(result_dir: Path) -> list[dict[str, Any]]:
    if result_dir.is_file():
        return _flatten(_read_json_rows(result_dir))

    merged = result_dir / "merged_results.json"
    if merged.exists():
        return _flatten(_read_json_rows(merged))

    rows: list[dict[str, Any]] = []
    for path in sorted(result_dir.glob("rank_*/results_incremental.jsonl")):
        for row in _read_json_rows(path):
            row["_file"] = str(path)
            rows.append(row)
    if rows:
        return _flatten(rows)

    ovo_jsons = sorted(result_dir.glob("minicpmv46_results_*.json"))
    if ovo_jsons:
        return _flatten(_read_json_rows(ovo_jsons[-1]))

    raise FileNotFoundError(f"No supported result records found under {result_dir}")


def _flatten(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row.get("test_info"), list):
            for index, item in enumerate(row["test_info"]):
                if isinstance(item, dict):
                    flat.append({**row, **item, "_question_index": index})
        else:
            flat.append(row)
    return flat


def _extract_letter(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"A", "B", "C", "D", "E"}:
        return text
    match = re.search(r"\b([A-E])\b", text)
    return match.group(1) if match else ""


def _correct(row: dict[str, Any]) -> bool | None:
    value = row.get("correct")
    if isinstance(value, bool):
        return value
    gt = _extract_letter(row.get("ground_truth") or row.get("answer_gt") or row.get("answer"))
    pred = _extract_letter(row.get("prediction") or row.get("response"))
    if gt and pred:
        return gt == pred
    return None


def _prediction(row: dict[str, Any]) -> str:
    return _extract_letter(row.get("prediction") or row.get("response"))


def _ground_truth(row: dict[str, Any]) -> str:
    return _extract_letter(row.get("ground_truth") or row.get("answer_gt") or row.get("answer"))


def _task(row: dict[str, Any]) -> str:
    value = row.get("task") or row.get("task_type") or row.get("category") or ""
    return str(value)


def _benchmark_group(benchmark: str, task: str) -> str:
    if benchmark.lower() == "ovo":
        return OVO_GROUPS.get(task, "unknown")
    return task or "unknown"


def _row_key(row: dict[str, Any]) -> str:
    if row.get("_key"):
        return str(row["_key"])
    parts = [
        row.get("id", ""),
        row.get("video", ""),
        row.get("video_path", ""),
        row.get("task", row.get("task_type", "")),
        row.get("time_stamp", ""),
        row.get("question", ""),
        row.get("_question_index", ""),
    ]
    return "|".join(str(part) for part in parts)


def _adaptive(row: dict[str, Any]) -> dict[str, Any]:
    profile = row.get("profile")
    if isinstance(profile, dict) and isinstance(profile.get("adaptive"), dict):
        return profile["adaptive"]
    adaptive = row.get("adaptive")
    return adaptive if isinstance(adaptive, dict) else {}


def _iteration(adaptive: dict[str, Any], index: int) -> dict[str, Any]:
    iterations = adaptive.get("iterations")
    if isinstance(iterations, list) and len(iterations) > index and isinstance(iterations[index], dict):
        return iterations[index]
    return {}


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _retrieved_candidate_scores(adaptive: dict[str, Any]) -> list[dict[str, Any]]:
    memory_ids = [int(value) for value in adaptive.get("memory_chunk_ids", []) or []]
    queue = adaptive.get("candidate_queue", []) or []
    by_id = {
        int(item.get("chunk_id")): item
        for item in queue
        if isinstance(item, dict) and isinstance(item.get("chunk_id"), int)
    }
    return [
        {
            "chunk_id": chunk_id,
            "candidate_score": by_id.get(chunk_id, {}).get("total_score"),
            "semantic_score": by_id.get(chunk_id, {}).get("semantic_score"),
            "event_score": by_id.get(chunk_id, {}).get("event_score"),
            "detail_score": by_id.get(chunk_id, {}).get("detail_score"),
            "cue_score": by_id.get(chunk_id, {}).get("cue_score"),
        }
        for chunk_id in memory_ids
    ]


def _gains(adaptive: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for item in adaptive.get("iterations", []) or []:
        if isinstance(item, dict) and isinstance(item.get("gain_vs_previous"), (int, float)):
            values.append(float(item["gain_vs_previous"]))
    return values


def _classification(base_correct: bool, prism_correct: bool) -> str:
    if not base_correct and prism_correct:
        return "rescued"
    if base_correct and not prism_correct:
        return "damaged"
    if base_correct and prism_correct:
        return "both_correct"
    return "both_wrong"


def _summarize_numeric(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return {"mean": None, "median": None}
    return {"mean": mean(values), "median": median(values)}


def _summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label = defaultdict(list)
    by_group = defaultdict(list)
    by_task = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
        by_group[row["group"]].append(row)
        by_task[row["task"]].append(row)

    def block(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "samples": len(items),
            "labels": dict(Counter(row["label"] for row in items)),
            "stop_reasons": dict(Counter(row["stop_reason"] for row in items if row["stop_reason"])),
            "initial_margin": _summarize_numeric(items, "iter0_answer_margin"),
            "entropy_confidence": _summarize_numeric(items, "iter0_entropy_confidence"),
            "visual_support": _summarize_numeric(items, "iter0_visual_support"),
            "initial_sufficiency": _summarize_numeric(items, "iter0_sufficiency"),
            "num_historical_frames": _summarize_numeric(items, "num_historical_frames"),
        }

    return {
        "overall": block(rows),
        "by_label": {label: block(items) for label, items in sorted(by_label.items())},
        "by_group": {group: block(items) for group, items in sorted(by_group.items())},
        "by_task": {task: block(items) for task, items in sorted(by_task.items())},
    }


def analyze_pair(benchmark: str, baseline_dir: Path, prism_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_rows = _load_results(baseline_dir)
    prism_rows = _load_results(prism_dir)
    baseline_by_key = {_row_key(row): row for row in baseline_rows}
    prism_by_key = {_row_key(row): row for row in prism_rows}
    common_keys = sorted(set(baseline_by_key) & set(prism_by_key))

    rows: list[dict[str, Any]] = []
    unmatched = {
        "baseline_only": len(set(baseline_by_key) - set(prism_by_key)),
        "prism_only": len(set(prism_by_key) - set(baseline_by_key)),
    }
    for key in common_keys:
        base = baseline_by_key[key]
        prism = prism_by_key[key]
        base_correct = _correct(base)
        prism_correct = _correct(prism)
        if base_correct is None or prism_correct is None:
            continue
        adaptive = _adaptive(prism)
        iter0 = _iteration(adaptive, 0)
        task = _task(prism) or _task(base)
        row = {
            "benchmark": benchmark,
            "sample_key": key,
            "task": task,
            "group": _benchmark_group(benchmark, task),
            "question": prism.get("question") or base.get("question") or "",
            "ground_truth": _ground_truth(prism) or _ground_truth(base),
            "recent6_prediction": _prediction(base),
            "prism_prediction": _prediction(prism),
            "recent6_correct": bool(base_correct),
            "prism_correct": bool(prism_correct),
            "label": _classification(bool(base_correct), bool(prism_correct)),
            "memory_triggered": bool(adaptive.get("memory_triggered")),
            "num_historical_frames": len(adaptive.get("memory_chunk_ids", []) or []),
            "stop_reason": adaptive.get("stop_reason", ""),
            "iter0_sufficiency": _num(iter0.get("sufficiency")),
            "iter0_answer_margin": _num(iter0.get("answer_margin")),
            "iter0_entropy_confidence": _num(iter0.get("entropy_confidence")),
            "iter0_visual_support": _num(iter0.get("visual_support_norm")),
            "final_sufficiency": _num(adaptive.get("final_sufficiency")),
            "retrieved_chunk_ids": json.dumps(adaptive.get("memory_chunk_ids", []) or []),
            "retrieved_candidate_scores": json.dumps(_retrieved_candidate_scores(adaptive)),
            "sufficiency_gains": json.dumps(_gains(adaptive)),
        }
        rows.append(row)
    metadata = {
        "benchmark": benchmark,
        "baseline_records": len(baseline_rows),
        "prism_records": len(prism_rows),
        "matched_records": len(rows),
        **unmatched,
    }
    return rows, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ovo-baseline", type=Path)
    parser.add_argument("--ovo-prism", type=Path)
    parser.add_argument("--streamingbench-baseline", type=Path)
    parser.add_argument("--streamingbench-prism", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    pairs = [
        ("OVO", args.ovo_baseline, args.ovo_prism),
        ("StreamingBench", args.streamingbench_baseline, args.streamingbench_prism),
    ]
    for benchmark, baseline_dir, prism_dir in pairs:
        if baseline_dir is None or prism_dir is None:
            continue
        rows, pair_metadata = analyze_pair(benchmark, baseline_dir, prism_dir)
        all_rows.extend(rows)
        metadata.append(pair_metadata)

    if not all_rows:
        raise SystemExit("No matched rows produced. Check the result directories.")

    csv_path = args.out_dir / "prism_rescue_damage_samples.csv"
    fieldnames = list(all_rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    json_path = args.out_dir / "prism_rescue_damage_summary.json"
    report = {"inputs": metadata, "summary": _summaries(all_rows)}
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved samples: {csv_path}")
    print(f"saved summary: {json_path}")
    print(json.dumps(report["summary"]["overall"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
