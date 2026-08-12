#!/usr/bin/env python3
"""Summarize OVO progressive_sufficiency_memory_heg diagnostic sweeps."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ovo_constants import BACKWARD_TASKS, REAL_TIME_TASKS, extract_br_answer  # noqa: E402


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and all(key in data for key in ("backward", "realtime", "forward")):
        return [*data.get("backward", []), *data.get("realtime", []), *data.get("forward", [])]
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    return []


def _load_result_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        return _flatten(_load_json_rows(path))
    merged = path / "merged_results.json"
    if merged.exists():
        return _flatten(_load_json_rows(merged))
    snapshots = sorted(path.glob("minicpmv46_results_*.json"))
    if snapshots:
        return _flatten(_load_json_rows(snapshots[-1]))
    rows: list[dict[str, Any]] = []
    for item in sorted(path.glob("rank_*/results_incremental.jsonl")):
        rows.extend(_load_json_rows(item))
    return _flatten(rows)


def _flatten(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in rows:
        if row.get("task") in set(BACKWARD_TASKS + REAL_TIME_TASKS):
            flat.append(row)
        for index, item in enumerate(row.get("test_info", []) or []):
            if isinstance(item, dict):
                flat.append({**item, "_parent_id": row.get("id"), "_question_index": index, "task": row.get("task")})
    return flat


def _metric_key(row: dict[str, Any]) -> str:
    return str(
        row.get("_key")
        or "|".join(
            str(row.get(key, ""))
            for key in ("id", "_parent_id", "_question_index", "video", "task", "question")
        )
    )


def _prediction(row: dict[str, Any]) -> str | None:
    value = row.get("prediction") or row.get("pred")
    if isinstance(value, str):
        pred = extract_br_answer(value)
        if pred:
            return pred
    return extract_br_answer(row.get("response"))


def _correct(row: dict[str, Any]) -> bool | None:
    value = row.get("correct")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    gt = str(row.get("ground_truth") or row.get("answer_gt") or "").strip().upper()
    pred = _prediction(row)
    if gt in {"A", "B", "C", "D"} and pred:
        return pred == gt
    return None


def _adaptive(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("adaptive"), dict):
        return row["adaptive"]
    profile = row.get("profile")
    if isinstance(profile, dict) and isinstance(profile.get("adaptive"), dict):
        return profile["adaptive"]
    return {}


def _numbers(rows: Iterable[dict[str, Any]], *keys: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        adaptive = _adaptive(row)
        for key in keys:
            value = adaptive.get(key, row.get(key))
            if isinstance(value, (int, float)):
                values.append(float(value))
                break
    return values


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _percent(count: int, total: int) -> float | None:
    return count / total if total else None


def _trigger_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for item in _adaptive(row).get("iterations", []) or []:
            reason = item.get("retrieval_trigger_reason")
            if reason:
                counts[str(reason)] += 1
    return counts


def _false_stop(row: dict[str, Any]) -> bool:
    if _correct(row) is not False:
        return False
    meta = _adaptive(row)
    return str(meta.get("stop_reason")) in {"sufficient_evidence", "sufficient_no_historical_advantage"}


def _parse_csv_value(value: str) -> Any:
    text = value.strip()
    if text == "":
        return None
    if text in {"True", "False"}:
        return text == "True"
    if text == "None":
        return None
    if text and text[0] in "[{\"'":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return value
    return value


def _load_oracle_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    if path.is_dir():
        source = path / "oracle_rows.jsonl"
        if not source.exists():
            return []
        path = source
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return [{key: _parse_csv_value(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    return _load_json_rows(path)


def _oracle_key(row: dict[str, Any]) -> str:
    return str(row.get("_key") or row.get("key") or _metric_key(row))


def _is_known_oracle_false_stop(row: dict[str, Any]) -> bool:
    mode = str(row.get("failure_mode") or row.get("oracle_failure_mode") or "")
    if "false_stop" in mode:
        return True
    if row.get("k0_correct") is False and row.get("minimum_correct_k") in {1, 2, 3, "1", "2", "3"}:
        return True
    if row.get("baseline_correct") is False and row.get("oracle_correct") is True:
        return True
    return False


def _heg_activated(row: dict[str, Any]) -> bool:
    for item in _adaptive(row).get("iterations", []) or []:
        if str(item.get("retrieval_trigger_reason")) in {
            "historical_evidence_gain",
            "low_sufficiency_and_historical_gain",
        }:
            return True
    return False


def _summarize_run(name: str, rows: list[dict[str, Any]], baseline_by_key: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if _correct(row) is not None]
    correct_count = sum(bool(_correct(row)) for row in scored)
    memory_frames = [int(_adaptive(row).get("num_memory_frames", 0) or 0) for row in rows]
    dist = Counter(memory_frames)
    paired = [(row, baseline_by_key.get(_metric_key(row))) for row in scored]
    paired = [(row, base) for row, base in paired if base is not None and _correct(base) is not None]
    rescued = sum(_correct(base) is False and _correct(row) is True for row, base in paired)
    damaged = sum(_correct(base) is True and _correct(row) is False for row, base in paired)
    baseline_correct = [(row, base) for row, base in paired if _correct(base) is True]
    new_triggers_on_baseline_correct = sum(_heg_activated(row) for row, _base in baseline_correct)
    damaged_after_new_trigger = sum(_heg_activated(row) and _correct(row) is False for row, _base in baseline_correct)
    return {
        "name": name,
        "samples": len(rows),
        "scored_samples": len(scored),
        "accuracy": correct_count / len(scored) if scored else None,
        "accuracy_count": correct_count,
        "backward_accuracy": _split_accuracy(scored, set(BACKWARD_TASKS)),
        "realtime_accuracy": _split_accuracy(scored, set(REAL_TIME_TASKS)),
        "rescued": rescued,
        "damaged": damaged,
        "net_rescue": rescued - damaged,
        "paired_with_baseline": len(paired),
        "false_stops": sum(_false_stop(row) for row in scored),
        "memory_trigger_rate": _percent(sum(bool(_adaptive(row).get("memory_triggered")) for row in rows), len(rows)),
        "avg_historical_frames": _mean([float(value) for value in memory_frames]),
        "historical_frame_counts": {str(index): dist.get(index, 0) for index in range(4)},
        "historical_frame_percent": {str(index): _percent(dist.get(index, 0), len(rows)) for index in range(4)},
        "trigger_counts": dict(_trigger_counts(rows)),
        "mean_latency_seconds": _mean(_numbers(rows, "end_to_end_time_seconds", "end_to_end_time")),
        "mean_ttft_seconds": _mean(_numbers(rows, "ttft_seconds")),
        "peak_gpu_allocated_mb": max(_numbers(rows, "gpu_peak_allocated_mb", "peak_allocated_gpu_mb"), default=None),
        "peak_gpu_reserved_mb": max(_numbers(rows, "gpu_peak_reserved_mb", "peak_reserved_gpu_mb"), default=None),
        "baseline_correct_new_heg_triggers": new_triggers_on_baseline_correct,
        "baseline_correct_damaged_after_new_heg_trigger": damaged_after_new_trigger,
    }


def _split_accuracy(rows: list[dict[str, Any]], tasks: set[str]) -> dict[str, Any]:
    subset = [row for row in rows if row.get("task") in tasks]
    correct = sum(bool(_correct(row)) for row in subset)
    return {"samples": len(subset), "accuracy": correct / len(subset) if subset else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True, help="Current PRISM result dir on the same subset.")
    parser.add_argument("--run", action="append", default=[], help="NAME=RESULT_DIR. Repeat for HEG thresholds.")
    parser.add_argument("--oracle-run", type=Path, help="Optional oracle_rows.jsonl/CSV source for known false-stop tracking.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    baseline_rows = _load_result_rows(args.baseline)
    baseline_by_key = {_metric_key(row): row for row in baseline_rows}
    summary = {
        "baseline": _summarize_run("baseline", baseline_rows, baseline_by_key),
        "runs": [],
        "oracle_false_stops": {},
    }
    for item in args.run:
        if "=" not in item:
            raise ValueError("--run must use NAME=RESULT_DIR")
        name, value = item.split("=", 1)
        rows = _load_result_rows(Path(value))
        summary["runs"].append(_summarize_run(name, rows, baseline_by_key))

    oracle_rows = [row for row in _load_oracle_rows(args.oracle_run) if _is_known_oracle_false_stop(row)]
    if oracle_rows:
        oracle_keys = {_oracle_key(row) for row in oracle_rows}
        summary["oracle_false_stops"]["known_count"] = len(oracle_keys)
        by_run: dict[str, Any] = {}
        for item in args.run:
            name, value = item.split("=", 1)
            rows_by_key = {_metric_key(row): row for row in _load_result_rows(Path(value))}
            matched = [rows_by_key[key] for key in oracle_keys if key in rows_by_key]
            by_run[name] = {
                "matched": len(matched),
                "activated": sum(_heg_activated(row) for row in matched),
                "correct": sum(_correct(row) is True for row in matched),
                "remain_wrong": sum(_correct(row) is False for row in matched),
            }
        summary["oracle_false_stops"]["by_run"] = by_run

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
