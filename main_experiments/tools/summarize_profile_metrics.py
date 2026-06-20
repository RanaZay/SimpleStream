#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


METRIC_KEYS = [
    "num_frames",
    "num_vision_tokens",
    "decode_time",
    "preprocess_time",
    "vision_preprocess_time_ms",
    "vision_encoder_time_ms",
    "vision_resampler_time_ms",
    "vision_projector_time_ms",
    "vision_hook_subtask_time_ms",
    "vision_total_frontend_time_ms",
    "non_vision_generate_time_ms",
    "prefill_forward_time_ms",
    "decode_forward_time_ms",
    "prefill_kv_time_ms",
    "generate_first_token_time_ms",
    "generate_tokens_time_ms",
    "st_vision_tower_ms",
    "st_projector_ms",
    "st_compress_features_ms",
    "st_prefill_kv_ms",
    "st_store_kv_ms",
    "st_retrieval_forward_ms",
    "st_reconstruct_kv_ms",
    "st_generate_first_token_ms",
    "st_generate_tokens_ms",
    "model_generate_time",
    "generate_time",
    "ttft_seconds",
    "end_to_end_time",
    "gpu_peak_allocated_mb",
    "gpu_peak_reserved_mb",
    "gpu_peak_extra_allocated_mb",
    "gpu_peak_extra_reserved_mb",
]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _metric_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": mean(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "max": max(values),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _records_from_ovo_item(item: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(item.get("test_info"), list):
        rows = []
        for index, test_info in enumerate(item["test_info"]):
            if not isinstance(test_info, dict):
                continue
            row = dict(test_info)
            row.setdefault("id", item.get("id"))
            row.setdefault("task", item.get("task"))
            row.setdefault("forward_index", index)
            rows.append(row)
        return rows
    return [item]


def _load_records(result_dir: Path) -> list[dict[str, Any]]:
    jsonl_paths = []
    direct = result_dir / "results_incremental.jsonl"
    if direct.exists():
        jsonl_paths.append(direct)
    jsonl_paths.extend(sorted(result_dir.glob("rank_*/results_incremental.jsonl")))
    if jsonl_paths:
        records: list[dict[str, Any]] = []
        for path in jsonl_paths:
            for item in _load_jsonl(path):
                records.extend(_records_from_ovo_item(item))
        return records

    final_jsons = sorted(result_dir.glob("*results_*.json"))
    if not final_jsons:
        raise FileNotFoundError(f"No results_incremental.jsonl or final results JSON found under {result_dir}")
    data = json.loads(final_jsons[-1].read_text())
    records = []
    if isinstance(data.get("results"), list):
        records.extend(data["results"])
    for key in ("backward", "realtime", "forward"):
        for item in data.get(key, []):
            records.extend(_records_from_ovo_item(item))
    return records


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    values_by_key: dict[str, list[float]] = {key: [] for key in METRIC_KEYS}
    correct = 0
    scored = 0
    errors = 0
    for record in records:
        if record.get("error"):
            errors += 1
        if "correct" in record:
            scored += 1
            if record.get("correct"):
                correct += 1
        for key in METRIC_KEYS:
            value = _as_float(record.get(key))
            if value is not None:
                values_by_key[key].append(value)

    metric_summary = {key: _metric_summary(values) for key, values in values_by_key.items()}
    return {
        "records": len(records),
        "scored_records": scored,
        "correct": correct,
        "accuracy": (100.0 * correct / scored) if scored else None,
        "errors": errors,
        "metrics": metric_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize latency and GPU-memory metrics from MiniCPM eval outputs.")
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    records = _load_records(args.result_dir)
    summary = summarize(records)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
