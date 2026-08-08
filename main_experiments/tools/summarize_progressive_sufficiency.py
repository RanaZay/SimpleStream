#!/usr/bin/env python3
"""Summarize progressive_sufficiency_memory diagnostics and baseline deltas."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def _json_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and all(key in value for key in ("backward", "realtime", "forward")):
        return [*value["backward"], *value["realtime"], *value["forward"]]
    if isinstance(value, dict) and isinstance(value.get("results"), list):
        return value["results"]
    return []


def _load(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        return _json_rows(path)
    merged = path / "merged_results.json"
    if merged.exists():
        return _json_rows(merged)
    ovo = sorted(path.glob("minicpmv46_results_*.json"))
    if ovo:
        return _json_rows(ovo[-1])
    rows: list[dict[str, Any]] = []
    for item in sorted(path.glob("rank_*/results_incremental.jsonl")):
        rows.extend(_json_rows(item))
    return rows


def _key(row: dict[str, Any], suffix: str = "") -> str:
    return str(row.get("_key") or f"{row.get('id', row.get('video', ''))}:{row.get('task', '')}:{row.get('question', '')}{suffix}")


def _flatten(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row.get("adaptive"), dict):
            flat.append({**row, "_metric_key": _key(row)})
        for index, item in enumerate(row.get("test_info", []) or []):
            if isinstance(item, dict) and isinstance(item.get("adaptive"), dict):
                flat.append({**item, "_metric_key": _key(row, f":{index}")})
    return flat


def _number(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        adaptive = row["adaptive"]
        value = adaptive.get(key, row.get(key))
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _correct(row: dict[str, Any]) -> bool | None:
    value = row.get("correct")
    if isinstance(value, bool):
        return value
    ground_truth = str(row.get("ground_truth", "")).strip().upper()
    response = str(row.get("response", "")).upper()
    match = re.search(r"\b([A-D])\b", response)
    if ground_truth in {"A", "B", "C", "D"} and match:
        return match.group(1) == ground_truth
    return None


def _average(values: list[float]) -> float:
    return mean(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = _flatten(_load(args.result))
    if not rows:
        raise SystemExit(f"No top-level adaptive records found under {args.result}")
    distribution = Counter(int(row["adaptive"].get("num_memory_frames", 0)) for row in rows)
    scored = [row for row in rows if _correct(row) is not None]
    correct = sum(bool(_correct(row)) for row in scored)
    summary: dict[str, Any] = {
        "samples": len(rows),
        "memory_trigger_rate": sum(bool(row["adaptive"].get("memory_triggered")) for row in rows) / len(rows),
        "historical_frame_distribution": {str(k): distribution.get(k, 0) for k in range(4)},
        "accuracy": correct / len(scored) if scored else None,
        "accuracy_samples": len(scored),
        "mean_extra_frames": _average(_number(rows, "num_extra_frames")),
        "mean_sufficiency_iterations": _average(_number(rows, "num_sufficiency_iterations")),
        "mean_ttft_seconds": _average(_number(rows, "ttft_seconds")),
        "mean_end_to_end_seconds": _average(_number(rows, "end_to_end_time_seconds")),
        "peak_allocated_gpu_mb": max(_number(rows, "peak_allocated_gpu_mb"), default=0.0),
        "peak_reserved_gpu_mb": max(_number(rows, "peak_reserved_gpu_mb"), default=0.0),
        "stop_reasons": dict(Counter(str(row["adaptive"].get("stop_reason")) for row in rows)),
    }
    if args.baseline:
        baseline_rows = {_key(row): row for row in _load(args.baseline)}
        paired = [(row, baseline_rows.get(row["_metric_key"])) for row in rows]
        paired = [(row, base) for row, base in paired if base is not None and _correct(row) is not None and _correct(base) is not None]
        rescued = sum(not bool(_correct(base)) and bool(_correct(row)) for row, base in paired)
        damaged = sum(bool(_correct(base)) and not bool(_correct(row)) for row, base in paired)
        summary.update({"paired_samples": len(paired), "rescued_baseline_errors": rescued, "damaged_baseline_successes": damaged, "net_rescue": rescued - damaged})

    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
