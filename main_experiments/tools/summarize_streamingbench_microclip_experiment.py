#!/usr/bin/env python3
"""Summarize StreamingBench-100 PRISM microclip controls.

This is read-only: it compares saved result files by question/sample ID and
reports Recent-6 vs anchor-only vs temporal-microclip vs sparse-history-3.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_results(path: Path) -> tuple[str, dict[int, dict[str, Any]]]:
    if path.is_file():
        payload = json.load(path.open(encoding="utf-8"))
        rows = payload.get("results", payload) if isinstance(payload, dict) else payload
        return str(path), {int(row.get("_index", row.get("question_id"))): row for row in rows}

    merged = sorted(
        path.glob("streaming_bench_minicpmv46_results_*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if merged:
        return load_results(merged[0])

    rank_files = sorted(path.glob("rank_*/results_incremental.jsonl"))
    if not rank_files:
        raise FileNotFoundError(f"No StreamingBench results found under {path}")
    rows_by_id: dict[int, dict[str, Any]] = {}
    for rank_file in rank_files:
        with rank_file.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows_by_id[int(row.get("_index", row.get("question_id")))] = row
    return "\n  ".join(str(item) for item in rank_files), rows_by_id


def correct(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("correct"))


def category(row: dict[str, Any] | None) -> str:
    if not row:
        return "unknown"
    return str(row.get("task_type") or row.get("category") or "unknown")


def adaptive(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    profile = row.get("profile") if isinstance(row.get("profile"), dict) else {}
    return row.get("adaptive") or profile.get("adaptive") or {}


def number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def hist_frames(row: dict[str, Any] | None) -> int:
    meta = adaptive(row)
    value = meta.get("final_historical_frames", meta.get("num_memory_frames", 0))
    return int(value) if isinstance(value, (int, float)) else 0


def vision_tokens(row: dict[str, Any] | None) -> float | None:
    if not row:
        return None
    value = number(row.get("num_vision_tokens"))
    if value is not None:
        return value
    profile = row.get("profile") if isinstance(row.get("profile"), dict) else {}
    return number(profile.get("num_vision_tokens"))


def latency(row: dict[str, Any] | None) -> float | None:
    if not row:
        return None
    profile = row.get("profile") if isinstance(row.get("profile"), dict) else {}
    for key in ("end_to_end_time_seconds", "model_generate_time_seconds", "generate_time"):
        value = number(profile.get(key, row.get(key)))
        if value is not None:
            return value
    return None


def ttft(row: dict[str, Any] | None) -> float | None:
    if not row:
        return None
    value = number(row.get("ttft_seconds"))
    if value is not None:
        return value
    profile = row.get("profile") if isinstance(row.get("profile"), dict) else {}
    return number(profile.get("ttft_seconds"))


def summarize_method(
    *,
    rows: dict[int, dict[str, Any]],
    recent_rows: dict[int, dict[str, Any]],
    sample_ids: list[int],
) -> dict[str, Any]:
    total = len(sample_ids)
    method_correct = sum(correct(rows.get(qid)) for qid in sample_ids)
    recent_correct = sum(correct(recent_rows.get(qid)) for qid in sample_ids)
    rescued = sum(not correct(recent_rows.get(qid)) and correct(rows.get(qid)) for qid in sample_ids)
    damaged = sum(correct(recent_rows.get(qid)) and not correct(rows.get(qid)) for qid in sample_ids)
    hist_counts = Counter(hist_frames(rows.get(qid)) for qid in sample_ids)
    return {
        "total": total,
        "correct": method_correct,
        "accuracy": method_correct / total if total else None,
        "recent_correct": recent_correct,
        "rescued": rescued,
        "damaged": damaged,
        "net_rescue": rescued - damaged,
        "historical_frame_usage": dict(sorted(hist_counts.items())),
        "mean_historical_frames": mean([float(hist_frames(rows.get(qid))) for qid in sample_ids]),
        "mean_vision_tokens": mean([value for qid in sample_ids if (value := vision_tokens(rows.get(qid))) is not None]),
        "mean_latency_seconds": mean([value for qid in sample_ids if (value := latency(rows.get(qid))) is not None]),
        "mean_ttft_seconds": mean([value for qid in sample_ids if (value := ttft(rows.get(qid))) is not None]),
    }


def category_breakdown(
    *,
    rows: dict[int, dict[str, Any]],
    recent_rows: dict[int, dict[str, Any]],
    sample_ids: list[int],
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for qid in sample_ids:
        groups[category(rows.get(qid) or recent_rows.get(qid))].append(qid)
    out: dict[str, Any] = {}
    for name, ids in sorted(groups.items()):
        out[name] = summarize_method(rows=rows, recent_rows=recent_rows, sample_ids=ids)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recent6", required=True)
    parser.add_argument("--anchor-only", required=True)
    parser.add_argument("--microclip", required=True)
    parser.add_argument("--sparse-top3", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", default="")
    args = parser.parse_args()

    recent_source, recent_rows = load_results(Path(args.recent6))
    anchor_source, anchor_rows = load_results(Path(args.anchor_only))
    microclip_source, microclip_rows = load_results(Path(args.microclip))
    sparse_source, sparse_rows = load_results(Path(args.sparse_top3))

    sample_ids = sorted(set(recent_rows) & set(anchor_rows) & set(microclip_rows) & set(sparse_rows))
    if not sample_ids:
        raise ValueError("No matched question IDs across all runs")

    methods = {
        "anchor_only": anchor_rows,
        "microclip": microclip_rows,
        "sparse_top3": sparse_rows,
    }
    summary: dict[str, Any] = {
        "sources": {
            "recent6": recent_source,
            "anchor_only": anchor_source,
            "microclip": microclip_source,
            "sparse_top3": sparse_source,
        },
        "matched_samples": len(sample_ids),
        "recent6": summarize_method(rows=recent_rows, recent_rows=recent_rows, sample_ids=sample_ids),
        "methods": {},
        "category_breakdown": {},
    }
    for name, rows in methods.items():
        summary["methods"][name] = summarize_method(rows=rows, recent_rows=recent_rows, sample_ids=sample_ids)
        summary["category_breakdown"][name] = category_breakdown(rows=rows, recent_rows=recent_rows, sample_ids=sample_ids)

    oracle_recent_anchor_micro = sum(
        correct(recent_rows.get(qid)) or correct(anchor_rows.get(qid)) or correct(microclip_rows.get(qid))
        for qid in sample_ids
    )
    recent_wrong = [qid for qid in sample_ids if not correct(recent_rows.get(qid))]
    microclip_only_rescues = [
        qid
        for qid in recent_wrong
        if correct(microclip_rows.get(qid)) and not correct(anchor_rows.get(qid))
    ]
    summary["oracle_recent_anchor_microclip"] = {
        "correct": oracle_recent_anchor_micro,
        "accuracy": oracle_recent_anchor_micro / len(sample_ids),
        "headroom_vs_recent6": (
            oracle_recent_anchor_micro - summary["recent6"]["correct"]
        ) / len(sample_ids),
        "recent6_wrong_samples": len(recent_wrong),
        "microclip_rescues_not_anchor": len(microclip_only_rescues),
        "microclip_rescue_question_ids_not_anchor": microclip_only_rescues,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.out_csv:
        csv_path = Path(args.out_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["method", "total", "correct", "accuracy", "rescued", "damaged", "net_rescue"],
            )
            writer.writeheader()
            writer.writerow({"method": "recent6", **{key: summary["recent6"][key] for key in writer.fieldnames[1:]}})
            for name in methods:
                writer.writerow({"method": name, **{key: summary["methods"][name][key] for key in writer.fieldnames[1:]}})

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
