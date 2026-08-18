#!/usr/bin/env python3
"""Paired comparison for full StreamingBench exact-six recent sampler runs."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_results(path: Path) -> tuple[str, list[dict[str, Any]]]:
    if path.is_file():
        payload = json.load(path.open(encoding="utf-8"))
        if isinstance(payload, dict):
            return str(path), list(payload.get("results", []))
        return str(path), list(payload)
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
    rows: list[dict[str, Any]] = []
    for rank_file in rank_files:
        rows.extend(read_jsonl(rank_file))
    dedup: dict[int, dict[str, Any]] = {}
    for row in rows:
        dedup[int(row["_index"])] = row
    return "\n  ".join(str(item) for item in rank_files), [dedup[key] for key in sorted(dedup)]


def is_correct(row: dict[str, Any]) -> bool:
    return bool(row.get("correct"))


def adaptive(row: dict[str, Any]) -> dict[str, Any]:
    profile = row.get("profile") if isinstance(row.get("profile"), dict) else {}
    return row.get("adaptive") or profile.get("adaptive") or {}


def numeric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    profile = row.get("profile") if isinstance(row.get("profile"), dict) else {}
    value = profile.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    meta = adaptive(row)
    value = meta.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "min": ordered[0],
        "p25": ordered[int(0.25 * (len(ordered) - 1))],
        "median": statistics.median(ordered),
        "p75": ordered[int(0.75 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def summarize_single(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(is_correct(row) for row in rows)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row.get("task_type", ""))].append(row)
    category = {}
    for name, group in sorted(by_category.items()):
        c = sum(is_correct(row) for row in group)
        category[name] = {"correct": c, "total": len(group), "accuracy": c / len(group) if group else None}

    frame_counts = Counter(int(row.get("num_frames", 0) or 0) for row in rows)
    selected_counts = Counter(int(numeric(row, "selected_frame_count") or row.get("num_frames", 0) or 0) for row in rows)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
        "category_accuracy": category,
        "num_frames_distribution": dict(sorted(frame_counts.items())),
        "selected_frame_count_distribution": dict(sorted(selected_counts.items())),
        "temporal_span_seconds": distribution(
            [value for row in rows if (value := numeric(row, "temporal_span_seconds")) is not None]
        ),
        "mean_adjacent_spacing_seconds": distribution(
            [value for row in rows if (value := numeric(row, "mean_adjacent_spacing_seconds")) is not None]
        ),
        "vision_tokens": distribution([value for row in rows if (value := numeric(row, "num_vision_tokens")) is not None]),
        "e2e_latency_seconds": distribution(
            [value for row in rows if (value := numeric(row, "end_to_end_time_seconds")) is not None]
        ),
        "ttft_seconds": distribution([value for row in rows if (value := numeric(row, "ttft_seconds")) is not None]),
        "gpu_peak_allocated_mb": distribution(
            [value for row in rows if (value := numeric(row, "gpu_peak_allocated_mb")) is not None]
        ),
    }


def expected_ids_from_annotations(path: Path) -> set[int]:
    data = json.load(path.open(encoding="utf-8"))
    total = 0
    for entry in data:
        total += len(entry.get("questions", []))
    return set(range(total))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", required=True)
    parser.add_argument("--uniform", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    current_path, current_rows = load_results(Path(args.current))
    uniform_path, uniform_rows = load_results(Path(args.uniform))
    current_by_id = {int(row["_index"]): row for row in current_rows}
    uniform_by_id = {int(row["_index"]): row for row in uniform_rows}
    duplicate_current = len(current_by_id) != len(current_rows)
    duplicate_uniform = len(uniform_by_id) != len(uniform_rows)
    current_ids = set(current_by_id)
    uniform_ids = set(uniform_by_id)
    common_ids = current_ids & uniform_ids
    expected_ids = expected_ids_from_annotations(Path(args.annotations))

    if duplicate_current or duplicate_uniform:
        raise AssertionError(f"duplicate IDs: current={duplicate_current} uniform={duplicate_uniform}")
    if current_ids != uniform_ids:
        raise AssertionError(
            f"ID mismatch: current_only={len(current_ids - uniform_ids)} uniform_only={len(uniform_ids - current_ids)}"
        )
    if common_ids != expected_ids:
        raise AssertionError(
            f"Expected ID mismatch: missing={len(expected_ids - common_ids)} extra={len(common_ids - expected_ids)}"
        )

    paired_rows = []
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    rescued = damaged = both_correct = both_wrong = 0
    for qid in sorted(common_ids):
        cur = current_by_id[qid]
        uni = uniform_by_id[qid]
        cur_ok = is_correct(cur)
        uni_ok = is_correct(uni)
        if cur_ok and uni_ok:
            group = "CC"
            both_correct += 1
        elif (not cur_ok) and uni_ok:
            group = "WC"
            rescued += 1
        elif cur_ok and not uni_ok:
            group = "CW"
            damaged += 1
        else:
            group = "WW"
            both_wrong += 1
        category = str(cur.get("task_type", ""))
        by_category[category][group] += 1
        paired_rows.append(
            {
                "question_id": qid,
                "category": category,
                "group": group,
                "ground_truth": cur.get("answer_gt"),
                "current_prediction": cur.get("response"),
                "uniform_prediction": uni.get("response"),
                "current_correct": cur_ok,
                "uniform_correct": uni_ok,
            }
        )

    category_pairing = {}
    for category, counts in sorted(by_category.items()):
        total = sum(counts.values())
        wc = counts.get("WC", 0)
        cw = counts.get("CW", 0)
        cc = counts.get("CC", 0)
        category_pairing[category] = {
            "total": total,
            "CC": cc,
            "WC_rescued": wc,
            "CW_damaged": cw,
            "WW": counts.get("WW", 0),
            "net_rescue": wc - cw,
            "current_accuracy": (cc + cw) / total if total else None,
            "uniform_accuracy": (cc + wc) / total if total else None,
            "accuracy_delta": (wc - cw) / total if total else None,
        }

    report = {
        "current_source": current_path,
        "uniform_source": uniform_path,
        "validity": {
            "current_rows": len(current_rows),
            "uniform_rows": len(uniform_rows),
            "common_ids": len(common_ids),
            "expected_ids": len(expected_ids),
            "zero_duplicate_ids": not duplicate_current and not duplicate_uniform,
            "common_scored_ids_equal_expected_ids": common_ids == expected_ids,
            "same_denominator": len(current_rows) == len(uniform_rows) == len(common_ids) == len(expected_ids),
            "current_errors": sum(1 for row in current_rows if row.get("error")),
            "uniform_errors": sum(1 for row in uniform_rows if row.get("error")),
        },
        "current_recent6": summarize_single(current_rows),
        "uniform_dense6": summarize_single(uniform_rows),
        "paired": {
            "samples": len(common_ids),
            "CC": both_correct,
            "WC_rescued": rescued,
            "CW_damaged": damaged,
            "WW": both_wrong,
            "net_rescue": rescued - damaged,
            "category_pairing": category_pairing,
        },
        "paired_rows": paired_rows,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n============================================================")
    print("StreamingBench Full Recent Sampler Paired Comparison")
    print("============================================================")
    print(f"current_recent6: {100 * report['current_recent6']['accuracy']:.2f}% ({report['current_recent6']['correct']}/{report['current_recent6']['total']})")
    print(f"uniform_dense6:  {100 * report['uniform_dense6']['accuracy']:.2f}% ({report['uniform_dense6']['correct']}/{report['uniform_dense6']['total']})")
    print(
        f"paired: rescued={rescued} damaged={damaged} net={rescued - damaged} "
        f"same_denominator={report['validity']['same_denominator']} errors="
        f"{report['validity']['current_errors']}/{report['validity']['uniform_errors']}"
    )
    print("Per-category pairing:")
    for category, row in category_pairing.items():
        print(
            f"  {category}: current={100 * row['current_accuracy']:.2f}% "
            f"uniform={100 * row['uniform_accuracy']:.2f}% "
            f"WC={row['WC_rescued']} CW={row['CW_damaged']} net={row['net_rescue']}"
        )


if __name__ == "__main__":
    main()
