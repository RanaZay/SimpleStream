#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def _grid_params(key: str) -> tuple[float, float, int]:
    match = re.fullmatch(r"s(-?\d+\.\d+)_g(\d+\.\d+)_r(\d+)", key)
    if not match:
        raise ValueError(f"Unexpected ledger grid key: {key}")
    return float(match.group(1)), float(match.group(2)), int(match.group(3))


def evaluate_policy(records: list[dict[str, Any]], *, base_key: str, grid_key: str) -> dict[str, Any]:
    score_threshold, merge_gap_seconds, min_positive_run = _grid_params(grid_key)
    total = len(records)
    base_correct = 0
    final_correct = 0
    triggered = 0
    rescued = 0
    damaged = 0
    rejected_no_map = 0
    rejected_not_cumulative = 0
    for row in records:
        base_ok = bool(row.get(f"{base_key}_correct"))
        base_correct += int(base_ok)
        ledger = row["ledger_grid"][grid_key]
        use_ledger = bool(row.get("cumulative_count_required") and ledger.get("mapped_option"))
        if not row.get("cumulative_count_required"):
            rejected_not_cumulative += 1
        elif not ledger.get("mapped_option"):
            rejected_no_map += 1
        final_pred = ledger["mapped_option"] if use_ledger else row.get(f"{base_key}_prediction")
        final_ok = bool(final_pred == row.get("ground_truth"))
        final_correct += int(final_ok)
        triggered += int(use_ledger)
        rescued += int((not base_ok) and final_ok)
        damaged += int(base_ok and not final_ok)
    return {
        "base_key": base_key,
        "grid_key": grid_key,
        "score_threshold": score_threshold,
        "merge_gap_seconds": merge_gap_seconds,
        "min_positive_run": min_positive_run,
        "base_correct": base_correct,
        "base_accuracy": base_correct / total if total else None,
        "final_correct": final_correct,
        "final_accuracy": final_correct / total if total else None,
        "triggered": triggered,
        "trigger_rate": triggered / total if total else None,
        "rescued": rescued,
        "damaged": damaged,
        "net": rescued - damaged,
        "rejected_no_map": rejected_no_map,
        "rejected_not_cumulative": rejected_not_cumulative,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge StreamingBench count-ledger shard outputs.")
    parser.add_argument("--shard-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    shard_files = sorted(args.shard_root.glob("shard_*/count_ledger_override_records.json"))
    if not shard_files:
        raise FileNotFoundError(f"No shard records found under {args.shard_root}/shard_*")

    records: list[dict[str, Any]] = []
    for path in shard_files:
        payload = json.load(path.open(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"Expected list in {path}")
        records.extend(payload)

    by_id: dict[int, dict[str, Any]] = {}
    duplicates: list[int] = []
    for row in records:
        qid = int(row["question_id"])
        if qid in by_id:
            duplicates.append(qid)
        by_id[qid] = row
    records = [by_id[qid] for qid in sorted(by_id)]
    grid_keys = sorted(set.intersection(*(set(row["ledger_grid"]) for row in records))) if records else []

    policies: list[dict[str, Any]] = []
    for base_key in ("recent", "prism"):
        for grid_key in grid_keys:
            policies.append(evaluate_policy(records, base_key=base_key, grid_key=grid_key))
    policies.sort(key=lambda item: (item["final_correct"], item["net"], -item["triggered"]), reverse=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "shard_root": str(args.shard_root),
        "shard_files": [str(path) for path in shard_files],
        "samples": len(records),
        "duplicate_ids": sorted(duplicates),
        "counting_questions": len(records),
        "cumulative_count_required": sum(1 for row in records if row.get("cumulative_count_required")),
        "recent_correct": sum(1 for row in records if row.get("recent_correct")),
        "prism_correct": sum(1 for row in records if row.get("prism_correct")),
        "policies": policies,
        "best_policy": policies[0] if policies else None,
        "mean_timing_ms": mean([row["timing_ms"] for row in records]) if records else None,
        "mean_historical_chunks": mean([row["decode"]["historical_chunks"] for row in records]) if records else None,
        "pattern_counts": dict(Counter("movie_clip" if "movie clip" in row["question"].lower() else "other" for row in records)),
    }
    (args.out_dir / "count_ledger_override_records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out_dir / "count_ledger_override_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.out_dir / "count_ledger_override_policies.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(policies[0]) if policies else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(policies)

    print("\n" + "=" * 80)
    print("StreamingBench Count Ledger Shard Summary")
    print("=" * 80)
    print(f"shards: {len(shard_files)}")
    print(f"samples: {summary['samples']}")
    print(f"duplicate_ids: {len(duplicates)}")
    print(f"cumulative_count_required: {summary['cumulative_count_required']}")
    print(f"Recent-6 Counting: {summary['recent_correct']}/{summary['samples']} = {100*summary['recent_correct']/summary['samples']:.2f}%")
    print(f"PRISM Counting: {summary['prism_correct']}/{summary['samples']} = {100*summary['prism_correct']/summary['samples']:.2f}%")
    print("\nTOP POLICIES")
    print("base,threshold,gap,minrun,final,acc,rescue,damage,net,trigger")
    for item in policies[:15]:
        print(
            f"{item['base_key']},"
            f"{item['score_threshold']:.3f},"
            f"{item['merge_gap_seconds']:.1f},"
            f"{item['min_positive_run']},"
            f"{item['final_correct']}/{summary['samples']},"
            f"{100*item['final_accuracy']:.2f}%,"
            f"{item['rescued']},"
            f"{item['damaged']},"
            f"{item['net']},"
            f"{item['triggered']}"
        )
    print(f"\nSaved: {args.out_dir}")


if __name__ == "__main__":
    main()
