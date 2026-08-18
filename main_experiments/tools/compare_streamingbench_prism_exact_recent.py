#!/usr/bin/env python3
"""Compare corrected exact-six StreamingBench Recent-6 against PRISM."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def extract_mcq_answer(response: str | None) -> str | None:
    if response is None:
        return None
    text = str(response).strip().upper()
    if text in {"A", "B", "C", "D"}:
        return text
    match = re.search(r"\b([ABCD])\b", text)
    return match.group(1) if match else None


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


def expected_ids_from_annotations(path: Path) -> set[int]:
    data = json.load(path.open(encoding="utf-8"))
    total = 0
    for entry in data:
        total += len(entry.get("questions", []))
    return set(range(total))


def adaptive(row: dict[str, Any]) -> dict[str, Any]:
    profile = row.get("profile") if isinstance(row.get("profile"), dict) else {}
    value = row.get("adaptive") or profile.get("adaptive") or {}
    return value if isinstance(value, dict) else {}


def nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def is_correct(row: dict[str, Any]) -> bool:
    return bool(row.get("correct"))


def pred_letter(row: dict[str, Any]) -> str | None:
    return extract_mcq_answer(row.get("response"))


def number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def numeric(row: dict[str, Any], key: str) -> float | None:
    value = number(row.get(key))
    if value is not None:
        return value
    profile = row.get("profile") if isinstance(row.get("profile"), dict) else {}
    value = number(profile.get(key))
    if value is not None:
        return value
    return number(adaptive(row).get(key))


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "mean": statistics.mean(ordered),
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
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
        "category_accuracy": {
            name: {
                "correct": sum(is_correct(row) for row in group),
                "total": len(group),
                "accuracy": sum(is_correct(row) for row in group) / len(group) if group else None,
            }
            for name, group in sorted(by_category.items())
        },
        "frame_count_distribution": dict(sorted(Counter(int(row.get("num_frames", 0) or 0) for row in rows).items())),
        "vision_tokens": stats([value for row in rows if (value := numeric(row, "num_vision_tokens")) is not None]),
        "latency_seconds": stats([value for row in rows if (value := numeric(row, "end_to_end_time_seconds")) is not None]),
        "ttft_seconds": stats([value for row in rows if (value := numeric(row, "ttft_seconds")) is not None]),
        "gpu_peak_allocated_mb": stats(
            [value for row in rows if (value := numeric(row, "gpu_peak_allocated_mb")) is not None]
        ),
    }


def selected_timestamps_from_current(row: dict[str, Any]) -> list[float]:
    values = adaptive(row).get("selected_timestamps")
    if isinstance(values, list):
        return [float(value) for value in values if number(value) is not None]
    return []


def selected_timestamps_from_prism(row: dict[str, Any]) -> list[float]:
    values = nested(adaptive(row), "baseline_recent_equivalence", "baseline_recent", "cdas", "selected_timestamps")
    if isinstance(values, list):
        return [float(value) for value in values if number(value) is not None]
    return []


def close_list(a: list[float], b: list[float], eps: float = 1e-5) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= eps for x, y in zip(a, b))


def memory_frames(row: dict[str, Any]) -> int:
    meta = adaptive(row)
    value = number(meta.get("final_historical_frames"))
    if value is not None:
        return int(value)
    return int(number(meta.get("num_memory_frames")) or 0)


def memory_ids(row: dict[str, Any]) -> list[int]:
    values = adaptive(row).get("memory_chunk_ids")
    if not isinstance(values, list):
        return []
    return [int(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]


def rollback_changed(meta: dict[str, Any]) -> bool | None:
    iterations = meta.get("iterations")
    if not isinstance(iterations, list) or not iterations:
        return None
    recent_ids = {
        int(value)
        for value in meta.get("recent_chunk_ids", [])
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    stop_memory = None
    for item in reversed(iterations):
        if isinstance(item, dict):
            stop_memory = [
                int(value)
                for value in item.get("context_chunk_ids", [])
                if isinstance(value, (int, float)) and not isinstance(value, bool) and int(value) not in recent_ids
            ]
            break
    if stop_memory is None:
        return None
    return sorted(stop_memory) != sorted(memory_ids({"adaptive": meta}))


def first_iteration(meta: dict[str, Any]) -> dict[str, Any]:
    iterations = meta.get("iterations")
    if isinstance(iterations, list) and iterations and isinstance(iterations[0], dict):
        return iterations[0]
    return {}


def top_candidate(meta: dict[str, Any]) -> dict[str, Any]:
    queue = meta.get("candidate_queue")
    if isinstance(queue, list) and queue and isinstance(queue[0], dict):
        return queue[0]
    return {}


def memory_behavior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metas = [adaptive(row) for row in rows]
    triggered = [row for row in rows if memory_frames(row) > 0]
    available_history = []
    temporal_violations = 0
    for meta in metas:
        starts = meta.get("history_candidate_start_time_seconds")
        if isinstance(starts, list):
            available_history.append(float(len(starts)))
        temporal_violations += int(number(meta.get("history_temporal_violation_count")) or 0)
    stop_reasons = Counter(str(meta.get("stop_reason", "")) for meta in metas)
    hist_counts = Counter(memory_frames(row) for row in rows)
    rollbacks = [rollback_changed(meta) for meta in metas if memory_frames({"adaptive": meta}) > 0]
    return {
        "memory_trigger_rate": len(triggered) / len(rows) if rows else None,
        "memory_triggered_samples": len(triggered),
        "avg_historical_frames": statistics.mean([memory_frames(row) for row in rows]) if rows else None,
        "historical_frame_distribution": dict(sorted(hist_counts.items())),
        "stop_reasons": dict(stop_reasons),
        "available_historical_chunks": stats(available_history),
        "history_temporal_violations": temporal_violations,
        "candidate1_gate_stats": "not_applicable_for_progressive_sufficiency_memory",
        "rollback_frequency": {
            "checked": sum(value is not None for value in rollbacks),
            "final_best_memory_differs_from_stopping_memory": sum(value is True for value in rollbacks),
        },
    }


def changed_sample_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = {
        "initial_sufficiency": lambda meta: number(first_iteration(meta).get("sufficiency")),
        "initial_answer_margin": lambda meta: number(first_iteration(meta).get("answer_margin")),
        "initial_entropy_confidence": lambda meta: number(first_iteration(meta).get("entropy_confidence")),
        "initial_visual_support": lambda meta: number(first_iteration(meta).get("visual_support_norm")),
        "top_candidate_total_score": lambda meta: number(top_candidate(meta).get("total_score")),
        "top_candidate_semantic_score": lambda meta: number(top_candidate(meta).get("semantic_score")),
        "top_candidate_temporal_distance_seconds": lambda meta: number(
            top_candidate(meta).get("candidate_temporal_distance_seconds")
        ),
        "historical_frames": lambda meta: float(int(number(meta.get("final_historical_frames")) or 0)),
    }
    out: dict[str, Any] = {}
    for field, getter in fields.items():
        values = []
        for row in rows:
            value = getter(adaptive(row))
            if value is not None:
                values.append(float(value))
        out[field] = stats(values)
    out["stop_reasons"] = dict(Counter(str(adaptive(row).get("stop_reason", "")) for row in rows))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", required=True)
    parser.add_argument("--prism", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    current_source, current_rows = load_results(Path(args.current))
    prism_source, prism_rows = load_results(Path(args.prism))
    current_by_id = {int(row["_index"]): row for row in current_rows}
    prism_by_id = {int(row["_index"]): row for row in prism_rows}
    current_ids = set(current_by_id)
    prism_ids = set(prism_by_id)
    common_ids = current_ids & prism_ids
    expected_ids = expected_ids_from_annotations(Path(args.annotations))
    duplicate_current = len(current_by_id) != len(current_rows)
    duplicate_prism = len(prism_by_id) != len(prism_rows)
    if duplicate_current or duplicate_prism:
        raise AssertionError(f"duplicate IDs: current={duplicate_current} prism={duplicate_prism}")
    if current_ids != prism_ids:
        raise AssertionError(
            f"ID mismatch: current_only={len(current_ids - prism_ids)} prism_only={len(prism_ids - current_ids)}"
        )
    if common_ids != expected_ids:
        raise AssertionError(
            f"Expected ID mismatch: missing={len(expected_ids - common_ids)} extra={len(common_ids - expected_ids)}"
        )

    counts = Counter()
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    paired_rows = []
    zero_memory = []
    zero_memory_mismatches = []
    memory_triggered_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for qid in sorted(common_ids):
        current = current_by_id[qid]
        prism = prism_by_id[qid]
        cur_ok = is_correct(current)
        pri_ok = is_correct(prism)
        if cur_ok and pri_ok:
            group = "CC"
        elif (not cur_ok) and pri_ok:
            group = "WC"
        elif cur_ok and not pri_ok:
            group = "CW"
        else:
            group = "WW"
        counts[group] += 1
        category = str(current.get("task_type", ""))
        by_category[category][group] += 1
        mem_frames = memory_frames(prism)
        if mem_frames > 0:
            memory_triggered_groups[group].append(prism)
        else:
            cur_ts = selected_timestamps_from_current(current)
            pri_ts = selected_timestamps_from_prism(prism)
            same_timestamps = close_list(cur_ts, pri_ts)
            same_answer = pred_letter(current) == pred_letter(prism)
            zero_memory.append(
                {
                    "question_id": qid,
                    "same_selected_timestamps": same_timestamps,
                    "same_prediction": same_answer,
                }
            )
            if not same_timestamps or not same_answer:
                zero_memory_mismatches.append(
                    {
                        "question_id": qid,
                        "category": category,
                        "same_selected_timestamps": same_timestamps,
                        "current_timestamps": cur_ts,
                        "prism_recent_timestamps": pri_ts,
                        "same_prediction": same_answer,
                        "current_prediction": pred_letter(current),
                        "prism_prediction": pred_letter(prism),
                        "current_response": current.get("response"),
                        "prism_response": prism.get("response"),
                    }
                )
        paired_rows.append(
            {
                "question_id": qid,
                "category": category,
                "group": group,
                "ground_truth": current.get("answer_gt"),
                "current_prediction": pred_letter(current),
                "prism_prediction": pred_letter(prism),
                "prism_memory_frames": mem_frames,
                "prism_memory_chunk_ids": memory_ids(prism),
                "prism_stop_reason": adaptive(prism).get("stop_reason"),
            }
        )

    category_pairing = {}
    for category, category_counts in sorted(by_category.items()):
        total = sum(category_counts.values())
        wc = category_counts.get("WC", 0)
        cw = category_counts.get("CW", 0)
        cc = category_counts.get("CC", 0)
        category_pairing[category] = {
            "total": total,
            "CC": cc,
            "WC_rescued": wc,
            "CW_damaged": cw,
            "WW": category_counts.get("WW", 0),
            "net_rescue": wc - cw,
            "current_accuracy": (cc + cw) / total if total else None,
            "prism_accuracy": (cc + wc) / total if total else None,
            "accuracy_delta": (wc - cw) / total if total else None,
        }

    memory_triggered_summary = {}
    for group, rows in sorted(memory_triggered_groups.items()):
        memory_triggered_summary[group] = {
            "n": len(rows),
            "metrics": changed_sample_metrics(rows),
            "category_counts": dict(Counter(str(row.get("task_type", "")) for row in rows)),
        }

    report = {
        "current_source": current_source,
        "prism_source": prism_source,
        "validity": {
            "current_rows": len(current_rows),
            "prism_rows": len(prism_rows),
            "common_ids": len(common_ids),
            "expected_ids": len(expected_ids),
            "zero_duplicate_ids": not duplicate_current and not duplicate_prism,
            "common_scored_ids_equal_expected_ids": common_ids == expected_ids,
            "same_denominator": len(current_rows) == len(prism_rows) == len(common_ids) == len(expected_ids),
            "current_errors": sum(1 for row in current_rows if row.get("error")),
            "prism_errors": sum(1 for row in prism_rows if row.get("error")),
        },
        "current_recent6": summarize_single(current_rows),
        "prism": summarize_single(prism_rows),
        "paired": {
            "samples": len(common_ids),
            "CC": counts.get("CC", 0),
            "WC_rescued": counts.get("WC", 0),
            "CW_damaged": counts.get("CW", 0),
            "WW": counts.get("WW", 0),
            "net_rescue": counts.get("WC", 0) - counts.get("CW", 0),
            "category_pairing": category_pairing,
        },
        "zero_memory_equivalence": {
            "samples": len(zero_memory),
            "same_selected_timestamps": sum(item["same_selected_timestamps"] for item in zero_memory),
            "same_prediction": sum(item["same_prediction"] for item in zero_memory),
            "mismatch_count": len(zero_memory_mismatches),
            "mismatches": zero_memory_mismatches,
        },
        "memory_behavior": memory_behavior(prism_rows),
        "memory_triggered_failure_analysis": memory_triggered_summary,
        "paired_rows": paired_rows,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    cur = report["current_recent6"]
    pri = report["prism"]
    print("\n============================================================")
    print("StreamingBench Corrected Recent-6 vs PRISM")
    print("============================================================")
    print(f"current_recent6: {100 * cur['accuracy']:.2f}% ({cur['correct']}/{cur['total']})")
    print(f"PRISM:           {100 * pri['accuracy']:.2f}% ({pri['correct']}/{pri['total']})")
    print(
        f"paired: WC={counts.get('WC', 0)} CW={counts.get('CW', 0)} "
        f"net={counts.get('WC', 0) - counts.get('CW', 0)} "
        f"errors={report['validity']['current_errors']}/{report['validity']['prism_errors']}"
    )
    zm = report["zero_memory_equivalence"]
    print(
        f"zero-memory equivalence: samples={zm['samples']} "
        f"same_timestamps={zm['same_selected_timestamps']} same_prediction={zm['same_prediction']} "
        f"mismatches={zm['mismatch_count']}"
    )
    mb = report["memory_behavior"]
    print(
        f"memory: trigger_rate={100 * mb['memory_trigger_rate']:.2f}% "
        f"avg_hist_frames={mb['avg_historical_frames']:.3f} "
        f"temporal_violations={mb['history_temporal_violations']}"
    )
    print("Per-category pairing:")
    for category, row in category_pairing.items():
        print(
            f"  {category}: current={100 * row['current_accuracy']:.2f}% "
            f"PRISM={100 * row['prism_accuracy']:.2f}% "
            f"WC={row['WC_rescued']} CW={row['CW_damaged']} net={row['net_rescue']}"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
