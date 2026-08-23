#!/usr/bin/env python3
"""Summarize SB-100 CLIP-MMR candidate-override validation.

This is a read-only analysis tool. It compares a corrected Recent-6 baseline,
the current CLIP-MMR PRISM controller, and the isolated
progressive_sufficiency_memory_clip_mmr_candidate_override run on the same
StreamingBench subset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_results(path: Path) -> tuple[str, list[dict[str, Any]]]:
    if path.is_file():
        if path.suffix == ".jsonl":
            return str(path), read_jsonl(path)
        payload = json.load(path.open(encoding="utf-8"))
        if isinstance(payload, dict):
            return str(path), list(payload.get("results", payload.get("rows", [])))
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


def load_oracle(path: Path, variant: str = "clip_mmr") -> tuple[str, dict[int, dict[str, Any]]]:
    if path.is_file():
        if path.suffix == ".jsonl":
            source = path
            rows = read_jsonl(path)
        else:
            payload = json.load(path.open(encoding="utf-8"))
            source = path
            rows = payload if isinstance(payload, list) else payload.get("results", [])
    else:
        jsonl = path / "retrieval_variant_oracle_results.jsonl"
        json_path = path / "retrieval_variant_oracle_results.json"
        if jsonl.exists():
            source = jsonl
            rows = read_jsonl(jsonl)
        elif json_path.exists():
            source = json_path
            payload = json.load(json_path.open(encoding="utf-8"))
            rows = payload if isinstance(payload, list) else payload.get("results", [])
        else:
            raise FileNotFoundError(f"No retrieval_variant_oracle_results JSON/JSONL found under {path}")

    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("retrieval_variant")) == variant:
            out[int(row["question_id"])] = row
    return str(source), out


def extract_mcq_answer(response: Any) -> str | None:
    if response is None:
        return None
    text = str(response).strip().upper()
    if text in {"A", "B", "C", "D"}:
        return text
    match = re.search(r"\b([ABCD])\b", text)
    return match.group(1) if match else None


def adaptive(row: dict[str, Any]) -> dict[str, Any]:
    profile = row.get("profile") if isinstance(row.get("profile"), dict) else {}
    meta = row.get("adaptive") or profile.get("adaptive") or {}
    return meta if isinstance(meta, dict) else {}


def number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def numeric(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = number(row.get(key))
        if value is not None:
            return value
        profile = row.get("profile") if isinstance(row.get("profile"), dict) else {}
        value = number(profile.get(key))
        if value is not None:
            return value
        value = number(adaptive(row).get(key))
        if value is not None:
            return value
    return None


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "mean": statistics.mean(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "max": ordered[-1],
    }


def prediction(row: dict[str, Any]) -> str | None:
    return extract_mcq_answer(row.get("response"))


def correct(row: dict[str, Any]) -> bool:
    return bool(row.get("correct"))


def selected_timestamps_current(row: dict[str, Any]) -> list[float]:
    values = adaptive(row).get("selected_timestamps")
    if isinstance(values, list):
        return [float(value) for value in values if number(value) is not None]
    return []


def selected_timestamps_prism(row: dict[str, Any]) -> list[float]:
    meta = adaptive(row)
    value: Any = meta
    for key in ("baseline_recent_equivalence", "baseline_recent", "cdas", "selected_timestamps"):
        if not isinstance(value, dict):
            return []
        value = value.get(key)
    if isinstance(value, list):
        return [float(item) for item in value if number(item) is not None]
    return []


def close_list(left: list[float], right: list[float], eps: float = 1e-5) -> bool:
    return len(left) == len(right) and all(abs(a - b) <= eps for a, b in zip(left, right))


def list_ints(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [int(item) for item in value if isinstance(item, (int, float)) and not isinstance(item, bool)]


def memory_chunk_ids(row: dict[str, Any]) -> list[int]:
    return list_ints(adaptive(row).get("memory_chunk_ids"))


def final_historical_frames(row: dict[str, Any]) -> int:
    meta = adaptive(row)
    for key in ("final_historical_frames", "num_memory_frames"):
        value = number(meta.get(key))
        if value is not None:
            return int(value)
    ids = memory_chunk_ids(row)
    return len(ids)


def first_iteration(row: dict[str, Any]) -> dict[str, Any]:
    iterations = adaptive(row).get("iterations")
    if isinstance(iterations, list) and iterations and isinstance(iterations[0], dict):
        return iterations[0]
    return {}


def iteration_memory_ids(row: dict[str, Any]) -> list[int]:
    meta = adaptive(row)
    recent = set(list_ints(meta.get("recent_chunk_ids")))
    ids: list[int] = []
    iterations = meta.get("iterations")
    if isinstance(iterations, list):
        for item in iterations:
            if not isinstance(item, dict):
                continue
            for chunk_id in list_ints(item.get("context_chunk_ids")):
                if chunk_id not in recent and chunk_id not in ids:
                    ids.append(chunk_id)
    return ids


def rollback_yes(row: dict[str, Any]) -> bool:
    retrieved = sorted(iteration_memory_ids(row))
    final = sorted(memory_chunk_ids(row))
    return bool(retrieved) and retrieved != final


def candidate_queue_top(row: dict[str, Any]) -> dict[str, Any]:
    meta = adaptive(row)
    queue = meta.get("candidate_queue")
    if isinstance(queue, list) and queue and isinstance(queue[0], dict):
        return queue[0]
    return {}


def method_summary(name: str, rows: list[dict[str, Any]], baseline_by_id: dict[int, dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    row_by_id = {int(row["_index"]): row for row in rows}
    rescued = damaged = 0
    if baseline_by_id:
        for qid, row in row_by_id.items():
            base = baseline_by_id[qid]
            rescued += int((not correct(base)) and correct(row))
            damaged += int(correct(base) and (not correct(row)))
    hist_counts = Counter(final_historical_frames(row) for row in rows)
    trigger_count = sum(final_historical_frames(row) > 0 for row in rows)
    return {
        "method": name,
        "samples": total,
        "correct": sum(correct(row) for row in rows),
        "accuracy": sum(correct(row) for row in rows) / total if total else None,
        "rescued_vs_recent6": rescued,
        "damaged_vs_recent6": damaged,
        "net_vs_recent6": rescued - damaged,
        "trigger_rate": trigger_count / total if total else None,
        "avg_hist": statistics.mean([final_historical_frames(row) for row in rows]) if rows else None,
        "hist_frame_distribution": dict(sorted(hist_counts.items())),
        "latency_seconds": stats(
            [value for row in rows if (value := numeric(row, "end_to_end_time_seconds", "latency_seconds")) is not None]
        ),
        "ttft_seconds": stats([value for row in rows if (value := numeric(row, "ttft_seconds")) is not None]),
        "vision_tokens": stats([value for row in rows if (value := numeric(row, "num_vision_tokens")) is not None]),
        "gpu_peak_allocated_mb": stats(
            [value for row in rows if (value := numeric(row, "gpu_peak_allocated_mb", "gpu_peak_memory_mb")) is not None]
        ),
    }


def trigger_decomposition(rows: list[dict[str, Any]], gamma: float, tau: float) -> dict[str, Any]:
    counts = Counter()
    low_total = 0
    override_total = 0
    disagreement_total = 0
    relevance_total = 0
    overlap = 0
    for row in rows:
        iter0 = first_iteration(row)
        suff = number(iter0.get("sufficiency"))
        low = bool(suff is not None and suff < tau)
        relevance = number(iter0.get("top1_unused_candidate_relevance"))
        if relevance is None:
            relevance = number(candidate_queue_top(row).get("retrieval_relevance"))
        strong_rel = bool(relevance is not None and relevance >= gamma)
        disagreement = bool(iter0.get("retrieval_disagreement"))
        override = bool(iter0.get("strong_candidate_disagreement_override")) or (strong_rel and disagreement)
        if low and override:
            counts["low_sufficiency_and_strong_candidate_disagreement_override"] += 1
        elif low:
            counts["low_sufficiency"] += 1
        elif override:
            counts["strong_candidate_disagreement_override"] += 1
        else:
            counts["none"] += 1
        low_total += int(low)
        override_total += int(override)
        disagreement_total += int(disagreement)
        relevance_total += int(strong_rel)
        overlap += int(low and override)
    return {
        "reason_counts": dict(counts),
        "total_override_triggered": override_total,
        "total_retrieval_disagreement": disagreement_total,
        f"total_top1_relevance_ge_{gamma}": relevance_total,
        "low_sufficiency_trigger_count": low_total,
        "low_sufficiency_candidate_override_overlap": overlap,
    }


def temporal_violations(rows: list[dict[str, Any]]) -> int:
    total = 0
    for row in rows:
        meta = adaptive(row)
        value = number(meta.get("history_temporal_violation_count"))
        if value is not None:
            total += int(value)
        else:
            for item in meta.get("candidate_queue") or []:
                if isinstance(item, dict) and item.get("history_temporal_violation"):
                    total += 1
    return total


def validity(
    recent_rows: list[dict[str, Any]],
    mmr_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    expected_samples: int,
) -> dict[str, Any]:
    maps = {
        "recent6": {int(row["_index"]): row for row in recent_rows},
        "clip_mmr": {int(row["_index"]): row for row in mmr_rows},
        "candidate_override": {int(row["_index"]): row for row in candidate_rows},
    }
    ids = {name: set(rows) for name, rows in maps.items()}
    common = set.intersection(*ids.values())
    zero_memory = [
        row
        for row in candidate_rows
        if final_historical_frames(row) == 0 and int(row["_index"]) in maps["recent6"]
    ]
    same_ts = 0
    same_pred = 0
    mismatches = []
    for row in zero_memory:
        qid = int(row["_index"])
        recent = maps["recent6"][qid]
        recent_ts = selected_timestamps_current(recent)
        cand_ts = selected_timestamps_prism(row)
        ts_ok = close_list(recent_ts, cand_ts)
        pred_ok = prediction(recent) == prediction(row)
        same_ts += int(ts_ok)
        same_pred += int(pred_ok)
        if not ts_ok or not pred_ok:
            mismatches.append(
                {
                    "question_id": qid,
                    "same_timestamps": ts_ok,
                    "recent_timestamps": recent_ts,
                    "candidate_recent_timestamps": cand_ts,
                    "same_prediction": pred_ok,
                    "recent_prediction": prediction(recent),
                    "candidate_prediction": prediction(row),
                }
            )
    return {
        "rows": {name: len(row_map) for name, row_map in maps.items()},
        "common_ids": len(common),
        "expected_samples": expected_samples,
        "exactly_expected_common_ids": len(common) == expected_samples,
        "duplicate_ids": {
            "recent6": len(recent_rows) - len(maps["recent6"]),
            "clip_mmr": len(mmr_rows) - len(maps["clip_mmr"]),
            "candidate_override": len(candidate_rows) - len(maps["candidate_override"]),
        },
        "errors": {
            "recent6": sum(1 for row in recent_rows if row.get("error")),
            "clip_mmr": sum(1 for row in mmr_rows if row.get("error")),
            "candidate_override": sum(1 for row in candidate_rows if row.get("error")),
        },
        "temporal_history_violations": {
            "clip_mmr": temporal_violations(mmr_rows),
            "candidate_override": temporal_violations(candidate_rows),
        },
        "zero_memory_equivalence": {
            "samples": len(zero_memory),
            "same_recent_six_timestamps": same_ts,
            "same_prediction_as_recent6": same_pred,
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
        },
    }


def oracle_rescuable_cases(
    oracle_by_id: dict[int, dict[str, Any]],
    candidate_by_id: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases = []
    summary = Counter()
    for qid in sorted(oracle_by_id):
        oracle = oracle_by_id[qid]
        min_k = oracle.get("minimum_k_to_correct")
        if min_k not in {1, 2, 3}:
            continue
        row = candidate_by_id.get(qid)
        if row is None:
            continue
        iter0 = first_iteration(row)
        retrieved_ids = iteration_memory_ids(row)
        final_ids = memory_chunk_ids(row)
        actual_k = len(retrieved_ids) if retrieved_ids else len(final_ids)
        final_ok = correct(row)
        rb = rollback_yes(row)
        if final_ok:
            summary["captured_by_new_controller"] += 1
        elif not retrieved_ids and not final_ids:
            summary["still_false_stopped"] += 1
        elif rb:
            summary["lost_to_rollback"] += 1
        else:
            summary["other_failure"] += 1
        branches = oracle.get("branches") or {}
        cases.append(
            {
                "question_id": qid,
                "category": oracle.get("category") or row.get("task_type"),
                "K0_prediction": (branches.get("0") or {}).get("prediction"),
                "oracle_minimum_K": min_k,
                "actual_candidate_override_retrieved_K": actual_k,
                "actual_retrieved_chunk_ids": retrieved_ids,
                "actual_final_memory_chunk_ids": final_ids,
                "final_prediction": prediction(row),
                "correct": final_ok,
                "trigger_reason": iter0.get("retrieval_trigger_reason") or adaptive(row).get("stop_reason"),
                "rollback": rb,
            }
        )
    summary["oracle_rescuable"] = len(cases)
    return cases, dict(summary)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def print_table(summaries: list[dict[str, Any]]) -> None:
    print("method | accuracy | rescued | damaged | net | trigger_rate | avg_hist")
    for row in summaries:
        acc = 100.0 * row["accuracy"] if row["accuracy"] is not None else 0.0
        trig = 100.0 * row["trigger_rate"] if row["trigger_rate"] is not None else 0.0
        print(
            f"{row['method']} | {acc:.2f}% ({row['correct']}/{row['samples']}) | "
            f"{row['rescued_vs_recent6']} | {row['damaged_vs_recent6']} | {row['net_vs_recent6']} | "
            f"{trig:.2f}% | {row['avg_hist']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recent", required=True, help="Corrected exact-six Recent-6 result dir/JSON.")
    parser.add_argument("--clip-mmr", required=True, help="Current CLIP-MMR PRISM controller result dir/JSON.")
    parser.add_argument("--candidate-override", required=True, help="Candidate-override result dir/JSON.")
    parser.add_argument("--oracle", required=True, help="Retrieval-variant oracle result dir/JSON/JSONL.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expected-samples", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.2995)
    parser.add_argument("--tau", type=float, default=0.62)
    args = parser.parse_args()

    recent_source, recent_rows = load_results(Path(args.recent))
    mmr_source, mmr_rows = load_results(Path(args.clip_mmr))
    candidate_source, candidate_rows = load_results(Path(args.candidate_override))
    oracle_source, oracle_by_id = load_oracle(Path(args.oracle), variant="clip_mmr")

    recent_by_id = {int(row["_index"]): row for row in recent_rows}
    candidate_by_id = {int(row["_index"]): row for row in candidate_rows}

    summaries = [
        method_summary("corrected_recent6", recent_rows, recent_by_id),
        method_summary("clip_mmr_prism", mmr_rows, recent_by_id),
        method_summary("candidate_override", candidate_rows, recent_by_id),
    ]
    cases, oracle_summary = oracle_rescuable_cases(oracle_by_id, candidate_by_id)
    q98 = next((case for case in cases if int(case["question_id"]) == 98), None)

    report = {
        "sources": {
            "recent": recent_source,
            "clip_mmr": mmr_source,
            "candidate_override": candidate_source,
            "oracle": oracle_source,
        },
        "validity": validity(recent_rows, mmr_rows, candidate_rows, args.expected_samples),
        "method_summaries": summaries,
        "candidate_override_trigger_decomposition": trigger_decomposition(candidate_rows, args.gamma, args.tau),
        "oracle_rescuable_cases": cases,
        "oracle_rescuable_summary": oracle_summary,
        "q98": q98,
    }

    cand = summaries[-1]
    if cand["accuracy"] is not None and cand["accuracy"] >= 0.76 and cand["rescued_vs_recent6"] >= 4 and cand["damaged_vs_recent6"] <= 1:
        recommendation = "keep candidate override for the next controlled check"
    elif q98 and q98.get("rollback"):
        recommendation = "move to rollback fix, because admission reaches useful evidence but final selection can discard it"
    else:
        recommendation = "test exact P3 rule before any full benchmark run"
    report["recommendation"] = recommendation

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidate_override_validation_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(out_dir / "candidate_override_oracle_rescuable_cases.csv", cases)

    print("\n================================================================================")
    print("StreamingBench-100 Candidate Override Validation")
    print("================================================================================")
    print_table(summaries)
    print("\nValidity:")
    print(json.dumps(report["validity"], indent=2, ensure_ascii=False))
    print("\nTrigger decomposition:")
    print(json.dumps(report["candidate_override_trigger_decomposition"], indent=2, ensure_ascii=False))
    print("\nOracle-rescuable CLIP-MMR K0 errors:")
    for case in cases:
        print(
            f"  qid={case['question_id']} minK={case['oracle_minimum_K']} "
            f"K0={case['K0_prediction']} actualK={case['actual_candidate_override_retrieved_K']} "
            f"ids={case['actual_retrieved_chunk_ids']} final={case['final_prediction']} "
            f"correct={case['correct']} trigger={case['trigger_reason']} rollback={case['rollback']}"
        )
    print("\nOracle-rescuable summary:")
    print(json.dumps(oracle_summary, indent=2, ensure_ascii=False))
    print(f"\nQ98: {json.dumps(q98, ensure_ascii=False)}")
    print(f"\nRecommendation: {recommendation}")
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
