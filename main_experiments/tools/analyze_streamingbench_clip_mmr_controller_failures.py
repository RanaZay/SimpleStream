#!/usr/bin/env python3
"""Join CLIP-MMR oracle branches with actual progressive PRISM trajectories.

This is a read-only diagnostic. It does not run MiniCPM inference and does not
modify PRISM behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_json_file(path: Path, pattern: str) -> Path:
    if path.is_file():
        return path
    matches = sorted(path.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No {pattern} found under {path}")
    return matches[0]


def load_oracle(path: Path, variant: str) -> tuple[Path, dict[int, dict[str, Any]]]:
    if path.is_dir():
        jsonl = path / "retrieval_variant_oracle_results.jsonl"
        if jsonl.exists():
            rows = read_jsonl(jsonl)
            return jsonl, {
                int(row["question_id"]): row
                for row in rows
                if str(row.get("retrieval_variant")) == variant
            }
        json_path = path / "retrieval_variant_oracle_results.json"
    else:
        json_path = path
    rows = read_json(json_path)
    return json_path, {
        int(row["question_id"]): row
        for row in rows
        if str(row.get("retrieval_variant")) == variant
    }


def load_actual(path: Path) -> tuple[Path, dict[int, dict[str, Any]]]:
    result_path = find_json_file(path, "streaming_bench_minicpmv46_results_*.json")
    data = read_json(result_path)
    rows = data.get("results", data if isinstance(data, list) else [])
    return result_path, {int(row.get("_index", row.get("question_id"))): row for row in rows}


def as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def pred_from_response(response: Any) -> str | None:
    if response is None:
        return None
    text = str(response).strip()
    if not text:
        return None
    for ch in text:
        up = ch.upper()
        if up in {"A", "B", "C", "D"}:
            return up
    return None


def oracle_pred(row: dict[str, Any], k: int) -> str | None:
    return (row.get("branches") or {}).get(str(k), {}).get("prediction")


def oracle_correct(row: dict[str, Any], k: int) -> bool:
    return bool((row.get("branches") or {}).get(str(k), {}).get("correct"))


def actual_adaptive(row: dict[str, Any]) -> dict[str, Any]:
    adaptive = row.get("adaptive")
    if isinstance(adaptive, dict):
        return adaptive
    profile = row.get("profile")
    if isinstance(profile, dict) and isinstance(profile.get("adaptive"), dict):
        return profile["adaptive"]
    return {}


def queue_ids(queue: list[dict[str, Any]], k: int = 3) -> list[int]:
    out: list[int] = []
    for item in queue[:k]:
        value = item.get("chunk_id")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append(int(value))
    return out


def same_prefix(left: list[int], right: list[int], k: int) -> bool:
    return left[:k] == right[:k] and len(left) >= k and len(right) >= k


def set_agreement(left: list[int], right: list[int], k: int) -> bool:
    return len(left) >= k and len(right) >= k and set(left[:k]) == set(right[:k])


def iteration_by_index(adaptive: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(item.get("iteration")): item
        for item in adaptive.get("iterations", [])
        if isinstance(item, dict) and isinstance(item.get("iteration"), (int, float))
    }


def retrieved_iteration_ids(adaptive: dict[str, Any]) -> list[int]:
    ids: list[int] = []
    for item in adaptive.get("iterations", []):
        added = item.get("added_chunk_id") if isinstance(item, dict) else None
        if isinstance(added, (int, float)) and not isinstance(added, bool):
            ids.append(int(added))
    return ids


def actual_memory_ids(adaptive: dict[str, Any]) -> list[int]:
    return [
        int(value)
        for value in adaptive.get("memory_chunk_ids", [])
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]


def actual_pred(row: dict[str, Any]) -> str | None:
    return pred_from_response(row.get("response"))


def stat(values: list[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(clean),
        "mean": mean(clean),
        "median": median(clean),
        "min": min(clean),
        "max": max(clean),
    }


def get_iter0_signal(adaptive: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    iterations = iteration_by_index(adaptive)
    iter0 = iterations.get(0, {})
    queue = adaptive.get("candidate_queue") or oracle.get("candidate_queue") or []
    top1 = queue[0] if queue else {}
    current_pred = iter0.get("predicted_option") or oracle_pred(oracle, 0)
    best_supported = top1.get("best_supported_option")
    return {
        "current_sufficiency": as_float(iter0.get("sufficiency")),
        "answer_margin": as_float(iter0.get("answer_margin")),
        "entropy_confidence": as_float(iter0.get("entropy_confidence")),
        "visual_support": as_float(iter0.get("visual_support_norm")),
        "top1_clip_relevance": as_float(top1.get("retrieval_relevance", top1.get("semantic_score"))),
        "top1_temporal_distance": as_float(top1.get("candidate_temporal_distance_seconds")),
        "best_supported_option": best_supported,
        "current_predicted_option": current_pred,
        "retrieval_disagreement": (
            best_supported is not None and current_pred is not None and str(best_supported) != str(current_pred)
        ),
    }


def classify_oracle_rescuable(
    oracle: dict[str, Any],
    actual: dict[str, Any],
) -> str:
    adaptive = actual_adaptive(actual)
    actual_is_correct = bool(actual.get("correct"))
    oracle_queue = queue_ids(oracle.get("candidate_queue") or [], 3)
    actual_queue = queue_ids(adaptive.get("candidate_queue") or [], 3)
    minimum_k = oracle.get("minimum_k_to_correct")
    if actual_is_correct:
        return "ACTUAL_RESCUED"
    if actual_queue and oracle_queue:
        if actual_queue[: min(3, len(actual_queue), len(oracle_queue))] != oracle_queue[: min(3, len(actual_queue), len(oracle_queue))]:
            return "QUEUE_MISMATCH"
    memory_ids = actual_memory_ids(adaptive)
    retrieved_ids = retrieved_iteration_ids(adaptive)
    reached_k = len(retrieved_ids)
    final_k = len(memory_ids)
    if not memory_ids and not retrieved_ids:
        return "FALSE_STOP_AT_K0"
    if isinstance(minimum_k, int) and reached_k < minimum_k:
        return "FALSE_STOP_BEFORE_REQUIRED_K"
    if isinstance(minimum_k, int):
        required_ids = oracle_queue[:minimum_k]
        reached_required = set(required_ids).issubset(set(retrieved_ids))
        final_has_required = set(required_ids).issubset(set(memory_ids))
        if reached_required and not final_has_required:
            return "CORRECT_CANDIDATES_RETRIEVED_BUT_ROLLBACK"
        if reached_required and final_has_required:
            return "CORRECT_CANDIDATES_RETRIEVED_BUT_FINAL_GENERATION_DIFFERS"
    if final_k < reached_k:
        return "CORRECT_CANDIDATES_RETRIEVED_BUT_ROLLBACK"
    return "OTHER"


def classify_damage(oracle: dict[str, Any], actual: dict[str, Any]) -> str:
    adaptive = actual_adaptive(actual)
    memory_ids = actual_memory_ids(adaptive)
    retrieved_ids = retrieved_iteration_ids(adaptive)
    if not memory_ids and not retrieved_ids:
        return "GENERATION_MISMATCH"
    if memory_ids and not oracle_correct(oracle, len(memory_ids)):
        return "RETRIEVAL_CHANGED_ANSWER_INCORRECTLY"
    if retrieved_ids and len(memory_ids) < len(retrieved_ids):
        return "ROLLBACK_SELECTED_HARMFUL_CONTEXT"
    return "UNNECESSARY_RETRIEVAL"


def summarize_signals(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    numeric_keys = [
        "current_sufficiency",
        "answer_margin",
        "entropy_confidence",
        "visual_support",
        "top1_clip_relevance",
        "top1_temporal_distance",
    ]
    out: dict[str, Any] = {}
    for label, rows in groups.items():
        row_out: dict[str, Any] = {"n": len(rows)}
        for key in numeric_keys:
            row_out[key] = stat([value for row in rows if (value := row.get(key)) is not None])
        row_out["retrieval_disagreement_rate"] = (
            sum(bool(row.get("retrieval_disagreement")) for row in rows) / len(rows)
            if rows
            else None
        )
        row_out["best_supported_option"] = dict(Counter(str(row.get("best_supported_option")) for row in rows))
        out[label] = row_out
    return out


def simulate_policy(
    name: str,
    joined: list[dict[str, Any]],
    choose_k,
) -> dict[str, Any]:
    correct = 0
    rescues = 0
    damages = 0
    frames = []
    for item in joined:
        oracle = item["oracle"]
        k = int(choose_k(item))
        k = max(0, min(3, k))
        is_correct = oracle_correct(oracle, k)
        k0_correct = oracle_correct(oracle, 0)
        correct += int(is_correct)
        rescues += int((not k0_correct) and is_correct)
        damages += int(k0_correct and not is_correct)
        frames.append(k)
    total = len(joined)
    return {
        "policy": name,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "rescues": rescues,
        "damages": damages,
        "net_rescue": rescues - damages,
        "memory_trigger_rate": sum(k > 0 for k in frames) / total if total else None,
        "avg_historical_frames": mean(frames) if frames else None,
    }


def actual_policy_k(item: dict[str, Any]) -> int:
    return len(actual_memory_ids(actual_adaptive(item["actual"])))


def policy_grid(joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    policies.append(simulate_policy("P0_actual_selected_K", joined, actual_policy_k))
    policies.append(
        simulate_policy(
            "P1_force_K1_when_actual_K0",
            joined,
            lambda item: 1 if actual_policy_k(item) == 0 else actual_policy_k(item),
        )
    )
    relevance_values = sorted(
        {
            round(float(item["signals"]["top1_clip_relevance"]), 4)
            for item in joined
            if item["signals"].get("top1_clip_relevance") is not None
        }
    )
    thresholds = relevance_values[:: max(1, len(relevance_values) // 6)] + relevance_values[-1:]
    for threshold in sorted(set(thresholds)):
        policies.append(
            simulate_policy(
                f"P2_force_K1_if_relevance>={threshold:.4f}",
                joined,
                lambda item, t=threshold: (
                    1
                    if actual_policy_k(item) == 0
                    and (item["signals"].get("top1_clip_relevance") or -1.0) >= t
                    else actual_policy_k(item)
                ),
            )
        )
    suff_thresholds = [0.55, 0.60, 0.62, 0.65, 0.70]
    rel_thresholds = thresholds[:: max(1, len(thresholds) // 4)] or [0.0]
    for rel_t in sorted(set(rel_thresholds)):
        for suff_t in suff_thresholds:
            policies.append(
                simulate_policy(
                    f"P3_K1_if_rel>={rel_t:.4f}_suff<{suff_t:.2f}_disagree",
                    joined,
                    lambda item, r=rel_t, s=suff_t: (
                        1
                        if actual_policy_k(item) == 0
                        and (item["signals"].get("top1_clip_relevance") or -1.0) >= r
                        and (item["signals"].get("current_sufficiency") or 999.0) < s
                        and bool(item["signals"].get("retrieval_disagreement"))
                        else actual_policy_k(item)
                    ),
                )
            )
    policies.append(
        simulate_policy(
            "P4_allow_K2_when_actual_K1",
            joined,
            lambda item: 2 if actual_policy_k(item) == 1 else actual_policy_k(item),
        )
    )
    return sorted(policies, key=lambda row: (row["accuracy"] or 0.0, row["net_rescue"]), reverse=True)


def format_pct(value: Any) -> str:
    if value is None:
        return "None"
    return f"{100.0 * float(value):.2f}%"


def print_case(item: dict[str, Any]) -> None:
    oracle = item["oracle"]
    actual = item["actual"]
    adaptive = actual_adaptive(actual)
    print("=" * 80)
    print(f"Question ID: {item['question_id']}")
    print(f"Category: {oracle.get('category')}")
    print(f"Question: {oracle.get('question')}")
    print(f"GT: {oracle.get('ground_truth')}")
    print(f"K0 prediction: {oracle_pred(oracle, 0)}")
    print(
        "Oracle: "
        f"K1={oracle_pred(oracle, 1)}/{oracle_correct(oracle, 1)} "
        f"K2={oracle_pred(oracle, 2)}/{oracle_correct(oracle, 2)} "
        f"K3={oracle_pred(oracle, 3)}/{oracle_correct(oracle, 3)} "
        f"minK={oracle.get('minimum_k_to_correct')}"
    )
    print("Oracle top3:")
    for candidate in (oracle.get("candidate_queue") or [])[:3]:
        print(
            "  "
            f"id={candidate.get('chunk_id')} "
            f"t=({candidate.get('start_time_seconds')}, {candidate.get('end_time_seconds')}) "
            f"rel={candidate.get('retrieval_relevance', candidate.get('semantic_score'))} "
            f"mmr={candidate.get('total_score')} "
            f"opt={candidate.get('best_supported_option')}"
        )
    print("Actual iterations:")
    for iteration in adaptive.get("iterations", []):
        print(
            "  "
            f"i={iteration.get('iteration')} added={iteration.get('added_chunk_id')} "
            f"pred={iteration.get('predicted_option')} "
            f"M={iteration.get('answer_margin')} "
            f"E={iteration.get('entropy_confidence')} "
            f"V={iteration.get('visual_support_norm')} "
            f"S={iteration.get('sufficiency')} "
            f"gain={iteration.get('gain_vs_previous')}"
        )
    print(f"Actual retrieved IDs: {retrieved_iteration_ids(adaptive)}")
    print(f"Actual best_memory IDs: {actual_memory_ids(adaptive)}")
    print(f"Stop reason: {adaptive.get('stop_reason')}")
    print(f"Final prediction: {actual_pred(actual)} correct={actual.get('correct')}")
    print(f"Rollback: {len(actual_memory_ids(adaptive)) < len(retrieved_iteration_ids(adaptive))}")
    print(f"Failure mode: {item.get('failure_mode') or item.get('damage_mode')}")


def write_outputs(out_dir: Path, joined: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clip_mmr_controller_failure_analysis.json").write_text(
        json.dumps({"summary": summary, "samples": joined}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fields = [
        "question_id",
        "category",
        "k0_correct",
        "actual_correct",
        "oracle_rescuable",
        "failure_mode",
        "damage_mode",
        "actual_k",
        "minimum_k_to_correct",
        "current_sufficiency",
        "answer_margin",
        "entropy_confidence",
        "visual_support",
        "top1_clip_relevance",
        "top1_temporal_distance",
        "best_supported_option",
        "current_predicted_option",
        "retrieval_disagreement",
    ]
    with (out_dir / "clip_mmr_controller_failure_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in joined:
            signals = item["signals"]
            writer.writerow(
                {
                    "question_id": item["question_id"],
                    "category": item["oracle"].get("category"),
                    "k0_correct": oracle_correct(item["oracle"], 0),
                    "actual_correct": item["actual"].get("correct"),
                    "oracle_rescuable": item["oracle_rescuable"],
                    "failure_mode": item.get("failure_mode"),
                    "damage_mode": item.get("damage_mode"),
                    "actual_k": actual_policy_k(item),
                    "minimum_k_to_correct": item["oracle"].get("minimum_k_to_correct"),
                    **signals,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", required=True, help="Retrieval variant oracle result dir or JSON/JSONL")
    parser.add_argument("--actual", required=True, help="Actual progressive_sufficiency_memory_clip_mmr result dir")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--variant", default="clip_mmr")
    args = parser.parse_args()

    oracle_path, oracle_by_id = load_oracle(Path(args.oracle), args.variant)
    actual_path, actual_by_id = load_actual(Path(args.actual))
    common = sorted(set(oracle_by_id) & set(actual_by_id))
    if not common:
        raise RuntimeError("No common question IDs between oracle and actual results")

    joined: list[dict[str, Any]] = []
    queue_counts = Counter()
    failure_counts = Counter()
    damage_counts = Counter()
    for qid in common:
        oracle = oracle_by_id[qid]
        actual = actual_by_id[qid]
        adaptive = actual_adaptive(actual)
        oracle_q = queue_ids(oracle.get("candidate_queue") or [], 3)
        actual_q = queue_ids(adaptive.get("candidate_queue") or [], 3)
        if same_prefix(oracle_q, actual_q, 1):
            queue_counts["top1_agree"] += 1
        if same_prefix(oracle_q, actual_q, 3):
            queue_counts["top3_ordered_agree"] += 1
        if set_agreement(oracle_q, actual_q, 3):
            queue_counts["top3_set_agree"] += 1
        if oracle_q and actual_q:
            queue_counts["queue_comparable"] += 1

        oracle_rescuable = (not oracle_correct(oracle, 0)) and oracle.get("minimum_k_to_correct") in {1, 2, 3}
        item = {
            "question_id": qid,
            "oracle": oracle,
            "actual": actual,
            "signals": get_iter0_signal(adaptive, oracle),
            "oracle_rescuable": oracle_rescuable,
        }
        if oracle_rescuable:
            mode = classify_oracle_rescuable(oracle, actual)
            item["failure_mode"] = mode
            failure_counts[mode] += 1
        if oracle_correct(oracle, 0) and not bool(actual.get("correct")):
            mode = classify_damage(oracle, actual)
            item["damage_mode"] = mode
            damage_counts[mode] += 1
        joined.append(item)

    oracle_rescuable = [item for item in joined if item["oracle_rescuable"]]
    damaged = [item for item in joined if "damage_mode" in item]
    signal_groups = {
        "oracle_rescuable_k0_wrong": [item["signals"] for item in oracle_rescuable],
        "k0_correct": [item["signals"] for item in joined if oracle_correct(item["oracle"], 0)],
        "actual_damaged": [item["signals"] for item in damaged],
    }
    policies = policy_grid(joined)
    summary = {
        "oracle_path": str(oracle_path),
        "actual_path": str(actual_path),
        "variant": args.variant,
        "matched_samples": len(common),
        "oracle_rescuable": len(oracle_rescuable),
        "failure_mode_counts": dict(failure_counts),
        "damaged_count": len(damaged),
        "damage_mode_counts": dict(damage_counts),
        "queue_consistency": {
            "comparable_samples": queue_counts["queue_comparable"],
            "top1_agreement": queue_counts["top1_agree"] / len(common),
            "top3_ordered_agreement": queue_counts["top3_ordered_agree"] / len(common),
            "top3_set_agreement": queue_counts["top3_set_agree"] / len(common),
        },
        "signal_comparison": summarize_signals(signal_groups),
        "policy_table": policies,
    }
    write_outputs(Path(args.out_dir), joined, summary)

    print("=" * 80)
    print("CLIP-MMR Controller Failure Analysis")
    print("=" * 80)
    print(f"Oracle: {oracle_path}")
    print(f"Actual: {actual_path}")
    print(f"Matched samples: {len(common)}")
    print(f"Oracle-rescuable K0 errors: {len(oracle_rescuable)}")
    print(f"Failure modes: {dict(failure_counts)}")
    qc = summary["queue_consistency"]
    print(
        "Queue consistency: "
        f"top1={format_pct(qc['top1_agreement'])} "
        f"top3_ordered={format_pct(qc['top3_ordered_agreement'])} "
        f"top3_set={format_pct(qc['top3_set_agreement'])}"
    )
    print(f"Actual damaged K0-correct samples: {len(damaged)}")
    print(f"Damage modes: {dict(damage_counts)}")

    print("\nOracle-rescuable samples")
    for item in oracle_rescuable:
        print_case(item)

    print("\nDamaged samples")
    for item in damaged:
        print_case(item)

    print("\nSignal comparison")
    print(json.dumps(summary["signal_comparison"], indent=2, ensure_ascii=False))

    print("\nCounterfactual policies")
    for row in policies[:20]:
        print(
            f"{row['policy']}: acc={format_pct(row['accuracy'])} "
            f"rescues={row['rescues']} damages={row['damages']} "
            f"net={row['net_rescue']} trigger={format_pct(row['memory_trigger_rate'])} "
            f"avg_hist={row['avg_historical_frames']}"
        )
    print(f"\nSaved: {Path(args.out_dir)}")


if __name__ == "__main__":
    main()
