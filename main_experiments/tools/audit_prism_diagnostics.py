#!/usr/bin/env python3
"""Read-only PRISM diagnostic audit over saved MiniCPM result JSONs.

This script does not run inference. It only inspects saved result metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


PSM_MODES = {
    "progressive_sufficiency_memory",
    "progressive_sufficiency_memory_heg",
    "progressive_sufficiency_memory_conservative_gate",
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]
        payload = json.load(path.open(encoding="utf-8"))
        return normalize_payload(payload)

    preferred_patterns = [
        "streaming_bench_minicpmv46_results_*.json",
        "minicpmv46_results_*.json",
        "merged_results.json",
    ]
    for pattern in preferred_patterns:
        matches = sorted(path.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
        if matches:
            return read_rows(matches[0])

    rows: list[dict[str, Any]] = []
    for item in sorted(path.glob("rank_*/results_incremental.jsonl")):
        rows.extend(read_rows(item))
    if rows:
        dedup: dict[str, dict[str, Any]] = {}
        for row in rows:
            dedup[stable_key(row)] = row
        return list(dedup.values())
    raise FileNotFoundError(f"No result JSON/JSONL files found under {path}")


def normalize_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return flatten_ovo(payload)
    if isinstance(payload, dict):
        if isinstance(payload.get("results"), list):
            return flatten_ovo(payload["results"])
        if all(isinstance(payload.get(key), list) for key in ("backward", "realtime", "forward")):
            return flatten_ovo([*payload["backward"], *payload["realtime"], *payload["forward"]])
    return []


def flatten_ovo(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row.get("adaptive"), dict):
            flat.append(row)
        test_info = row.get("test_info")
        if isinstance(test_info, list):
            for index, item in enumerate(test_info):
                if isinstance(item, dict) and isinstance(item.get("adaptive"), dict):
                    merged = dict(item)
                    for key in ("video", "video_id", "id", "task", "category", "task_type"):
                        merged.setdefault(key, row.get(key))
                    merged.setdefault("_index", row.get("_index", row.get("id")))
                    merged.setdefault("_subindex", index)
                    flat.append(merged)
    return flat


def stable_key(row: dict[str, Any]) -> str:
    if row.get("_index") is not None and row.get("_subindex") is not None:
        return f"{row.get('_index')}:{row.get('_subindex')}"
    return str(
        row.get("_key")
        or row.get("_index")
        or row.get("id")
        or f"{row.get('video', row.get('video_id', ''))}:{row.get('task', row.get('task_type', ''))}:{row.get('question', '')}"
    )


def adaptive(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("adaptive") if isinstance(row.get("adaptive"), dict) else {}


def profile(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("profile") if isinstance(row.get("profile"), dict) else {}


def as_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            out.append(int(item))
    return out


def as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def final_answer(row: dict[str, Any]) -> str | None:
    for key in ("prediction", "pred", "answer", "response", "model_answer"):
        value = row.get(key)
        if value is None:
            continue
        match = re.search(r"\b([A-E])\b", str(value).upper())
        if match:
            return match.group(1)
    return None


def gt_answer(row: dict[str, Any]) -> str | None:
    for key in ("answer_gt", "ground_truth", "gt", "label", "correct_answer"):
        value = row.get(key)
        if value is None:
            continue
        match = re.search(r"\b([A-E])\b", str(value).upper())
        if match:
            return match.group(1)
    return None


def is_correct(row: dict[str, Any]) -> bool | None:
    if isinstance(row.get("correct"), bool):
        return bool(row["correct"])
    pred = final_answer(row)
    gt = gt_answer(row)
    if pred and gt:
        return pred == gt
    return None


def category(row: dict[str, Any]) -> str:
    return str(row.get("task_type") or row.get("category") or row.get("task") or "unknown")


def number_summary(values: list[float]) -> dict[str, Any]:
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {"n": len(clean), "mean": mean(clean), "min": min(clean), "max": max(clean)}


def accuracy(correctness: list[bool | None]) -> dict[str, Any]:
    known = [bool(v) for v in correctness if v is not None]
    return {
        "known": len(known),
        "correct": sum(known),
        "accuracy": (sum(known) / len(known)) if known else None,
    }


def decoded_chunks(row: dict[str, Any]) -> int | None:
    prof = profile(row)
    value = prof.get("decoded_chunks", row.get("decoded_chunks"))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    base = adaptive(row).get("baseline_recent_equivalence", {}).get("baseline_recent", {})
    value = base.get("decoded_chunks")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return None


def available_history(ad: dict[str, Any]) -> int:
    start = ad.get("history_search_start")
    end = ad.get("history_search_end")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        return max(0, int(end) - int(start) + 1)
    return len(ad.get("candidate_queue") or [])


def candidate_map(ad: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for candidate in ad.get("candidate_queue") or []:
        if isinstance(candidate, dict) and candidate.get("chunk_id") is not None:
            out[int(candidate["chunk_id"])] = candidate
    return out


def iteration_memory(iteration: dict[str, Any], recent_ids: set[int]) -> list[int]:
    return [chunk_id for chunk_id in as_int_list(iteration.get("context_chunk_ids")) if chunk_id not in recent_ids]


def option_scorer_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mechanism: dict[str, dict[str, Any]] = {}
    records_by_mech: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        ad = adaptive(row)
        iterations = ad.get("iterations") or []
        if not iterations:
            continue
        iter0 = iterations[0]
        if not isinstance(iter0, dict):
            continue
        mech = str(iter0.get("option_scoring_mechanism") or "missing")
        records_by_mech[mech].append((row, iter0))

    for mech, records in sorted(records_by_mech.items()):
        margins = [as_float(it.get("answer_margin")) for _, it in records]
        confidences = [as_float(it.get("entropy_confidence")) for _, it in records]
        iter0_preds = [str(it.get("predicted_option")) if it.get("predicted_option") is not None else None for _, it in records]
        final_preds = [final_answer(row) for row, _ in records]
        agreements = [
            bool(a == b)
            for a, b in zip(iter0_preds, final_preds)
            if a is not None and b is not None
        ]
        iter0_correct = [
            (pred == gt_answer(row)) if pred is not None and gt_answer(row) is not None else None
            for pred, (row, _) in zip(iter0_preds, records)
        ]
        final_correct = [is_correct(row) for row, _ in records]
        by_mechanism[mech] = {
            "samples": len(records),
            "iter0_final_agreement_known": len(agreements),
            "iter0_final_agreement": sum(agreements) / len(agreements) if agreements else None,
            "mean_margin": mean([v for v in margins if v is not None]) if any(v is not None for v in margins) else None,
            "mean_entropy_confidence": mean([v for v in confidences if v is not None]) if any(v is not None for v in confidences) else None,
            "iteration0_correctness": accuracy(iter0_correct),
            "final_correctness": accuracy(final_correct),
        }
    for required in ("direct_option_logits", "sequence_log_likelihood", "minimal_one_token_decode_fallback"):
        by_mechanism.setdefault(
            required,
            {
                "samples": 0,
                "iter0_final_agreement_known": 0,
                "iter0_final_agreement": None,
                "mean_margin": None,
                "mean_entropy_confidence": None,
                "iteration0_correctness": accuracy([]),
                "final_correctness": accuracy([]),
            },
        )
    return by_mechanism


def history_decode_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decoded: list[float] = []
    recent_counts: list[float] = []
    history_counts: list[float] = []
    zero_history = 0
    lt64 = 0
    ge64 = 0
    timestamp_violations: list[dict[str, Any]] = []
    decoded_less_than_recent_plus_history: list[dict[str, Any]] = []

    for row in rows:
        ad = adaptive(row)
        recent_ids = as_int_list(ad.get("recent_chunk_ids"))
        history_count = available_history(ad)
        dc = decoded_chunks(row)
        if dc is not None:
            decoded.append(float(dc))
        recent_counts.append(float(len(recent_ids)))
        history_counts.append(float(history_count))
        zero_history += int(history_count == 0)
        lt64 += int(history_count < 64)
        ge64 += int(history_count >= 64)
        if dc is not None and recent_ids and dc < len(recent_ids) + history_count:
            decoded_less_than_recent_plus_history.append({"key": stable_key(row), "decoded_chunks": dc, "recent": recent_ids, "history_count": history_count})

        earliest_recent = min(recent_ids) if recent_ids else None
        for candidate in ad.get("candidate_queue") or []:
            if not isinstance(candidate, dict):
                continue
            cid = candidate.get("chunk_id")
            if earliest_recent is not None and isinstance(cid, (int, float)) and int(cid) >= earliest_recent:
                timestamp_violations.append(
                    {
                        "key": stable_key(row),
                        "category": category(row),
                        "candidate_chunk_id": int(cid),
                        "earliest_recent_chunk_id": earliest_recent,
                    }
                )
    return {
        "decoded_chunks": number_summary(decoded),
        "recent_chunk_count": number_summary(recent_counts),
        "available_historical_chunks": number_summary(history_counts),
        "samples_with_0_historical_chunks": zero_history,
        "samples_with_lt64_historical_chunks": lt64,
        "samples_with_ge64_historical_chunks": ge64,
        "candidate_not_older_than_recent_violations": timestamp_violations[:50],
        "candidate_not_older_than_recent_violation_count": len(timestamp_violations),
        "decoded_less_than_recent_plus_history_count": len(decoded_less_than_recent_plus_history),
        "decoded_less_than_recent_plus_history_examples": decoded_less_than_recent_plus_history[:20],
    }


def temporal_distance_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: list[tuple[float, float]] = []
    material: list[dict[str, Any]] = []
    for row in rows:
        ad = adaptive(row)
        recent_ids = as_int_list(ad.get("recent_chunk_ids"))
        if not recent_ids:
            continue
        earliest_recent = min(recent_ids)
        for candidate in ad.get("candidate_queue") or []:
            if not isinstance(candidate, dict):
                continue
            cid = candidate.get("chunk_id")
            ts = as_float(candidate.get("timestamp"))
            if not isinstance(cid, (int, float)) or ts is None:
                continue
            chunk_dist = float(earliest_recent - int(cid))
            timestamp_dist = float(earliest_recent - ts)
            pairs.append((chunk_dist, timestamp_dist))
            if abs(chunk_dist - timestamp_dist) > max(2.0, 0.25 * max(abs(chunk_dist), 1.0)):
                material.append(
                    {
                        "key": stable_key(row),
                        "category": category(row),
                        "candidate_chunk_id": int(cid),
                        "earliest_recent_chunk_id": earliest_recent,
                        "candidate_timestamp_seconds": ts,
                        "chunk_index_distance": chunk_dist,
                        "timestamp_distance_seconds": timestamp_dist,
                    }
                )
    corr = None
    if len(pairs) >= 2:
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        mx, my = mean(xs), mean(ys)
        num = sum((x - mx) * (y - my) for x, y in pairs)
        denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        deny = math.sqrt(sum((y - my) ** 2 for y in ys))
        corr = num / (denx * deny) if denx and deny else None
    return {
        "pairs": len(pairs),
        "pearson_correlation": corr,
        "material_difference_count": len(material),
        "material_difference_examples": material[:50],
        "note": "Uses candidate timestamp seconds and recent chunk indices; exact recent chunk timestamps are not saved separately.",
    }


def rollback_audit(rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]] | None) -> dict[str, Any]:
    baseline_by_key = {stable_key(row): row for row in baseline_rows or []}
    total_triggered = 0
    mismatch = 0
    breakdown = Counter()
    examples: list[dict[str, Any]] = []
    for row in rows:
        ad = adaptive(row)
        if not ad.get("memory_triggered"):
            continue
        total_triggered += 1
        recent_ids = set(as_int_list(ad.get("recent_chunk_ids")))
        iterations = [it for it in ad.get("iterations") or [] if isinstance(it, dict)]
        stop_memory = iteration_memory(iterations[-1], recent_ids) if iterations else []
        final_memory = as_int_list(ad.get("memory_chunk_ids"))
        differs = stop_memory != final_memory
        mismatch += int(differs)
        base = baseline_by_key.get(stable_key(row))
        if base is not None and is_correct(base) is not None and is_correct(row) is not None:
            if not is_correct(base) and is_correct(row):
                label = "rescued"
            elif is_correct(base) and not is_correct(row):
                label = "damaged"
            elif is_correct(base) and is_correct(row):
                label = "both_correct"
            else:
                label = "both_wrong"
        else:
            label = "unknown_without_baseline"
        if differs:
            breakdown[label] += 1
            examples.append(
                {
                    "key": stable_key(row),
                    "category": category(row),
                    "stop_reason": ad.get("stop_reason"),
                    "stop_iteration_memory": stop_memory,
                    "final_best_memory": final_memory,
                    "iteration_sufficiencies": [it.get("sufficiency") for it in iterations],
                    "iteration_selected_memory": [iteration_memory(it, recent_ids) for it in iterations],
                    "outcome_vs_baseline": label,
                }
            )
    return {
        "memory_triggered_samples": total_triggered,
        "final_best_memory_differs_from_stopping_iteration": mismatch,
        "breakdown_for_mismatches": dict(breakdown),
        "examples": examples[:50],
    }


def candidate1_gate_audit(rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]] | None) -> dict[str, Any]:
    baseline_by_key = {stable_key(row): row for row in baseline_rows or []}
    checked = 0
    later_pass_count = 0
    examples: list[dict[str, Any]] = []
    for row in rows:
        ad = adaptive(row)
        recent_ids = set(as_int_list(ad.get("recent_chunk_ids")))
        for iteration in ad.get("iterations") or []:
            if not isinstance(iteration, dict):
                continue
            gate = iteration.get("conservative_gate")
            if not isinstance(gate, dict):
                continue
            selected_ids = set(as_int_list(iteration.get("context_chunk_ids"))) - recent_ids
            unused = [
                c for c in ad.get("candidate_queue") or []
                if isinstance(c, dict)
                and isinstance(c.get("chunk_id"), (int, float))
                and int(c["chunk_id"]) not in selected_ids
                and int(c["chunk_id"]) not in recent_ids
            ]
            if not unused:
                continue
            checked += 1
            first = unused[0]
            threshold = as_float(gate.get("candidate_threshold")) or 0.0
            td_threshold = as_float(gate.get("temporal_distance_threshold")) or 0.0
            first_score = as_float(first.get("total_score")) or -float("inf")
            first_distance = min(abs(int(first["chunk_id"]) - rid) for rid in recent_ids) if recent_ids else None
            first_pass = first_score >= threshold and first_distance is not None and first_distance > td_threshold
            later_passing = []
            for cand in unused[1:]:
                score = as_float(cand.get("total_score")) or -float("inf")
                distance = min(abs(int(cand["chunk_id"]) - rid) for rid in recent_ids) if recent_ids else None
                if distance is not None and score >= threshold and distance > td_threshold:
                    later_passing.append((cand, score, distance))
            if (not first_pass) and later_passing:
                later_pass_count += 1
                best_later, best_score, best_distance = max(later_passing, key=lambda item: (item[1], item[2]))
                base = baseline_by_key.get(stable_key(row))
                if base is not None and is_correct(base) is not None and is_correct(row) is not None:
                    oracle = "historical_helped" if (not is_correct(base) and is_correct(row)) else (
                        "historical_hurt" if (is_correct(base) and not is_correct(row)) else "no_changed_correctness"
                    )
                else:
                    oracle = "unknown_without_baseline"
                examples.append(
                    {
                        "key": stable_key(row),
                        "benchmark_category": category(row),
                        "iteration": iteration.get("iteration"),
                        "gate_reason": gate.get("reason"),
                        "candidate1_chunk_id": first.get("chunk_id"),
                        "candidate1_score": first_score,
                        "candidate1_temporal_distance": first_distance,
                        "later_passing_chunk_id": best_later.get("chunk_id"),
                        "later_passing_score": best_score,
                        "later_passing_temporal_distance": best_distance,
                        "max_unused_candidate_score": max((as_float(c.get("total_score")) or -float("inf")) for c in unused),
                        "oracle": oracle,
                    }
                )
    return {
        "gate_iterations_checked": checked,
        "candidate1_fails_but_later_candidate_passes": later_pass_count,
        "examples": examples[:100],
    }


def code_path_decode_finding() -> dict[str, Any]:
    return {
        "progressive_sufficiency_memory": {
            "memory_would_trigger": True,
            "memory_gate_reason": "progressive_sufficiency_pending",
            "config_memory_search_chunks_used_for_decode": "max(config.memory_anchors, config.memory_search_chunks)",
            "decode_recent_hint": "max(recent_frames_only, config.max_window + memory_search_chunks)",
            "chunks_passed_to_select_progressive_sufficiency_memory": "all chunks decoded by decode_video_to_chunks_qwen using decode_recent_hint",
        },
        "progressive_sufficiency_memory_heg": {
            "memory_would_trigger": True,
            "memory_gate_reason": "progressive_sufficiency_heg_pending",
            "config_memory_search_chunks_used_for_decode": "max(config.memory_anchors, config.memory_search_chunks)",
            "decode_recent_hint": "max(recent_frames_only, config.max_window + memory_search_chunks)",
            "chunks_passed_to_select_progressive_sufficiency_memory": "all chunks decoded by decode_video_to_chunks_qwen using decode_recent_hint",
        },
        "progressive_sufficiency_memory_conservative_gate": {
            "memory_would_trigger": True,
            "memory_gate_reason": "progressive_sufficiency_pending",
            "config_memory_search_chunks_used_for_decode": "max(config.memory_anchors, config.memory_search_chunks)",
            "decode_recent_hint": "max(recent_frames_only, config.max_window + memory_search_chunks)",
            "chunks_passed_to_select_progressive_sufficiency_memory": "all chunks decoded by decode_video_to_chunks_qwen using decode_recent_hint",
        },
        "bug_or_inconsistency": (
            "Not a legacy question-wording gate in this code: _memory_trigger_decision returns True for all "
            "progressive_sufficiency_like modes. However, PRISM history availability is still coupled to "
            "config.memory_search_chunks in the upstream broad decode. If config.memory_search_chunks is less than "
            "MINICPM_PSM_HISTORY_SEARCH_CHUNKS, PRISM cannot access its configured horizon even though it requests 64 internally."
        ),
    }


def audit_one(name: str, result: Path, baseline: Path | None) -> dict[str, Any]:
    rows = read_rows(result)
    baseline_rows = read_rows(baseline) if baseline else None
    psm_rows = [row for row in rows if adaptive(row)]
    return {
        "benchmark": name,
        "result_path": str(result),
        "baseline_path": str(baseline) if baseline else None,
        "rows_with_adaptive_metadata": len(psm_rows),
        "code_path_decode_finding": code_path_decode_finding(),
        "A_option_scorer_audit": option_scorer_audit(psm_rows),
        "B_history_decode_audit": history_decode_audit(psm_rows),
        "C_temporal_distance_audit": temporal_distance_audit(psm_rows),
        "D_best_context_rollback_audit": rollback_audit(psm_rows, baseline_rows),
        "E_candidate1_gate_audit": candidate1_gate_audit(psm_rows, baseline_rows),
    }


def print_human(report: dict[str, Any]) -> None:
    print(f"\n{'=' * 80}\n{report['benchmark']}\n{'=' * 80}")
    print(f"Result: {report['result_path']}")
    print(f"Rows with adaptive metadata: {report['rows_with_adaptive_metadata']}")
    print("\nUpstream decode path:")
    for mode, finding in report["code_path_decode_finding"].items():
        if mode == "bug_or_inconsistency":
            continue
        print(f"  {mode}:")
        print(f"    memory_would_trigger: {finding['memory_would_trigger']}")
        print(f"    gate reason: {finding['memory_gate_reason']}")
        print(f"    decode_recent_hint: {finding['decode_recent_hint']}")
    print(f"  Finding: {report['code_path_decode_finding']['bug_or_inconsistency']}")

    print("\nA. Option scorer audit")
    for mech, stats in sorted(report["A_option_scorer_audit"].items()):
        print(
            f"  {mech}: n={stats['samples']} "
            f"agree_iter0_final={stats['iter0_final_agreement']} "
            f"mean_margin={stats['mean_margin']} "
            f"mean_entropy_conf={stats['mean_entropy_confidence']} "
            f"iter0_acc={stats['iteration0_correctness']['accuracy']} "
            f"final_acc={stats['final_correctness']['accuracy']}"
        )

    hist = report["B_history_decode_audit"]
    print("\nB. History decode audit")
    print(f"  decoded_chunks: {hist['decoded_chunks']}")
    print(f"  recent_chunk_count: {hist['recent_chunk_count']}")
    print(f"  available_historical_chunks: {hist['available_historical_chunks']}")
    print(f"  samples with 0 history: {hist['samples_with_0_historical_chunks']}")
    print(f"  samples with <64 history: {hist['samples_with_lt64_historical_chunks']}")
    print(f"  samples with >=64 history: {hist['samples_with_ge64_historical_chunks']}")
    print(f"  candidate older-than-recent violations: {hist['candidate_not_older_than_recent_violation_count']}")

    temp = report["C_temporal_distance_audit"]
    print("\nC. Temporal distance audit")
    print(f"  pairs={temp['pairs']} pearson={temp['pearson_correlation']}")
    print(f"  material differences={temp['material_difference_count']}")

    rb = report["D_best_context_rollback_audit"]
    print("\nD. Best-context rollback audit")
    print(f"  memory_triggered={rb['memory_triggered_samples']}")
    print(f"  final best_memory != stopping memory: {rb['final_best_memory_differs_from_stopping_iteration']}")
    print(f"  breakdown: {rb['breakdown_for_mismatches']}")

    gate = report["E_candidate1_gate_audit"]
    print("\nE. Candidate-1 gate audit")
    print(f"  gate iterations checked={gate['gate_iterations_checked']}")
    print(f"  candidate #1 fails but later candidate passes={gate['candidate1_fails_but_later_candidate_passes']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ovo", type=Path, help="OVO PRISM result dir or JSON")
    parser.add_argument("--ovo-baseline", type=Path, help="Optional OVO baseline dir or JSON")
    parser.add_argument("--streamingbench", type=Path, help="StreamingBench PRISM result dir or JSON")
    parser.add_argument("--streamingbench-baseline", type=Path, help="Optional StreamingBench baseline dir or JSON")
    parser.add_argument("--out-json", type=Path, help="Optional path for full audit JSON")
    args = parser.parse_args()

    reports = []
    if args.ovo:
        reports.append(audit_one("OVO", args.ovo, args.ovo_baseline))
    if args.streamingbench:
        reports.append(audit_one("StreamingBench", args.streamingbench, args.streamingbench_baseline))
    if not reports:
        raise SystemExit("Provide --ovo and/or --streamingbench")

    output = {"reports": reports}
    for report in reports:
        print_human(report)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nSaved full JSON audit: {args.out_json}")


if __name__ == "__main__":
    main()
