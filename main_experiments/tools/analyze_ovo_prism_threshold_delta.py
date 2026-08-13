#!/usr/bin/env python3
"""Offline t062-vs-t074 PRISM retrieval-delta diagnostics for OVO.

This is a read-only analysis tool. It consumes completed result directories,
compares rows with the duplicate-free id_task_question key, and reports which
logged inference-time signals distinguish useful extra retrieval from
harmful/wasted extra retrieval. It does not rerun inference, train a model, or
change PRISM.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ovo_constants import BACKWARD_TASKS, REAL_TIME_TASKS, extract_br_answer  # noqa: E402


LETTERS = ("A", "B", "C", "D", "E")
TASKS = tuple(BACKWARD_TASKS + REAL_TIME_TASKS)
POSITIVE_GROUPS = {"EXTRA_RETRIEVAL_HELPFUL", "DIFFERENT_K_HELPFUL"}
NEGATIVE_GROUPS = {
    "EXTRA_RETRIEVAL_HARMFUL",
    "EXTRA_RETRIEVAL_NEUTRAL_CORRECT",
    "EXTRA_RETRIEVAL_NEUTRAL_WRONG",
    "DIFFERENT_K_HARMFUL",
}


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    if isinstance(data, dict) and all(key in data for key in ("backward", "realtime", "forward")):
        return [*data.get("backward", []), *data.get("realtime", []), *data.get("forward", [])]
    return []


def _flatten(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in rows:
        if row.get("task") in set(TASKS):
            flat.append(row)
        for index, item in enumerate(row.get("test_info", []) or []):
            if isinstance(item, dict):
                flat.append({**item, "_parent_id": row.get("id"), "_question_index": index, "task": row.get("task")})
    return flat


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
        for row in _load_json_rows(item):
            row["_file"] = str(item)
            rows.append(row)
    return _flatten(rows)


def _primary_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("_parent_id") or "")


def _key(row: dict[str, Any]) -> str:
    return "|".join([_primary_id(row), str(row.get("task") or ""), str(row.get("question") or "")])


def _dedupe_by_key(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = _key(row)
        if key.strip("|"):
            by_key[key].append(row)
    duplicate_keys = {key: value for key, value in by_key.items() if len(value) > 1}
    return (
        {key: value[0] for key, value in by_key.items()},
        {
            "rows": len(rows),
            "unique_keys": len(by_key),
            "duplicate_keys": len(duplicate_keys),
            "duplicate_rows": sum(len(value) - 1 for value in duplicate_keys.values()),
        },
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
    if gt in LETTERS and pred:
        return pred == gt
    return None


def _adaptive(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("adaptive"), dict):
        return row["adaptive"]
    profile = row.get("profile")
    if isinstance(profile, dict) and isinstance(profile.get("adaptive"), dict):
        return profile["adaptive"]
    return {}


def _iterations(row: dict[str, Any]) -> list[dict[str, Any]]:
    iterations = _adaptive(row).get("iterations")
    return [item for item in iterations if isinstance(item, dict)] if isinstance(iterations, list) else []


def _memory_frames(row: dict[str, Any]) -> int:
    return int(_adaptive(row).get("num_memory_frames", 0) or 0)


def _float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def _numeric_summary(values: Iterable[float | None]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    return {
        "count": len(clean),
        "mean": mean(clean) if clean else None,
        "median": median(clean) if clean else None,
        "p10": _percentile(clean, 0.10),
        "p25": _percentile(clean, 0.25),
        "p75": _percentile(clean, 0.75),
        "p90": _percentile(clean, 0.90),
    }


def _top_probs(probs: Any) -> tuple[float | None, float | None, str | None]:
    if not isinstance(probs, dict):
        return None, None, None
    values: list[tuple[str, float]] = []
    for letter, value in probs.items():
        number = _float(value)
        if number is not None:
            values.append((str(letter), number))
    if not values:
        return None, None, None
    ordered = sorted(values, key=lambda item: (-item[1], item[0]))
    top1 = ordered[0][1]
    top2 = ordered[1][1] if len(ordered) > 1 else None
    return top1, top2, ordered[0][0]


def _entropy_from_probs(probs: Any) -> float | None:
    if not isinstance(probs, dict):
        return None
    values = [_float(value) for value in probs.values()]
    clean = [value for value in values if value is not None and value > 0]
    if not clean:
        return None
    return -sum(value * math.log(value) for value in clean)


def _js_divergence(left: Any, right: Any) -> float | None:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None
    keys = sorted(set(left) | set(right))
    p = [_float(left.get(key)) or 0.0 for key in keys]
    q = [_float(right.get(key)) or 0.0 for key in keys]
    ps = sum(p)
    qs = sum(q)
    if ps <= 0 or qs <= 0:
        return None
    p = [value / ps for value in p]
    q = [value / qs for value in q]
    m = [(a + b) / 2.0 for a, b in zip(p, q)]

    def kl(a: list[float], b: list[float]) -> float:
        return sum(x * math.log(x / y) for x, y in zip(a, b) if x > 0 and y > 0)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _best_candidate(row: dict[str, Any], decision_k: int) -> dict[str, Any]:
    queue = _adaptive(row).get("candidate_queue") or []
    if isinstance(queue, list) and 0 <= decision_k < len(queue) and isinstance(queue[decision_k], dict):
        return queue[decision_k]
    return {}


def _recent_first_id(row: dict[str, Any]) -> int | None:
    ids = _adaptive(row).get("recent_chunk_ids") or row.get("final_chunk_ids") or []
    ints = [int(value) for value in ids if isinstance(value, int) or str(value).lstrip("-").isdigit()]
    return min(ints) if ints else None


def _temporal_distance(row: dict[str, Any], candidate: dict[str, Any]) -> float | None:
    first_recent = _recent_first_id(row)
    chunk_id = candidate.get("chunk_id")
    if first_recent is None or not isinstance(chunk_id, (int, float)):
        return None
    return float(first_recent - int(chunk_id))


def _candidate_temporal_distance_seconds(row: dict[str, Any], candidate: dict[str, Any]) -> float | None:
    question_time = _float(row.get("time_stamp") or row.get("ask_time") or row.get("clue_time"))
    candidate_time = _float(candidate.get("timestamp"))
    if question_time is None or candidate_time is None:
        return None
    return question_time - candidate_time


def _extract_signals(row: dict[str, Any], decision_k: int) -> dict[str, float | None]:
    iterations = _iterations(row)
    current = iterations[decision_k] if 0 <= decision_k < len(iterations) else {}
    candidate = _best_candidate(row, decision_k)
    top1, top2, _pred = _top_probs(current.get("option_probabilities"))
    current_support = _float(current.get("visual_support_norm"))
    best_total = _float(candidate.get("total_score"))
    best_semantic = _float(candidate.get("semantic_score"))
    temporal_chunks = _temporal_distance(row, candidate)
    search_start = _float(_adaptive(row).get("history_search_start"))
    search_end = _float(_adaptive(row).get("history_search_end"))
    norm_temporal = None
    if temporal_chunks is not None and search_start is not None and search_end is not None and search_end > search_start:
        norm_temporal = temporal_chunks / max(1.0, search_end - search_start + 1.0)
    signals: dict[str, float | None] = {
        "top1_probability": top1,
        "top2_probability": top2,
        "answer_margin": _float(current.get("answer_margin")),
        "entropy": _entropy_from_probs(current.get("option_probabilities")),
        "normalized_entropy": _float(current.get("normalized_entropy")),
        "entropy_confidence": _float(current.get("entropy_confidence")),
        "current_sufficiency": _float(current.get("sufficiency")),
        "current_visual_support": current_support,
        "support_for_predicted_answer": current_support,
        "candidate_total_score": best_total,
        "candidate_semantic_score": best_semantic,
        "candidate_event_score": _float(candidate.get("event_score")),
        "candidate_detail_score": _float(candidate.get("detail_score")),
        "candidate_diversity_score": _float(candidate.get("diversity_score")),
        "candidate_semantic_cue_score": _float(candidate.get("cue_score")),
        "temporal_distance_chunks": temporal_chunks,
        "normalized_temporal_distance": norm_temporal,
        "temporal_distance_seconds": _candidate_temporal_distance_seconds(row, candidate),
        "candidate_minus_current_support": (
            best_total - current_support if best_total is not None and current_support is not None else None
        ),
        "candidate_semantic_minus_current_support": (
            best_semantic - current_support if best_semantic is not None and current_support is not None else None
        ),
        "historical_advantage": _float(current.get("historical_advantage")),
        "historical_ratio": _float(current.get("historical_ratio")),
        "heg": _float(current.get("heg")),
        "heg_current": _float(current.get("heg_current")),
        "heg_alternative": _float(current.get("heg_alternative")),
        "evidence_conflict": _float(current.get("evidence_conflict")),
        "historical_option_margin": _float(current.get("historical_option_margin")),
    }
    option_supports = current.get("option_supports") or candidate.get("option_supports")
    if isinstance(option_supports, dict):
        values = sorted([float(value) for value in option_supports.values() if isinstance(value, (int, float))], reverse=True)
        if values:
            signals["max_support_across_options"] = values[0]
        if len(values) > 1:
            signals["support_gap_top2_options"] = values[0] - values[1]
    return signals


def _classify(base: dict[str, Any], run: dict[str, Any]) -> str:
    base_k = _memory_frames(base)
    run_k = _memory_frames(run)
    base_correct = _correct(base)
    run_correct = _correct(run)
    if base_k == 0 and run_k == 0:
        return "SAME_STOP"
    if run_k > base_k:
        if base_k == 0:
            if base_correct is False and run_correct is True:
                return "EXTRA_RETRIEVAL_HELPFUL"
            if base_correct is True and run_correct is False:
                return "EXTRA_RETRIEVAL_HARMFUL"
            if base_correct is True and run_correct is True:
                return "EXTRA_RETRIEVAL_NEUTRAL_CORRECT"
            return "EXTRA_RETRIEVAL_NEUTRAL_WRONG"
        if base_correct is False and run_correct is True:
            return "DIFFERENT_K_HELPFUL"
        if base_correct is True and run_correct is False:
            return "DIFFERENT_K_HARMFUL"
        if base_correct is True and run_correct is True:
            return "DIFFERENT_K_NEUTRAL_CORRECT"
        return "DIFFERENT_K_NEUTRAL_WRONG"
    if base_k == run_k:
        return "SAME_K"
    return "LOWER_K_THAN_BASELINE"


def _auc(labels: list[int], scores: list[float]) -> float | None:
    pairs = [(score, label) for score, label in zip(scores, labels) if math.isfinite(score)]
    pos = sum(label == 1 for _score, label in pairs)
    neg = sum(label == 0 for _score, label in pairs)
    if pos == 0 or neg == 0:
        return None
    ordered = sorted(pairs, key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        avg_rank = (index + 1 + end) / 2.0
        rank_sum += avg_rank * sum(label == 1 for _score, label in ordered[index:end])
        index = end
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def _signal_distributions(samples: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    signal_names = sorted({name for sample in samples for name in sample.get("signals", {})})
    distributions: dict[str, Any] = {}
    auc_rows: list[dict[str, Any]] = []
    for name in signal_names:
        positive_values = [sample["signals"].get(name) for sample in samples if sample["binary_label"] == 1]
        negative_values = [sample["signals"].get(name) for sample in samples if sample["binary_label"] == 0]
        labels: list[int] = []
        values: list[float] = []
        for sample in samples:
            value = sample["signals"].get(name)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                labels.append(int(sample["binary_label"]))
                values.append(float(value))
        auc = _auc(labels, values) if labels else None
        auc_abs = max(auc, 1.0 - auc) if auc is not None else None
        distributions[name] = {
            "positive": _numeric_summary(positive_values),
            "negative": _numeric_summary(negative_values),
            "auc_positive_high": auc,
            "auc_best_direction": auc_abs,
            "higher_values_predict": "positive" if auc is not None and auc >= 0.5 else "negative",
        }
        auc_rows.append(
            {
                "signal": name,
                "available": len(values),
                "auc_positive_high": auc,
                "auc_best_direction": auc_abs,
                "higher_values_predict": "positive" if auc is not None and auc >= 0.5 else "negative",
            }
        )
    auc_rows.sort(key=lambda row: (row["auc_best_direction"] is not None, row["auc_best_direction"] or -1), reverse=True)
    return distributions, auc_rows


def _stability(samples: list[dict[str, Any]]) -> dict[str, Any]:
    transitions: list[dict[str, Any]] = []
    by_sample: list[dict[str, Any]] = []
    for sample in samples:
        iterations = _iterations(sample["run_row"])
        if len(iterations) < 2:
            continue
        preds = [item.get("predicted_option") for item in iterations]
        top1s = [_top_probs(item.get("option_probabilities"))[0] for item in iterations]
        margins = [_float(item.get("answer_margin")) for item in iterations]
        dist_changes: list[float | None] = []
        pred_changes = []
        for index in range(1, len(iterations)):
            pred_changed = preds[index] != preds[index - 1]
            confidence_change = (
                top1s[index] - top1s[index - 1]
                if top1s[index] is not None and top1s[index - 1] is not None
                else None
            )
            margin_change = (
                margins[index] - margins[index - 1]
                if margins[index] is not None and margins[index - 1] is not None
                else None
            )
            divergence = _js_divergence(
                iterations[index - 1].get("option_probabilities"),
                iterations[index].get("option_probabilities"),
            )
            pred_changes.append(pred_changed)
            dist_changes.append(divergence)
            transitions.append(
                {
                    "key": sample["key"],
                    "group": sample["group"],
                    "task": sample["task"],
                    "from_k": index - 1,
                    "to_k": index,
                    "prediction_changed": pred_changed,
                    "confidence_change": confidence_change,
                    "margin_change": margin_change,
                    "distribution_change_js": divergence,
                }
            )
        by_sample.append(
            {
                "key": sample["key"],
                "group": sample["group"],
                "task": sample["task"],
                "predictions": preds,
                "any_prediction_changed": any(pred_changes),
                "num_prediction_changes": sum(pred_changes),
                "mean_distribution_change_js": mean([v for v in dist_changes if v is not None]) if any(v is not None for v in dist_changes) else None,
            }
        )

    by_group: dict[str, Any] = {}
    for group, rows in _group_by(transitions, "group").items():
        by_group[group] = {
            "transitions": len(rows),
            "prediction_change_rate": sum(bool(row["prediction_changed"]) for row in rows) / len(rows) if rows else None,
            "confidence_change": _numeric_summary(row.get("confidence_change") for row in rows),
            "margin_change": _numeric_summary(row.get("margin_change") for row in rows),
            "distribution_change_js": _numeric_summary(row.get("distribution_change_js") for row in rows),
        }
    return {"by_group": by_group, "transitions": transitions, "by_sample": by_sample}


def _group_by(rows: Iterable[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(row)
    return dict(grouped)


def _candidate_agreement(samples: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    missing = Counter()
    for sample in samples:
        decision_k = sample["decision_k"]
        current = _iterations(sample["run_row"])[decision_k] if decision_k < len(_iterations(sample["run_row"])) else {}
        candidate = _best_candidate(sample["run_row"], decision_k)
        current_pred = current.get("predicted_option")
        option_supports = candidate.get("option_supports") or current.get("candidate_option_supports")
        if not isinstance(option_supports, dict):
            missing["option_supports"] += 1
            continue
        supports = {str(k): _float(v) for k, v in option_supports.items()}
        supports = {k: v for k, v in supports.items() if v is not None}
        if not supports or not current_pred:
            missing["current_pred_or_supports"] += 1
            continue
        history_pred = max(supports.items(), key=lambda item: (item[1], item[0]))[0]
        current_support = supports.get(str(current_pred))
        alternatives = [value for key, value in supports.items() if key != str(current_pred)]
        best_alt = max(alternatives) if alternatives else None
        rows.append(
            {
                "key": sample["key"],
                "group": sample["group"],
                "task": sample["task"],
                "current_pred": current_pred,
                "history_pred": history_pred,
                "history_agrees_with_current": history_pred == str(current_pred),
                "history_support_current": current_support,
                "history_support_best_alternative": best_alt,
                "alternative_advantage": best_alt - current_support if best_alt is not None and current_support is not None else None,
            }
        )
    summary = {}
    for group, items in _group_by(rows, "group").items():
        summary[group] = {
            "count": len(items),
            "agreement_rate": sum(bool(item["history_agrees_with_current"]) for item in items) / len(items) if items else None,
            "history_support_current": _numeric_summary(item.get("history_support_current") for item in items),
            "history_support_best_alternative": _numeric_summary(item.get("history_support_best_alternative") for item in items),
            "alternative_advantage": _numeric_summary(item.get("alternative_advantage") for item in items),
        }
    return {
        "available_rows": len(rows),
        "missing": dict(missing),
        "summary_by_group": summary,
        "rows": rows,
        "note": "Threshold-only PRISM logs generally do not include per-candidate option CLIP support; this section is populated only when such fields are already present.",
    }


def _temporal_analysis(samples: list[dict[str, Any]]) -> dict[str, Any]:
    bucket_defs = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 64)]
    rows = [sample for sample in samples if sample["signals"].get("temporal_distance_chunks") is not None]
    by_group = {
        group: {
            "temporal_distance_chunks": _numeric_summary(item["signals"].get("temporal_distance_chunks") for item in items),
            "temporal_distance_seconds": _numeric_summary(item["signals"].get("temporal_distance_seconds") for item in items),
        }
        for group, items in _group_by(rows, "group").items()
    }
    buckets: dict[str, Any] = {}
    for lo, hi in bucket_defs:
        label = f"{lo}-{hi}"
        items = [
            item
            for item in rows
            if isinstance(item["signals"].get("temporal_distance_chunks"), (int, float))
            and lo <= float(item["signals"]["temporal_distance_chunks"]) < hi
        ]
        counts = Counter(item["outcome"] for item in items)
        total = len(items)
        buckets[label] = {
            "count": total,
            "rescued": counts.get("rescued", 0),
            "damaged": counts.get("damaged", 0),
            "neutral": counts.get("neutral", 0),
            "rescue_rate": counts.get("rescued", 0) / total if total else None,
            "damage_rate": counts.get("damaged", 0) / total if total else None,
            "neutral_rate": counts.get("neutral", 0) / total if total else None,
        }
    return {"by_group": by_group, "buckets_by_chunk_distance": buckets}


def _task_analysis(samples: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for task in TASKS:
        items = [sample for sample in samples if sample["task"] == task]
        if not items:
            continue
        out[task] = {
            "count": len(items),
            "groups": dict(Counter(item["group"] for item in items)),
            "outcomes": dict(Counter(item["outcome"] for item in items)),
            "memory_frames_t074": _numeric_summary(item["run_k"] for item in items),
            "current_sufficiency": _numeric_summary(item["signals"].get("current_sufficiency") for item in items),
            "candidate_total_score": _numeric_summary(item["signals"].get("candidate_total_score") for item in items),
        }
    return out


def _evaluate_rule(samples: list[dict[str, Any]], rule: dict[str, Any], tasks: set[str] | None = None) -> dict[str, Any]:
    selected = [sample for sample in samples if tasks is None or sample["task"] in tasks]
    rescued = damaged = allowed = blocked = 0
    correct = 0
    false_stops = 0
    memory_frames = 0
    for sample in selected:
        use_t074 = _rule_decision(sample, rule)
        chosen_correct = sample["run_correct"] if use_t074 else sample["base_correct"]
        chosen_k = sample["run_k"] if use_t074 else sample["base_k"]
        correct += int(bool(chosen_correct))
        false_stops += int(chosen_correct is False and chosen_k == 0)
        memory_frames += int(chosen_k)
        if use_t074:
            allowed += 1
            if sample["base_correct"] is False and sample["run_correct"] is True:
                rescued += 1
            if sample["base_correct"] is True and sample["run_correct"] is False:
                damaged += 1
        else:
            blocked += 1
    return {
        "samples": len(selected),
        "micro_accuracy": correct / len(selected) if selected else None,
        "rescued": rescued,
        "damaged": damaged,
        "net_rescue": rescued - damaged,
        "false_stops": false_stops,
        "memory_trigger_rate_proxy": sum(1 for sample in selected if (_rule_decision(sample, rule) and sample["run_k"] > 0) or (not _rule_decision(sample, rule) and sample["base_k"] > 0)) / len(selected) if selected else None,
        "avg_historical_frames_proxy": memory_frames / len(selected) if selected else None,
        "extra_retrieval_allowed": allowed,
        "extra_retrieval_blocked": blocked,
    }


def _rule_decision(sample: dict[str, Any], rule: dict[str, Any]) -> bool:
    suff = sample["signals"].get("current_sufficiency")
    cand = sample["signals"].get("candidate_total_score")
    adv = sample["signals"].get("candidate_minus_current_support")
    if sample["run_k"] <= sample["base_k"]:
        return False
    kind = rule["kind"]
    if kind == "suff_lt":
        return suff is not None and suff < rule["tau"]
    if kind == "suff_lt_and_candidate_gt":
        return suff is not None and cand is not None and suff < rule["tau"] and cand > rule["candidate_threshold"]
    if kind == "suff_lt_and_advantage_gt":
        return suff is not None and adv is not None and suff < rule["tau"] and adv > rule["advantage_threshold"]
    if kind == "two_zone_candidate_advantage":
        if suff is None:
            return False
        if suff < rule["tau_low"]:
            return True
        if suff >= rule["tau_high"]:
            return False
        ok_candidate = cand is not None and cand > rule["candidate_threshold"]
        ok_advantage = adv is not None and adv > rule["advantage_threshold"]
        return ok_candidate and ok_advantage
    return False


def _candidate_thresholds(values: Iterable[float | None]) -> list[float]:
    clean = sorted({round(float(value), 6) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))})
    if len(clean) <= 12:
        return clean
    return sorted({clean[int((len(clean) - 1) * q)] for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]})


def _rule_search(samples: list[dict[str, Any]]) -> dict[str, Any]:
    suffs = _candidate_thresholds(sample["signals"].get("current_sufficiency") for sample in samples)
    cands = _candidate_thresholds(sample["signals"].get("candidate_total_score") for sample in samples)
    advs = _candidate_thresholds(sample["signals"].get("candidate_minus_current_support") for sample in samples)
    rules: list[dict[str, Any]] = []
    for tau in suffs:
        rules.append({"kind": "suff_lt", "tau": tau})
        for candidate_threshold in cands:
            rules.append({"kind": "suff_lt_and_candidate_gt", "tau": tau, "candidate_threshold": candidate_threshold})
        for advantage_threshold in advs:
            rules.append({"kind": "suff_lt_and_advantage_gt", "tau": tau, "advantage_threshold": advantage_threshold})
    for tau_low in suffs:
        for tau_high in suffs:
            if tau_high <= tau_low:
                continue
            for candidate_threshold in cands:
                for advantage_threshold in advs:
                    rules.append(
                        {
                            "kind": "two_zone_candidate_advantage",
                            "tau_low": tau_low,
                            "tau_high": tau_high,
                            "candidate_threshold": candidate_threshold,
                            "advantage_threshold": advantage_threshold,
                        }
                    )
    scored = []
    for rule in rules:
        metrics = _evaluate_rule(samples, rule)
        scored.append({"rule": rule, "metrics": metrics})
    scored.sort(
        key=lambda item: (
            item["metrics"]["micro_accuracy"] or -1,
            item["metrics"]["net_rescue"],
            -(item["metrics"]["memory_trigger_rate_proxy"] or 9),
            -(item["metrics"]["avg_historical_frames_proxy"] or 9),
        ),
        reverse=True,
    )
    return {"searched_rules": len(scored), "top_rules": scored[:25]}


def _heldout_validation(samples: list[dict[str, Any]], top_rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks = [task for task in TASKS if any(sample["task"] == task for sample in samples)]
    rows = []
    for entry in top_rules[:10]:
        rule = entry["rule"]
        heldout = []
        for task in tasks:
            metrics = _evaluate_rule(samples, rule, tasks={task})
            heldout.append({"task": task, **metrics})
        rows.append(
            {
                "rule": rule,
                "heldout": heldout,
                "mean_micro_accuracy": mean([item["micro_accuracy"] for item in heldout if item["micro_accuracy"] is not None]) if heldout else None,
                "mean_net_rescue": mean([item["net_rescue"] for item in heldout]) if heldout else None,
                "mean_memory_trigger_rate_proxy": mean([item["memory_trigger_rate_proxy"] for item in heldout if item["memory_trigger_rate_proxy"] is not None]) if heldout else None,
            }
        )
    return rows


def _examples(samples: list[dict[str, Any]], limit: int) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for group, items in _group_by(samples, "group").items():
        out[group] = [
            {
                "key": item["key"],
                "task": item["task"],
                "question": item["question"],
                "t062_prediction": item["base_prediction"],
                "t074_prediction": item["run_prediction"],
                "t062_k": item["base_k"],
                "t074_k": item["run_k"],
                "signals": item["signals"],
            }
            for item in items[:limit]
        ]
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    signal_names = sorted({name for row in rows for name in row.get("signals", {})})
    fields = [
        "key",
        "group",
        "outcome",
        "task",
        "question",
        "base_prediction",
        "run_prediction",
        "base_k",
        "run_k",
        "decision_k",
        *signal_names,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {field: row.get(field) for field in fields}
            flat.update({name: row.get("signals", {}).get(name) for name in signal_names})
            writer.writerow(flat)


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# OVO PRISM t062 vs t074 Delta Diagnostics",
        "",
        "Read-only offline analysis. Signals are inference-time logged fields only; labels are used only for offline evaluation.",
        "",
        "## A. Delta Set Counts",
        "",
    ]
    for group, count in summary["delta_counts"].items():
        lines.append(f"- {group}: {count}")
    lines.extend(["", "## D. Top ROC-AUC Signals", ""])
    for row in summary["auc_ranking"][:15]:
        auc = row["auc_best_direction"]
        if auc is None:
            continue
        lines.append(
            f"- {row['signal']}: AUC={auc:.3f}, direction={row['higher_values_predict']}, n={row['available']}"
        )
    lines.extend(["", "## I. Best Offline Rules", ""])
    for entry in summary["offline_rule_search"]["top_rules"][:10]:
        metrics = entry["metrics"]
        lines.append(
            f"- `{entry['rule']}`: micro={metrics['micro_accuracy']:.4f}, "
            f"net={metrics['net_rescue']}, mem_rate={metrics['memory_trigger_rate_proxy']:.4f}, "
            f"avg_frames={metrics['avg_historical_frames_proxy']:.4f}"
        )
    lines.extend(["", "## K. Recommended Next Controller", "", summary["recommendation"]])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t062", type=Path, required=True, help="Threshold 0.62 result dir on the fixed subset.")
    parser.add_argument("--t074", type=Path, required=True, help="Threshold 0.74 result dir on the same fixed subset.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()

    t062_rows, t062_dupes = _dedupe_by_key(_load_result_rows(args.t062))
    t074_rows, t074_dupes = _dedupe_by_key(_load_result_rows(args.t074))
    common = sorted(set(t062_rows) & set(t074_rows))
    samples: list[dict[str, Any]] = []
    skipped = Counter()
    for key in common:
        base = t062_rows[key]
        run = t074_rows[key]
        base_correct = _correct(base)
        run_correct = _correct(run)
        if base_correct is None or run_correct is None:
            skipped["unscored"] += 1
            continue
        group = _classify(base, run)
        base_k = _memory_frames(base)
        run_k = _memory_frames(run)
        if run_k > base_k:
            decision_k = base_k
            signals = _extract_signals(run, decision_k)
        else:
            decision_k = min(base_k, max(0, len(_iterations(run)) - 1))
            signals = _extract_signals(run, decision_k) if _iterations(run) else {}
        if group in POSITIVE_GROUPS:
            binary_label = 1
            outcome = "rescued"
        elif group in NEGATIVE_GROUPS:
            binary_label = 0
            if base_correct is True and run_correct is False:
                outcome = "damaged"
            else:
                outcome = "neutral"
        else:
            binary_label = None
            outcome = "other"
        samples.append(
            {
                "key": key,
                "group": group,
                "outcome": outcome,
                "binary_label": binary_label,
                "task": str(run.get("task") or base.get("task") or ""),
                "question": str(run.get("question") or base.get("question") or ""),
                "base_correct": base_correct,
                "run_correct": run_correct,
                "base_prediction": _prediction(base),
                "run_prediction": _prediction(run),
                "base_k": base_k,
                "run_k": run_k,
                "decision_k": decision_k,
                "signals": signals,
                "base_row": base,
                "run_row": run,
            }
        )

    comparison_samples = [sample for sample in samples if sample["binary_label"] in {0, 1}]
    distributions, auc_ranking = _signal_distributions(comparison_samples)
    rule_search = _rule_search(comparison_samples)
    heldout = _heldout_validation(comparison_samples, rule_search["top_rules"])
    delta_counts = dict(Counter(sample["group"] for sample in samples))
    validation = {
        "t062": t062_dupes,
        "t074": t074_dupes,
        "common_keys": len(common),
        "scored_common_keys": len(samples),
        "skipped": dict(skipped),
        "match_key": "id_task_question",
    }
    if rule_search["top_rules"]:
        best = rule_search["top_rules"][0]
        recommendation = (
            "Do not change official PRISM yet. The next candidate should be a simple two-zone, "
            "training-free gate only if its held-out task metrics remain stable: retrieve when "
            "current sufficiency is clearly low; stop when sufficiency is clearly high; in the "
            "ambiguous band, require strong candidate score and candidate-vs-current support advantage. "
            f"Best offline rule found here: {best['rule']}."
        )
    else:
        recommendation = (
            "Do not change official PRISM yet. No offline rule could be evaluated from available logged signals."
        )

    public_samples = []
    for sample in samples:
        public = {key: value for key, value in sample.items() if key not in {"base_row", "run_row"}}
        public_samples.append(public)

    summary = {
        "validation": validation,
        "delta_counts": delta_counts,
        "examples": _examples(samples, args.examples),
        "signal_distributions": distributions,
        "auc_ranking": auc_ranking,
        "answer_stability": _stability(samples),
        "candidate_answer_agreement": _candidate_agreement(comparison_samples),
        "temporal_distance": _temporal_analysis(comparison_samples),
        "per_task": _task_analysis(comparison_samples),
        "offline_rule_search": rule_search,
        "heldout_validation": heldout,
        "recommendation": recommendation,
        "sample_classifications": public_samples,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.out_dir / "sample_classifications.csv", samples)
    (args.out_dir / "report.md").write_text(_render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
