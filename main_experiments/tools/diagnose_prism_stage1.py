#!/usr/bin/env python3
"""Deep diagnostics for Recent-6 vs PRISM Stage-1 analysis.

This script is intentionally analysis-only. It reads completed result files and
does not import or modify PRISM model/retrieval behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


OVO_GROUPS = {
    "EPM": "backward",
    "HLD": "backward",
    "ASI": "backward",
    "OCR": "real_time",
    "OJR": "real_time",
    "ACR": "real_time",
    "STU": "real_time",
    "ATR": "real_time",
    "FPD": "real_time",
    "REC": "forward",
    "SSR": "forward",
    "CRR": "forward",
}


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get("results"), list):
        return value["results"]
    if isinstance(value, dict) and all(key in value for key in ("backward", "realtime", "forward")):
        rows: list[dict[str, Any]] = []
        for group in ("backward", "realtime", "forward"):
            for row in value[group]:
                if isinstance(row, dict):
                    rows.append({**row, "_source_group": group})
        return rows
    return []


def _flatten(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        row = dict(row)
        row.setdefault("_record_index", row_index)
        if isinstance(row.get("test_info"), list):
            for index, item in enumerate(row["test_info"]):
                if isinstance(item, dict):
                    merged = {**row, **item, "_question_index": index}
                    flat.append(merged)
        else:
            flat.append(row)
    return flat


def _load_records(result_dir: Path) -> list[dict[str, Any]]:
    if result_dir.is_file():
        rows = _flatten(_read_json_rows(result_dir))
        for index, row in enumerate(rows):
            row["_file"] = str(result_dir)
            row["_load_index"] = index
        return rows

    merged = result_dir / "merged_results.json"
    if merged.exists():
        rows = _flatten(_read_json_rows(merged))
        for index, row in enumerate(rows):
            row["_file"] = str(merged)
            row["_load_index"] = index
        return rows

    rows: list[dict[str, Any]] = []
    for path in sorted(result_dir.glob("rank_*/results_incremental.jsonl")):
        for row in _read_json_rows(path):
            row["_file"] = str(path)
            rows.append(row)
    if rows:
        rows = _flatten(rows)
        for index, row in enumerate(rows):
            row["_load_index"] = index
        return rows

    ovo_jsons = sorted(result_dir.glob("minicpmv46_results_*.json"))
    if ovo_jsons:
        path = ovo_jsons[-1]
        rows = _flatten(_read_json_rows(path))
        for index, row in enumerate(rows):
            row["_file"] = str(path)
            row["_load_index"] = index
        return rows

    raise FileNotFoundError(f"No supported result records found under {result_dir}")


def _extract_letter(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"A", "B", "C", "D", "E"}:
        return text
    match = re.search(r"\b([A-E])\b", text)
    return match.group(1) if match else ""


def _prediction(row: dict[str, Any]) -> str:
    return _extract_letter(row.get("prediction") or row.get("pred") or row.get("response"))


def _ground_truth(row: dict[str, Any]) -> str:
    return _extract_letter(row.get("ground_truth") or row.get("answer_gt") or row.get("answer"))


def _correct_with_source(row: dict[str, Any]) -> tuple[bool | None, str]:
    value = row.get("correct")
    if isinstance(value, bool):
        return value, "correct_field"
    gt = _ground_truth(row)
    pred = _prediction(row)
    if gt and pred:
        return gt == pred, "letter_compare"
    if not gt and not pred:
        return None, "missing_ground_truth_and_prediction"
    if not gt:
        return None, "missing_ground_truth"
    return None, "missing_prediction"


def _task(row: dict[str, Any]) -> str:
    return str(row.get("task") or row.get("task_type") or row.get("category") or "")


def _group(benchmark: str, task: str) -> str:
    if benchmark == "ovo":
        return OVO_GROUPS.get(task, "unknown")
    return task or "unknown"


def _primary_key(row: dict[str, Any]) -> str:
    if row.get("_key"):
        return f"_key:{row['_key']}"
    parts = [
        row.get("id", ""),
        row.get("video", ""),
        row.get("video_path", ""),
        row.get("task", row.get("task_type", "")),
        row.get("time_stamp", ""),
        row.get("question", ""),
        row.get("_question_index", ""),
    ]
    return "fallback:" + "|".join(str(part) for part in parts)


def _sample_key_candidates(row: dict[str, Any]) -> dict[str, str]:
    return {
        "primary": _primary_key(row),
        "id": str(row.get("id", "")),
        "video_task_question_time": "|".join(
            str(row.get(key, ""))
            for key in ("video", "video_path", "task", "task_type", "time_stamp", "question")
        ),
        "question": str(row.get("question", "")),
    }


def _adaptive(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("adaptive"), dict):
        return row["adaptive"]
    profile = row.get("profile")
    if isinstance(profile, dict) and isinstance(profile.get("adaptive"), dict):
        return profile["adaptive"]
    return {}


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _baseline_chunk_ids(row: dict[str, Any]) -> list[int]:
    for key in ("final_chunk_ids", "selected_chunk_ids"):
        ids = _int_list(row.get(key))
        if ids:
            return ids
    profile = row.get("profile")
    if isinstance(profile, dict):
        ids = _int_list(profile.get("selected_chunk_ids"))
        if ids:
            return ids
    return []


def _prism_ids(row: dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
    adaptive = _adaptive(row)
    recent = _int_list(adaptive.get("recent_chunk_ids"))
    memory = _int_list(adaptive.get("memory_chunk_ids"))
    final = _int_list(adaptive.get("final_selected_chunk_ids"))
    if not final:
        final = _int_list(row.get("final_chunk_ids"))
    if not recent and final:
        memory_set = set(memory)
        recent = [chunk_id for chunk_id in final if chunk_id not in memory_set]
    return recent, memory, final


def _actual_memory(row: dict[str, Any]) -> bool:
    recent, memory, final = _prism_ids(row)
    if memory:
        return True
    return bool(set(final) - set(recent)) if recent and final else False


def _iteration(row: dict[str, Any], index: int) -> dict[str, Any]:
    adaptive = _adaptive(row)
    iterations = adaptive.get("iterations")
    if isinstance(iterations, list) and index < len(iterations) and isinstance(iterations[index], dict):
        return iterations[index]
    return {}


def _classification(base_correct: bool, prism_correct: bool) -> str:
    if base_correct and prism_correct:
        return "both_correct"
    if base_correct and not prism_correct:
        return "damaged"
    if not base_correct and prism_correct:
        return "rescued"
    return "both_wrong"


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _numeric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    values = sorted(values)
    p10 = values[max(0, min(len(values) - 1, int(round(0.10 * (len(values) - 1)))))]
    p90 = values[max(0, min(len(values) - 1, int(round(0.90 * (len(values) - 1)))))]
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "p10": p10,
        "p90": p90,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _paired_records(
    baseline_rows: list[dict[str, Any]], prism_rows: list[dict[str, Any]]
) -> tuple[list[tuple[str, dict[str, Any], dict[str, Any]]], dict[str, Any]]:
    base_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prism_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in baseline_rows:
        base_by_key[_primary_key(row)].append(row)
    for row in prism_rows:
        prism_by_key[_primary_key(row)].append(row)

    common = sorted(set(base_by_key) & set(prism_by_key))
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    duplicate_examples: list[dict[str, Any]] = []
    for key in common:
        base_items = sorted(base_by_key[key], key=lambda item: int(item.get("_load_index", 0)))
        prism_items = sorted(prism_by_key[key], key=lambda item: int(item.get("_load_index", 0)))
        if len(base_items) != 1 or len(prism_items) != 1:
            duplicate_examples.append(
                {
                    "key": key,
                    "baseline_count": len(base_items),
                    "prism_count": len(prism_items),
                    "baseline_files": sorted({str(item.get("_file", "")) for item in base_items}),
                    "prism_files": sorted({str(item.get("_file", "")) for item in prism_items}),
                }
            )
        for base, prism in zip(base_items, prism_items):
            pairs.append((key, base, prism))

    diag = {
        "baseline_records": len(baseline_rows),
        "prism_records": len(prism_rows),
        "baseline_unique_primary_keys": len(base_by_key),
        "prism_unique_primary_keys": len(prism_by_key),
        "common_primary_keys": len(common),
        "paired_records_after_duplicate_zip": len(pairs),
        "baseline_duplicate_records": sum(max(0, len(items) - 1) for items in base_by_key.values()),
        "prism_duplicate_records": sum(max(0, len(items) - 1) for items in prism_by_key.values()),
        "baseline_only_keys": len(set(base_by_key) - set(prism_by_key)),
        "prism_only_keys": len(set(prism_by_key) - set(base_by_key)),
        "matching_key": "_key if present else id|video|video_path|task|task_type|time_stamp|question|_question_index",
        "duplicate_examples": duplicate_examples[:20],
        "baseline_only_examples": sorted(set(base_by_key) - set(prism_by_key))[:20],
        "prism_only_examples": sorted(set(prism_by_key) - set(base_by_key))[:20],
    }
    return pairs, diag


def _row_summary(
    benchmark: str,
    key: str,
    base: dict[str, Any],
    prism: dict[str, Any],
) -> dict[str, Any] | None:
    base_correct, base_correct_source = _correct_with_source(base)
    prism_correct, prism_correct_source = _correct_with_source(prism)
    if base_correct is None or prism_correct is None:
        return None

    adaptive = _adaptive(prism)
    iter0 = _iteration(prism, 0)
    recent_ids, memory_ids, final_ids = _prism_ids(prism)
    base_ids = _baseline_chunk_ids(base)
    actual_memory = bool(set(final_ids) - set(recent_ids)) if recent_ids and final_ids else bool(memory_ids)
    label = _classification(bool(base_correct), bool(prism_correct))
    gains = [
        float(item.get("gain_vs_previous"))
        for item in adaptive.get("iterations", []) or []
        if isinstance(item, dict) and isinstance(item.get("gain_vs_previous"), (int, float))
    ]

    return {
        "benchmark": benchmark,
        "key": key,
        "task": _task(prism) or _task(base),
        "group": _group(benchmark, _task(prism) or _task(base)),
        "question": str(prism.get("question") or base.get("question") or ""),
        "ground_truth": _ground_truth(prism) or _ground_truth(base),
        "baseline_prediction": _prediction(base),
        "prism_prediction": _prediction(prism),
        "baseline_correct": bool(base_correct),
        "prism_correct": bool(prism_correct),
        "label": label,
        "baseline_correct_source": base_correct_source,
        "prism_correct_source": prism_correct_source,
        "baseline_chunk_ids": json.dumps(base_ids),
        "prism_recent_chunk_ids": json.dumps(recent_ids),
        "prism_memory_chunk_ids": json.dumps(memory_ids),
        "prism_final_selected_chunk_ids": json.dumps(final_ids),
        "num_historical_frames": len(set(final_ids) - set(recent_ids)) if recent_ids and final_ids else len(memory_ids),
        "actual_memory_retrieval": actual_memory,
        "memory_triggered": bool(adaptive.get("memory_triggered")),
        "stop_reason": adaptive.get("stop_reason"),
        "iter0_prediction": iter0.get("predicted_option"),
        "iter0_answer_margin": _safe_float(iter0.get("answer_margin")),
        "iter0_entropy_confidence": _safe_float(iter0.get("entropy_confidence")),
        "iter0_normalized_entropy": _safe_float(iter0.get("normalized_entropy")),
        "iter0_visual_support": _safe_float(iter0.get("visual_support_norm")),
        "iter0_visual_support_raw": _safe_float(iter0.get("visual_support_raw")),
        "iter0_sufficiency": _safe_float(iter0.get("sufficiency")),
        "final_sufficiency": _safe_float(adaptive.get("final_sufficiency")),
        "sufficiency_gains": json.dumps(gains),
        "candidate_scores": json.dumps(
            [
                {
                    "chunk_id": item.get("chunk_id"),
                    "total_score": item.get("total_score"),
                    "semantic_score": item.get("semantic_score"),
                    "event_score": item.get("event_score"),
                    "detail_score": item.get("detail_score"),
                    "cue_score": item.get("cue_score"),
                }
                for item in adaptive.get("candidate_queue", []) or []
                if isinstance(item, dict) and item.get("chunk_id") in set(memory_ids)
            ],
            ensure_ascii=False,
        ),
        "baseline_prompt_available": any(key in base for key in ("prompt", "answer_prompt", "final_prompt")),
        "prism_prompt_available": any(key in prism for key in ("prompt", "answer_prompt", "final_prompt")),
        "baseline_decode_backend": base.get("decode_backend"),
        "prism_decode_backend": prism.get("decode_backend"),
        "baseline_file": base.get("_file"),
        "prism_file": prism.get("_file"),
    }


def _scoreability_report(
    benchmark: str,
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    source_counts = Counter()
    unscored: list[dict[str, Any]] = []
    scored = 0
    for key, base, prism in pairs:
        base_correct, base_source = _correct_with_source(base)
        prism_correct, prism_source = _correct_with_source(prism)
        source_counts[f"baseline:{base_source}"] += 1
        source_counts[f"prism:{prism_source}"] += 1
        if base_correct is None or prism_correct is None:
            if len(unscored) < 30:
                unscored.append(
                    {
                        "benchmark": benchmark,
                        "key": key,
                        "task": _task(prism) or _task(base),
                        "question": prism.get("question") or base.get("question"),
                        "baseline_source": base_source,
                        "prism_source": prism_source,
                        "baseline_ground_truth_fields": {
                            name: base.get(name) for name in ("ground_truth", "answer_gt", "answer")
                        },
                        "prism_ground_truth_fields": {
                            name: prism.get(name) for name in ("ground_truth", "answer_gt", "answer")
                        },
                        "baseline_response": base.get("response"),
                        "prism_response": prism.get("response"),
                        "baseline_keys": sorted(base.keys()),
                        "prism_keys": sorted(prism.keys()),
                    }
                )
        else:
            scored += 1
    return {
        "paired_records": len(pairs),
        "scored_pairs": scored,
        "unscored_pairs": len(pairs) - scored,
        "correctness_source_counts": dict(source_counts),
        "unscored_examples": unscored,
    }


def _count_table(rows: list[dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    counts: dict[tuple[Any, ...], Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        counts[key][row["label"]] += 1
        totals[key] += 1
    out: list[dict[str, Any]] = []
    for key, counter in sorted(counts.items()):
        record = {field: value for field, value in zip(key_fields, key)}
        record.update(
            {
                "samples": totals[key],
                "rescued": counter.get("rescued", 0),
                "damaged": counter.get("damaged", 0),
                "both_correct": counter.get("both_correct", 0),
                "both_wrong": counter.get("both_wrong", 0),
                "net_rescue": counter.get("rescued", 0) - counter.get("damaged", 0),
            }
        )
        out.append(record)
    return out


def _same_context(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    no_memory = [row for row in rows if int(row.get("num_historical_frames") or 0) == 0]
    same = []
    diff = []
    by_label: dict[str, Counter] = defaultdict(Counter)
    for row in no_memory:
        answer_equal = row["baseline_prediction"] == row["prism_prediction"]
        by_label[row["label"]]["same_answer" if answer_equal else "different_answer"] += 1
        (same if answer_equal else diff).append(row)
    return {
        "no_historical_frame_samples": len(no_memory),
        "baseline_answer_equals_prism_answer": len(same),
        "baseline_answer_differs_from_prism_answer": len(diff),
        "by_label": {label: dict(counter) for label, counter in sorted(by_label.items())},
        "known_call_path_difference": (
            "PRISM performs an iteration-0 option-scoring forward pass before final generation; "
            "when memory is not selected, final generation uses the same recent frames and prompt "
            "unless record fields show otherwise. Prompts are generally not stored in result JSONL."
        ),
    }, diff


def _iteration0(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if row.get("iter0_prediction")]
    disagree = [
        row
        for row in available
        if str(row.get("iter0_prediction")) != str(row.get("baseline_prediction"))
    ]
    by_label = defaultdict(Counter)
    for row in available:
        by_label[row["label"]]["agree" if row not in disagree else "disagree"] += 1
    return {
        "samples_with_iteration0_prediction": len(available),
        "iteration0_vs_baseline_disagreements": len(disagree),
        "disagreement_rate": (len(disagree) / len(available)) if available else None,
        "by_label": {label: dict(counter) for label, counter in sorted(by_label.items())},
        "reason": (
            "iteration0 is produced by PRISM's option-scoring forward pass over option letters; "
            "Recent-6 baseline is a normal generation call. They are diagnostic cousins, not the "
            "same decoding path."
        ),
    }


def _calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mechanisms = Counter()
    examples: list[dict[str, Any]] = []
    margin_values: list[float] = []
    entropy_values: list[float] = []
    seen_labels: set[str] = set()
    for row in rows:
        adaptive_row = None
        # The row summary does not carry probabilities; this block is filled by
        # callers via the attached "_prism_raw" row.
        raw = row.get("_prism_raw")
        if isinstance(raw, dict):
            iter0 = _iteration(raw, 0)
            mechanisms[str(iter0.get("option_scoring_mechanism", ""))] += 1
            if isinstance(iter0.get("answer_margin"), (int, float)):
                margin_values.append(float(iter0["answer_margin"]))
            if isinstance(iter0.get("entropy_confidence"), (int, float)):
                entropy_values.append(float(iter0["entropy_confidence"]))
            probs = iter0.get("option_probabilities")
            if isinstance(probs, dict) and row["label"] not in seen_labels:
                rel_logits = {}
                logps = {key: math.log(max(float(value), 1e-12)) for key, value in probs.items()}
                if logps:
                    max_logp = max(logps.values())
                    rel_logits = {key: value - max_logp for key, value in logps.items()}
                examples.append(
                    {
                        "label": row["label"],
                        "benchmark": row["benchmark"],
                        "task": row["task"],
                        "question": row["question"],
                        "ground_truth": row["ground_truth"],
                        "baseline_prediction": row["baseline_prediction"],
                        "iter0_prediction": row.get("iter0_prediction"),
                        "prism_prediction": row["prism_prediction"],
                        "option_probabilities": probs,
                        "relative_logits_from_log_probabilities": rel_logits,
                        "answer_margin": iter0.get("answer_margin"),
                        "entropy_confidence": iter0.get("entropy_confidence"),
                        "option_token_ids": iter0.get("option_token_ids"),
                        "option_scoring_mechanism": iter0.get("option_scoring_mechanism"),
                    }
                )
                seen_labels.add(row["label"])
        else:
            adaptive_row = raw
    return {
        "softmax_temperature": 1.0,
        "scoring_mechanism_counts": dict(mechanisms),
        "margin": _numeric_summary(margin_values),
        "entropy_confidence": _numeric_summary(entropy_values),
        "raw_logits_available_in_records": False,
        "relative_logits_note": (
            "Saved JSONL stores option probabilities, not raw vocabulary logits. "
            "Relative log-probabilities are recoverable up to an additive constant."
        ),
        "option_length_note": (
            "Current scoring uses option letters A-E. When each label is a unique one-token ID, "
            "the direct last-position label logits are used; full option text length does not enter "
            "the probability except through the prompt context."
        ),
        "representative_examples": examples,
    }


def _visual_support(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get("iter0_visual_support")
        if not isinstance(value, (int, float)):
            continue
        groups[f"label:{row['label']}"].append(float(value))
        groups[f"actual_memory:{bool(row['actual_memory_retrieval'])}"].append(float(value))
        if row["actual_memory_retrieval"] and row["label"] == "rescued":
            groups["memory_helpful_proxy"].append(float(value))
        if row["actual_memory_retrieval"] and row["label"] == "damaged":
            groups["memory_harmful_proxy"].append(float(value))
        if row["actual_memory_retrieval"] and row["label"] in {"both_wrong", "damaged"}:
            groups["memory_not_helpful_proxy"].append(float(value))
    return {key: _numeric_summary(values) for key, values in sorted(groups.items())}


def _diagnose_benchmark(
    benchmark: str,
    baseline_dir: Path,
    prism_dir: Path,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_rows = _load_records(baseline_dir)
    prism_rows = _load_records(prism_dir)
    pairs, record_diag = _paired_records(baseline_rows, prism_rows)
    scoreability = _scoreability_report(benchmark, pairs)
    rows: list[dict[str, Any]] = []
    for key, base, prism in pairs:
        summary = _row_summary(benchmark, key, base, prism)
        if summary is not None:
            summary["_prism_raw"] = prism
            rows.append(summary)

    csv_rows = [{key: value for key, value in row.items() if key != "_prism_raw"} for row in rows]
    _write_csv(out_dir / f"{benchmark}_paired_scored_samples.csv", csv_rows)
    _write_csv(out_dir / f"{benchmark}_same_context_differences.csv", [
        {key: value for key, value in row.items() if key != "_prism_raw"}
        for row in _same_context(rows)[1]
    ])
    _write_csv(out_dir / f"{benchmark}_true_memory_by_task.csv", _count_table(rows, ["benchmark", "task", "actual_memory_retrieval"]))
    _write_csv(out_dir / f"{benchmark}_true_memory_by_group.csv", _count_table(rows, ["benchmark", "group", "actual_memory_retrieval"]))

    diag = {
        "record_matching": record_diag,
        "scoreability": scoreability,
        "same_context_consistency": _same_context(rows)[0],
        "true_memory_overall": _count_table(rows, ["benchmark", "actual_memory_retrieval"]),
        "true_memory_by_task": _count_table(rows, ["benchmark", "task", "actual_memory_retrieval"]),
        "true_memory_by_group": _count_table(rows, ["benchmark", "group", "actual_memory_retrieval"]),
        "iteration0_vs_recent6": _iteration0(rows),
        "option_score_calibration": _calibration(rows),
        "visual_support": _visual_support(rows),
    }
    return rows, diag


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ovo-baseline", type=Path, required=True)
    parser.add_argument("--ovo-prism", type=Path, required=True)
    parser.add_argument("--streamingbench-baseline", type=Path, required=True)
    parser.add_argument("--streamingbench-prism", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/prism_stage1_deep_diagnostics"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for benchmark, baseline_dir, prism_dir in (
        ("ovo", args.ovo_baseline, args.ovo_prism),
        ("streamingbench", args.streamingbench_baseline, args.streamingbench_prism),
    ):
        rows, diag = _diagnose_benchmark(benchmark, baseline_dir, prism_dir, args.out_dir)
        all_rows.extend(rows)
        diagnostics[benchmark] = diag

    all_csv_rows = [{key: value for key, value in row.items() if key != "_prism_raw"} for row in all_rows]
    _write_csv(args.out_dir / "all_paired_scored_samples.csv", all_csv_rows)
    _write_csv(args.out_dir / "all_true_memory_by_task.csv", _count_table(all_rows, ["benchmark", "task", "actual_memory_retrieval"]))

    diagnostics["combined"] = {
        "same_context_consistency": _same_context(all_rows)[0],
        "true_memory_overall": _count_table(all_rows, ["benchmark", "actual_memory_retrieval"]),
        "iteration0_vs_recent6": _iteration0(all_rows),
        "option_score_calibration": _calibration(all_rows),
        "visual_support": _visual_support(all_rows),
        "stage1_validity_note": (
            "Use true_memory_* tables for conclusions about PRISM retrieval. "
            "Non-memory rescue/damage reflects generation/scoring/path differences, not memory benefit."
        ),
    }
    with (args.out_dir / "stage1_deep_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, ensure_ascii=False)

    for benchmark in ("ovo", "streamingbench"):
        diag = diagnostics[benchmark]
        print("=" * 90)
        print(benchmark)
        print("record_matching:", json.dumps(diag["record_matching"], indent=2)[:3000])
        print("scoreability:", json.dumps(diag["scoreability"], indent=2)[:3000])
        print("same_context:", json.dumps(diag["same_context_consistency"], indent=2))
        print("true_memory_overall:", json.dumps(diag["true_memory_overall"], indent=2))
        print("iteration0:", json.dumps(diag["iteration0_vs_recent6"], indent=2))
        print("visual_support:", json.dumps(diag["visual_support"], indent=2)[:3000])
    print("=" * 90)
    print(f"saved: {args.out_dir}")


if __name__ == "__main__":
    main()
