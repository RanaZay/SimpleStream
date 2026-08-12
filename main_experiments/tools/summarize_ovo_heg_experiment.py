#!/usr/bin/env python3
"""Summarize OVO PRISM diagnostic sweeps on a fixed subset."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ovo_constants import BACKWARD_TASKS, REAL_TIME_TASKS, extract_br_answer  # noqa: E402


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and all(key in data for key in ("backward", "realtime", "forward")):
        return [*data.get("backward", []), *data.get("realtime", []), *data.get("forward", [])]
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    return []


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
        rows.extend(_load_json_rows(item))
    return _flatten(rows)


def _flatten(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in rows:
        if row.get("task") in set(BACKWARD_TASKS + REAL_TIME_TASKS):
            flat.append(row)
        for index, item in enumerate(row.get("test_info", []) or []):
            if isinstance(item, dict):
                flat.append({**item, "_parent_id": row.get("id"), "_question_index": index, "task": row.get("task")})
    return flat


def _metric_key(row: dict[str, Any]) -> str:
    return str(
        row.get("_key")
        or "|".join(
            str(row.get(key, ""))
            for key in ("id", "_parent_id", "_question_index", "video", "task", "question")
        )
    )


def _question_index(row: dict[str, Any]) -> Any:
    value = row.get("_question_index")
    if value is None:
        value = row.get("question_index")
    return value


def _primary_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("_parent_id") or "")


def _sample_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("_key") or row.get("key") or _primary_id(row)),
            str(row.get("task") or row.get("task_type") or ""),
            str(_question_index(row) if _question_index(row) is not None else ""),
            str(row.get("question") or ""),
        ]
    )


def _key_variants(row: dict[str, Any]) -> dict[str, str]:
    qidx = _question_index(row)
    variants = {
        "metric_key": _metric_key(row),
        "sample_key": _sample_key(row),
        "id_task_question": "|".join([_primary_id(row), str(row.get("task") or ""), str(row.get("question") or "")]),
        "video_task_question": "|".join([str(row.get("video") or row.get("video_path") or ""), str(row.get("task") or ""), str(row.get("question") or "")]),
        "task_question": "|".join([str(row.get("task") or ""), str(row.get("question") or "")]),
        "question": str(row.get("question") or ""),
    }
    if qidx is not None:
        variants["id_qidx_task_question"] = "|".join(
            [_primary_id(row), str(qidx), str(row.get("task") or ""), str(row.get("question") or "")]
        )
    explicit = row.get("_key") or row.get("key")
    if explicit:
        variants["explicit_key"] = str(explicit)
    return {name: value for name, value in variants.items() if value}


def _index_by_variant(rows: list[dict[str, Any]], variant: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = _key_variants(row).get(variant)
        if key:
            out[key].append(row)
    return dict(out)


def _duplicate_summary(rows: list[dict[str, Any]], variants: Iterable[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for variant in variants:
        index = _index_by_variant(rows, variant)
        duplicate_keys = {key: items for key, items in index.items() if len(items) > 1}
        summary[variant] = {
            "rows": len(rows),
            "unique_keys": len(index),
            "duplicate_keys": len(duplicate_keys),
            "duplicate_rows": sum(len(items) - 1 for items in duplicate_keys.values()),
        }
    return summary


def _best_matching_variant(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    variants: Iterable[str],
) -> tuple[str | None, dict[str, Any]]:
    diagnostics: dict[str, Any] = {}
    best_variant: str | None = None
    best_score = (-1, -1, -1)
    for variant in variants:
        left = _index_by_variant(left_rows, variant)
        right = _index_by_variant(right_rows, variant)
        common = set(left) & set(right)
        left_duplicate_rows = sum(len(items) - 1 for items in left.values() if len(items) > 1)
        right_duplicate_rows = sum(len(items) - 1 for items in right.values() if len(items) > 1)
        diagnostics[variant] = {
            "left_unique_keys": len(left),
            "right_unique_keys": len(right),
            "common_unique_keys": len(common),
            "left_duplicate_rows": left_duplicate_rows,
            "right_duplicate_rows": right_duplicate_rows,
            "unambiguous": left_duplicate_rows == 0 and right_duplicate_rows == 0,
        }
        has_common = len(common) > 0
        unambiguous = left_duplicate_rows == 0 and right_duplicate_rows == 0
        score = (has_common, unambiguous, len(common), -left_duplicate_rows - right_duplicate_rows, len(left) + len(right))
        if score > best_score:
            best_score = score
            best_variant = variant
    return best_variant, diagnostics


def _max_common_keys(diagnostics: dict[str, Any]) -> dict[str, Any]:
    if not diagnostics:
        return {"count": 0, "variants": []}
    max_count = max(int(item.get("common_unique_keys") or 0) for item in diagnostics.values())
    return {
        "count": max_count,
        "variants": [
            name
            for name, item in sorted(diagnostics.items())
            if int(item.get("common_unique_keys") or 0) == max_count
        ],
    }


def _matched_rows(
    oracle_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    variant: str | None,
) -> list[dict[str, Any]]:
    if not variant:
        return []
    rows_by_key = _index_by_variant(result_rows, variant)
    matched: list[dict[str, Any]] = []
    seen: set[str] = set()
    for oracle_row in oracle_rows:
        key = _key_variants(oracle_row).get(variant)
        if not key or key in seen:
            continue
        seen.add(key)
        matches = rows_by_key.get(key) or []
        if matches:
            matched.append(matches[0])
    return matched


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
    if gt in {"A", "B", "C", "D"} and pred:
        return pred == gt
    return None


def _adaptive(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("adaptive"), dict):
        return row["adaptive"]
    profile = row.get("profile")
    if isinstance(profile, dict) and isinstance(profile.get("adaptive"), dict):
        return profile["adaptive"]
    return {}


def _numbers(rows: Iterable[dict[str, Any]], *keys: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        adaptive = _adaptive(row)
        for key in keys:
            value = adaptive.get(key, row.get(key))
            if isinstance(value, (int, float)):
                values.append(float(value))
                break
    return values


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _percent(count: int, total: int) -> float | None:
    return count / total if total else None


def _trigger_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for item in _adaptive(row).get("iterations", []) or []:
            reason = item.get("retrieval_trigger_reason")
            if reason:
                counts[str(reason)] += 1
    return counts


def _false_stop(row: dict[str, Any]) -> bool:
    if _correct(row) is not False:
        return False
    meta = _adaptive(row)
    return str(meta.get("stop_reason")) in {"sufficient_evidence", "sufficient_no_historical_advantage"}


def _parse_csv_value(value: str) -> Any:
    text = value.strip()
    if text == "":
        return None
    if text in {"True", "False"}:
        return text == "True"
    if text == "None":
        return None
    if text and text[0] in "[{\"'":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return value
    return value


def _load_oracle_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    if path.is_dir():
        source = path / "oracle_rows.jsonl"
        if not source.exists():
            return []
        path = source
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return [{key: _parse_csv_value(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    return _load_json_rows(path)


def _is_known_oracle_false_stop(row: dict[str, Any]) -> bool:
    mode = str(row.get("failure_mode") or row.get("oracle_failure_mode") or "")
    if "false_stop" in mode:
        return True
    if row.get("k0_correct") is False and row.get("minimum_correct_k") in {1, 2, 3, "1", "2", "3"}:
        return True
    if row.get("baseline_correct") is False and row.get("oracle_correct") is True:
        return True
    return False


def _heg_activated(row: dict[str, Any]) -> bool:
    for item in _adaptive(row).get("iterations", []) or []:
        if str(item.get("retrieval_trigger_reason")) in {
            "historical_evidence_gain",
            "low_sufficiency_and_historical_gain",
        }:
            return True
    return False


def _memory_frames(row: dict[str, Any]) -> int:
    return int(_adaptive(row).get("num_memory_frames", 0) or 0)


def _memory_activated(row: dict[str, Any]) -> bool:
    return bool(_adaptive(row).get("memory_triggered")) or _memory_frames(row) > 0


def _official_macro_accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_section_task: dict[str, dict[str, list[bool]]] = {
        "backward": defaultdict(list),
        "realtime": defaultdict(list),
    }
    for row in rows:
        correct = _correct(row)
        if correct is None:
            continue
        task = str(row.get("task") or "")
        if task in BACKWARD_TASKS:
            by_section_task["backward"][task].append(bool(correct))
        elif task in REAL_TIME_TASKS:
            by_section_task["realtime"][task].append(bool(correct))

    sections: dict[str, Any] = {}
    section_avgs: list[float] = []
    for section, tasks in (("backward", BACKWARD_TASKS), ("realtime", REAL_TIME_TASKS)):
        task_summary: dict[str, Any] = {}
        task_accs: list[float] = []
        for task in tasks:
            values = by_section_task[section].get(task, [])
            if not values:
                continue
            correct_count = sum(values)
            accuracy = correct_count / len(values)
            task_accs.append(accuracy)
            task_summary[task] = {"samples": len(values), "correct": correct_count, "accuracy": accuracy}
        average = sum(task_accs) / len(task_accs) if task_accs else None
        if average is not None:
            section_avgs.append(average)
        sections[section] = {"average": average, "tasks": task_summary}
    return {
        "total_avg": sum(section_avgs) / len(section_avgs) if section_avgs else None,
        "backward_avg": sections["backward"]["average"],
        "realtime_avg": sections["realtime"]["average"],
        "sections": sections,
        "aggregation": "OVO printed-style macro: per-task accuracy, split average, then total over backward/realtime.",
    }


def _summarize_run(name: str, rows: list[dict[str, Any]], baseline_by_key: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if _correct(row) is not None]
    correct_count = sum(bool(_correct(row)) for row in scored)
    memory_frames = [_memory_frames(row) for row in rows]
    dist = Counter(memory_frames)
    paired = [(row, baseline_by_key.get(_metric_key(row))) for row in scored]
    paired = [(row, base) for row, base in paired if base is not None and _correct(base) is not None]
    rescued = sum(_correct(base) is False and _correct(row) is True for row, base in paired)
    damaged = sum(_correct(base) is True and _correct(row) is False for row, base in paired)
    baseline_correct = [(row, base) for row, base in paired if _correct(base) is True]
    new_triggers_on_baseline_correct = sum(_memory_frames(row) > _memory_frames(base) for row, base in baseline_correct)
    damaged_after_new_trigger = sum(
        _memory_frames(row) > _memory_frames(base) and _correct(row) is False for row, base in baseline_correct
    )
    return {
        "name": name,
        "samples": len(rows),
        "scored_samples": len(scored),
        "accuracy": correct_count / len(scored) if scored else None,
        "accuracy_count": correct_count,
        "backward_accuracy": _split_accuracy(scored, set(BACKWARD_TASKS)),
        "realtime_accuracy": _split_accuracy(scored, set(REAL_TIME_TASKS)),
        "official_macro_accuracy": _official_macro_accuracy(scored),
        "rescued": rescued,
        "damaged": damaged,
        "net_rescue": rescued - damaged,
        "paired_with_baseline": len(paired),
        "false_stops": sum(_false_stop(row) for row in scored),
        "memory_trigger_rate": _percent(sum(bool(_adaptive(row).get("memory_triggered")) for row in rows), len(rows)),
        "avg_historical_frames": _mean([float(value) for value in memory_frames]),
        "historical_frame_counts": {str(index): dist.get(index, 0) for index in range(4)},
        "historical_frame_percent": {str(index): _percent(dist.get(index, 0), len(rows)) for index in range(4)},
        "trigger_counts": dict(_trigger_counts(rows)),
        "mean_latency_seconds": _mean(_numbers(rows, "end_to_end_time_seconds", "end_to_end_time")),
        "mean_ttft_seconds": _mean(_numbers(rows, "ttft_seconds")),
        "peak_gpu_allocated_mb": max(_numbers(rows, "gpu_peak_allocated_mb", "peak_allocated_gpu_mb"), default=None),
        "peak_gpu_reserved_mb": max(_numbers(rows, "gpu_peak_reserved_mb", "peak_reserved_gpu_mb"), default=None),
        "baseline_correct_extra_retrievals": new_triggers_on_baseline_correct,
        "baseline_correct_damaged_after_extra_retrieval": damaged_after_new_trigger,
    }


def _split_accuracy(rows: list[dict[str, Any]], tasks: set[str]) -> dict[str, Any]:
    subset = [row for row in rows if row.get("task") in tasks]
    correct = sum(bool(_correct(row)) for row in subset)
    return {"samples": len(subset), "accuracy": correct / len(subset) if subset else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True, help="Current PRISM result dir on the same subset.")
    parser.add_argument("--run", action="append", default=[], help="NAME=RESULT_DIR. Repeat for sweep thresholds.")
    parser.add_argument("--oracle-run", type=Path, help="Optional oracle_rows.jsonl/CSV source for known false-stop tracking.")
    parser.add_argument("--subset", type=Path, help="Optional fixed OVO subset annotation JSON for oracle overlap diagnostics.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    baseline_rows = _load_result_rows(args.baseline)
    baseline_by_key = {_metric_key(row): row for row in baseline_rows}
    run_rows: dict[str, list[dict[str, Any]]] = {}
    for item in args.run:
        if "=" not in item:
            raise ValueError("--run must use NAME=RESULT_DIR")
        name, value = item.split("=", 1)
        run_rows[name] = _load_result_rows(Path(value))
    summary = {
        "baseline": _summarize_run("baseline", baseline_rows, baseline_by_key),
        "runs": [],
        "oracle_false_stops": {},
        "selection": {},
    }
    for name, rows in run_rows.items():
        summary["runs"].append(_summarize_run(name, rows, baseline_by_key))

    oracle_rows = [row for row in _load_oracle_rows(args.oracle_run) if _is_known_oracle_false_stop(row)]
    if oracle_rows:
        variants = [
            "explicit_key",
            "sample_key",
            "id_qidx_task_question",
            "id_task_question",
            "video_task_question",
            "task_question",
            "question",
            "metric_key",
        ]
        baseline_variant, baseline_overlap = _best_matching_variant(oracle_rows, baseline_rows, variants)
        subset_rows = _flatten(_load_json_rows(args.subset)) if args.subset else []
        subset_variant, subset_overlap = _best_matching_variant(oracle_rows, subset_rows, variants) if subset_rows else (None, {})
        oracle_in_subset = _matched_rows(oracle_rows, subset_rows, subset_variant) if subset_rows else []
        matched_baseline_oracle_rows = []
        if baseline_variant:
            baseline_keys = set(_index_by_variant(baseline_rows, baseline_variant))
            seen_keys: set[str] = set()
            for row in oracle_rows:
                key = _key_variants(row).get(baseline_variant)
                if key and key in baseline_keys and key not in seen_keys:
                    matched_baseline_oracle_rows.append(row)
                    seen_keys.add(key)
        summary["oracle_false_stops"] = {
            "known_count": len({_sample_key(row) for row in oracle_rows}),
            "oracle_rows_after_filter": len(oracle_rows),
            "result_match_key_variant": baseline_variant,
            "subset_match_key_variant": subset_variant,
            "match_policy": (
                "Prefer key formats with nonzero overlap and no duplicate rows on either side. "
                "Loose question-only overlaps are reported for diagnostics but not used when an unambiguous key overlaps."
            ),
            "key_overlap_diagnostics_vs_baseline": baseline_overlap,
            "key_overlap_diagnostics_vs_subset": subset_overlap,
            "max_loose_overlap_vs_baseline": _max_common_keys(baseline_overlap),
            "max_loose_overlap_vs_subset": _max_common_keys(subset_overlap),
            "duplicate_diagnostics": {
                "oracle": _duplicate_summary(oracle_rows, variants),
                "baseline": _duplicate_summary(baseline_rows, variants),
                "subset": _duplicate_summary(subset_rows, variants) if subset_rows else {},
            },
            "present_in_baseline_result": len(matched_baseline_oracle_rows),
            "present_in_subset": len(oracle_in_subset),
        }
        by_run: dict[str, Any] = {}
        for name, rows in run_rows.items():
            variant, overlap = _best_matching_variant(oracle_rows, rows, variants)
            matched = _matched_rows(oracle_rows, rows, variant)
            by_run[name] = {
                "match_key_variant": variant,
                "key_overlap_diagnostics": overlap,
                "matched": len(matched),
                "activated": sum(_memory_activated(row) for row in matched),
                "correct": sum(_correct(row) is True for row in matched),
                "remain_wrong": sum(_correct(row) is False for row in matched),
            }
        summary["oracle_false_stops"]["by_run"] = by_run

    ranked = sorted(
        summary["runs"],
        key=lambda item: (
            item.get("net_rescue") if isinstance(item.get("net_rescue"), int) else -10**9,
            (item.get("backward_accuracy") or {}).get("accuracy") or -1.0,
            -(item.get("damaged") if isinstance(item.get("damaged"), int) else 10**9),
            -(item.get("avg_historical_frames") or 10**9),
        ),
        reverse=True,
    )
    if ranked:
        summary["selection"] = {
            "best_threshold": ranked[0]["name"],
            "criteria": [
                "net rescue",
                "backward gain",
                "limited damage",
                "low extra-frame cost",
            ],
            "ranking": [item["name"] for item in ranked],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
