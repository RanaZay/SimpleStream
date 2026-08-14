#!/usr/bin/env python3
"""Offline replay for PRISM conservative-gate candidates.

The replay is read-only. It uses a source PRISM run that already contains
candidate queues and sufficiency traces. If the conservative gate stops, the
sample is replayed as Recent-6 baseline. If it retrieves and the source run
also retrieved, the source PRISM outcome is reused. If the gate retrieves but
the source run did not, the outcome is marked unknown and excluded from scored
accuracy because no inference exists for that counterfactual.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


OVO_GROUPS = {
    "EPM": "backward",
    "HLD": "backward",
    "ASI": "backward",
    "OCR": "realtime",
    "OJR": "realtime",
    "ACR": "realtime",
    "STU": "realtime",
    "ATR": "realtime",
    "FPD": "realtime",
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
            rows.extend(value.get(group, []))
        return rows
    return []


def _flatten(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row.get("test_info"), list):
            for index, item in enumerate(row["test_info"]):
                if isinstance(item, dict):
                    flat.append({**row, **item, "_question_index": index})
        else:
            flat.append(row)
    return flat


def _load(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        return _flatten(_read_json_rows(path))
    merged = path / "merged_results.json"
    if merged.exists():
        return _flatten(_read_json_rows(merged))
    rows: list[dict[str, Any]] = []
    for item in sorted(path.glob("rank_*/results_incremental.jsonl")):
        rows.extend(_read_json_rows(item))
    if rows:
        return _flatten(rows)
    ovo = sorted(path.glob("minicpmv46_results_*.json"))
    if ovo:
        return _flatten(_read_json_rows(ovo[-1]))
    raise FileNotFoundError(f"No result rows found under {path}")


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _key(row: dict[str, Any]) -> str:
    return "|".join(_norm_text(row.get(name)) for name in ("id", "task", "question"))


def _letter(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"A", "B", "C", "D", "E"}:
        return text
    match = re.search(r"\b([A-E])\b", text)
    return match.group(1) if match else ""


def _correct(row: dict[str, Any]) -> bool | None:
    value = row.get("correct")
    if isinstance(value, bool):
        return value
    gt = _letter(row.get("ground_truth") or row.get("answer_gt") or row.get("answer"))
    pred = _letter(row.get("prediction") or row.get("response"))
    if gt and pred:
        return gt == pred
    return None


def _prediction(row: dict[str, Any]) -> str:
    return _letter(row.get("prediction") or row.get("response"))


def _adaptive(row: dict[str, Any]) -> dict[str, Any]:
    profile = row.get("profile")
    if isinstance(profile, dict) and isinstance(profile.get("adaptive"), dict):
        return profile["adaptive"]
    adaptive = row.get("adaptive")
    return adaptive if isinstance(adaptive, dict) else {}


def _first_iteration(adaptive: dict[str, Any]) -> dict[str, Any]:
    iterations = adaptive.get("iterations")
    if isinstance(iterations, list) and iterations and isinstance(iterations[0], dict):
        return iterations[0]
    return {}


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def _best_candidate(adaptive: dict[str, Any], recent_ids: list[int]) -> tuple[float | None, float | None, int | None]:
    queue = [item for item in adaptive.get("candidate_queue", []) or [] if isinstance(item, dict)]
    if not queue:
        return None, None, None
    candidate = queue[0]
    chunk_id = int(candidate.get("chunk_id"))
    score = _num(candidate.get("total_score"))
    distance = min(abs(int(value) - chunk_id) for value in recent_ids) if recent_ids else None
    return score, float(distance) if distance is not None else None, chunk_id


def _gate(
    *,
    sufficiency: float | None,
    candidate_score: float | None,
    temporal_distance: float | None,
    tau_low: float,
    tau_high: float,
    candidate_threshold: float,
    temporal_threshold: float,
) -> tuple[bool, str]:
    if sufficiency is None:
        return False, "missing_sufficiency"
    if sufficiency < tau_low:
        return True, "low_sufficiency"
    if sufficiency >= tau_high:
        return False, "high_sufficiency"
    if candidate_score is not None and temporal_distance is not None:
        if candidate_score > candidate_threshold and temporal_distance > temporal_threshold:
            return True, "ambiguous_strong_temporal_candidate"
        if candidate_score <= candidate_threshold and temporal_distance <= temporal_threshold:
            return False, "ambiguous_weak_near_candidate"
        if candidate_score <= candidate_threshold:
            return False, "ambiguous_weak_candidate"
        return False, "ambiguous_near_candidate"
    return False, "ambiguous_missing_candidate_signal"


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return mean(clean) if clean else None


def _auc(rows: list[dict[str, Any]], signal: str) -> dict[str, Any]:
    pairs = [
        (float(row[signal]), int(row["positive"]))
        for row in rows
        if isinstance(row.get(signal), (int, float))
    ]
    positives = [value for value, label in pairs if label == 1]
    negatives = [value for value, label in pairs if label == 0]
    if not positives or not negatives:
        return {"available": len(pairs), "auc_positive_high": None, "auc_best_direction": None}
    wins = ties = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                ties += 1.0
    auc = (wins + 0.5 * ties) / (len(positives) * len(negatives))
    return {
        "available": len(pairs),
        "auc_positive_high": auc,
        "auc_best_direction": max(auc, 1.0 - auc),
        "higher_values_predict": "positive" if auc >= 0.5 else "negative",
    }


def _ovo_macro(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        correct = row.get("replay_correct")
        if isinstance(correct, bool):
            by_task[str(row.get("task") or "")].append(correct)
    task_acc = {
        task: sum(values) / len(values)
        for task, values in by_task.items()
        if values
    }
    backward = [task_acc[task] for task, group in OVO_GROUPS.items() if group == "backward" and task in task_acc]
    realtime = [task_acc[task] for task, group in OVO_GROUPS.items() if group == "realtime" and task in task_acc]
    total_values = [*backward, *realtime]
    return {
        "task_accuracy": task_acc,
        "backward_avg": mean(backward) if backward else None,
        "realtime_avg": mean(realtime) if realtime else None,
        "total_avg": mean(total_values) if total_values else None,
    }


def _summarize_replay(rows: list[dict[str, Any]], benchmark: str) -> dict[str, Any]:
    scored = [row for row in rows if isinstance(row.get("replay_correct"), bool)]
    base_scored = [row for row in rows if isinstance(row.get("baseline_correct"), bool)]
    rescued = sum(row["baseline_correct"] is False and row.get("replay_correct") is True for row in rows)
    damaged = sum(row["baseline_correct"] is True and row.get("replay_correct") is False for row in rows)
    distribution = Counter(int(row.get("replay_memory_frames") or 0) for row in rows if not row.get("unknown_outcome"))
    summary: dict[str, Any] = {
        "samples": len(rows),
        "scored_samples": len(scored),
        "unknown_outcomes": sum(bool(row.get("unknown_outcome")) for row in rows),
        "accuracy": sum(bool(row["replay_correct"]) for row in scored) / len(scored) if scored else None,
        "baseline_accuracy": (
            sum(bool(row["baseline_correct"]) for row in base_scored) / len(base_scored) if base_scored else None
        ),
        "rescued": rescued,
        "damaged": damaged,
        "net_rescue": rescued - damaged,
        "memory_trigger_rate": sum(bool(row.get("replay_memory_frames")) for row in rows) / len(rows) if rows else None,
        "avg_historical_frames": _mean(row.get("replay_memory_frames") for row in rows),
        "historical_frame_distribution": {str(index): distribution.get(index, 0) for index in range(4)},
        "gate_reasons": dict(Counter(str(row.get("gate_reason")) for row in rows)),
        "mean_latency_seconds": _mean(row.get("replay_latency_seconds") for row in rows),
        "mean_ttft_seconds": _mean(row.get("replay_ttft_seconds") for row in rows),
    }
    if benchmark == "ovo":
        summary["ovo_macro"] = _ovo_macro(rows)
        summary["false_stops"] = sum(
            row["baseline_correct"] is False and not bool(row.get("replay_memory_frames")) for row in rows
        )
    else:
        by_category: dict[str, list[bool]] = defaultdict(list)
        for row in rows:
            if isinstance(row.get("replay_correct"), bool):
                by_category[str(row.get("task") or "unknown")].append(bool(row["replay_correct"]))
        summary["category_accuracy"] = {
            key: {"accuracy": sum(values) / len(values), "correct": sum(values), "total": len(values)}
            for key, values in sorted(by_category.items())
        }
    return summary


def _source_signal_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signal_rows: list[dict[str, Any]] = []
    for row in rows:
        base_correct = row.get("baseline_correct")
        source_correct = row.get("source_correct")
        if not isinstance(base_correct, bool) or not isinstance(source_correct, bool):
            continue
        memory_frames = int(row.get("source_memory_frames") or 0)
        if memory_frames <= 0:
            group = "no_memory"
            positive = None
        elif not base_correct and source_correct:
            group = "rescued_memory"
            positive = 1
        elif base_correct and not source_correct:
            group = "damaged_memory"
            positive = 0
        else:
            group = "neutral_memory"
            positive = 0
        item = {**row, "source_group": group, "positive": positive}
        signal_rows.append(item)
    return signal_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["ovo", "streamingbench"], required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--source-prism", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tau-low", type=float, nargs="+", default=[0.60, 0.62, 0.64])
    parser.add_argument("--tau-high", type=float, nargs="+", default=[0.70, 0.72, 0.74])
    parser.add_argument("--candidate-threshold", type=float, nargs="+", default=[0.52, 0.54, 0.56])
    parser.add_argument("--temporal-distance-threshold", type=float, nargs="+", default=[5.0, 10.0, 20.0])
    args = parser.parse_args()

    baseline = {_key(row): row for row in _load(args.baseline)}
    source = {_key(row): row for row in _load(args.source_prism)}
    common_keys = sorted(set(baseline) & set(source))
    base_source_rows: list[dict[str, Any]] = []
    for key in common_keys:
        base = baseline[key]
        src = source[key]
        adaptive = _adaptive(src)
        iteration0 = _first_iteration(adaptive)
        recent_ids = [int(value) for value in adaptive.get("recent_chunk_ids", []) or []]
        candidate_score, temporal_distance, candidate_id = _best_candidate(adaptive, recent_ids)
        first_candidate = (adaptive.get("candidate_queue") or [{}])[0] if adaptive.get("candidate_queue") else {}
        candidate_semantic = _num(first_candidate.get("semantic_score"))
        current_support = _num(iteration0.get("visual_support_norm"))
        base_source_rows.append(
            {
                "key": key,
                "task": str(src.get("task") or src.get("task_type") or src.get("category") or ""),
                "baseline_correct": _correct(base),
                "source_correct": _correct(src),
                "baseline_prediction": _prediction(base),
                "source_prediction": _prediction(src),
                "source_memory_frames": int(adaptive.get("num_memory_frames") or 0),
                "source_latency_seconds": _num(adaptive.get("end_to_end_time_seconds") or src.get("end_to_end_time")),
                "source_ttft_seconds": _num(adaptive.get("ttft_seconds") or src.get("ttft_seconds")),
                "baseline_latency_seconds": _num(base.get("end_to_end_time") or base.get("end_to_end_time_seconds")),
                "baseline_ttft_seconds": _num(base.get("ttft_seconds")),
                "current_sufficiency": _num(iteration0.get("sufficiency")),
                "candidate_total_score": candidate_score,
                "candidate_semantic_score": candidate_semantic,
                "candidate_semantic_minus_current_support": (
                    candidate_semantic - current_support
                    if candidate_semantic is not None and current_support is not None
                    else None
                ),
                "temporal_distance_chunks": temporal_distance,
                "best_candidate_chunk_id": candidate_id,
            }
        )

    runs: list[dict[str, Any]] = []
    replay_tables: dict[str, list[dict[str, Any]]] = {}
    for tau_low in args.tau_low:
        for tau_high in args.tau_high:
            if tau_high <= tau_low:
                continue
            for candidate_threshold in args.candidate_threshold:
                for temporal_threshold in args.temporal_distance_threshold:
                    name = (
                        f"low{tau_low:.2f}_high{tau_high:.2f}_"
                        f"cand{candidate_threshold:.2f}_td{temporal_threshold:g}"
                    )
                    replay_rows: list[dict[str, Any]] = []
                    for row in base_source_rows:
                        retrieve, reason = _gate(
                            sufficiency=row["current_sufficiency"],
                            candidate_score=row["candidate_total_score"],
                            temporal_distance=row["temporal_distance_chunks"],
                            tau_low=tau_low,
                            tau_high=tau_high,
                            candidate_threshold=candidate_threshold,
                            temporal_threshold=temporal_threshold,
                        )
                        source_retrieved = int(row.get("source_memory_frames") or 0) > 0
                        unknown = bool(retrieve and not source_retrieved)
                        if unknown:
                            replay_correct = None
                            replay_prediction = ""
                            replay_memory_frames = 1
                            latency = None
                            ttft = None
                        elif retrieve:
                            replay_correct = row["source_correct"]
                            replay_prediction = row["source_prediction"]
                            replay_memory_frames = row["source_memory_frames"]
                            latency = row["source_latency_seconds"]
                            ttft = row["source_ttft_seconds"]
                        else:
                            replay_correct = row["baseline_correct"]
                            replay_prediction = row["baseline_prediction"]
                            replay_memory_frames = 0
                            latency = row["baseline_latency_seconds"]
                            ttft = row["baseline_ttft_seconds"]
                        replay_rows.append(
                            {
                                **row,
                                "config": name,
                                "gate_retrieve": retrieve,
                                "gate_reason": reason,
                                "unknown_outcome": unknown,
                                "replay_correct": replay_correct,
                                "replay_prediction": replay_prediction,
                                "replay_memory_frames": replay_memory_frames,
                                "replay_latency_seconds": latency,
                                "replay_ttft_seconds": ttft,
                            }
                        )
                    summary = _summarize_replay(replay_rows, args.benchmark)
                    summary.update(
                        {
                            "name": name,
                            "tau_low": tau_low,
                            "tau_high": tau_high,
                            "candidate_threshold": candidate_threshold,
                            "temporal_distance_threshold": temporal_threshold,
                        }
                    )
                    runs.append(summary)
                    replay_tables[name] = replay_rows

    signal_rows = _source_signal_rows(base_source_rows)
    auc_rows = [
        {**_auc([row for row in signal_rows if row.get("positive") is not None], signal), "signal": signal}
        for signal in (
            "candidate_total_score",
            "candidate_semantic_score",
            "temporal_distance_chunks",
            "candidate_semantic_minus_current_support",
            "current_sufficiency",
        )
    ]
    distributions: dict[str, dict[str, Any]] = {}
    for group in ("rescued_memory", "damaged_memory", "neutral_memory", "no_memory"):
        items = [row for row in signal_rows if row["source_group"] == group]
        distributions[group] = {
            "count": len(items),
            "candidate_total_score_mean": _mean(row.get("candidate_total_score") for row in items),
            "candidate_semantic_score_mean": _mean(row.get("candidate_semantic_score") for row in items),
            "temporal_distance_chunks_mean": _mean(row.get("temporal_distance_chunks") for row in items),
            "current_sufficiency_mean": _mean(row.get("current_sufficiency") for row in items),
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "benchmark": args.benchmark,
        "validation": {
            "baseline_rows": len(baseline),
            "source_rows": len(source),
            "common_keys": len(common_keys),
            "match_key": "id_task_question",
        },
        "runs": runs,
        "source_signal_distributions": distributions,
        "source_signal_auc": auc_rows,
        "note": "Unknown outcomes mean the gate would retrieve but the source PRISM run did not, so no existing inference output can score that counterfactual.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.out_dir / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "name",
            "samples",
            "scored_samples",
            "unknown_outcomes",
            "accuracy",
            "baseline_accuracy",
            "rescued",
            "damaged",
            "net_rescue",
            "memory_trigger_rate",
            "avg_historical_frames",
            "mean_latency_seconds",
            "mean_ttft_seconds",
        ]
        if args.benchmark == "ovo":
            fields.extend(["ovo_macro_total", "ovo_backward", "ovo_realtime", "false_stops"])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            row = {field: run.get(field) for field in fields}
            if args.benchmark == "ovo":
                macro = run.get("ovo_macro", {})
                row["ovo_macro_total"] = macro.get("total_avg")
                row["ovo_backward"] = macro.get("backward_avg")
                row["ovo_realtime"] = macro.get("realtime_avg")
            writer.writerow(row)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
