#!/usr/bin/env python3
"""Offline PRISM sufficiency calibration over saved diagnostics.

No inference is run. The script reads saved PRISM result JSON/JSONL files and
optional causal-control/oracle outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, median
from typing import Any


FORMULAS = {
    "S0_current": lambda m, e, v: 0.50 * m + 0.20 * e + 0.30 * v,
    "S1_50M_50V": lambda m, e, v: 0.50 * m + 0.50 * v,
    "S2_40M_60V": lambda m, e, v: 0.40 * m + 0.60 * v,
    "S3_30M_70V": lambda m, e, v: 0.30 * m + 0.70 * v,
    "S4_M_times_V": lambda m, e, v: m * v,
    "S5_sqrt_M_times_V": lambda m, e, v: math.sqrt(max(0.0, m * v)),
    "S6_min_M_V": lambda m, e, v: min(m, v),
    "S7_collapsed_conf_50V": lambda m, e, v: 0.50 * ((m + e) / 2.0) + 0.50 * v,
}

ACTIVATION_RATES = [0.10, 0.15, 0.20, 0.25]


def read_json_or_jsonl(path: Path) -> Any:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    return json.load(path.open(encoding="utf-8"))


def flatten_ovo(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row.get("adaptive"), dict):
            flat.append(row)
        for index, item in enumerate(row.get("test_info") or []):
            if isinstance(item, dict) and isinstance(item.get("adaptive"), dict):
                merged = dict(item)
                for key in ("video", "video_id", "id", "task", "category", "task_type"):
                    merged.setdefault(key, row.get(key))
                merged.setdefault("_index", row.get("_index", row.get("id")))
                merged.setdefault("_subindex", index)
                flat.append(merged)
    return flat


def normalize_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return flatten_ovo(payload)
    if isinstance(payload, dict):
        if isinstance(payload.get("results"), list):
            return flatten_ovo(payload["results"])
        if all(isinstance(payload.get(key), list) for key in ("backward", "realtime", "forward")):
            return flatten_ovo([*payload["backward"], *payload["realtime"], *payload["forward"]])
    return []


def load_results(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        return normalize_payload(read_json_or_jsonl(path))
    for pattern in (
        "streaming_bench_minicpmv46_results_*.json",
        "minicpmv46_results_*.json",
        "merged_results.json",
    ):
        matches = sorted(path.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
        if matches:
            return load_results(matches[0])
    rows: list[dict[str, Any]] = []
    for item in sorted(path.glob("rank_*/results_incremental.jsonl")):
        rows.extend(normalize_payload(read_json_or_jsonl(item)))
    if not rows:
        raise FileNotFoundError(f"No result rows found under {path}")
    dedup: dict[str, dict[str, Any]] = {}
    for row in rows:
        dedup[stable_key(row)] = row
    return list(dedup.values())


def stable_key(row: dict[str, Any]) -> str:
    if row.get("_index") is not None and row.get("_subindex") is not None:
        return f"{row.get('_index')}:{row.get('_subindex')}"
    return str(
        row.get("_key")
        or row.get("_index")
        or row.get("question_id")
        or row.get("id")
        or f"{row.get('video', row.get('video_id', ''))}:{row.get('task', row.get('task_type', ''))}:{row.get('question', '')}"
    )


def extract_option(text: Any) -> str | None:
    if text is None:
        return None
    match = re.search(r"\b([A-E])\b", str(text).upper())
    return match.group(1) if match else None


def gt_answer(row: dict[str, Any]) -> str | None:
    for key in ("answer_gt", "ground_truth", "gt", "label", "correct_answer"):
        pred = extract_option(row.get(key))
        if pred:
            return pred
    return None


def row_correct(row: dict[str, Any]) -> bool | None:
    if isinstance(row.get("correct"), bool):
        return bool(row["correct"])
    gt = gt_answer(row)
    pred = None
    for key in ("prediction", "pred", "answer", "response", "model_answer"):
        pred = extract_option(row.get(key))
        if pred:
            break
    if gt and pred:
        return gt == pred
    return None


def as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (denx * deny) if denx and deny else None


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    output = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            output[indexed[k][0]] = avg_rank
        i = j
    return output


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return pearson(ranks(xs), ranks(ys))


def roc_auc(labels: list[bool], scores: list[float]) -> float | None:
    pairs = [(bool(label), float(score)) for label, score in zip(labels, scores) if math.isfinite(float(score))]
    positives = [score for label, score in pairs if label]
    negatives = [score for label, score in pairs if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    ties = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                ties += 1.0
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def quantiles(values: list[float]) -> dict[str, float | None]:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None}

    def q(prob: float) -> float:
        if len(clean) == 1:
            return clean[0]
        pos = prob * (len(clean) - 1)
        low = int(math.floor(pos))
        high = int(math.ceil(pos))
        if low == high:
            return clean[low]
        return clean[low] * (high - pos) + clean[high] * (pos - low)

    return {
        "n": len(clean),
        "mean": mean(clean),
        "median": median(clean),
        "p25": q(0.25),
        "p75": q(0.75),
    }


def threshold_for_low_activation(scores: list[float], activation_rate: float) -> float | None:
    clean = sorted(v for v in scores if math.isfinite(v))
    if not clean:
        return None
    index = max(0, min(len(clean) - 1, math.ceil(float(activation_rate) * len(clean)) - 1))
    return clean[index]


def load_memory_helpful_labels(path: Path | None) -> dict[str, bool]:
    if path is None:
        return {}
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        labels: dict[str, bool] = {}
        for row in rows:
            key = stable_key(row)
            recent = str(row.get("recent_only_correct", "")).strip().lower()
            retrieved = str(row.get("retrieved_memory_correct", "")).strip().lower()
            if recent in {"true", "false"} and retrieved in {"true", "false"}:
                labels[key] = recent == "false" and retrieved == "true"
        return labels
    payload = read_json_or_jsonl(path)
    if isinstance(payload, dict):
        if isinstance(payload.get("rows"), list):
            rows = payload["rows"]
        elif isinstance(payload.get("results"), list):
            rows = payload["results"]
        else:
            rows = []
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    labels: dict[str, bool] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = stable_key(row)
        if isinstance(row.get("memory_helpful"), bool):
            labels[key] = bool(row["memory_helpful"])
            continue
        cond = row.get("conditions")
        if isinstance(cond, dict):
            recent = cond.get("recent_only_control") or {}
            retrieved = cond.get("retrieved_memory") or {}
            if isinstance(recent.get("correct"), bool) and isinstance(retrieved.get("correct"), bool):
                labels[key] = (not bool(recent["correct"])) and bool(retrieved["correct"])
                continue
        if isinstance(row.get("recent_only_correct"), bool) and isinstance(row.get("retrieved_memory_correct"), bool):
            labels[key] = (not bool(row["recent_only_correct"])) and bool(row["retrieved_memory_correct"])
    return labels


def extract_records(
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]] | None,
    memory_labels: dict[str, bool],
) -> list[dict[str, Any]]:
    baseline_by_key = {stable_key(row): row for row in baseline_rows or []}
    records: list[dict[str, Any]] = []
    for row in rows:
        adaptive = row.get("adaptive") if isinstance(row.get("adaptive"), dict) else {}
        iterations = adaptive.get("iterations") or []
        if not iterations or not isinstance(iterations[0], dict):
            continue
        iter0 = iterations[0]
        m = as_float(iter0.get("answer_margin"))
        e = as_float(iter0.get("entropy_confidence"))
        v = as_float(iter0.get("visual_support_norm"))
        if m is None or e is None or v is None:
            continue
        gt = gt_answer(row)
        iter0_pred = extract_option(iter0.get("predicted_option"))
        iter0_correct = (iter0_pred == gt) if iter0_pred and gt else None
        base = baseline_by_key.get(stable_key(row))
        baseline_correct = row_correct(base) if base is not None else None
        current_correct = baseline_correct if baseline_correct is not None else iter0_correct
        formulas = {name: fn(m, e, v) for name, fn in FORMULAS.items()}
        records.append(
            {
                "key": stable_key(row),
                "category": str(row.get("task_type") or row.get("category") or row.get("task") or "unknown"),
                "M": m,
                "E": e,
                "V": v,
                "iter0_prediction": iter0_pred,
                "ground_truth": gt,
                "iter0_correct": iter0_correct,
                "baseline_current_correct": baseline_correct,
                "current_correct": current_correct,
                "memory_helpful": memory_labels.get(stable_key(row)),
                "confidently_wrong": bool(iter0_correct is False and m >= 0.90),
                **formulas,
            }
        )
    return records


def summarize_formula(records: list[dict[str, Any]], formula: str) -> dict[str, Any]:
    target_records = [row for row in records if isinstance(row.get("current_correct"), bool)]
    labels = [bool(row["current_correct"]) for row in target_records]
    scores = [float(row[formula]) for row in target_records]
    correct_scores = [float(row[formula]) for row in target_records if bool(row["current_correct"])]
    wrong_scores = [float(row[formula]) for row in target_records if not bool(row["current_correct"])]

    helpful_records = [row for row in records if isinstance(row.get("memory_helpful"), bool)]
    helpful_labels = [bool(row["memory_helpful"]) for row in helpful_records]
    helpful_scores_low_positive = [-float(row[formula]) for row in helpful_records]

    confidently_wrong = [row for row in records if row.get("confidently_wrong")]
    current_correct = [row for row in target_records if bool(row["current_correct"])]
    tradeoffs: dict[str, Any] = {}
    for rate in ACTIVATION_RATES:
        threshold = threshold_for_low_activation(scores, rate)
        if threshold is None:
            tradeoffs[f"{int(rate * 100)}pct_activation"] = {
                "threshold": None,
                "confident_wrong_false_stop_rate": None,
                "correct_unnecessary_retrieval_rate": None,
            }
            continue
        false_stop = [
            float(row[formula]) > threshold
            for row in confidently_wrong
        ]
        unnecessary = [
            float(row[formula]) <= threshold
            for row in current_correct
        ]
        tradeoffs[f"{int(rate * 100)}pct_activation"] = {
            "threshold": threshold,
            "confident_wrong_false_stop_rate": (
                sum(false_stop) / len(false_stop) if false_stop else None
            ),
            "correct_unnecessary_retrieval_rate": (
                sum(unnecessary) / len(unnecessary) if unnecessary else None
            ),
        }

    return {
        "correctness_auc_high_suff_predicts_correct": roc_auc(labels, scores) if labels else None,
        "memory_helpful_auc_low_suff_predicts_helpful": (
            roc_auc(helpful_labels, helpful_scores_low_positive) if helpful_labels else None
        ),
        "current_correct_distribution": quantiles(correct_scores),
        "current_wrong_distribution": quantiles(wrong_scores),
        "confidently_wrong_count": len(confidently_wrong),
        "tradeoffs": tradeoffs,
    }


def summarize_benchmark(
    name: str,
    result_path: Path | None,
    baseline_path: Path | None,
    memory_labels_path: Path | None,
) -> dict[str, Any] | None:
    if result_path is None:
        return None
    rows = load_results(result_path)
    baseline_rows = load_results(baseline_path) if baseline_path else None
    labels = load_memory_helpful_labels(memory_labels_path)
    records = extract_records(rows, baseline_rows, labels)
    m_values = [row["M"] for row in records]
    e_values = [row["E"] for row in records]
    v_values = [row["V"] for row in records]
    formulas = {name: summarize_formula(records, name) for name in FORMULAS}
    return {
        "benchmark": name,
        "result_path": str(result_path),
        "baseline_path": str(baseline_path) if baseline_path else None,
        "memory_labels_path": str(memory_labels_path) if memory_labels_path else None,
        "records": len(records),
        "current_correct_source": "baseline_correct_if_available_else_iteration0_correct",
        "memory_helpful_labels": sum(1 for row in records if isinstance(row.get("memory_helpful"), bool)),
        "confidently_wrong_count": sum(1 for row in records if row.get("confidently_wrong")),
        "M_E_correlation": {
            "pearson": pearson(m_values, e_values),
            "spearman": spearman(m_values, e_values),
        },
        "M_V_correlation": {
            "pearson": pearson(m_values, v_values),
            "spearman": spearman(m_values, v_values),
        },
        "E_V_correlation": {
            "pearson": pearson(e_values, v_values),
            "spearman": spearman(e_values, v_values),
        },
        "formulas": formulas,
        "records_table": records,
    }


def combined_table(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_bench = {report["benchmark"]: report for report in reports if report}
    rows: list[dict[str, Any]] = []
    for formula in FORMULAS:
        sb = by_bench.get("StreamingBench", {}).get("formulas", {}).get(formula, {})
        ovo = by_bench.get("OVO", {}).get("formulas", {}).get(formula, {})
        sb_trade = (sb.get("tradeoffs") or {}).get("15pct_activation", {})
        ovo_trade = (ovo.get("tradeoffs") or {}).get("15pct_activation", {})
        rows.append(
            {
                "formula": formula,
                "SB_correctness_AUC": sb.get("correctness_auc_high_suff_predicts_correct"),
                "SB_memory_helpful_AUC": sb.get("memory_helpful_auc_low_suff_predicts_helpful"),
                "OVO_correctness_AUC": ovo.get("correctness_auc_high_suff_predicts_correct"),
                "OVO_memory_helpful_AUC": ovo.get("memory_helpful_auc_low_suff_predicts_helpful"),
                "SB_confident_wrong_false_stop_at_15pct": sb_trade.get("confident_wrong_false_stop_rate"),
                "SB_correct_unnecessary_retrieval_at_15pct": sb_trade.get("correct_unnecessary_retrieval_rate"),
                "OVO_confident_wrong_false_stop_at_15pct": ovo_trade.get("confident_wrong_false_stop_rate"),
                "OVO_correct_unnecessary_retrieval_at_15pct": ovo_trade.get("correct_unnecessary_retrieval_rate"),
            }
        )
    return rows


def print_report(reports: list[dict[str, Any]], table: list[dict[str, Any]]) -> None:
    for report in reports:
        print(f"\n{'=' * 80}\n{report['benchmark']}\n{'=' * 80}")
        print(f"records: {report['records']}")
        print(f"current_correct_source: {report['current_correct_source']}")
        print(f"memory_helpful_labels: {report['memory_helpful_labels']}")
        print(f"confidently_wrong_count: {report['confidently_wrong_count']}")
        print(f"M/E Pearson: {report['M_E_correlation']['pearson']}")
        print(f"M/E Spearman: {report['M_E_correlation']['spearman']}")
        print("\nformula correctness/memory AUC:")
        for formula, payload in report["formulas"].items():
            print(
                f"  {formula}: "
                f"correct_AUC={payload['correctness_auc_high_suff_predicts_correct']} "
                f"memory_helpful_AUC_low={payload['memory_helpful_auc_low_suff_predicts_helpful']}"
            )
        print("\n15% activation tradeoff:")
        for formula, payload in report["formulas"].items():
            trade = payload["tradeoffs"]["15pct_activation"]
            print(
                f"  {formula}: threshold={trade['threshold']} "
                f"conf_wrong_false_stop={trade['confident_wrong_false_stop_rate']} "
                f"correct_unnecessary_retrieval={trade['correct_unnecessary_retrieval_rate']}"
            )

    print(f"\n{'=' * 80}\nCOMBINED TABLE\n{'=' * 80}")
    headers = list(table[0]) if table else []
    print("\t".join(headers))
    for row in table:
        print("\t".join("" if row[h] is None else str(row[h]) for h in headers))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streamingbench", type=Path)
    parser.add_argument("--streamingbench-baseline", type=Path)
    parser.add_argument("--streamingbench-memory-labels", type=Path, help="Optional controlled causal/oracle JSON/JSONL/CSV-derived JSON labels.")
    parser.add_argument("--ovo", type=Path)
    parser.add_argument("--ovo-baseline", type=Path)
    parser.add_argument("--ovo-memory-labels", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-csv", type=Path)
    args = parser.parse_args()

    reports = [
        report
        for report in (
            summarize_benchmark(
                "StreamingBench",
                args.streamingbench,
                args.streamingbench_baseline,
                args.streamingbench_memory_labels,
            ),
            summarize_benchmark("OVO", args.ovo, args.ovo_baseline, args.ovo_memory_labels),
        )
        if report is not None
    ]
    if not reports:
        raise SystemExit("Provide --streamingbench and/or --ovo")
    table = combined_table(reports)
    print_report(reports, table)

    output = {
        "formula_definitions": list(FORMULAS),
        "activation_rates": ACTIVATION_RATES,
        "reports": [
            {key: value for key, value in report.items() if key != "records_table"}
            for report in reports
        ],
        "combined_table": table,
    }
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nSaved JSON: {args.out_json}")
    if args.out_csv:
        write_csv(args.out_csv, table)
        print(f"Saved CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
