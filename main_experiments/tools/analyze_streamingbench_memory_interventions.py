#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


TAGS = {
    "right now": r"\bright now\b|\bcurrently\b|\bright currently\b|\bnow\b",
    "just now": r"\bjust now\b|\bjust\b|\bjust happened\b",
    "so far / in total": r"\bso far\b|\bin total\b|\btotal\b|\boverall\b|\bup to now\b",
    "next / likely / might": r"\bnext\b|\blikely\b|\bmight\b|\bwill\b|\babout to\b|\bgoing to\b",
    "before": r"\bbefore\b|\bprevious\b|\bpreviously\b|\bearlier\b",
    "after": r"\bafter\b|\bfollowing\b|\bthen\b",
    "beginning": r"\bbeginning\b|\bat the start\b|\binitially\b|\bfirst\b",
    "why": r"\bwhy\b",
    "where": r"\bwhere\b",
    "how many": r"\bhow many\b|\bcount\b|\bnumber of\b",
}


INTENT_ORDER = [
    "PRESENT_STATE",
    "RECENT_EVENT",
    "PROSPECTIVE",
    "CUMULATIVE_HISTORY",
    "HISTORICAL",
    "CAUSAL",
    "OTHER",
]


def extract_answer(text: Any) -> str | None:
    if text is None:
        return None
    match = re.search(r"\b([A-E])\b", str(text).upper())
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
        rows = payload.get("results", payload if isinstance(payload, list) else [])
        return str(path), rows

    merged = sorted(
        path.glob("streaming_bench_minicpmv46_results_*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if merged:
        return load_results(merged[0])

    rank_files = sorted(path.glob("rank_*/results_incremental.jsonl"))
    if not rank_files:
        raise FileNotFoundError(f"No saved StreamingBench results found under {path}")
    rows: list[dict[str, Any]] = []
    for rank_file in rank_files:
        rows.extend(read_jsonl(rank_file))
    dedup: dict[Any, dict[str, Any]] = {}
    for row in rows:
        dedup[row.get("_index", row.get("_key"))] = row
    return "\n  ".join(str(item) for item in rank_files), sorted(
        dedup.values(), key=lambda item: int(item.get("_index", 0))
    )


def timestamp_to_seconds(ts: Any) -> float:
    parts = [float(part) for part in re.findall(r"\d+(?:\.\d+)?", str(ts))]
    if len(parts) >= 3:
        return parts[-3] * 3600.0 + parts[-2] * 60.0 + parts[-1]
    if len(parts) == 2:
        return parts[0] * 60.0 + parts[1]
    return parts[0] if parts else 0.0


def load_annotations(path: Path) -> dict[int, tuple[dict[str, Any], dict[str, Any]]]:
    data = json.load(path.open(encoding="utf-8"))
    flat: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for entry in data:
        questions = sorted(entry.get("questions", []), key=lambda item: timestamp_to_seconds(item.get("time_stamp")))
        for question in questions:
            flat.append((entry, question))
    return {index: item for index, item in enumerate(flat)}


def format_options(question: dict[str, Any]) -> list[str]:
    raw = question.get("options") or question.get("choices") or []
    if isinstance(raw, dict):
        return [f"{letter}. {raw[letter]}" for letter in "ABCDE" if letter in raw]
    if isinstance(raw, list):
        output = []
        for index, value in enumerate(raw):
            letter = chr(65 + index)
            text = str(value)
            output.append(text if text.strip().startswith(f"{letter}.") else f"{letter}. {text}")
        return output
    return []


def semantic_intent(question: str, category: str) -> tuple[str, str]:
    text = question.lower().strip()
    causal = bool(re.search(r"\bwhy\b|\breason\b|\bcause\b|\bbecause\b", text))
    prospective = bool(re.search(r"\bnext\b|\bmight\b|\blikely\b|\bwill\b|\bgoing to\b|\babout to\b", text))
    cumulative = bool(re.search(r"\bso far\b|\bin total\b|\btotal\b|\boverall\b|\bup to now\b|\bthroughout\b", text))
    historical = bool(re.search(r"\bbefore\b|\bpreviously\b|\bprevious\b|\bearlier\b|\bat the beginning\b|\bin the beginning\b|\bfirst\b", text))
    recent = bool(re.search(r"\bjust\b|\bjust now\b|\brecently\b|\blast\b|\bwhat did .* do\b", text))
    present = bool(re.search(r"\bright now\b|\bnow\b|\bcurrently\b|\bvisible\b|\bwearing\b|\bholding\b|\bis\b|\bare\b|\bwhere\b", text))

    if prospective or category == "Prospective Reasoning":
        return "PROSPECTIVE", "asks about likely/next future continuation"
    if cumulative:
        return "CUMULATIVE_HISTORY", "asks for accumulated state/count over time"
    if causal or category == "Causal Reasoning":
        return "CAUSAL", "asks for reason/cause"
    if historical:
        return "HISTORICAL", "refers to earlier/previous/beginning context"
    if recent:
        return "RECENT_EVENT", "asks about a just-recent event"
    if present:
        return "PRESENT_STATE", "asks about current visible state"
    return "OTHER", "no clear temporal intent"


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def numeric_values(values: list[Any]) -> list[float]:
    out = []
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append(float(value))
    return out


def summarize_numbers(values: list[float]) -> dict[str, float | int | None]:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(clean),
        "mean": mean(clean),
        "median": median(clean),
        "min": min(clean),
        "max": max(clean),
    }


def candidate_by_id(adaptive: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for item in as_list(adaptive.get("candidate_queue")) + as_list(adaptive.get("memory_scores")):
        if isinstance(item, dict) and item.get("chunk_id") is not None:
            out[int(item["chunk_id"])] = item
    return out


def gate_records(adaptive: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for item in as_list(adaptive.get("iterations")):
        if isinstance(item, dict) and isinstance(item.get("conservative_gate"), dict):
            gate = dict(item["conservative_gate"])
            gate["iteration"] = item.get("iteration")
            records.append(gate)
    return records


def context_differs(base: dict[str, Any], ours: dict[str, Any]) -> bool | None:
    adaptive = ours.get("adaptive") or {}
    eq = adaptive.get("baseline_recent_equivalence") or {}
    if "final_equals_recent" in eq:
        return not bool(eq["final_equals_recent"])
    if adaptive.get("memory_chunk_ids"):
        return True
    if base.get("final_chunk_ids") is not None and ours.get("final_chunk_ids") is not None:
        return base.get("final_chunk_ids") != ours.get("final_chunk_ids")
    return None


def build_changed_rows(
    base_rows: list[dict[str, Any]],
    ours_rows: list[dict[str, Any]],
    annotations: dict[int, tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    base = {int(row["_index"]): row for row in base_rows if row.get("_index") is not None}
    ours = {int(row["_index"]): row for row in ours_rows if row.get("_index") is not None}
    rows: list[dict[str, Any]] = []

    for index in sorted(set(base) & set(ours)):
        b_row = base[index]
        o_row = ours[index]
        b_correct = bool(b_row.get("correct"))
        o_correct = bool(o_row.get("correct"))
        if b_correct == o_correct:
            continue

        label = "WC" if o_correct else "CW"
        adaptive = o_row.get("adaptive") or {}
        category = str(o_row.get("task_type") or b_row.get("task_type") or "")
        question = str(o_row.get("question") or b_row.get("question") or "")
        intent, intent_reason = semantic_intent(question, category)
        recent_ids = [int(value) for value in as_list(adaptive.get("recent_chunk_ids"))]
        memory_ids = [int(value) for value in as_list(adaptive.get("memory_chunk_ids"))]
        final_ids = [int(value) for value in as_list(o_row.get("final_chunk_ids"))]
        base_final_ids = [int(value) for value in as_list(b_row.get("final_chunk_ids"))]
        nearest_recent_distances = [
            min(abs(memory_id - recent_id) for recent_id in recent_ids) if recent_ids else None
            for memory_id in memory_ids
        ]
        query_distances = [
            max(recent_ids) - memory_id if recent_ids else None
            for memory_id in memory_ids
        ]
        candidates = candidate_by_id(adaptive)
        memory_candidate_scores = [candidates.get(memory_id, {}) for memory_id in memory_ids]
        iterations = as_list(adaptive.get("iterations"))
        gates = gate_records(adaptive)
        first_gate = gates[0] if gates else {}
        last_iteration = iterations[-1] if iterations and isinstance(iterations[-1], dict) else {}
        sufficiencies = [
            float(item["sufficiency"])
            for item in iterations
            if isinstance(item, dict) and isinstance(item.get("sufficiency"), (int, float))
        ]
        gains = [
            float(item["gain_vs_previous"])
            for item in iterations
            if isinstance(item, dict) and isinstance(item.get("gain_vs_previous"), (int, float))
        ]
        tags = [tag for tag, pattern in TAGS.items() if re.search(pattern, question.lower())]
        _video_entry, ann_question = annotations.get(index, ({}, {}))

        row = {
            "question_id": index,
            "key": o_row.get("_key") or b_row.get("_key"),
            "video_id": o_row.get("video") or b_row.get("video"),
            "timestamp": o_row.get("time_stamp") or b_row.get("time_stamp"),
            "category": category,
            "question": question,
            "options": format_options(ann_question),
            "label": label,
            "temporal_intent": intent,
            "temporal_intent_reason": intent_reason,
            "tags": tags,
            "ground_truth": o_row.get("answer_gt") or b_row.get("answer_gt"),
            "simplestream_prediction": extract_answer(b_row.get("response")),
            "ours_prediction": extract_answer(o_row.get("response")),
            "simplestream_raw": b_row.get("response"),
            "ours_raw": o_row.get("response"),
            "simplestream_final_chunk_ids": base_final_ids,
            "ours_recent_chunk_ids": recent_ids,
            "ours_memory_chunk_ids": memory_ids,
            "ours_final_chunk_ids": final_ids,
            "num_recent_frames": len(recent_ids),
            "num_memory_frames": int(adaptive.get("num_memory_frames") or len(memory_ids)),
            "memory_nearest_recent_distances": nearest_recent_distances,
            "memory_query_distances": query_distances,
            "memory_candidate_scores": memory_candidate_scores,
            "candidate_queue": as_list(adaptive.get("candidate_queue")),
            "iterations": iterations,
            "sufficiency_scores": sufficiencies,
            "sufficiency_gains": gains,
            "initial_sufficiency": sufficiencies[0] if sufficiencies else None,
            "final_sufficiency": adaptive.get("final_sufficiency"),
            "best_candidate_total_score": first_gate.get("best_unused_candidate_total_score"),
            "best_candidate_distance": first_gate.get("best_unused_candidate_temporal_distance_chunks"),
            "gate_reason_initial": first_gate.get("reason"),
            "gate_threshold_tau_low": first_gate.get("tau_low"),
            "gate_threshold_tau_high": first_gate.get("tau_high"),
            "gate_candidate_threshold": first_gate.get("candidate_threshold"),
            "gate_temporal_distance_threshold": first_gate.get("temporal_distance_threshold"),
            "stop_reason": adaptive.get("stop_reason"),
            "context_differs": context_differs(b_row, o_row),
            "baseline_recent_equivalence": adaptive.get("baseline_recent_equivalence"),
            "last_iteration": last_iteration,
            "adaptive_config": adaptive.get("config"),
            "full_adaptive": adaptive,
        }
        rows.append(row)
    return rows


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key == "full_adaptive":
            continue
        if isinstance(value, (list, dict)):
            out[key] = json.dumps(value, ensure_ascii=False)
        else:
            out[key] = value
    return out


def print_counter(title: str, rows: list[dict[str, Any]], key: str) -> None:
    print(f"\n{title}")
    print("value,WC,CW,net,total")
    values = sorted({row.get(key) for row in rows}, key=lambda value: str(value))
    for value in values:
        wc = sum(1 for row in rows if row.get(key) == value and row["label"] == "WC")
        cw = sum(1 for row in rows if row.get(key) == value and row["label"] == "CW")
        print(f"{value},{wc},{cw},{wc - cw},{wc + cw}")


def print_numeric_summary(title: str, rows: list[dict[str, Any]], getter) -> None:
    print(f"\n{title}")
    print("label,n,mean,median,min,max")
    for label in ("WC", "CW"):
        values: list[float] = []
        for row in rows:
            if row["label"] == label:
                value = getter(row)
                if isinstance(value, list):
                    values.extend(numeric_values(value))
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    values.append(float(value))
        stats = summarize_numbers(values)
        print(
            f"{label},{stats['n']},{stats['mean']},{stats['median']},"
            f"{stats['min']},{stats['max']}"
        )


def print_ranked_characteristics(rows: list[dict[str, Any]], label: str) -> None:
    other = "CW" if label == "WC" else "WC"
    print(f"\nRANKED CHARACTERISTICS FOR {label}")
    items: list[tuple[float, str, int, int]] = []
    fields = ["temporal_intent", "category", "stop_reason", "gate_reason_initial"]
    for field in fields:
        values = {row.get(field) for row in rows}
        for value in values:
            a = sum(1 for row in rows if row["label"] == label and row.get(field) == value)
            b = sum(1 for row in rows if row["label"] == other and row.get(field) == value)
            if a:
                score = a / max(1, b + 1)
                items.append((score, f"{field}={value}", a, b))
    for score, name, a, b in sorted(items, reverse=True)[:12]:
        print(f"{name}: {label}={a}, {other}={b}, separation_score={score:.2f}")


def print_prospective_and_harmful_notes(rows: list[dict[str, Any]]) -> None:
    print("\nPROSPECTIVE WC CASES")
    for row in rows:
        if row["label"] == "WC" and row["temporal_intent"] == "PROSPECTIVE":
            print("-" * 80)
            print(f"Q{row['question_id']} {row['video_id']} {row['timestamp']}")
            print(row["question"])
            print(f"GT={row['ground_truth']} base={row['simplestream_prediction']} ours={row['ours_prediction']}")
            print(f"memory={row['ours_memory_chunk_ids']} recent={row['ours_recent_chunk_ids']}")
            print(f"dist_to_recent={row['memory_nearest_recent_distances']} scores={row['memory_candidate_scores']}")
            print(f"stop={row['stop_reason']} initial_gate={row['gate_reason_initial']} final_suff={row['final_sufficiency']}")
            print("memory_visual_content: not saved in result JSON; only chunk IDs/scores are available")

    print("\nREPRESENTATIVE HARMFUL CW CASES")
    harmful = [
        row for row in rows
        if row["label"] == "CW"
        and (
            row["temporal_intent"] in {"PRESENT_STATE", "CUMULATIVE_HISTORY"}
            or "right now" in row["tags"]
            or "how many" in row["tags"]
        )
    ]
    for row in harmful[:20]:
        print("-" * 80)
        print(f"Q{row['question_id']} {row['category']} {row['video_id']} {row['timestamp']}")
        print(row["question"])
        print(f"intent={row['temporal_intent']} tags={row['tags']}")
        print(f"GT={row['ground_truth']} base={row['simplestream_prediction']} ours={row['ours_prediction']}")
        print(f"memory={row['ours_memory_chunk_ids']} recent={row['ours_recent_chunk_ids']}")
        print(f"dist_to_recent={row['memory_nearest_recent_distances']} scores={row['memory_candidate_scores']}")
        print(f"stop={row['stop_reason']} initial_gate={row['gate_reason_initial']} final_suff={row['final_sufficiency']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze StreamingBench WC/CW memory interventions.")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--ours", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    base_source, base_rows = load_results(args.baseline)
    ours_source, ours_rows = load_results(args.ours)
    annotations = load_annotations(args.annotations)
    rows = build_changed_rows(base_rows, ours_rows, annotations)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.out_dir / "streamingbench_memory_interventions_64.json"
    csv_path = args.out_dir / "streamingbench_memory_interventions_64.csv"
    summary_path = args.out_dir / "streamingbench_memory_interventions_summary.json"

    payload = {
        "baseline_source": base_source,
        "ours_source": ours_source,
        "annotations": str(args.annotations),
        "changed_count": len(rows),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_rows = [flatten_for_csv(row) for row in rows]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()) if csv_rows else [])
        writer.writeheader()
        writer.writerows(csv_rows)

    summary = {
        "changed_count": len(rows),
        "label_counts": Counter(row["label"] for row in rows),
        "intent_by_label": {
            label: Counter(row["temporal_intent"] for row in rows if row["label"] == label)
            for label in ("WC", "CW")
        },
        "category_by_label": {
            label: Counter(row["category"] for row in rows if row["label"] == label)
            for label in ("WC", "CW")
        },
        "stop_reason_by_label": {
            label: Counter(row["stop_reason"] for row in rows if row["label"] == label)
            for label in ("WC", "CW")
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Baseline source: {base_source}")
    print(f"Our source: {ours_source}")
    print(f"Changed samples: {len(rows)}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved summary: {summary_path}")

    print_counter("TEMPORAL INTENT DISTRIBUTION", rows, "temporal_intent")
    print_counter("CATEGORY DISTRIBUTION", rows, "category")
    print_counter("STOP REASON DISTRIBUTION", rows, "stop_reason")
    print_counter("INITIAL GATE REASON DISTRIBUTION", rows, "gate_reason_initial")
    print_numeric_summary("MEMORY TEMPORAL DISTANCE TO NEAREST RECENT CHUNK", rows, lambda row: row["memory_nearest_recent_distances"])
    print_numeric_summary("MEMORY TEMPORAL DISTANCE TO QUERY/LATEST RECENT CHUNK", rows, lambda row: row["memory_query_distances"])
    print_numeric_summary("MEMORY FRAME COUNT", rows, lambda row: row["num_memory_frames"])
    print_numeric_summary("INITIAL SUFFICIENCY", rows, lambda row: row["initial_sufficiency"])
    print_numeric_summary("FINAL SUFFICIENCY", rows, lambda row: row["final_sufficiency"])
    print_numeric_summary("BEST CANDIDATE TOTAL SCORE", rows, lambda row: row["best_candidate_total_score"])
    print_numeric_summary("BEST CANDIDATE DISTANCE", rows, lambda row: row["best_candidate_distance"])

    print("\nTAG DISTRIBUTION")
    print("tag,WC,CW,net")
    for tag in TAGS:
        wc = sum(1 for row in rows if row["label"] == "WC" and tag in row["tags"])
        cw = sum(1 for row in rows if row["label"] == "CW" and tag in row["tags"])
        print(f"{tag},{wc},{cw},{wc-cw}")

    print_ranked_characteristics(rows, "WC")
    print_ranked_characteristics(rows, "CW")
    print_prospective_and_harmful_notes(rows)


if __name__ == "__main__":
    main()
