#!/usr/bin/env python3
"""Option-distribution instability analysis for strict StreamingBench microclips.

This is an analysis-only forward-pass tool. It does not modify PRISM and does
not generate final answers. It rebuilds fixed K0/anchor/microclip contexts for
the same saved StreamingBench-100 samples, calls PRISM's option/sufficiency
scorer for each context, then analyzes whether K0->anchor instability predicts
strict microclip value using previously saved oracle generation outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

build_prompt = None
decode_video_to_chunks_qwen = None
resolve_video_path = None
select_recent_window_frames = None
timestamp_to_seconds = None
RecentWindowQAModel = None


def default_qa_device() -> str:
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


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
    dedup: dict[Any, dict[str, Any]] = {}
    for row in rows:
        dedup[row.get("_index", row.get("_key"))] = row
    return "\n  ".join(str(item) for item in rank_files), sorted(
        dedup.values(),
        key=lambda item: int(item.get("_index", 0)),
    )


def load_strict_oracle(path: Path) -> tuple[str, dict[int, dict[str, Any]]]:
    if path.is_dir():
        jsonl = path / "strict_microclip_oracle_results.jsonl"
        if jsonl.exists():
            rows = read_jsonl(jsonl)
            return str(jsonl), {int(row["question_id"]): row for row in rows}
        json_path = path / "strict_microclip_oracle_results.json"
        if json_path.exists():
            return load_strict_oracle(json_path)
    if path.is_file() and path.suffix == ".jsonl":
        rows = read_jsonl(path)
        return str(path), {int(row["question_id"]): row for row in rows}
    if path.is_file():
        payload = json.load(path.open(encoding="utf-8"))
        rows = payload.get("results", payload) if isinstance(payload, dict) else payload
        return str(path), {int(row["question_id"]): row for row in rows}
    raise FileNotFoundError(f"No strict oracle result file found under {path}")


def adaptive(row: dict[str, Any]) -> dict[str, Any]:
    profile = row.get("profile") if isinstance(row.get("profile"), dict) else {}
    return row.get("adaptive") or profile.get("adaptive") or {}


def load_annotations(path: Path, video_dir: str) -> dict[int, dict[str, Any]]:
    if build_prompt is None or resolve_video_path is None or timestamp_to_seconds is None:
        raise RuntimeError("StreamingBench helpers were not initialized")
    data = json.load(path.open(encoding="utf-8"))
    tasks: list[dict[str, Any]] = []
    for entry in data:
        video_path_raw = entry["video_path"]
        video_path = resolve_video_path(video_path_raw, video_dir)
        questions = sorted(entry.get("questions", []), key=lambda item: timestamp_to_seconds(item.get("time_stamp")))
        for question in questions:
            tasks.append(
                {
                    "_index": len(tasks),
                    "video": entry.get("video_id") or Path(video_path_raw).stem,
                    "video_path": video_path,
                    "video_path_raw": video_path_raw,
                    "question_obj": question,
                    "prompt": build_prompt(question),
                    "task_type": question.get("task_type") or entry.get("task_type"),
                }
            )
    return {int(task["_index"]): task for task in tasks}


def extract_mcq_answer(text: Any) -> str | None:
    if text is None:
        return None
    import re

    match = re.search(r"\b([A-E])\b", str(text).upper())
    return match.group(1) if match else None


def answer_gt_from(saved: dict[str, Any], question: dict[str, Any]) -> str | None:
    for value in (
        saved.get("answer_gt"),
        saved.get("ground_truth"),
        question.get("answer"),
        question.get("answer_gt"),
    ):
        answer = extract_mcq_answer(str(value)) if value is not None else None
        if answer:
            return answer
    return None


def number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def chunk_id(chunk: Any) -> int:
    return int(getattr(chunk, "chunk_index"))


def chunk_bounds(chunk: Any) -> tuple[float, float]:
    timestamps = getattr(chunk, "frame_timestamps", None) or []
    numeric = [float(ts) for ts in timestamps if isinstance(ts, (int, float)) and math.isfinite(float(ts))]
    if numeric:
        return min(numeric), max(numeric)
    start = number(getattr(chunk, "start_time", None))
    end = number(getattr(chunk, "end_time", None))
    if start is None:
        start = float(chunk_id(chunk))
    if end is None:
        end = start
    return (end, start) if end < start else (start, end)


def chunk_anchor_time(chunk: Any) -> float:
    timestamps = getattr(chunk, "frame_timestamps", None) or []
    numeric = [float(ts) for ts in timestamps if isinstance(ts, (int, float)) and math.isfinite(float(ts))]
    if numeric:
        return numeric[len(numeric) // 2]
    start, end = chunk_bounds(chunk)
    return 0.5 * (start + end)


def recent_start_time(chunks: list[Any]) -> float:
    if not chunks:
        raise ValueError("No recent chunks")
    return min(chunk_bounds(chunk)[0] for chunk in chunks)


def frame_records_from_chunks(chunks: list[Any], recent_start: float, epsilon: float) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for chunk in chunks:
        frames = list(getattr(chunk, "frames", []) or [])
        if not frames:
            continue
        timestamps = getattr(chunk, "frame_timestamps", None) or []
        numeric = [float(ts) for ts in timestamps if isinstance(ts, (int, float)) and math.isfinite(float(ts))]
        start, end = chunk_bounds(chunk)
        if len(numeric) != len(frames):
            if len(frames) == 1:
                numeric = [chunk_anchor_time(chunk)]
            elif end > start:
                step = (end - start) / max(1, len(frames) - 1)
                numeric = [start + index * step for index in range(len(frames))]
            else:
                numeric = [start + index * 1e-4 for index in range(len(frames))]
        for index, (frame, ts) in enumerate(zip(frames, numeric)):
            if float(ts) < float(recent_start) - epsilon:
                records.append(
                    {
                        "image": frame,
                        "timestamp": float(ts),
                        "chunk_id": chunk_id(chunk),
                        "frame_index": index,
                        "key": (chunk_id(chunk), index, round(float(ts), 6)),
                    }
                )
    records.sort(key=lambda item: (item["timestamp"], item["chunk_id"], item["frame_index"]))
    return records


def choose_anchor_record(anchor_chunk: Any, records: list[dict[str, Any]], anchor_time: float) -> dict[str, Any] | None:
    candidates = [record for record in records if int(record["chunk_id"]) == chunk_id(anchor_chunk)]
    if not candidates:
        return None
    return min(candidates, key=lambda item: (abs(float(item["timestamp"]) - anchor_time), int(item["frame_index"])))


def strict_microclip_packet(
    anchor_chunk: Any,
    records: list[dict[str, Any]],
    anchor_time: float,
) -> list[dict[str, Any]]:
    anchor = choose_anchor_record(anchor_chunk, records, anchor_time)
    if anchor is None:
        return []
    previous = [
        record for record in records if float(record["timestamp"]) < float(anchor["timestamp"]) and record["key"] != anchor["key"]
    ]
    following = [
        record for record in records if float(record["timestamp"]) > float(anchor["timestamp"]) and record["key"] != anchor["key"]
    ]
    selected: list[dict[str, Any]] = []
    if previous:
        selected.append(max(previous, key=lambda item: (item["timestamp"], item["chunk_id"], item["frame_index"])))
    selected.append(anchor)
    if following:
        selected.append(min(following, key=lambda item: (item["timestamp"], item["chunk_id"], item["frame_index"])))
    unique = {record["key"]: record for record in selected}
    return sorted(unique.values(), key=lambda item: (item["timestamp"], item["chunk_id"], item["frame_index"]))


def frame_chunk(record: dict[str, Any], offset: int) -> Any:
    timestamp = float(record["timestamp"])
    return SimpleNamespace(
        frames=[record["image"]],
        frame_timestamps=[timestamp],
        start_time=timestamp,
        end_time=timestamp,
        chunk_index=-(100000 + offset),
        fps=1.0,
    )


def score_context(
    qa: Any,
    *,
    chunks: list[Any],
    prompt: str,
    options: list[dict[str, str]],
    scorer: Any,
    evaluator: Any,
) -> dict[str, Any]:
    score, elapsed_ms = evaluator(
        qa,
        chunks,
        prompt,
        options,
        scorer,
        0.50,
        0.20,
        0.30,
    )
    return {
        "predicted_option": score.get("predicted_option"),
        "option_probabilities": score.get("option_probabilities"),
        "answer_margin": score.get("answer_margin"),
        "normalized_entropy": score.get("normalized_entropy"),
        "entropy_confidence": score.get("entropy_confidence"),
        "visual_support": score.get("visual_support_norm"),
        "visual_support_raw": score.get("visual_support_raw"),
        "sufficiency": score.get("sufficiency"),
        "option_scoring_mechanism": score.get("option_scoring_mechanism"),
        "option_forward_ms": score.get("option_forward_ms"),
        "sufficiency_ms": score.get("sufficiency_ms"),
        "elapsed_ms": float(elapsed_ms),
    }


def probs(score: dict[str, Any]) -> dict[str, float]:
    raw = score.get("option_probabilities") or {}
    return {str(key): float(value) for key, value in raw.items()}


def entropy(dist: dict[str, float]) -> float:
    return -sum(value * math.log(value) for value in dist.values() if value > 0.0)


def js_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    keys = sorted(set(left) | set(right))
    p = {key: float(left.get(key, 0.0)) for key in keys}
    q = {key: float(right.get(key, 0.0)) for key in keys}
    total_p = sum(p.values()) or 1.0
    total_q = sum(q.values()) or 1.0
    p = {key: value / total_p for key, value in p.items()}
    q = {key: value / total_q for key, value in q.items()}
    m = {key: 0.5 * (p[key] + q[key]) for key in keys}

    def kl(a: dict[str, float], b: dict[str, float]) -> float:
        return sum(a[key] * math.log(a[key] / b[key]) for key in keys if a[key] > 0.0 and b[key] > 0.0)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def total_variation(left: dict[str, float], right: dict[str, float]) -> float:
    keys = sorted(set(left) | set(right))
    return 0.5 * sum(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys)


def top1(dist: dict[str, float]) -> float:
    return max(dist.values()) if dist else 0.0


def margin(dist: dict[str, float]) -> float:
    ordered = sorted(dist.values(), reverse=True)
    if len(ordered) < 2:
        return 0.0
    return float(ordered[0] - ordered[1])


def argmax(dist: dict[str, float]) -> str | None:
    if not dist:
        return None
    return max(dist, key=lambda key: (dist[key], -ord(str(key)[0])))


def add_instability(record: dict[str, Any]) -> None:
    p0 = probs(record["P0"])
    p1 = probs(record["P1"])
    pc = probs(record["Pclip"])
    record["signals"] = {
        "JS_01": js_divergence(p0, p1),
        "TV_01": total_variation(p0, p1),
        "pred_changed_01": argmax(p0) != argmax(p1),
        "top1_delta_01": top1(p1) - top1(p0),
        "margin_delta_01": margin(p1) - margin(p0),
        "entropy_delta_01": entropy(p1) - entropy(p0),
        "JS_1clip": js_divergence(p1, pc),
        "TV_1clip": total_variation(p1, pc),
        "pred_changed_1clip": argmax(p1) != argmax(pc),
        "top1_delta_1clip": top1(pc) - top1(p1),
        "margin_delta_1clip": margin(pc) - margin(p1),
        "entropy_delta_1clip": entropy(pc) - entropy(p1),
        "candidate_total_score": record.get("candidate_total_score"),
        "candidate_semantic_score": record.get("candidate_semantic_score"),
        "candidate_temporal_distance_seconds": record.get("candidate_temporal_distance_seconds"),
        "K0_sufficiency": record["P0"].get("sufficiency"),
        "K0_visual_support": record["P0"].get("visual_support"),
        "K0_answer_margin": record["P0"].get("answer_margin"),
    }


def strict_label(oracle_row: dict[str, Any]) -> str:
    branches = oracle_row.get("branches") or {}
    k0_correct = bool((branches.get("K0") or {}).get("correct"))
    anchor_correct = bool((branches.get("anchor") or {}).get("correct"))
    microclip_correct = bool((branches.get("microclip") or {}).get("correct"))
    if not k0_correct and not anchor_correct and microclip_correct:
        return "MICROCLIP_HELPFUL"
    if k0_correct and not microclip_correct:
        return "MICROCLIP_HARMFUL"
    return "MICROCLIP_NEUTRAL"


def anchor_label(oracle_row: dict[str, Any]) -> str:
    branches = oracle_row.get("branches") or {}
    k0_correct = bool((branches.get("K0") or {}).get("correct"))
    anchor_correct = bool((branches.get("anchor") or {}).get("correct"))
    if not k0_correct and anchor_correct:
        return "ANCHOR_HELPFUL"
    if k0_correct and not anchor_correct:
        return "ANCHOR_HARMFUL"
    return "ANCHOR_NEUTRAL"


def dist(values: list[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"n": 0}
    ordered = sorted(clean)
    return {
        "n": len(ordered),
        "mean": statistics.mean(ordered),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "p25": ordered[int(0.25 * (len(ordered) - 1))],
        "p75": ordered[int(0.75 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def auc(labels: list[bool], scores: list[float]) -> float | None:
    pairs = [(label, float(score)) for label, score in zip(labels, scores) if math.isfinite(float(score))]
    positives = [score for label, score in pairs if label]
    negatives = [score for label, score in pairs if not label]
    if not positives or not negatives:
        return None
    wins = ties = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                ties += 1.0
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def signal_stats(records: list[dict[str, Any]], signal_names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_label[str(record["microclip_label"])].append(record)
    for signal in signal_names:
        out[signal] = {}
        for label in ("MICROCLIP_HELPFUL", "MICROCLIP_HARMFUL", "MICROCLIP_NEUTRAL"):
            values = [
                float(value)
                for item in by_label.get(label, [])
                if (value := item["signals"].get(signal)) is not None and isinstance(value, (int, float, bool))
            ]
            out[signal][label] = dist(values)
        values = [
            float(item["signals"][signal])
            for item in records
            if item["signals"].get(signal) is not None and isinstance(item["signals"].get(signal), (int, float, bool))
        ]
        labels = [
            item["microclip_label"] == "MICROCLIP_HELPFUL"
            for item in records
            if item["signals"].get(signal) is not None and isinstance(item["signals"].get(signal), (int, float, bool))
        ]
        out[signal]["helpful_vs_rest_auc"] = auc(labels, values) if values else None
        out[signal]["auc_note"] = "statistically_unstable_n_helpful_lt_5"
    return out


def outcome(oracle_row: dict[str, Any], branch: str) -> bool:
    return bool(((oracle_row.get("branches") or {}).get(branch) or {}).get("correct"))


def simulate_rule(records: list[dict[str, Any]], name: str, predicate: Any) -> dict[str, Any]:
    total = len(records)
    expansions = [record for record in records if predicate(record)]
    correct = 0
    rescues = damages = neutral = 0
    for record in records:
        oracle = record["oracle_outcomes"]
        expand = predicate(record)
        final_correct = outcome(oracle, "microclip") if expand else outcome(oracle, "anchor")
        correct += int(final_correct)
        if expand and record["microclip_label"] == "MICROCLIP_HELPFUL":
            rescues += 1
        elif expand and record["microclip_label"] == "MICROCLIP_HARMFUL":
            damages += 1
        elif expand:
            neutral += 1
    return {
        "rule": name,
        "expansions": len(expansions),
        "microclip_only_rescues_captured": rescues,
        "microclip_damages_triggered": damages,
        "neutral_expansions": neutral,
        "simulated_correct": correct,
        "simulated_accuracy": correct / total if total else None,
        "mean_additional_historical_frames": (
            statistics.mean([float(record.get("microclip_num_frames", 0)) for record in expansions])
            if expansions
            else 0.0
        ),
    }


def rule_grid(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    js_thresholds = [0.001, 0.003, 0.005, 0.01, 0.02, 0.04]
    margin_thresholds = [0.02, 0.05, 0.10, 0.20, 0.35]
    score_thresholds = [0.35, 0.45, 0.50, 0.55, 0.60]
    distance_thresholds = [1.0, 5.0, 10.0, 20.0, 40.0]
    rules: list[dict[str, Any]] = [
        simulate_rule(records, "never_expand_anchor", lambda _r: False),
        simulate_rule(records, "always_expand_microclip", lambda r: bool(r.get("anchor_available"))),
        simulate_rule(records, "expand_if_pred_changed_01", lambda r: bool(r["signals"].get("pred_changed_01"))),
    ]
    for js in js_thresholds:
        rules.append(simulate_rule(records, f"JS_01>{js}", lambda r, js=js: float(r["signals"].get("JS_01") or 0.0) > js))
        for score in score_thresholds:
            rules.append(
                simulate_rule(
                    records,
                    f"JS_01>{js}_and_total>{score}",
                    lambda r, js=js, score=score: float(r["signals"].get("JS_01") or 0.0) > js
                    and float(r["signals"].get("candidate_total_score") or -1.0) > score,
                )
            )
        for margin_th in margin_thresholds:
            rules.append(
                simulate_rule(
                    records,
                    f"JS_01>{js}_and_abs_margin_delta>{margin_th}",
                    lambda r, js=js, margin_th=margin_th: float(r["signals"].get("JS_01") or 0.0) > js
                    and abs(float(r["signals"].get("margin_delta_01") or 0.0)) > margin_th,
                )
            )
        for distance in distance_thresholds:
            rules.append(
                simulate_rule(
                    records,
                    f"JS_01>{js}_and_dist>{distance}",
                    lambda r, js=js, distance=distance: float(r["signals"].get("JS_01") or 0.0) > js
                    and float(r["signals"].get("candidate_temporal_distance_seconds") or -1.0) > distance,
                )
            )
    for margin_th in margin_thresholds:
        for score in score_thresholds:
            rules.append(
                simulate_rule(
                    records,
                    f"abs_margin_delta>{margin_th}_and_total>{score}",
                    lambda r, margin_th=margin_th, score=score: abs(float(r["signals"].get("margin_delta_01") or 0.0)) > margin_th
                    and float(r["signals"].get("candidate_total_score") or -1.0) > score,
                )
            )
    return sorted(
        rules,
        key=lambda item: (
            -int(item["microclip_only_rescues_captured"]),
            int(item["microclip_damages_triggered"]),
            -float(item["simulated_accuracy"]),
            int(item["expansions"]),
        ),
    )


def leave_one_rescue_out(records: list[dict[str, Any]], candidate_rules: list[dict[str, Any]]) -> dict[str, Any]:
    helpful_ids = [int(record["question_id"]) for record in records if record["microclip_label"] == "MICROCLIP_HELPFUL"]
    if not helpful_ids:
        return {"helpful_ids": [], "results": []}
    result_rows: list[dict[str, Any]] = []

    def predicate_from_name(rule_name: str) -> Any:
        if rule_name == "never_expand_anchor":
            return lambda _r: False
        if rule_name == "always_expand_microclip":
            return lambda r: bool(r.get("anchor_available"))
        if rule_name == "expand_if_pred_changed_01":
            return lambda r: bool(r["signals"].get("pred_changed_01"))
        if rule_name.startswith("JS_01>") and "_and_total>" in rule_name:
            left, right = rule_name.split("_and_total>", maxsplit=1)
            js = float(left.replace("JS_01>", ""))
            score = float(right)
            return lambda r, js=js, score=score: float(r["signals"].get("JS_01") or 0.0) > js and float(
                r["signals"].get("candidate_total_score") or -1.0
            ) > score
        if rule_name.startswith("JS_01>") and "_and_abs_margin_delta>" in rule_name:
            left, right = rule_name.split("_and_abs_margin_delta>", maxsplit=1)
            js = float(left.replace("JS_01>", ""))
            margin_th = float(right)
            return lambda r, js=js, margin_th=margin_th: float(r["signals"].get("JS_01") or 0.0) > js and abs(
                float(r["signals"].get("margin_delta_01") or 0.0)
            ) > margin_th
        if rule_name.startswith("JS_01>") and "_and_dist>" in rule_name:
            left, right = rule_name.split("_and_dist>", maxsplit=1)
            js = float(left.replace("JS_01>", ""))
            distance = float(right)
            return lambda r, js=js, distance=distance: float(r["signals"].get("JS_01") or 0.0) > js and float(
                r["signals"].get("candidate_temporal_distance_seconds") or -1.0
            ) > distance
        if rule_name.startswith("JS_01>"):
            js = float(rule_name.replace("JS_01>", ""))
            return lambda r, js=js: float(r["signals"].get("JS_01") or 0.0) > js
        if rule_name.startswith("abs_margin_delta>") and "_and_total>" in rule_name:
            left, right = rule_name.split("_and_total>", maxsplit=1)
            margin_th = float(left.replace("abs_margin_delta>", ""))
            score = float(right)
            return lambda r, margin_th=margin_th, score=score: abs(float(r["signals"].get("margin_delta_01") or 0.0)) > margin_th and float(
                r["signals"].get("candidate_total_score") or -1.0
            ) > score
        return lambda _r: False

    for held_out in helpful_ids:
        train = [record for record in records if int(record["question_id"]) != held_out]
        rescored = [
            simulate_rule(train, item["rule"], predicate_from_name(item["rule"]))
            for item in candidate_rules
        ]
        best = max(
            rescored,
            key=lambda item: (
                int(item["microclip_only_rescues_captured"]),
                -int(item["microclip_damages_triggered"]),
                float(item["simulated_accuracy"]),
                -int(item["expansions"]),
            ),
        )
        held_record = next(record for record in records if int(record["question_id"]) == held_out)
        selected = bool(predicate_from_name(best["rule"])(held_record))
        result_rows.append({"held_out_question_id": held_out, "selected_by_train_rule": selected, "rule": best["rule"]})
    return {"helpful_ids": helpful_ids, "results": result_rows}


def category_analysis(records: list[dict[str, Any]], signal_names: list[str]) -> dict[str, Any]:
    wanted = {
        "Clips Summarize",
        "Prospective Reasoning",
        "Counting",
        "Action Recognition",
        "Object Recognition",
    }
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if str(record.get("category")) in wanted:
            groups[str(record.get("category"))].append(record)
    out: dict[str, Any] = {}
    for category, group in sorted(groups.items()):
        out[category] = {
            "n": len(group),
            "labels": dict(Counter(record["microclip_label"] for record in group)),
            "signals": {
                signal: dist([
                    float(record["signals"][signal])
                    for record in group
                    if record["signals"].get(signal) is not None
                    and isinstance(record["signals"].get(signal), (int, float, bool))
                ])
                for signal in signal_names
            },
        }
    return out


def write_csv(path: Path, records: list[dict[str, Any]], signal_names: list[str]) -> None:
    fields = [
        "question_id",
        "category",
        "ground_truth",
        "microclip_label",
        "anchor_label",
        "anchor_available",
        "microclip_num_frames",
        "candidate_total_score",
        "candidate_semantic_score",
        "candidate_temporal_distance_seconds",
        "P0_predicted_option",
        "P1_predicted_option",
        "Pclip_predicted_option",
        *signal_names,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {
                "question_id": record["question_id"],
                "category": record.get("category"),
                "ground_truth": record.get("ground_truth"),
                "microclip_label": record.get("microclip_label"),
                "anchor_label": record.get("anchor_label"),
                "anchor_available": record.get("anchor_available"),
                "microclip_num_frames": record.get("microclip_num_frames"),
                "candidate_total_score": record.get("candidate_total_score"),
                "candidate_semantic_score": record.get("candidate_semantic_score"),
                "candidate_temporal_distance_seconds": record.get("candidate_temporal_distance_seconds"),
                "P0_predicted_option": record["P0"].get("predicted_option"),
                "P1_predicted_option": record["P1"].get("predicted_option"),
                "Pclip_predicted_option": record["Pclip"].get("predicted_option"),
            }
            row.update({signal: record["signals"].get(signal) for signal in signal_names})
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", required=True, help="Saved temporal_microclip StreamingBench-100 result dir/JSON.")
    parser.add_argument("--strict-oracle", required=True, help="Saved strict_microclip_oracle result dir/JSON/JSONL.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--qa-model", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--qa-device", default=None)
    parser.add_argument("--attn-implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"))
    parser.add_argument("--max-qa-tokens", type=int, default=256)
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--recent-window", type=int, default=6)
    parser.add_argument("--context-time", type=float, default=70.0)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    global RecentWindowQAModel, build_prompt, decode_video_to_chunks_qwen
    global resolve_video_path, select_recent_window_frames, timestamp_to_seconds

    from lib.minicpm.baseline import RecentWindowQAModel as _RecentWindowQAModel
    from lib.minicpm.baseline import select_recent_window_frames as _select_recent_window_frames
    from lib.minicpm.progressive_sufficiency import _evaluate_sufficiency
    from lib.minicpm.progressive_sufficiency import _extract_mcq_options
    from lib.minicpm.progressive_sufficiency import _get_clip_scorer
    from lib.shared.recent_window import decode_video_to_chunks_qwen as _decode_video_to_chunks_qwen
    from main_experiments.minicpm_v46.streamingbench.eval_baseline import build_prompt as _build_prompt
    from main_experiments.minicpm_v46.streamingbench.eval_baseline import resolve_video_path as _resolve_video_path
    from main_experiments.minicpm_v46.streamingbench.eval_baseline import timestamp_to_seconds as _timestamp_to_seconds

    RecentWindowQAModel = _RecentWindowQAModel
    build_prompt = _build_prompt
    decode_video_to_chunks_qwen = _decode_video_to_chunks_qwen
    resolve_video_path = _resolve_video_path
    select_recent_window_frames = _select_recent_window_frames
    timestamp_to_seconds = _timestamp_to_seconds

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_path, saved_rows = load_results(Path(args.source_results))
    strict_source, strict_rows = load_strict_oracle(Path(args.strict_oracle))
    if args.max_samples > 0:
        saved_rows = saved_rows[: args.max_samples]
    annotations = load_annotations(Path(args.annotations), args.video_dir)

    qa = RecentWindowQAModel(
        model_name=args.qa_model,
        device=args.qa_device or default_qa_device(),
        max_new_tokens=args.max_qa_tokens,
        attn_implementation=args.attn_implementation,
    )
    scorer = _get_clip_scorer(qa)

    completed: dict[int, dict[str, Any]] = {}
    jsonl_path = out_dir / "microclip_instability_scores.jsonl"
    if jsonl_path.exists():
        for row in read_jsonl(jsonl_path):
            completed[int(row["question_id"])] = row

    epsilon = 1e-6
    with jsonl_path.open("a", encoding="utf-8") as handle:
        for ordinal, saved in enumerate(saved_rows, start=1):
            qid = int(saved.get("_index", ordinal - 1))
            if qid in completed:
                print(f"[{ordinal}/{len(saved_rows)}] skip qid={qid}", flush=True)
                continue
            if qid not in strict_rows:
                raise KeyError(f"Question index {qid} is absent from strict oracle results")
            task = annotations.get(qid)
            if task is None:
                raise KeyError(f"Question index {qid} is absent from annotations")
            question = task["question_obj"]
            prompt = task["prompt"]
            options = _extract_mcq_options(prompt)
            if not options:
                print(f"[{ordinal}/{len(saved_rows)}] skip non-MCQ qid={qid}", flush=True)
                continue
            gt = answer_gt_from(saved, question)
            ts_sec = float(timestamp_to_seconds(question["time_stamp"]))
            video_end = ts_sec + 1e-4
            broad_start = max(0.0, ts_sec - max(float(args.context_time), float(args.chunk_duration)))
            decode_recent_hint = max(
                int(math.ceil(float(args.context_time) / max(float(args.chunk_duration), 1e-6))),
                int(args.recent_window) + 64,
            )

            print(f"[{ordinal}/{len(saved_rows)}] qid={qid}", flush=True)
            broad_chunks, broad_backend = decode_video_to_chunks_qwen(
                video_path=task["video_path"],
                chunk_duration=args.chunk_duration,
                fps=args.fps,
                recent_frames_only=decode_recent_hint,
                video_start=broad_start,
                video_end=video_end,
            )
            broad_by_id = {chunk_id(chunk): chunk for chunk in broad_chunks}
            recent_video_start = max(0.0, video_end - float(args.recent_window) * float(args.chunk_duration))
            recent_selection = select_recent_window_frames(
                qa=qa,
                video_path=task["video_path"],
                chunk_duration=args.chunk_duration,
                fps=args.fps,
                recent_frames_only=args.recent_window,
                video_start=recent_video_start,
                video_end=video_end,
                cdas_config=None,
            )
            recent_chunks = list(recent_selection.selected_chunks)
            r_start = recent_start_time(recent_chunks)
            historical_chunks = [chunk for chunk in broad_chunks if chunk_bounds(chunk)[1] < r_start - epsilon]
            records = frame_records_from_chunks(historical_chunks, r_start, epsilon)

            source_meta = adaptive(saved)
            candidate_queue = list(source_meta.get("candidate_queue") or [])
            anchor_meta = candidate_queue[0] if candidate_queue else {}
            anchor_chunk = broad_by_id.get(int(anchor_meta["chunk_id"])) if "chunk_id" in anchor_meta else None
            anchor_time = number(anchor_meta.get("timestamp"))
            if anchor_chunk is not None and anchor_time is None:
                anchor_time = chunk_anchor_time(anchor_chunk)

            anchor_records: list[dict[str, Any]] = []
            microclip_records: list[dict[str, Any]] = []
            if anchor_chunk is not None and anchor_time is not None:
                anchor_record = choose_anchor_record(anchor_chunk, records, anchor_time)
                if anchor_record is not None:
                    anchor_records = [anchor_record]
                microclip_records = strict_microclip_packet(anchor_chunk, records, anchor_time)
            for record in [*anchor_records, *microclip_records]:
                if not float(record["timestamp"]) < r_start - epsilon:
                    raise AssertionError(
                        f"Temporal violation qid={qid}: history frame {record['timestamp']} recent_start={r_start}"
                    )

            anchor_chunks = [frame_chunk(record, index) for index, record in enumerate(anchor_records)]
            microclip_chunks = [frame_chunk(record, index) for index, record in enumerate(microclip_records)]
            p0 = score_context(
                qa,
                chunks=recent_chunks,
                prompt=prompt,
                options=options,
                scorer=scorer,
                evaluator=_evaluate_sufficiency,
            )
            p1 = score_context(
                qa,
                chunks=[*anchor_chunks, *recent_chunks],
                prompt=prompt,
                options=options,
                scorer=scorer,
                evaluator=_evaluate_sufficiency,
            )
            pclip = score_context(
                qa,
                chunks=[*microclip_chunks, *recent_chunks],
                prompt=prompt,
                options=options,
                scorer=scorer,
                evaluator=_evaluate_sufficiency,
            )
            candidate_end = number(anchor_meta.get("end_time_seconds"))
            if candidate_end is None and anchor_chunk is not None:
                candidate_end = chunk_bounds(anchor_chunk)[1]
            output = {
                "question_id": qid,
                "key": saved.get("_key"),
                "video_id": task["video"],
                "category": saved.get("task_type") or task.get("task_type"),
                "question": question.get("question"),
                "options": question.get("options"),
                "ground_truth": gt,
                "candidate_anchor_id": anchor_meta.get("chunk_id"),
                "candidate_total_score": anchor_meta.get("total_score"),
                "candidate_semantic_score": anchor_meta.get("semantic_score"),
                "candidate_temporal_distance_seconds": (
                    float(r_start - candidate_end) if candidate_end is not None else None
                ),
                "anchor_available": bool(anchor_records),
                "microclip_num_frames": len(microclip_records),
                "microclip_timestamps": [float(record["timestamp"]) for record in microclip_records],
                "decode": {
                    "source_path": source_path,
                    "strict_oracle_source": strict_source,
                    "broad_backend": broad_backend,
                    "recent_start_time_seconds": r_start,
                    "recent_chunk_ids": [chunk_id(chunk) for chunk in recent_chunks],
                    "historical_frame_records": len(records),
                },
                "P0": p0,
                "P1": p1,
                "Pclip": pclip,
                "microclip_label": strict_label(strict_rows[qid]),
                "anchor_label": anchor_label(strict_rows[qid]),
                "oracle_outcomes": strict_rows[qid],
            }
            add_instability(output)
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            handle.flush()
            completed[qid] = output

    records = [completed[key] for key in sorted(completed)]
    signal_names = [
        "JS_01",
        "TV_01",
        "pred_changed_01",
        "top1_delta_01",
        "margin_delta_01",
        "entropy_delta_01",
        "JS_1clip",
        "TV_1clip",
        "pred_changed_1clip",
        "top1_delta_1clip",
        "margin_delta_1clip",
        "entropy_delta_1clip",
        "candidate_total_score",
        "candidate_semantic_score",
        "candidate_temporal_distance_seconds",
        "K0_sufficiency",
        "K0_visual_support",
        "K0_answer_margin",
    ]
    rules = rule_grid(records)
    summary = {
        "samples": len(records),
        "label_counts": dict(Counter(record["microclip_label"] for record in records)),
        "anchor_label_counts": dict(Counter(record["anchor_label"] for record in records)),
        "signal_stats": signal_stats(records, signal_names),
        "rules_top20": rules[:20],
        "baselines": {
            "K0_recent6_accuracy": sum(outcome(record["oracle_outcomes"], "K0") for record in records) / len(records),
            "never_expand_anchor_accuracy": rules[[rule["rule"] for rule in rules].index("never_expand_anchor")]["simulated_accuracy"],
            "always_expand_microclip_accuracy": rules[[rule["rule"] for rule in rules].index("always_expand_microclip")]["simulated_accuracy"],
        },
        "leave_one_rescue_out": leave_one_rescue_out(records, rules),
        "category_analysis": category_analysis(records, signal_names[:12]),
        "decision_notes": [
            "AUCs are statistically unstable because MICROCLIP_HELPFUL count is expected to be very small.",
            "Rules simulate final=Microclip if expanded else Anchor, using strict-oracle final-generation outcomes.",
            "No ground truth is used in context construction or signal computation.",
        ],
    }
    (out_dir / "microclip_instability_scores.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "microclip_instability_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(out_dir / "microclip_instability_scores.csv", records, signal_names)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
