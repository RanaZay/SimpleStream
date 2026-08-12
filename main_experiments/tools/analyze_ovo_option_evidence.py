#!/usr/bin/env python3
"""Offline option-specific historical evidence analysis for OVO oracle rows.

This script does not rerun MiniCPM and does not modify PRISM. It consumes a
completed ``ovo_prism_oracle_headroom.py`` run, computes CLIP similarities for
``question + option`` against Recent-6 and PRISM historical candidate frames,
then reports how well those evidence signals predict oracle-useful history.
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
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ovo_constants import BACKWARD_TASKS, REAL_TIME_TASKS  # noqa: E402

MCQ_TASKS = set(BACKWARD_TASKS + REAL_TIME_TASKS)
LETTERS = "ABCDE"


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _find_oracle_rows(path: Path) -> Path:
    if path.is_file():
        return path
    for name in ("oracle_rows.jsonl", "official_oracle_rows.jsonl"):
        candidate = path / name
        if candidate.exists():
            return candidate
    matches = sorted(path.glob("**/oracle_rows.jsonl"))
    if matches:
        return matches[-1]
    raise FileNotFoundError(f"No oracle_rows.jsonl found under {path}")


def _maybe_load_summary(path: Path) -> dict[str, Any]:
    if path.is_file():
        search_dir = path.parent
    else:
        search_dir = path
    for name in ("oracle_summary.json", "official_oracle_reconciliation_summary.json"):
        candidate = search_dir / name
        if candidate.exists():
            value = _load_json(candidate)
            return value if isinstance(value, dict) else {}
    return {}


def _check_official_sanity(summary: dict[str, Any], *, require: bool) -> dict[str, Any]:
    reconciliation = summary.get("official_protocol_reconciliation")
    checks = reconciliation.get("sanity_checks") if isinstance(reconciliation, dict) else None
    if not isinstance(checks, dict) or not checks:
        result = {
            "status": "missing",
            "message": "No official K=0 sanity checks found in summary.",
        }
        if require:
            raise SystemExit(result["message"])
        return result
    failed = {key: value for key, value in checks.items() if not value.get("passes_rounding_check")}
    result = {
        "status": "passed" if not failed else "failed",
        "checks": checks,
    }
    if failed and require:
        raise SystemExit(f"Official sanity checks failed: {failed}")
    return result


def _annotations_by_id(path: Path) -> dict[str, dict[str, Any]]:
    value = _load_json(path)
    if not isinstance(value, list):
        raise ValueError(f"Expected OVO annotation list at {path}")
    return {str(item["id"]): item for item in value if isinstance(item, dict) and "id" in item}


def _clean_option(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^[A-Ea-e]\s*[\).:-]\s*", "", text).strip()
    return text


def _option_records(row: dict[str, Any], annotations: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    anno = annotations.get(str(row.get("id")))
    raw_options = anno.get("options") if isinstance(anno, dict) else None
    if not isinstance(raw_options, list):
        raw_options = row.get("options")
    if not isinstance(raw_options, list):
        return []
    options = []
    for idx, option in enumerate(raw_options[: len(LETTERS)]):
        text = _clean_option(option)
        if text:
            options.append({"letter": LETTERS[idx], "text": text})
    return options


def _video_path(row: dict[str, Any], chunked_dir: Path) -> Path:
    return chunked_dir / f"{row.get('id')}.mp4"


def _chunk_ids(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _candidate_ids(row: dict[str, Any], max_candidates: int) -> list[int]:
    ids: list[int] = []
    queue = row.get("candidate_queue")
    if isinstance(queue, list):
        for item in queue:
            if not isinstance(item, dict):
                continue
            try:
                ids.append(int(item["chunk_id"]))
            except (KeyError, TypeError, ValueError):
                continue
            if len(ids) >= max_candidates:
                break
    if not ids:
        ids = _chunk_ids(row.get("candidate_top3_chunk_ids"))[:max_candidates]
    return ids


def _recent_ids(row: dict[str, Any], recent_window: int) -> list[int]:
    branches = row.get("branches")
    if isinstance(branches, list) and branches and isinstance(branches[0], dict):
        ids = _chunk_ids(branches[0].get("context_chunk_ids") or branches[0].get("memory_chunk_ids"))
        if ids:
            return ids[-recent_window:]
    return _chunk_ids(row.get("baseline_final_chunk_ids") or row.get("final_chunk_ids"))[-recent_window:]


def _safe_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    values = sorted(x for x in values if math.isfinite(x))
    if not values:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p25": None, "p75": None, "p90": None}

    def pct(p: float) -> float:
        pos = (len(values) - 1) * p
        low = int(math.floor(pos))
        high = int(math.ceil(pos))
        if low == high:
            return values[low]
        frac = pos - low
        return values[low] * (1.0 - frac) + values[high] * frac

    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "p10": pct(0.10),
        "p25": pct(0.25),
        "p75": pct(0.75),
        "p90": pct(0.90),
    }


def _roc_auc(labels: list[bool], scores: list[float]) -> float | None:
    pairs = [(float(score), bool(label)) for label, score in zip(labels, scores) if math.isfinite(float(score))]
    positives = sum(1 for _score, label in pairs if label)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None
    pairs.sort(key=lambda item: item[0])
    rank_sum = 0.0
    idx = 0
    while idx < len(pairs):
        j = idx + 1
        while j < len(pairs) and pairs[j][0] == pairs[idx][0]:
            j += 1
        avg_rank = (idx + 1 + j) / 2.0
        for k in range(idx, j):
            if pairs[k][1]:
                rank_sum += avg_rank
        idx = j
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _pr_at_threshold(labels: list[bool], scores: list[float], threshold: float) -> dict[str, Any]:
    pred = [score >= threshold for score in scores]
    tp = sum(1 for y, p in zip(labels, pred) if y and p)
    fp = sum(1 for y, p in zip(labels, pred) if not y and p)
    fn = sum(1 for y, p in zip(labels, pred) if y and not p)
    tn = sum(1 for y, p in zip(labels, pred) if not y and not p)
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_positive": tp + fp,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
    }


def _threshold_report(labels: list[bool], scores: list[float], *, include_zero: bool = False) -> list[dict[str, Any]]:
    valid = sorted(float(score) for score in scores if math.isfinite(float(score)))
    if not valid:
        return []
    thresholds: list[float] = []
    if include_zero:
        thresholds.append(0.0)
    for keep_rate in (0.05, 0.10, 0.20, 0.30, 0.50):
        idx = min(len(valid) - 1, max(0, int(math.floor((1.0 - keep_rate) * len(valid)))))
        thresholds.append(valid[idx])
    seen: set[float] = set()
    out = []
    for threshold in thresholds:
        rounded = round(float(threshold), 8)
        if rounded in seen:
            continue
        seen.add(rounded)
        out.append(_pr_at_threshold(labels, scores, float(threshold)))
    return out


def _top_margin(values: dict[str, float]) -> float | None:
    vals = sorted(values.values(), reverse=True)
    if len(vals) < 2:
        return None
    return vals[0] - vals[1]


def _argmax(values: dict[str, float]) -> str | None:
    if not values:
        return None
    return max(values, key=lambda key: values[key])


def _predicted_letter(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    match = re.search(r"\b([A-E])\b", text)
    return match.group(1) if match else None


def _oracle_correct_memory_ids(row: dict[str, Any]) -> set[int]:
    ids: set[int] = set()
    branches = row.get("branches")
    if not isinstance(branches, list):
        return ids
    for branch in branches:
        if not isinstance(branch, dict) or int(branch.get("k", -1)) == 0 or not branch.get("correct"):
            continue
        ids.update(_chunk_ids(branch.get("memory_chunk_ids")))
    return ids


def _failure_label(row: dict[str, Any]) -> str:
    mode = str(row.get("failure_mode") or "")
    if mode:
        return mode
    k0 = bool(row.get("k0_correct"))
    prism = bool(row.get("prism_correct"))
    used_memory = bool(row.get("prism_memory_chunk_ids"))
    helpful = row.get("minimum_correct_k") in {1, 2, 3}
    if not k0 and used_memory and prism:
        return "successful_retrieval"
    if k0 and used_memory and not prism:
        return "damaged_retrieval"
    if not k0 and not used_memory and helpful:
        return "false_stop"
    if not k0 and helpful:
        return "oracle_memory_helpful"
    if not k0:
        return "oracle_memory_not_helpful"
    return "recent6_correct"


def _score_chunks(scorer: Any, text: str, chunks: list[Any]) -> dict[int, float]:
    flat_frames: list[Any] = []
    owners: list[int] = []
    for chunk in chunks:
        chunk_id = int(chunk.chunk_index)
        for frame in chunk.frames:
            flat_frames.append(frame)
            owners.append(chunk_id)
    if not flat_frames:
        return {}
    scores = scorer.score(text, flat_frames)
    by_chunk: dict[int, float] = {}
    for owner, score in zip(owners, scores):
        by_chunk[owner] = max(by_chunk.get(owner, -1.0), float(score))
    return by_chunk


def _compute_row_evidence(
    *,
    row: dict[str, Any],
    annotations: dict[str, dict[str, Any]],
    chunked_dir: Path,
    scorer: Any,
    recent_window: int,
    max_candidates: int,
    chunk_duration: float,
    fps: float,
) -> dict[str, Any] | None:
    from lib.shared.recent_window import decode_video_to_chunks_qwen

    task = str(row.get("task") or "")
    if task not in MCQ_TASKS:
        return None
    options = _option_records(row, annotations)
    if len(options) < 2:
        return None

    chunks, backend = decode_video_to_chunks_qwen(
        video_path=str(_video_path(row, chunked_dir)),
        chunk_duration=chunk_duration,
        fps=fps,
        recent_frames_only=None,
    )
    by_id = {int(chunk.chunk_index): chunk for chunk in chunks}
    recent_ids = [chunk_id for chunk_id in _recent_ids(row, recent_window) if chunk_id in by_id]
    if not recent_ids:
        recent_ids = [int(chunk.chunk_index) for chunk in chunks[-recent_window:]]
    candidate_ids = [chunk_id for chunk_id in _candidate_ids(row, max_candidates) if chunk_id in by_id]
    recent_chunks = [by_id[chunk_id] for chunk_id in recent_ids]
    candidate_chunks = [by_id[chunk_id] for chunk_id in candidate_ids]

    question = str(row.get("question") or "").strip()
    recent_support: dict[str, float] = {}
    best_historical_support: dict[str, float] = {}
    best_historical_chunk_id: dict[str, int | None] = {}
    historical_support: dict[str, dict[str, float]] = {}
    for option in options:
        letter = option["letter"]
        text = f"{question} {option['text']}".strip()
        recent_scores = _score_chunks(scorer, text, recent_chunks)
        historical_scores = _score_chunks(scorer, text, candidate_chunks)
        recent_support[letter] = max(recent_scores.values()) if recent_scores else -1.0
        historical_support[letter] = {str(key): value for key, value in sorted(historical_scores.items())}
        if historical_scores:
            best_chunk = max(historical_scores, key=lambda key: historical_scores[key])
            best_historical_chunk_id[letter] = int(best_chunk)
            best_historical_support[letter] = float(historical_scores[best_chunk])
        else:
            best_historical_chunk_id[letter] = None
            best_historical_support[letter] = -1.0

    evidence_gain = {
        letter: best_historical_support[letter] - recent_support[letter]
        for letter in recent_support
    }
    max_heg_letter = _argmax(evidence_gain)
    current = _predicted_letter(row.get("iter0_predicted_option") or row.get("baseline_prediction"))
    heg_current = evidence_gain.get(current) if current else None
    alternative = {letter: value for letter, value in evidence_gain.items() if letter != current}
    heg_alternative_letter = _argmax(alternative)
    recent_option = _argmax(recent_support)
    historical_option = _argmax(best_historical_support)
    oracle_memory_ids = _oracle_correct_memory_ids(row)
    max_heg_chunk_id = best_historical_chunk_id.get(max_heg_letter) if max_heg_letter else None

    out = {
        "key": row.get("key"),
        "id": row.get("id"),
        "task": task,
        "group": row.get("group"),
        "question": row.get("question"),
        "ground_truth": row.get("ground_truth"),
        "current_predicted_option": current,
        "baseline_prediction": row.get("baseline_prediction"),
        "prism_prediction": row.get("prism_prediction"),
        "k0_correct": bool(row.get("k0_correct")),
        "prism_correct": bool(row.get("prism_correct")),
        "oracle_memory_helpful": row.get("minimum_correct_k") in {1, 2, 3},
        "minimum_correct_k": row.get("minimum_correct_k"),
        "failure_mode": _failure_label(row),
        "prism_used_memory": bool(row.get("prism_memory_chunk_ids")),
        "recent_chunk_ids": recent_ids,
        "candidate_chunk_ids": candidate_ids,
        "decode_backend": backend,
        "recent_support": recent_support,
        "historical_support": historical_support,
        "best_historical_support": best_historical_support,
        "evidence_gain": evidence_gain,
        "HEG": evidence_gain[max_heg_letter] if max_heg_letter else None,
        "HEG_option": max_heg_letter,
        "HEG_candidate_chunk_id": max_heg_chunk_id,
        "HEG_current": heg_current,
        "HEG_alternative": alternative[heg_alternative_letter] if heg_alternative_letter else None,
        "HEG_alternative_option": heg_alternative_letter,
        "historical_option": historical_option,
        "recent_option": recent_option,
        "evidence_conflict": bool(historical_option and recent_option and historical_option != recent_option),
        "historical_option_margin": _top_margin(best_historical_support),
        "HEG_candidate_in_oracle_correct_prefix": bool(
            max_heg_chunk_id is not None and int(max_heg_chunk_id) in oracle_memory_ids
        ),
        "oracle_correct_memory_chunk_ids": sorted(oracle_memory_ids),
        "iter0_answer_margin": row.get("iter0_answer_margin"),
        "iter0_entropy_confidence": row.get("iter0_entropy_confidence"),
        "iter0_visual_support": row.get("iter0_visual_support"),
        "iter0_sufficiency": row.get("iter0_sufficiency"),
        "prism_final_sufficiency": row.get("prism_final_sufficiency"),
        "historical_advantage": row.get("historical_advantage"),
    }
    conflict = bool(out["evidence_conflict"])
    out["HEG_alternative_conflict_score"] = (
        out["HEG_alternative"] if conflict and isinstance(out["HEG_alternative"], (int, float)) else -1.0
    )
    return out


def _rows_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(row)
    return dict(grouped)


def _distribution_block(rows: list[dict[str, Any]], signals: list[str]) -> dict[str, Any]:
    return {
        signal: _numeric_summary([
            value for row in rows if (value := _safe_float(row.get(signal))) is not None
        ])
        for signal in signals
    }


def _boolean_signal_report(labels: list[bool], values: list[bool]) -> dict[str, Any]:
    tp = sum(1 for y, p in zip(labels, values) if y and p)
    fp = sum(1 for y, p in zip(labels, values) if not y and p)
    fn = sum(1 for y, p in zip(labels, values) if y and not p)
    tn = sum(1 for y, p in zip(labels, values) if not y and not p)
    return {
        "true_count": sum(1 for value in values if value),
        "false_count": sum(1 for value in values if not value),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
    }


def _signal_eval(rows: list[dict[str, Any]], signals: list[str]) -> dict[str, Any]:
    target_rows = [row for row in rows if not row.get("k0_correct")]
    labels = [bool(row.get("oracle_memory_helpful")) for row in target_rows]
    out: dict[str, Any] = {
        "population": "Recent-6-wrong MCQ OVO rows",
        "samples": len(target_rows),
        "positives_oracle_memory_helpful": sum(1 for label in labels if label),
        "negatives_oracle_memory_not_helpful": sum(1 for label in labels if not label),
        "signals": {},
    }
    for signal in signals:
        valid_rows = [row for row in target_rows if _safe_float(row.get(signal)) is not None]
        valid_labels = [bool(row.get("oracle_memory_helpful")) for row in valid_rows]
        scores = [float(row[signal]) for row in valid_rows]
        out["signals"][signal] = {
            "samples": len(valid_rows),
            "roc_auc": _roc_auc(valid_labels, scores),
            "thresholds": _threshold_report(valid_labels, scores, include_zero=signal.startswith("HEG") or signal == "historical_advantage"),
        }
    values = [bool(row.get("evidence_conflict")) for row in target_rows]
    out["signals"]["evidence_conflict"] = _boolean_signal_report(labels, values)
    valid_rows = [row for row in target_rows if _safe_float(row.get("HEG_alternative_conflict_score")) is not None]
    out["signals"]["HEG_alternative_combined_with_evidence_conflict"] = {
        "samples": len(valid_rows),
        "roc_auc": _roc_auc(
            [bool(row.get("oracle_memory_helpful")) for row in valid_rows],
            [float(row["HEG_alternative_conflict_score"]) for row in valid_rows],
        ),
        "thresholds": _threshold_report(
            [bool(row.get("oracle_memory_helpful")) for row in valid_rows],
            [float(row["HEG_alternative_conflict_score"]) for row in valid_rows],
            include_zero=True,
        ),
        "definition": "score = HEG_alternative when evidence_conflict else -1",
    }
    return out


def _best_rule(signal_report: dict[str, Any]) -> dict[str, Any]:
    best_name = None
    best_auc = -1.0
    for name, payload in signal_report.get("signals", {}).items():
        auc = payload.get("roc_auc") if isinstance(payload, dict) else None
        if isinstance(auc, (int, float)) and auc > best_auc:
            best_name = name
            best_auc = float(auc)
    if best_name is None:
        return {"rule": None, "reason": "No scalar signal had a defined ROC-AUC."}
    thresholds = signal_report["signals"][best_name].get("thresholds", [])
    balanced = []
    for item in thresholds:
        precision = item.get("precision")
        recall = item.get("recall")
        if isinstance(precision, (int, float)) and isinstance(recall, (int, float)):
            balanced.append((2 * precision * recall / (precision + recall + 1e-12), item))
    threshold = max(balanced, key=lambda item: item[0])[1] if balanced else None
    return {
        "best_signal_by_auc": best_name,
        "roc_auc": best_auc,
        "suggested_simple_rule": (
            f"retrieve/use history when {best_name} >= {threshold['threshold']:.6f}"
            if threshold is not None
            else f"rank by {best_name}"
        ),
        "threshold_metrics": threshold,
        "caution": "This is diagnostic only; no classifier was trained and no PRISM code was changed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-run", type=Path, required=True, help="Oracle output directory or oracle_rows.jsonl.")
    parser.add_argument("--anno-path", type=Path, default=Path("data/ovo_bench/ovo_bench_new.json"))
    parser.add_argument("--chunked-dir", type=Path, default=Path("data/ovo_bench/chunked_videos"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--clip-model", default=os.environ.get("MINICPM_PSM_CLIP_MODEL", "openai/clip-vit-base-patch32"))
    parser.add_argument("--clip-device", default=os.environ.get("MINICPM_PSM_CLIP_DEVICE", "cuda:0"))
    parser.add_argument("--recent-window", type=int, default=6)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--require-official-sanity", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = _find_oracle_rows(args.oracle_run)
    summary = _maybe_load_summary(args.oracle_run)
    sanity = _check_official_sanity(summary, require=args.require_official_sanity)
    rows = _load_jsonl(rows_path)
    annotations = _annotations_by_id(args.anno_path)
    mcq_rows = [row for row in rows if str(row.get("task") or "") in MCQ_TASKS]
    if args.max_samples > 0:
        mcq_rows = mcq_rows[: args.max_samples]

    from lib.minicpm.referential_memory import AnswerGroundedFrameScorer

    scorer = AnswerGroundedFrameScorer(model_name=args.clip_model, device=args.clip_device)
    evidence_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, row in enumerate(mcq_rows, start=1):
        try:
            evidence = _compute_row_evidence(
                row=row,
                annotations=annotations,
                chunked_dir=args.chunked_dir,
                scorer=scorer,
                recent_window=args.recent_window,
                max_candidates=args.max_candidates,
                chunk_duration=args.chunk_duration,
                fps=args.fps,
            )
            if evidence is not None:
                evidence_rows.append(evidence)
        except Exception as exc:  # noqa: BLE001
            errors.append({"key": row.get("key"), "id": row.get("id"), "task": row.get("task"), "error": str(exc), "error_type": type(exc).__name__})
        if index % 25 == 0:
            print(f"[{index}/{len(mcq_rows)}] analyzed={len(evidence_rows)} errors={len(errors)}", flush=True)

    signals = [
        "iter0_answer_margin",
        "iter0_entropy_confidence",
        "iter0_visual_support",
        "iter0_sufficiency",
        "historical_advantage",
        "HEG",
        "HEG_current",
        "HEG_alternative",
        "historical_option_margin",
    ]
    signal_report = _signal_eval(evidence_rows, signals)
    groups = {
        "false_stops": [row for row in evidence_rows if row.get("failure_mode") == "C_false_stop_oracle_memory_could_rescue" or row.get("failure_mode") == "false_stop"],
        "successful_retrievals": [row for row in evidence_rows if row.get("failure_mode") == "B_retrieved_and_rescued" or row.get("failure_mode") == "successful_retrieval"],
        "damaged_retrievals": [row for row in evidence_rows if row.get("failure_mode") == "E_retrieved_and_damaged" or row.get("failure_mode") == "damaged_retrieval"],
        "oracle_memory_helpful": [row for row in evidence_rows if (not row.get("k0_correct")) and row.get("oracle_memory_helpful")],
        "oracle_memory_not_helpful": [row for row in evidence_rows if (not row.get("k0_correct")) and not row.get("oracle_memory_helpful")],
    }
    max_heg_rows = [row for row in evidence_rows if not row.get("k0_correct") and row.get("oracle_memory_helpful")]
    max_heg_hits = sum(1 for row in max_heg_rows if row.get("HEG_candidate_in_oracle_correct_prefix"))
    report = {
        "inputs": {
            "oracle_rows": str(rows_path),
            "annotation_path": str(args.anno_path),
            "chunked_dir": str(args.chunked_dir),
            "clip_model": args.clip_model,
            "clip_device": args.clip_device,
            "recent_window": args.recent_window,
            "max_candidates": args.max_candidates,
        },
        "official_sanity": sanity,
        "population": {
            "oracle_rows_total": len(rows),
            "mcq_rows_requested": len(mcq_rows),
            "mcq_rows_analyzed": len(evidence_rows),
            "errors": len(errors),
            "task_counts": dict(Counter(str(row.get("task")) for row in evidence_rows)),
            "failure_mode_counts": dict(Counter(str(row.get("failure_mode")) for row in evidence_rows)),
        },
        "distributions_overall": _distribution_block(evidence_rows, signals + ["HEG_alternative_conflict_score"]),
        "distributions_by_comparison_group": {
            name: {"samples": len(items), **_distribution_block(items, signals + ["HEG_alternative_conflict_score"])}
            for name, items in groups.items()
        },
        "signal_prediction_recent6_wrong": signal_report,
        "max_heg_candidate_oracle_alignment": {
            "population": "Recent-6-wrong and oracle-memory-helpful MCQ rows",
            "samples": len(max_heg_rows),
            "max_heg_candidate_in_any_oracle_correct_prefix": max_heg_hits,
            "rate": max_heg_hits / len(max_heg_rows) if max_heg_rows else None,
        },
        "best_signal_or_rule": _best_rule(signal_report),
        "notes": [
            "Signals are computed from question/options and frozen Recent-6/history frames only.",
            "Ground truth is used only to define oracle_memory_helpful labels from completed oracle branches.",
            "No classifier was trained and MiniCPM predictions were not regenerated.",
        ],
    }

    (args.out_dir / "option_evidence_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in evidence_rows),
        encoding="utf-8",
    )
    if errors:
        (args.out_dir / "option_evidence_errors.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in errors),
            encoding="utf-8",
        )
    with (args.out_dir / "option_evidence_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "key", "id", "task", "group", "failure_mode", "ground_truth",
            "current_predicted_option", "baseline_prediction", "prism_prediction",
            "k0_correct", "prism_correct", "oracle_memory_helpful", "minimum_correct_k",
            "prism_used_memory", "HEG", "HEG_option", "HEG_candidate_chunk_id",
            "HEG_current", "HEG_alternative", "HEG_alternative_option",
            "historical_option", "recent_option", "evidence_conflict",
            "historical_option_margin", "HEG_candidate_in_oracle_correct_prefix",
            "iter0_answer_margin", "iter0_entropy_confidence", "iter0_visual_support",
            "iter0_sufficiency", "prism_final_sufficiency", "historical_advantage",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in evidence_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    (args.out_dir / "option_evidence_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"saved: {args.out_dir}")


if __name__ == "__main__":
    main()
