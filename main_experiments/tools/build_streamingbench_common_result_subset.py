#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


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
    return []


def _flatten(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row.get("test_info"), list):
            for index, item in enumerate(row["test_info"]):
                if isinstance(item, dict):
                    flat.append({**row, **item, "_question_index": index})
        else:
            flat.append(row)
    return flat


def _load_results(path: Path) -> list[dict[str, Any]]:
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
    raise FileNotFoundError(f"No result rows found under {path}")


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _video_basename(value: Any) -> str:
    return Path(str(value or "")).name


def _result_key(row: dict[str, Any]) -> tuple[str, str, str]:
    video = row.get("video") or row.get("video_path") or row.get("video_path_raw") or row.get("id")
    task = row.get("task_type") or row.get("task") or row.get("category")
    return (_video_basename(video), _norm(task), _norm(row.get("question")))


def _question_key(video_path: Any, question: dict[str, Any]) -> tuple[str, str, str]:
    task = question.get("task_type") or question.get("task") or question.get("category")
    return (_video_basename(video_path), _norm(task), _norm(question.get("question")))


def _load_annotation(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a StreamingBench annotation list")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export the exact StreamingBench annotation subset represented by the "
            "intersection of two result directories. The output can be used as SB_ANNO_PATH."
        )
    )
    parser.add_argument("--anno-path", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--source-prism", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    args = parser.parse_args()

    baseline_keys = {_result_key(row) for row in _load_results(args.baseline)}
    source_keys = {_result_key(row) for row in _load_results(args.source_prism)}
    target_keys = baseline_keys & source_keys

    annotations = _load_annotation(args.anno_path)
    subset: list[dict[str, Any]] = []
    matched_keys: set[tuple[str, str, str]] = set()
    for entry in annotations:
        video_path = entry.get("video_path")
        questions = [
            question
            for question in entry.get("questions", [])
            if _question_key(video_path, question) in target_keys
        ]
        if not questions:
            continue
        for question in questions:
            matched_keys.add(_question_key(video_path, question))
        copied = dict(entry)
        copied["questions"] = questions
        subset.append(copied)

    missing = sorted(target_keys - matched_keys)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(subset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "annotation_videos": len(subset),
        "annotation_questions": sum(len(entry.get("questions", [])) for entry in subset),
        "baseline_unique_keys": len(baseline_keys),
        "source_unique_keys": len(source_keys),
        "common_result_keys": len(target_keys),
        "matched_annotation_keys": len(matched_keys),
        "missing_annotation_keys": len(missing),
        "missing_examples": [
            {"video": video, "task": task, "question": question}
            for video, task, question in missing[:20]
        ],
        "out_path": str(args.out_path),
    }
    summary_path = args.out_path.with_suffix(args.out_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
