#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def _video_basename(value: str | None) -> str:
    return Path(str(value or "")).name


def _question_key(video: str, question: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _video_basename(video),
        str(question.get("time_stamp", "")).strip(),
        re.sub(r"\s+", " ", str(question.get("question", "")).strip()),
    )


def _wrong_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _video_basename(row.get("video") or row.get("video_path") or row.get("video_path_raw")),
        str(row.get("time_stamp", "")).strip(),
        re.sub(r"\s+", " ", str(row.get("question", "")).strip()),
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _select_balanced_videos(wrong_rows: list[dict[str, Any]], max_videos: int) -> list[str]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in wrong_rows:
        by_task[str(row.get("task_type", "Unknown"))].append(row)

    selected: list[str] = []
    selected_set: set[str] = set()
    task_names = sorted(by_task, key=lambda task: len(by_task[task]), reverse=True)

    while len(selected) < max_videos:
        progressed = False
        for task in task_names:
            while by_task[task]:
                row = by_task[task].pop(0)
                video = _video_basename(row.get("video") or row.get("video_path") or row.get("video_path_raw"))
                if video and video not in selected_set:
                    selected.append(video)
                    selected_set.add(video)
                    progressed = True
                    break
            if len(selected) >= max_videos:
                break
        if not progressed:
            break
    return selected


def _filter_questions(
    annotations: list[dict[str, Any]],
    selected_videos: set[str],
    wrong_keys: set[tuple[str, str, str]],
    wrong_only: bool,
) -> list[dict[str, Any]]:
    subset: list[dict[str, Any]] = []
    for entry in annotations:
        video_name = _video_basename(entry.get("video_path"))
        if video_name not in selected_videos:
            continue
        questions = []
        for question in entry.get("questions", []):
            if not wrong_only or _question_key(video_name, question) in wrong_keys:
                questions.append(question)
        if questions:
            copied = dict(entry)
            copied["questions"] = questions
            subset.append(copied)
    return subset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a small StreamingBench failure subset from a wrong-example JSONL. "
            "The output keeps the original StreamingBench annotation format, so it can "
            "be passed directly to eval_adaptive_dist.py via SB_ANNO_PATH."
        )
    )
    parser.add_argument("--wrong-jsonl", type=Path, required=True)
    parser.add_argument("--anno-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/streamingbench_semantic_failure20"))
    parser.add_argument("--max-videos", type=int, default=20)
    args = parser.parse_args()

    wrong_rows = _load_jsonl(args.wrong_jsonl)
    with args.anno_path.open(encoding="utf-8") as handle:
        annotations = json.load(handle)

    selected_videos = _select_balanced_videos(wrong_rows, args.max_videos)
    selected_set = set(selected_videos)
    wrong_keys = {_wrong_key(row) for row in wrong_rows}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    full_video_subset = _filter_questions(annotations, selected_set, wrong_keys, wrong_only=False)
    wrong_only_subset = _filter_questions(annotations, selected_set, wrong_keys, wrong_only=True)

    full_path = args.out_dir / "questions_failure20_full_videos.json"
    wrong_path = args.out_dir / "questions_failure20_wrong_only.json"
    manifest_path = args.out_dir / "manifest.json"

    full_path.write_text(json.dumps(full_video_subset, ensure_ascii=False, indent=2), encoding="utf-8")
    wrong_path.write_text(json.dumps(wrong_only_subset, ensure_ascii=False, indent=2), encoding="utf-8")

    wrong_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in wrong_rows:
        video = _video_basename(row.get("video") or row.get("video_path") or row.get("video_path_raw"))
        if video in selected_set:
            wrong_by_video[video].append(row)

    manifest = {
        "selected_videos": selected_videos,
        "max_videos": args.max_videos,
        "full_video_annotation": str(full_path),
        "wrong_only_annotation": str(wrong_path),
        "videos": [
            {
                "video": video,
                "wrong_count": len(wrong_by_video[video]),
                "wrong_tasks": sorted({str(row.get("task_type", "")) for row in wrong_by_video[video]}),
                "wrong_questions": [
                    {
                        "task_type": row.get("task_type"),
                        "time_stamp": row.get("time_stamp"),
                        "question": row.get("question"),
                        "response": row.get("response"),
                        "answer_gt": row.get("answer_gt"),
                        "selected_chunk_ids": row.get("selected_chunk_ids"),
                    }
                    for row in wrong_by_video[video]
                ],
            }
            for video in selected_videos
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"selected videos: {len(selected_videos)}")
    print(f"full-video questions: {sum(len(entry['questions']) for entry in full_video_subset)}")
    print(f"wrong-only questions: {sum(len(entry['questions']) for entry in wrong_only_subset)}")
    print(f"full-video anno: {full_path}")
    print(f"wrong-only anno: {wrong_path}")
    print(f"manifest: {manifest_path}")
    for video in selected_videos:
        tasks = ", ".join(sorted({str(row.get("task_type", "")) for row in wrong_by_video[video]}))
        print(f"- {video}: {len(wrong_by_video[video])} wrong [{tasks}]")


if __name__ == "__main__":
    main()
