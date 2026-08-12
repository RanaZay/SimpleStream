#!/usr/bin/env python3
"""Build a fixed OVO MCQ subset for progressive_sufficiency_memory_heg sweeps."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ovo_constants import BACKWARD_TASKS, REAL_TIME_TASKS  # noqa: E402

DEFAULT_REQUIRED_TASKS = ("EPM", "HLD", "ASI", "STU", "ATR", "FPD")


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def _stable_key(item: dict[str, Any]) -> str:
    return "|".join(
        str(item.get(key, ""))
        for key in ("id", "video", "task", "question")
    )


def _round_robin_fill(
    selected: list[dict[str, Any]],
    selected_keys: set[str],
    pools: dict[str, list[dict[str, Any]]],
    task_order: list[str],
    target_total: int,
) -> None:
    exhausted = False
    while len(selected) < target_total and not exhausted:
        exhausted = True
        for task in task_order:
            pool = pools.get(task, [])
            while pool:
                item = pool.pop(0)
                key = _stable_key(item)
                if key not in selected_keys:
                    selected.append(item)
                    selected_keys.add(key)
                    exhausted = False
                    break
            if len(selected) >= target_total:
                break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno-path", type=Path, default=Path("data/ovo_bench/ovo_bench_new.json"))
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--target-total", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--required-tasks", default=",".join(DEFAULT_REQUIRED_TASKS))
    parser.add_argument("--min-per-required-task", type=int, default=50)
    args = parser.parse_args()

    if args.target_total < 1:
        raise ValueError("--target-total must be positive")
    if args.min_per_required_task < 1:
        raise ValueError("--min-per-required-task must be positive")

    annotations = _load_json(args.anno_path)
    mcq_tasks = set(BACKWARD_TASKS + REAL_TIME_TASKS)
    required_tasks = [task.strip() for task in args.required_tasks.split(",") if task.strip()]
    unknown = sorted(set(required_tasks) - mcq_tasks)
    if unknown:
        raise ValueError(f"Required tasks are not OVO MCQ tasks: {unknown}")

    rng = random.Random(args.seed)
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    original_index: dict[str, int] = {}
    for index, item in enumerate(annotations):
        task = item.get("task")
        if task in mcq_tasks:
            pools[str(task)].append(item)
            original_index[_stable_key(item)] = index
    for pool in pools.values():
        rng.shuffle(pool)

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    per_task_target = min(args.min_per_required_task, max(1, args.target_total // max(1, len(required_tasks))))
    for task in required_tasks:
        for item in pools.get(task, [])[:per_task_target]:
            key = _stable_key(item)
            if key not in selected_keys:
                selected.append(item)
                selected_keys.add(key)

    realtime_priority = [task for task in REAL_TIME_TASKS if task in pools]
    backward_priority = [task for task in BACKWARD_TASKS if task in pools]
    fill_order = [*realtime_priority, *backward_priority]
    _round_robin_fill(selected, selected_keys, pools, fill_order, args.target_total)

    if len(selected) < args.target_total:
        raise RuntimeError(
            f"Only selected {len(selected)} MCQ samples, fewer than requested {args.target_total}"
        )

    selected = sorted(selected[: args.target_total], key=lambda item: original_index.get(_stable_key(item), 10**12))
    counts = Counter(str(item.get("task")) for item in selected)
    split_counts = {
        "backward": sum(counts[task] for task in BACKWARD_TASKS),
        "realtime": sum(counts[task] for task in REAL_TIME_TASKS),
    }
    missing_required = [task for task in required_tasks if counts[task] == 0]
    if missing_required:
        raise RuntimeError(f"Subset is missing required tasks: {missing_required}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_path = args.out_path.with_suffix(".summary.json")
    summary = {
        "anno_path": str(args.anno_path),
        "out_path": str(args.out_path),
        "target_total": args.target_total,
        "selected_total": len(selected),
        "seed": args.seed,
        "required_tasks": required_tasks,
        "min_per_required_task": args.min_per_required_task,
        "task_counts": dict(sorted(counts.items())),
        "split_counts": split_counts,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
