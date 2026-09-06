#!/usr/bin/env python3
"""Summarize StreamBench-v0.3 official LLaMA-judge output."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

SUBTASK_ORDER = ["OS", "LM", "SM", "CI", "KG", "SF"]


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _acc(rows: list[dict]) -> tuple[int, int, float]:
    total = len(rows)
    correct = sum(int(row.get("score", 0)) for row in rows)
    return correct, total, (100.0 * correct / total if total else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recent6", type=Path, required=True)
    parser.add_argument("--prism", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    runs = {"recent6": _load_jsonl(args.recent6), "prism": _load_jsonl(args.prism)}
    summary: dict[str, dict] = {}
    for method, rows in runs.items():
        by_class = defaultdict(list)
        for row in rows:
            by_class[str(row.get("class", ""))].append(row)
        correct, total, overall = _acc(rows)
        summary[method] = {
            "correct": correct,
            "total": total,
            "accuracy": overall,
            "subtasks": {},
        }
        for subtask in SUBTASK_ORDER:
            c, n, a = _acc(by_class[subtask])
            summary[method]["subtasks"][subtask] = {"correct": c, "total": n, "accuracy": a}

    print("\nStreamBench-v0.3 Official-Style LLaMA-Judge Results")
    print("method | OS | LM | SM | CI | KG | SF | Overall")
    for method in ("recent6", "prism"):
        vals = [f"{summary[method]['subtasks'][s]['accuracy']:.2f}" for s in SUBTASK_ORDER]
        print(f"{method} | " + " | ".join(vals) + f" | {summary[method]['accuracy']:.2f}")
    delta_vals = [
        summary["prism"]["subtasks"][s]["accuracy"] - summary["recent6"]["subtasks"][s]["accuracy"]
        for s in SUBTASK_ORDER
    ]
    delta_overall = summary["prism"]["accuracy"] - summary["recent6"]["accuracy"]
    print("delta | " + " | ".join(f"{v:+.2f}" for v in delta_vals) + f" | {delta_overall:+.2f}")

    print("\nCounts")
    for method in ("recent6", "prism"):
        print(f"{method}: {summary[method]['correct']}/{summary[method]['total']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
