#!/usr/bin/env python3
"""Export StreamBench-v0.3 answers to official judge input format.

The local StreamBench-v0.3 runner saves one row per breakpoint with a
``methods`` list.  The official StreamBench evaluation judges each open-ended
answer against the reference answer with an LLM judge.  This script reshapes our
saved outputs into one JSON file per method with the fields expected by the
judge: question, label, predict, and class.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_result_files(paths: list[Path]) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, int, str], dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("results", []):
            key = (
                int(row.get("video_index", -1)),
                int(row.get("breakpoint_index", -1)),
                str(row.get("video_name", "")),
            )
            if key not in by_key:
                by_key[key] = dict(row)
                by_key[key]["methods"] = []
            existing_methods = {
                str(item.get("method"))
                for item in by_key[key].get("methods", [])
                if isinstance(item, dict)
            }
            for item in row.get("methods", []):
                method = str(item.get("method")) if isinstance(item, dict) else ""
                if method and method not in existing_methods:
                    by_key[key]["methods"].append(item)
                    existing_methods.add(method)
    rows = list(by_key.values())
    rows.sort(
        key=lambda row: (
            int(row.get("video_index", -1)),
            int(row.get("breakpoint_index", -1)),
            int(row.get("_index", -1)),
        )
    )
    return rows


def _iter_method_rows(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        selected = None
        for item in row.get("methods", []):
            if item.get("method") == method:
                selected = item
                break
        if selected is None:
            continue
        if selected.get("error"):
            continue
        prediction = str(selected.get("prediction") or "").strip()
        if not prediction:
            continue
        out.append(
            {
                "id": f"{row.get('video_index')}:{row.get('breakpoint_index')}",
                "video_index": row.get("video_index"),
                "breakpoint_index": row.get("breakpoint_index"),
                "video_name": row.get("video_name"),
                "class": row.get("subtask"),
                "class_1": row.get("class_1"),
                "class_2": row.get("class_2"),
                "time": row.get("time"),
                "question": row.get("question"),
                "label": row.get("answer_gt"),
                "predict": prediction,
                "method": method,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        required=True,
        help="One or more streambench_v0_3_smoke_results.json files.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["recent6", "prism"])
    args = parser.parse_args()

    rows = _load_result_files(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"loaded_rows: {len(rows)}")

    for method in args.methods:
        method_rows = _iter_method_rows(rows, method)
        out_path = args.out_dir / f"streambench_v0_3_{method}_official_input.json"
        out_path.write_text(json.dumps(method_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{method}: {len(method_rows)} -> {out_path}")


if __name__ == "__main__":
    main()
