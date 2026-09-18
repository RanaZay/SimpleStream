"""Run the pinned StreamChat author judge and reject incomplete evaluation outputs.

This deliberately does not use our legacy yes/no-only judge. Judge time is an
offline evaluation cost, not PRISM inference latency.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from importlib import metadata as package_metadata
import json
import math
from pathlib import Path
import subprocess
import sys

AUTHOR_COMMIT = "f8bcd22ef68cb69f59bcae29ed9fe93f75c077be"
SUBTASKS = {"OS", "LM", "SM", "CI", "KG", "SF"}


def index_rows(rows, expected):
    if expected <= 0:
        raise ValueError('Expected question count must be positive')
    indexed = {}
    for row in rows:
        key = row.get("id")
        if not isinstance(key, str) or not key or key in indexed:
            raise ValueError(f"Missing or duplicate question ID: {key!r}")
        if row.get("class") not in SUBTASKS:
            raise ValueError(f"Unknown subtask for {key}")
        for field in ("question", "label", "predict"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing {field} for {key}; do not silently drop failed inference")
        indexed[key] = row
    if len(indexed) != expected:
        raise ValueError(f"Expected {expected} unique questions, got {len(indexed)}")
    return indexed


def audit(inputs, outputs, expected=1838):
    source = index_rows(inputs, expected)
    judged = index_rows(outputs, expected)
    if source.keys() != judged.keys():
        raise ValueError("Judge coverage does not match input IDs")
    groups = defaultdict(list)
    for key, row in judged.items():
        for field in ("question", "label", "predict", "class"):
            if row[field] != source[key][field]:
                raise ValueError(f"Judge changed {field} for {key}")
        verdict = row.get("llama_pred")
        score = row.get("score")
        if verdict not in ("yes", "no"):
            raise ValueError(f"Invalid judge verdict for {key}: {verdict!r}")
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= 5):
            raise ValueError(f"Invalid judge score for {key}: {score!r}")
        groups[row["class"]].append(verdict == "yes")
    correct = sum(sum(values) for values in groups.values())
    per_task = {task: {"correct": sum(values), "n": len(values),
                       "accuracy_percent": 100 * sum(values) / len(values)}
                for task, values in sorted(groups.items())}
    return {"n": expected, "correct": correct, "accuracy_percent": 100 * correct / expected,
            "subtasks": per_task, "complete": True,
            "metric": "author_judge_yes_fraction", "judge_time_in_latency": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Pinned hmxiong/StreamChat checkout")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-model", required=True, help="Prefer a fixed local model snapshot")
    parser.add_argument("--expected", type=int, default=1838)
    args = parser.parse_args()
    source = args.source.resolve()
    script = source / "eval_video_qa_with_llama3_ours.py"
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != AUTHOR_COMMIT:
        raise ValueError(f"Expected StreamChat commit {AUTHOR_COMMIT}, got {commit}")
    subprocess.run(["git", "-C", str(source), "diff", "--exit-code", "HEAD", "--", script.name], check=True)
    inputs = json.loads(args.input.read_text())
    index_rows(inputs, args.expected)
    # An immutable output directory prevents stale judgments from masquerading as a new run.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"author_commit": commit, "judge_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                "judge_model": args.judge_model, "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "expected_questions": args.expected, "python": sys.executable,
                "protocol": "unmodified_author_script", "status": "running"}
    manifest['packages'] = {}
    for package in ('torch', 'transformers', 'accelerate'):
        try:
            manifest['packages'][package] = package_metadata.version(package)
        except package_metadata.PackageNotFoundError:
            manifest['packages'][package] = None
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    command = [sys.executable, str(script), "--predict_file", str(args.input.resolve()),
               "--output_dir", str(args.output_dir.resolve()), "--output_name", "judged",
               "--llama3_path", args.judge_model, "--num_chunks", "1", "--chunk_idx", "0"]
    try:
        subprocess.run(command, check=True)
        outputs = [json.loads(line) for line in (args.output_dir / "judged.json").read_text().splitlines() if line.strip()]
        summary = audit(inputs, outputs, args.expected)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        manifest["status"] = "complete"
        print(json.dumps(summary, indent=2))
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
