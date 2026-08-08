#!/usr/bin/env python3
"""Evaluate Recent-6 plus prefixes of the PSM candidate queue on baseline errors."""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from lib.minicpm.adaptive import AdaptiveWindowConfig  # noqa: E402
from lib.minicpm.baseline import RecentWindowQAModel  # noqa: E402
from lib.minicpm.progressive_sufficiency import (  # noqa: E402
    _PSM_HISTORY_INSTRUCTION,
    _candidate_metadata,
    _rank_candidates,
)
from lib.shared.recent_window import decode_video_to_chunks_qwen, extract_mcq_answer  # noqa: E402
from main_experiments.minicpm_v46.streamingbench.eval_baseline import (  # noqa: E402
    build_prompt,
    timestamp_to_seconds,
)


def _options(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return []


def _video(source: dict[str, Any], video_dir: Path) -> Path:
    for key in ("video_path", "video_path_raw", "video"):
        value = source.get(key)
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        for candidate in ([path] if path.is_absolute() else []) + [video_dir / path.name, video_dir / value]:
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Cannot resolve video for {source.get('_key', source.get('video'))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-jsonl", type=Path, required=True, help="Baseline-wrong StreamingBench JSONL with options.")
    parser.add_argument("--video-dir", type=Path, default=Path("data/streamingbench/videos"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qa-model", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--qa-device", default="auto")
    parser.add_argument("--max-examples", type=int, default=0)
    args = parser.parse_args()

    with args.examples_jsonl.open(encoding="utf-8") as handle:
        sources = [json.loads(line) for line in handle if line.strip()]
    sources = [row for row in sources if row.get("correct") is not True]
    if args.max_examples > 0:
        sources = sources[: args.max_examples]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    qa = RecentWindowQAModel(
        model_name=args.qa_model,
        device=args.qa_device,
        max_new_tokens=256,
        attn_implementation=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
    )
    config = AdaptiveWindowConfig(
        mode="progressive_sufficiency_memory",
        min_window=6,
        mid_window=6,
        max_window=6,
        memory_anchors=3,
        memory_search_chunks=64,
    )
    config.validate()
    minimum_counts: Counter[str] = Counter()
    output_path = args.output_dir / "oracle_results.jsonl"
    with output_path.open("w", encoding="utf-8") as output:
        for index, source in enumerate(sources, start=1):
            options = _options(source.get("options"))
            if len(options) < 2:
                raise ValueError(f"Example {index} has no usable options: {source.get('_key')}")
            prompt = build_prompt({"question": source.get("question", ""), "options": options})
            timestamp = float(timestamp_to_seconds(str(source.get("time_stamp", "0:00:00"))))
            video_path = _video(source, args.video_dir)
            chunks, backend = decode_video_to_chunks_qwen(
                video_path=str(video_path),
                chunk_duration=1.0,
                fps=1.0,
                recent_frames_only=70,
                video_start=max(0.0, timestamp - 70.0),
                video_end=timestamp + 1e-4,
            )
            recent = chunks[-6:]
            older = chunks[:-6][-64:]
            queue, ranking_ms = _rank_candidates(qa, older, prompt, config, 12, 2)
            answer_gt = extract_mcq_answer(str(source.get("answer_gt") or source.get("answer") or ""))
            branches: list[dict[str, Any]] = []
            minimum: int | None = None
            for k in range(4):
                memory = sorted([item["chunk"] for item in queue[:k]], key=lambda chunk: int(chunk.chunk_index))
                context = [*memory, *recent]
                answer_prompt = f"{_PSM_HISTORY_INSTRUCTION}{prompt}" if memory else prompt
                response = qa.generate_from_frames([frame for chunk in context for frame in chunk.frames], answer_prompt)
                prediction = extract_mcq_answer(response)
                correct = bool(answer_gt and prediction == answer_gt)
                if correct and minimum is None:
                    minimum = k
                branches.append(
                    {
                        "k": k,
                        "chunk_ids": [int(chunk.chunk_index) for chunk in context],
                        "response": response,
                        "prediction": prediction,
                        "correct": correct,
                    }
                )
            minimum_counts[str(minimum) if minimum is not None else "not_rescued"] += 1
            record = {
                "_key": source.get("_key", f"oracle:{index}"),
                "video": str(video_path),
                "question": source.get("question", ""),
                "options": options,
                "answer_gt": answer_gt,
                "decode_backend": backend,
                "candidate_ranking_ms": ranking_ms,
                "candidate_queue": [_candidate_metadata(item) for item in queue],
                "branches": branches,
                "minimum_candidates_to_correct": minimum,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(f"[{index}/{len(sources)}] minimum_candidates={minimum}", flush=True)

    summary = {"samples": len(sources), "minimum_candidates_distribution": dict(minimum_counts)}
    (args.output_dir / "oracle_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
