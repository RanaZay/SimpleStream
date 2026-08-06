#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from lib.minicpm.recent_description import RecentDescriptionQAModel, query_recent_window
from main_experiments.minicpm_v46.streamingbench import eval_baseline_dist as dist_sb


def _consume_recent_description_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--recent-desc-frames",
        type=int,
        default=int(os.environ.get("MINICPM_RECENT_DESC_FRAMES", "6")),
    )
    parser.add_argument(
        "--recent-desc-max-tokens",
        type=int,
        default=int(os.environ.get("MINICPM_RECENT_DESC_MAX_TOKENS", "256")),
    )
    parser.add_argument(
        "--recent-desc-max-words",
        type=int,
        default=int(os.environ.get("MINICPM_RECENT_DESC_MAX_WORDS", "45")),
    )
    parser.add_argument(
        "--recent-desc-max-prompt-chars",
        type=int,
        default=int(os.environ.get("MINICPM_RECENT_DESC_MAX_PROMPT_CHARS", "6000")),
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    os.environ["MINICPM_RECENT_DESC_FRAMES"] = str(args.recent_desc_frames)
    os.environ["MINICPM_RECENT_DESC_MAX_TOKENS"] = str(args.recent_desc_max_tokens)
    os.environ["MINICPM_RECENT_DESC_MAX_WORDS"] = str(args.recent_desc_max_words)
    os.environ["MINICPM_RECENT_DESC_MAX_PROMPT_CHARS"] = str(args.recent_desc_max_prompt_chars)
    return args


def _print_recent_description_summary(results: list[dict], frame_selection: str = "recent") -> None:
    summary = dist_sb.compute_summary(results)
    label = "All-Frames" if frame_selection == "all" else "Recent-Window"
    recent = os.environ.get("MINICPM_RECENT_DESC_FRAMES", "6")
    print("\n" + "=" * 60)
    print(f"StreamingBench {label} Results (MiniCPM-V-4.6 + RecentFrameDescriptions(recent={recent}))")
    print("=" * 60)
    for row in summary["tasks"]:
        print(f"  {row['task_type']}: {row['accuracy']:.2f}% ({row['correct']}/{row['total']})")
    overall = summary["overall"]
    print(f"\n  Overall: {overall['accuracy']:.2f}% ({overall['correct']}/{overall['total']})")
    print(f"  Errors: {summary['error_count']}")
    print("=" * 60)


def main() -> None:
    _consume_recent_description_args()
    dist_sb.RecentWindowQAModel = RecentDescriptionQAModel
    dist_sb.query_recent_window = query_recent_window
    dist_sb.print_summary = _print_recent_description_summary
    dist_sb.main()


if __name__ == "__main__":
    main()
