#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from lib.minicpm.story_memory import StoryMemoryQAModel, query_recent_window
from main_experiments.minicpm_v46.streamingbench import eval_baseline_dist as dist_sb


def _consume_story_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--story-recent-frames", type=int, default=int(os.environ.get("MINICPM_STORY_RECENT_FRAMES", "6")))
    parser.add_argument("--story-batch-size", type=int, default=int(os.environ.get("MINICPM_STORY_BATCH_SIZE", "8")))
    parser.add_argument("--story-max-items", type=int, default=int(os.environ.get("MINICPM_STORY_MAX_ITEMS", "96")))
    parser.add_argument("--story-max-prompt-chars", type=int, default=int(os.environ.get("MINICPM_STORY_MAX_PROMPT_CHARS", "9000")))
    parser.add_argument("--story-desc-max-tokens", type=int, default=int(os.environ.get("MINICPM_STORY_DESC_MAX_TOKENS", "192")))
    parser.add_argument("--story-describe-stride", type=int, default=int(os.environ.get("MINICPM_STORY_DESCRIBE_STRIDE", "1")))
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    os.environ["MINICPM_STORY_RECENT_FRAMES"] = str(args.story_recent_frames)
    os.environ["MINICPM_STORY_BATCH_SIZE"] = str(args.story_batch_size)
    os.environ["MINICPM_STORY_MAX_ITEMS"] = str(args.story_max_items)
    os.environ["MINICPM_STORY_MAX_PROMPT_CHARS"] = str(args.story_max_prompt_chars)
    os.environ["MINICPM_STORY_DESC_MAX_TOKENS"] = str(args.story_desc_max_tokens)
    os.environ["MINICPM_STORY_DESCRIBE_STRIDE"] = str(args.story_describe_stride)
    return args


def _print_story_summary(results: list[dict], frame_selection: str = "recent") -> None:
    summary = dist_sb.compute_summary(results)
    label = "All-Frames" if frame_selection == "all" else "Recent-Window"
    recent = os.environ.get("MINICPM_STORY_RECENT_FRAMES", "6")
    stride = os.environ.get("MINICPM_STORY_DESCRIBE_STRIDE", "1")
    print("\n" + "=" * 60)
    print(f"StreamingBench {label} Results (MiniCPM-V-4.6 + TextualStoryMemory(recent={recent}, stride={stride}))")
    print("=" * 60)
    for row in summary["tasks"]:
        print(f"  {row['task_type']}: {row['accuracy']:.2f}% ({row['correct']}/{row['total']})")
    overall = summary["overall"]
    print(f"\n  Overall: {overall['accuracy']:.2f}% ({overall['correct']}/{overall['total']})")
    print(f"  Errors: {summary['error_count']}")
    print("=" * 60)


def main() -> None:
    _consume_story_args()
    dist_sb.RecentWindowQAModel = StoryMemoryQAModel
    dist_sb.query_recent_window = query_recent_window
    dist_sb.print_summary = _print_story_summary
    dist_sb.main()


if __name__ == "__main__":
    main()
