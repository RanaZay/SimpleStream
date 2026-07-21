#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from lib.minicpm import baseline as baseline_mod
from lib.minicpm.story_memory import StoryMemoryQAModel, query_recent_window
from main_experiments.minicpm_v46.ovo import eval_baseline as ovo_eval


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


def main() -> None:
    story_args = _consume_story_args()
    baseline_mod.RecentWindowQAModel = StoryMemoryQAModel
    ovo_eval.RecentWindowQAModel = StoryMemoryQAModel
    baseline_mod.query_recent_window = query_recent_window
    ovo_eval.query_recent_window = query_recent_window
    ovo_eval.MODEL_LABEL = (
        "MiniCPM-V-4.6 + TextualStoryMemory"
        f"(recent={story_args.story_recent_frames}, stride={story_args.story_describe_stride})"
    )
    ovo_eval.main()


if __name__ == "__main__":
    main()
