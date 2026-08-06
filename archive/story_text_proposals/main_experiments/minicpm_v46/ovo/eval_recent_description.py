#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from lib.minicpm import baseline as baseline_mod
from lib.minicpm.recent_description import RecentDescriptionQAModel, query_recent_window
from main_experiments.minicpm_v46.ovo import eval_baseline as ovo_eval


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


def main() -> None:
    args = _consume_recent_description_args()
    baseline_mod.RecentWindowQAModel = RecentDescriptionQAModel
    ovo_eval.RecentWindowQAModel = RecentDescriptionQAModel
    baseline_mod.query_recent_window = query_recent_window
    ovo_eval.query_recent_window = query_recent_window
    ovo_eval.MODEL_LABEL = f"MiniCPM-V-4.6 + RecentFrameDescriptions(recent={args.recent_desc_frames})"
    ovo_eval.main()


if __name__ == "__main__":
    main()
