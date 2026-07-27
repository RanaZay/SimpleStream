#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from lib.minicpm import baseline as baseline_mod
from lib.minicpm.recursive_story_memory import RecursiveStoryMemoryQAModel, query_recent_window
from main_experiments.minicpm_v46.ovo import eval_baseline as ovo_eval


def _consume_rsm_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rsm-recent-frames", type=int, default=int(os.environ.get("MINICPM_RSM_RECENT_FRAMES", "6")))
    parser.add_argument("--rsm-update-batch", type=int, default=int(os.environ.get("MINICPM_RSM_UPDATE_BATCH", "4")))
    parser.add_argument("--rsm-max-story-tokens", type=int, default=int(os.environ.get("MINICPM_RSM_MAX_STORY_TOKENS", "256")))
    parser.add_argument(
        "--rsm-rewrite-max-new-tokens",
        type=int,
        default=int(os.environ.get("MINICPM_RSM_REWRITE_MAX_NEW_TOKENS", "384")),
    )
    parser.add_argument(
        "--rsm-compress-max-new-tokens",
        type=int,
        default=int(os.environ.get("MINICPM_RSM_COMPRESS_MAX_NEW_TOKENS", "320")),
    )
    parser.add_argument(
        "--rsm-max-compression-attempts",
        type=int,
        default=int(os.environ.get("MINICPM_RSM_MAX_COMPRESSION_ATTEMPTS", "1")),
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    os.environ["MINICPM_RSM_RECENT_FRAMES"] = str(args.rsm_recent_frames)
    os.environ["MINICPM_RSM_UPDATE_BATCH"] = str(args.rsm_update_batch)
    os.environ["MINICPM_RSM_MAX_STORY_TOKENS"] = str(args.rsm_max_story_tokens)
    os.environ["MINICPM_RSM_REWRITE_MAX_NEW_TOKENS"] = str(args.rsm_rewrite_max_new_tokens)
    os.environ["MINICPM_RSM_COMPRESS_MAX_NEW_TOKENS"] = str(args.rsm_compress_max_new_tokens)
    os.environ["MINICPM_RSM_MAX_COMPRESSION_ATTEMPTS"] = str(args.rsm_max_compression_attempts)
    return args


def main() -> None:
    rsm_args = _consume_rsm_args()
    baseline_mod.RecentWindowQAModel = RecursiveStoryMemoryQAModel
    ovo_eval.RecentWindowQAModel = RecursiveStoryMemoryQAModel
    baseline_mod.query_recent_window = query_recent_window
    ovo_eval.query_recent_window = query_recent_window
    ovo_eval.MODEL_LABEL = (
        "MiniCPM-V-4.6 + RecursiveStoryMemory"
        f"(recent={rsm_args.rsm_recent_frames}, batch={rsm_args.rsm_update_batch}, "
        f"L_max={rsm_args.rsm_max_story_tokens})"
    )
    ovo_eval.main()


if __name__ == "__main__":
    main()
