#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from lib.minicpm.recursive_story_memory import RecursiveStoryMemoryQAModel, query_recent_window
from main_experiments.minicpm_v46.streamingbench import eval_baseline_dist as dist_sb


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


def _print_rsm_summary(results: list[dict], frame_selection: str = "recent") -> None:
    summary = dist_sb.compute_summary(results)
    label = "All-Frames" if frame_selection == "all" else "Recent-Window"
    recent = os.environ.get("MINICPM_RSM_RECENT_FRAMES", "6")
    batch = os.environ.get("MINICPM_RSM_UPDATE_BATCH", "4")
    l_max = os.environ.get("MINICPM_RSM_MAX_STORY_TOKENS", "256")
    print("\n" + "=" * 60)
    print(
        f"StreamingBench {label} Results "
        f"(MiniCPM-V-4.6 + RecursiveStoryMemory(recent={recent}, batch={batch}, L_max={l_max}))"
    )
    print("=" * 60)
    for row in summary["tasks"]:
        print(f"  {row['task_type']}: {row['accuracy']:.2f}% ({row['correct']}/{row['total']})")
    overall = summary["overall"]
    print(f"\n  Overall: {overall['accuracy']:.2f}% ({overall['correct']}/{overall['total']})")
    print(f"  Errors: {summary['error_count']}")
    print("=" * 60)


def main() -> None:
    _consume_rsm_args()
    dist_sb.RecentWindowQAModel = RecursiveStoryMemoryQAModel
    dist_sb.query_recent_window = query_recent_window
    dist_sb.print_summary = _print_rsm_summary
    dist_sb.main()


if __name__ == "__main__":
    main()
