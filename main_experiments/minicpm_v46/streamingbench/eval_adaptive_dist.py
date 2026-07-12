#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from lib.minicpm.adaptive import query_recent_window
from main_experiments.minicpm_v46.streamingbench import eval_baseline_dist as dist_sb


def _consume_adaptive_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--adaptive-mode",
        choices=[
            "adaptive",
            "adaptive_dedup",
            "adaptive_memory",
            "adaptive_dedup_memory",
            "foveated",
            "foveated_memory",
            "online_memory",
            "semantic_memory",
            "semantic_episodic_memory",
            "bound_semantic_episodic_memory",
            "gated_semantic_episodic_memory",
            "strict_gated_semantic_memory",
            "question_aware_memory",
        ],
        default=os.environ.get("MINICPM_ADAPTIVE_MODE", "adaptive"),
    )
    parser.add_argument("--adaptive-min-window", type=int, default=int(os.environ.get("MINICPM_ADAPTIVE_MIN_WINDOW", "4")))
    parser.add_argument("--adaptive-mid-window", type=int, default=int(os.environ.get("MINICPM_ADAPTIVE_MID_WINDOW", "6")))
    parser.add_argument("--adaptive-max-window", type=int, default=int(os.environ.get("MINICPM_ADAPTIVE_MAX_WINDOW", "8")))
    parser.add_argument("--adaptive-dedup-threshold", type=float, default=float(os.environ.get("MINICPM_ADAPTIVE_DEDUP_THRESHOLD", "4.0")))
    parser.add_argument("--adaptive-dedup-min-frames", type=int, default=int(os.environ.get("MINICPM_ADAPTIVE_DEDUP_MIN_FRAMES", "4")))
    parser.add_argument("--adaptive-memory-anchors", type=int, default=int(os.environ.get("MINICPM_ADAPTIVE_MEMORY_ANCHORS", "2")))
    parser.add_argument("--adaptive-memory-search-chunks", type=int, default=int(os.environ.get("MINICPM_ADAPTIVE_MEMORY_SEARCH_CHUNKS", "0")))
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    os.environ["MINICPM_ADAPTIVE_MODE"] = args.adaptive_mode
    os.environ["MINICPM_ADAPTIVE_MIN_WINDOW"] = str(args.adaptive_min_window)
    os.environ["MINICPM_ADAPTIVE_MID_WINDOW"] = str(args.adaptive_mid_window)
    os.environ["MINICPM_ADAPTIVE_MAX_WINDOW"] = str(args.adaptive_max_window)
    os.environ["MINICPM_ADAPTIVE_DEDUP_THRESHOLD"] = str(args.adaptive_dedup_threshold)
    os.environ["MINICPM_ADAPTIVE_DEDUP_MIN_FRAMES"] = str(args.adaptive_dedup_min_frames)
    os.environ["MINICPM_ADAPTIVE_MEMORY_ANCHORS"] = str(args.adaptive_memory_anchors)
    os.environ["MINICPM_ADAPTIVE_MEMORY_SEARCH_CHUNKS"] = str(args.adaptive_memory_search_chunks)
    return args


def _print_adaptive_summary(results: list[dict], frame_selection: str = "recent") -> None:
    summary = dist_sb.compute_summary(results)
    mode = os.environ.get("MINICPM_ADAPTIVE_MODE", "adaptive")
    label = "All-Frames" if frame_selection == "all" else "Recent-Window"
    print("\n" + "=" * 60)
    print(f"StreamingBench {label} Results (MiniCPM-V-4.6 + AdaptiveSimpleStream({mode}))")
    print("=" * 60)
    for row in summary["tasks"]:
        print(f"  {row['task_type']}: {row['accuracy']:.2f}% ({row['correct']}/{row['total']})")
    overall = summary["overall"]
    print(f"\n  Overall: {overall['accuracy']:.2f}% ({overall['correct']}/{overall['total']})")
    print(f"  Errors: {summary['error_count']}")
    print("=" * 60)


def main() -> None:
    _consume_adaptive_args()
    dist_sb.query_recent_window = query_recent_window
    dist_sb.print_summary = _print_adaptive_summary
    dist_sb.main()


if __name__ == "__main__":
    main()
