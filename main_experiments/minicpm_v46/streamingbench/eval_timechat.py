#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from lib.minicpm.timechat import (
    TimeChatMiniCPMQAModel,
    query_all_frames,
    query_recent_window,
)
from main_experiments.minicpm_v46.streamingbench import eval_baseline as base_sb


def _consume_timechat_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--timechat-retention",
        "--timechat_retention",
        dest="timechat_retention",
        type=float,
        default=float(os.environ.get("MINICPM_TIMECHAT_RETENTION_RATIO", "0.8")),
        help="TimeChat-Online DTD visual-token retention ratio.",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    os.environ["MINICPM_TIMECHAT_RETENTION_RATIO"] = str(args.timechat_retention)
    return args


def _print_timechat_summary(results: list[dict], frame_selection: str = "recent") -> None:
    summary = base_sb.compute_summary(results)
    print("\n" + "=" * 60)
    label = "All-Frames" if frame_selection == "all" else "Recent-Window"
    retention_percent = int(round(float(os.environ.get("MINICPM_TIMECHAT_RETENTION_RATIO", "0.8")) * 100))
    print(f"StreamingBench {label} Results (MiniCPM-V-4.6 + TimeChat-DTD(retention={retention_percent}%))")
    print("=" * 60)
    for row in summary["tasks"]:
        print(f"  {row['task_type']}: {row['accuracy']:.2f}% ({row['correct']}/{row['total']})")
    overall = summary["overall"]
    print(f"\n  Overall: {overall['accuracy']:.2f}% ({overall['correct']}/{overall['total']})")
    print(f"  Errors: {summary['error_count']}")
    print("=" * 60)


def main() -> None:
    _consume_timechat_args()
    base_sb.RecentWindowQAModel = TimeChatMiniCPMQAModel
    base_sb.query_all_frames = query_all_frames
    base_sb.query_recent_window = query_recent_window
    base_sb.print_summary = _print_timechat_summary
    base_sb.main()


if __name__ == "__main__":
    main()

