#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from lib.minicpm.windowquant import (
    WindowQuantConfig,
    WindowQuantMiniCPMQAModel,
    query_all_frames,
    query_recent_window,
)
from main_experiments.minicpm_v46.streamingbench import eval_baseline_dist as dist_sb


def _consume_windowquant_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--wq-window-size", type=int, default=int(os.environ.get("MINICPM_WINDOWQUANT_WINDOW_SIZE", "16")))
    parser.add_argument("--wq-low-bits", type=int, default=int(os.environ.get("MINICPM_WINDOWQUANT_LOW_BITS", "4")))
    parser.add_argument("--wq-high-bits", type=int, default=int(os.environ.get("MINICPM_WINDOWQUANT_HIGH_BITS", "8")))
    parser.add_argument("--wq-high-ratio", type=float, default=float(os.environ.get("MINICPM_WINDOWQUANT_HIGH_RATIO", "0.25")))
    parser.add_argument(
        "--wq-protect-recent-windows",
        type=int,
        default=int(os.environ.get("MINICPM_WINDOWQUANT_PROTECT_RECENT_WINDOWS", "1")),
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    os.environ["MINICPM_WINDOWQUANT_WINDOW_SIZE"] = str(args.wq_window_size)
    os.environ["MINICPM_WINDOWQUANT_LOW_BITS"] = str(args.wq_low_bits)
    os.environ["MINICPM_WINDOWQUANT_HIGH_BITS"] = str(args.wq_high_bits)
    os.environ["MINICPM_WINDOWQUANT_HIGH_RATIO"] = str(args.wq_high_ratio)
    os.environ["MINICPM_WINDOWQUANT_PROTECT_RECENT_WINDOWS"] = str(args.wq_protect_recent_windows)
    WindowQuantConfig.from_env().validate()
    return args


def _print_windowquant_summary(results: list[dict], frame_selection: str = "recent") -> None:
    summary = dist_sb.compute_summary(results)
    print("\n" + "=" * 60)
    label = "All-Frames" if frame_selection == "all" else "Recent-Window"
    print(f"StreamingBench {label} Results (MiniCPM-V-4.6 + WindowQuant)")
    print("=" * 60)
    for row in summary["tasks"]:
        print(f"  {row['task_type']}: {row['accuracy']:.2f}% ({row['correct']}/{row['total']})")
    overall = summary["overall"]
    print(f"\n  Overall: {overall['accuracy']:.2f}% ({overall['correct']}/{overall['total']})")
    print(f"  Errors: {summary['error_count']}")
    print("=" * 60)


def main() -> None:
    _consume_windowquant_args()
    dist_sb.RecentWindowQAModel = WindowQuantMiniCPMQAModel
    dist_sb.query_all_frames = query_all_frames
    dist_sb.query_recent_window = query_recent_window
    dist_sb.print_summary = _print_windowquant_summary
    dist_sb.main()


if __name__ == "__main__":
    main()
