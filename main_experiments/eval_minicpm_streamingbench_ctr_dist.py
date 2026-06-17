#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import main_experiments.eval_minicpm_streamingbench_dist as dist_sb
from lib.recent_window_eval_minicpm_ctr import (
    CTRMiniCPMQAModel,
    query_all_frames,
    query_recent_window,
)


def _consume_ctr_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--ctr-budget",
        "--ctr_budget",
        dest="ctr_budget",
        type=int,
        default=int(os.environ.get("MINICPM_CTR_TOKEN_BUDGET", "50")),
        help="CTR visual-token budget per frame.",
    )
    parser.add_argument(
        "--ctr-tau",
        "--ctr_tau",
        dest="ctr_tau",
        type=float,
        default=float(os.environ.get("MINICPM_CTR_TAU", "0.9")),
        help="CTR temporal similarity threshold.",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    os.environ["MINICPM_CTR_TOKEN_BUDGET"] = str(args.ctr_budget)
    os.environ["MINICPM_CTR_TAU"] = str(args.ctr_tau)
    return args


def _print_ctr_summary(results: list[dict], frame_selection: str = "recent") -> None:
    summary = dist_sb.compute_summary(results)
    print("\n" + "=" * 60)
    label = "All-Frames" if frame_selection == "all" else "Recent-Window"
    budget = os.environ.get("MINICPM_CTR_TOKEN_BUDGET", "50")
    tau = os.environ.get("MINICPM_CTR_TAU", "0.9")
    print(f"StreamingBench {label} Results (MiniCPM-V-4.6 + CTR(G={budget}, tau={tau}))")
    print("=" * 60)
    for row in summary["tasks"]:
        print(f"  {row['task_type']}: {row['accuracy']:.2f}% ({row['correct']}/{row['total']})")
    overall = summary["overall"]
    print(f"\n  Overall: {overall['accuracy']:.2f}% ({overall['correct']}/{overall['total']})")
    print(f"  Errors: {summary['error_count']}")
    print("=" * 60)


def main() -> None:
    _consume_ctr_args()
    dist_sb.RecentWindowQAModel = CTRMiniCPMQAModel
    dist_sb.query_all_frames = query_all_frames
    dist_sb.query_recent_window = query_recent_window
    dist_sb.print_summary = _print_ctr_summary
    dist_sb.main()


if __name__ == "__main__":
    main()
