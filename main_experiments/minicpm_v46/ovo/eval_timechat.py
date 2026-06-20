#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import lib.minicpm.baseline as minicpm_eval
from lib.minicpm.timechat import (
    TimeChatMiniCPMQAModel,
    query_all_frames,
    query_recent_window,
)
from lib.shared.recent_window import calculate_ovo_scores
from main_experiments.minicpm_v46.ovo import eval_baseline as base_ovo


def _consume_timechat_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--timechat-retention",
        "--timechat_retention",
        dest="timechat_retention",
        type=float,
        default=float(os.environ.get("MINICPM_TIMECHAT_RETENTION_RATIO", "0.8")),
        help="TimeChat-Online DTD visual-token retention ratio, e.g. 1.0, 0.8, or 0.4.",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    os.environ["MINICPM_TIMECHAT_RETENTION_RATIO"] = str(args.timechat_retention)
    return args


def _print_timechat_ovo_results(
    model_label: str,
    backward_results: list[dict],
    realtime_results: list[dict],
    forward_results: list[dict],
) -> None:
    summary = calculate_ovo_scores(backward_results, realtime_results, forward_results)
    print("\n" + "=" * 60)
    print(f"OVO-Bench Results ({model_label})")
    print("=" * 60)

    category_scores: list[float] = []
    for section_name, title in (
        ("backward", "Backward Tracing"),
        ("realtime", "Real-time Perception"),
        ("forward", "Forward Responding"),
    ):
        rows = summary[section_name]
        if not rows:
            continue
        print(f"\n{title}:")
        accs: list[float] = []
        for task, stats in rows.items():
            print(f"  {task}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")
            accs.append(float(stats["accuracy"]))
        avg = sum(accs) / len(accs)
        category_scores.append(avg)
        print(f"  {title.split()[0]} Avg.: {avg:.2f}%")

    if category_scores:
        total_avg = sum(category_scores) / len(category_scores)
        print(f"\n{'=' * 60}")
        print(f"Total Avg.: {total_avg:.2f}%")
        print("=" * 60)


def main() -> None:
    args = _consume_timechat_args()
    retention_percent = int(round(float(args.timechat_retention) * 100))

    minicpm_eval.query_all_frames = query_all_frames
    minicpm_eval.query_recent_window = query_recent_window
    base_ovo.RecentWindowQAModel = TimeChatMiniCPMQAModel
    base_ovo.MODEL_LABEL = f"MiniCPM-V-4.6 + TimeChat-DTD(retention={retention_percent}%)"
    base_ovo.print_ovo_results = _print_timechat_ovo_results
    base_ovo.main()


if __name__ == "__main__":
    main()

