#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import lib.minicpm.baseline as minicpm_eval
from lib.minicpm.windowquant import (
    WindowQuantConfig,
    WindowQuantMiniCPMQAModel,
    query_all_frames,
    query_recent_window,
)
from lib.shared.recent_window import calculate_ovo_scores
from main_experiments.minicpm_v46.ovo import eval_baseline as base_ovo


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


def _print_windowquant_ovo_results(
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
    args = _consume_windowquant_args()
    minicpm_eval.query_all_frames = query_all_frames
    minicpm_eval.query_recent_window = query_recent_window
    base_ovo.RecentWindowQAModel = WindowQuantMiniCPMQAModel
    base_ovo.MODEL_LABEL = (
        "MiniCPM-V-4.6 + WindowQuant"
        f"(win={args.wq_window_size}, {args.wq_low_bits}/{args.wq_high_bits}b)"
    )
    base_ovo.print_ovo_results = _print_windowquant_ovo_results
    base_ovo.main()


if __name__ == "__main__":
    main()
