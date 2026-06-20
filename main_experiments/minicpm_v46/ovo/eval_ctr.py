#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import lib.minicpm.baseline as minicpm_eval
from main_experiments.minicpm_v46.ovo import eval_baseline as base_ovo
from lib.shared.recent_window import calculate_ovo_scores
from lib.minicpm.ctr import (
    CTRMiniCPMQAModel,
    query_all_frames,
    query_recent_window,
)


def _consume_ctr_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--ctr_budget",
        type=int,
        default=int(os.environ.get("MINICPM_CTR_TOKEN_BUDGET", "50")),
        help="CTR visual-token budget per frame.",
    )
    parser.add_argument(
        "--ctr_tau",
        type=float,
        default=float(os.environ.get("MINICPM_CTR_TAU", "0.9")),
        help="CTR temporal similarity threshold.",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    os.environ["MINICPM_CTR_TOKEN_BUDGET"] = str(args.ctr_budget)
    os.environ["MINICPM_CTR_TAU"] = str(args.ctr_tau)
    return args


def _print_ctr_ovo_results(
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
    ctr_args = _consume_ctr_args()

    # Keep the original OVO runner, but swap its MiniCPM evaluator and the
    # query functions referenced by the imported evaluation helpers.
    minicpm_eval.query_all_frames = query_all_frames
    minicpm_eval.query_recent_window = query_recent_window
    base_ovo.RecentWindowQAModel = CTRMiniCPMQAModel
    base_ovo.MODEL_LABEL = (
        f"MiniCPM-V-4.6 + CTR(G={ctr_args.ctr_budget}, tau={ctr_args.ctr_tau})"
    )
    base_ovo.print_ovo_results = _print_ctr_ovo_results
    base_ovo.main()


if __name__ == "__main__":
    main()
