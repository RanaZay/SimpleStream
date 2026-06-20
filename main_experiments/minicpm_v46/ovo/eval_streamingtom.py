#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import lib.minicpm.baseline as minicpm_eval
from main_experiments.minicpm_v46.ovo import eval_baseline as base_ovo
from lib.shared.recent_window import calculate_ovo_scores
from lib.minicpm.streamingtom import (
    StreamingTOMMiniCPMQAModel,
    query_all_frames,
    query_recent_window,
)


def _consume_streamingtom_args() -> argparse.Namespace:
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
    parser.add_argument(
        "--oqm_retrieval_max_tokens",
        type=int,
        default=int(os.environ.get("MINICPM_OQM_RETRIEVAL_MAX_TOKENS", "12544")),
        help="Maximum visual KV tokens retrieved by OQM.",
    )
    parser.add_argument(
        "--oqm_bits",
        type=int,
        default=int(os.environ.get("MINICPM_OQM_QUANTIZATION_BITS", "4")),
        help="OQM KV quantization bits.",
    )
    parser.add_argument(
        "--oqm_init_tokens",
        type=int,
        default=int(os.environ.get("MINICPM_OQM_INIT_TOKEN_COUNT", "14")),
        help="Initial prompt tokens preserved unquantized in OQM.",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    os.environ["MINICPM_CTR_TOKEN_BUDGET"] = str(args.ctr_budget)
    os.environ["MINICPM_CTR_TAU"] = str(args.ctr_tau)
    os.environ["MINICPM_OQM_GROUP_SIZE"] = str(args.ctr_budget)
    os.environ["MINICPM_OQM_RETRIEVAL_MAX_TOKENS"] = str(args.oqm_retrieval_max_tokens)
    os.environ["MINICPM_OQM_QUANTIZATION_BITS"] = str(args.oqm_bits)
    os.environ["MINICPM_OQM_INIT_TOKEN_COUNT"] = str(args.oqm_init_tokens)
    os.environ.setdefault("MINICPM_OQM_ENABLE_QUANTIZATION", "1")
    return args


def _print_streamingtom_ovo_results(
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
    args = _consume_streamingtom_args()
    minicpm_eval.query_all_frames = query_all_frames
    minicpm_eval.query_recent_window = query_recent_window
    base_ovo.RecentWindowQAModel = StreamingTOMMiniCPMQAModel
    base_ovo.MODEL_LABEL = (
        "MiniCPM-V-4.6 + StreamingTOM"
        f"(G={args.ctr_budget}, tau={args.ctr_tau}, "
        f"OQM={args.oqm_bits}-bit, retrieve={args.oqm_retrieval_max_tokens})"
    )
    base_ovo.print_ovo_results = _print_streamingtom_ovo_results
    base_ovo.main()


if __name__ == "__main__":
    main()
