#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from main_experiments.minicpm_v46.streamingbench import eval_baseline_dist as dist_sb
from lib.minicpm.streamingtom import (
    StreamingTOMMiniCPMQAModel,
    query_all_frames,
    query_recent_window,
)


def _consume_streamingtom_args() -> argparse.Namespace:
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
    parser.add_argument(
        "--oqm-retrieval-max-tokens",
        "--oqm_retrieval_max_tokens",
        dest="oqm_retrieval_max_tokens",
        type=int,
        default=int(os.environ.get("MINICPM_OQM_RETRIEVAL_MAX_TOKENS", "12544")),
        help="Maximum visual KV tokens retrieved by OQM.",
    )
    parser.add_argument(
        "--oqm-bits",
        "--oqm_bits",
        dest="oqm_bits",
        type=int,
        default=int(os.environ.get("MINICPM_OQM_QUANTIZATION_BITS", "4")),
        help="OQM KV quantization bits.",
    )
    parser.add_argument(
        "--oqm-init-tokens",
        "--oqm_init_tokens",
        dest="oqm_init_tokens",
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


def _print_streamingtom_summary(results: list[dict], frame_selection: str = "recent") -> None:
    summary = dist_sb.compute_summary(results)
    print("\n" + "=" * 60)
    label = "All-Frames" if frame_selection == "all" else "Recent-Window"
    budget = os.environ.get("MINICPM_CTR_TOKEN_BUDGET", "50")
    tau = os.environ.get("MINICPM_CTR_TAU", "0.9")
    bits = os.environ.get("MINICPM_OQM_QUANTIZATION_BITS", "4")
    retrieval = os.environ.get("MINICPM_OQM_RETRIEVAL_MAX_TOKENS", "12544")
    print(
        f"StreamingBench {label} Results "
        f"(MiniCPM-V-4.6 + StreamingTOM(G={budget}, tau={tau}, OQM={bits}-bit, retrieve={retrieval}))"
    )
    print("=" * 60)
    for row in summary["tasks"]:
        print(f"  {row['task_type']}: {row['accuracy']:.2f}% ({row['correct']}/{row['total']})")
    overall = summary["overall"]
    print(f"\n  Overall: {overall['accuracy']:.2f}% ({overall['correct']}/{overall['total']})")
    print(f"  Errors: {summary['error_count']}")
    print("=" * 60)


def main() -> None:
    _consume_streamingtom_args()
    dist_sb.RecentWindowQAModel = StreamingTOMMiniCPMQAModel
    dist_sb.query_all_frames = query_all_frames
    dist_sb.query_recent_window = query_recent_window
    dist_sb.print_summary = _print_streamingtom_summary
    dist_sb.main()


if __name__ == "__main__":
    main()
