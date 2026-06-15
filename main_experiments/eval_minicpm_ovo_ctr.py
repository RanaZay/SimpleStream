#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lib.recent_window_eval_minicpm as minicpm_eval
import main_experiments.eval_minicpm_ovo as base_ovo
from lib.recent_window_eval_minicpm_ctr import (
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
    base_ovo.main()


if __name__ == "__main__":
    main()
