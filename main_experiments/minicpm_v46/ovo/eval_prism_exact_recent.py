#!/usr/bin/env python3
"""OVO eval for PRISM with corrected exact-six Recent-6.

This is an isolated validation entry point. It leaves official PRISM unchanged
and only patches the baseline Recent-6 selector that PRISM receives.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from main_experiments.tools.determinism import configure_determinism

SEED = configure_determinism()

import lib.minicpm.adaptive as adaptive_mod  # noqa: E402
from lib.minicpm import baseline as baseline_mod  # noqa: E402
from main_experiments.minicpm_v46.ovo import eval_adaptive as ovo_adaptive  # noqa: E402
from main_experiments.minicpm_v46.ovo import eval_baseline as ovo_eval  # noqa: E402
from main_experiments.minicpm_v46.streamingbench.eval_prism_exact_recent_dist import (  # noqa: E402
    select_exact_current_recent_frames,
)


def main() -> None:
    adaptive_args = ovo_adaptive._consume_adaptive_args()
    os.environ["MINICPM_SEED"] = str(SEED)
    adaptive_mod.select_recent_window_frames = select_exact_current_recent_frames
    baseline_mod.query_recent_window = adaptive_mod.query_recent_window
    ovo_eval.query_recent_window = adaptive_mod.query_recent_window
    ovo_eval.MODEL_LABEL = f"MiniCPM-V-4.6 + AdaptiveSimpleStream({adaptive_args.adaptive_mode}, exact-current6)"
    ovo_eval.main()


if __name__ == "__main__":
    main()
