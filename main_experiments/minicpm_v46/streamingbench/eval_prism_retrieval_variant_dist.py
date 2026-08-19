#!/usr/bin/env python3
"""Distributed StreamingBench PRISM eval with an isolated retrieval variant."""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from lib.minicpm import progressive_sufficiency as psm  # noqa: E402
from lib.minicpm.prism_retrieval_variants import rank_candidates  # noqa: E402
from main_experiments.minicpm_v46.streamingbench import eval_prism_exact_recent_dist as exact_eval  # noqa: E402


def _install_retrieval_variant() -> None:
    variant = os.environ.get("MINICPM_PSM_RETRIEVAL_VARIANT", "current").strip()
    if variant in {"", "current"}:
        return

    def patched_rank_candidates(
        qa: Any,
        older_chunks: list[Any],
        prompt: str,
        config: Any,
        candidate_pool: int,
        min_temporal_gap: int,
    ) -> tuple[list[dict[str, Any]], float]:
        queue, elapsed_ms, _stats = rank_candidates(
            qa=qa,
            older_chunks=older_chunks,
            prompt=prompt,
            config=config,
            candidate_pool=candidate_pool,
            min_temporal_gap=min_temporal_gap,
            variant=variant,
            mmr_lambda=float(os.environ.get("MINICPM_PSM_MMR_LAMBDA", "0.80")),
        )
        return queue, elapsed_ms

    psm._rank_candidates = patched_rank_candidates


def main() -> None:
    _install_retrieval_variant()
    exact_eval.main()


if __name__ == "__main__":
    main()
