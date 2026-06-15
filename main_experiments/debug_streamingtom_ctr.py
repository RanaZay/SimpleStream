#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.streamingtom_ctr import CTRConfig, CausalTemporalReducer


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test StreamingTOM CTR on synthetic visual tokens.")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    base = torch.randn(args.tokens, args.dim)
    frames = []
    current = base
    for frame_idx in range(args.frames):
        if frame_idx == 0:
            current = base
        else:
            motion = torch.randn_like(current) * 0.08
            current = 0.95 * current + motion
        frames.append(current.clone())
    visual_tokens = torch.stack(frames, dim=0)

    reducer = CausalTemporalReducer(
        CTRConfig(
            token_budget=args.budget,
            similarity_threshold=args.tau,
        )
    )
    output = reducer.reduce_stream(visual_tokens)

    print("input_shape:", tuple(visual_tokens.shape))
    print("output_shape:", tuple(output.tokens.shape))
    print(json.dumps(output.metadata_dicts, indent=2))


if __name__ == "__main__":
    main()
