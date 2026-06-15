#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.recent_window_eval_minicpm_ctr import CTRMiniCPMQAModel
from lib.streamingtom_ctr import CTRConfig


def _make_images(num_images: int) -> list[Image.Image]:
    colors = [(128, 128, 128), (160, 120, 90), (90, 140, 180), (180, 170, 90)]
    return [
        Image.new("RGB", (224, 224), color=colors[index % len(colors)])
        for index in range(num_images)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug MiniCPM-V 4.6 generation through the CTR path.")
    parser.add_argument("--model_path", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_images", type=int, default=2)
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--question", default="Describe the images briefly.")
    args = parser.parse_args()

    os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "false")
    os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "1")

    qa = CTRMiniCPMQAModel(
        model_name=args.model_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        attn_implementation=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
        ctr_config=CTRConfig(
            token_budget=args.budget,
            similarity_threshold=args.tau,
        ),
    )
    answer = qa.generate_from_frames(_make_images(args.num_images), args.question)
    summary = {
        "answer": answer,
        "num_images": args.num_images,
        "ctr_budget": args.budget,
        "ctr_tau": args.tau,
        "tokens_before": qa._last_num_vision_tokens_before,
        "tokens_after": qa._last_num_vision_tokens_after,
        "vision_encode_ms": qa._last_ctr_vision_encode_ms,
        "compress_features_ms": qa._last_ctr_compress_features_ms,
        "ttft_seconds": qa._last_ttft_seconds,
        "model_generate_seconds": qa._last_model_generate_seconds,
        "ctr_metadata": qa._last_ctr_metadata,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
