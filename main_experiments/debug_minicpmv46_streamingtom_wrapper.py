#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.recent_window_eval_minicpm_streamingtom import StreamingTOMMiniCPMQAModel
from lib.streamingtom_ctr import CTRConfig
from lib.streamingtom_oqm import OQMConfig


def _make_images(num_images: int) -> list[Image.Image]:
    colors = [(128, 128, 128), (160, 120, 90), (90, 140, 180), (180, 170, 90)]
    return [
        Image.new("RGB", (224, 224), color=colors[index % len(colors)])
        for index in range(num_images)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug the full MiniCPM-V + CTR + OQM wrapper.")
    parser.add_argument("--model_path", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_images", type=int, default=2)
    parser.add_argument("--ctr_budget", type=int, default=50)
    parser.add_argument("--ctr_tau", type=float, default=0.9)
    parser.add_argument("--oqm_retrieval_max_tokens", type=int, default=50)
    parser.add_argument("--oqm_bits", type=int, default=4)
    parser.add_argument("--oqm_init_tokens", type=int, default=14)
    parser.add_argument("--max_new_tokens", type=int, default=12)
    parser.add_argument("--question", default="Describe the images briefly.")
    args = parser.parse_args()

    os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "false")
    os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "1")

    qa = StreamingTOMMiniCPMQAModel(
        model_name=args.model_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        attn_implementation=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
        ctr_config=CTRConfig(
            token_budget=args.ctr_budget,
            similarity_threshold=args.ctr_tau,
        ),
        oqm_config=OQMConfig(
            retrieval_max_tokens=args.oqm_retrieval_max_tokens,
            enable_quantization=True,
            quantization_bits=args.oqm_bits,
            group_size=args.ctr_budget,
            init_token_count=args.oqm_init_tokens,
            sliding_window_size=args.oqm_retrieval_max_tokens,
        ),
    )
    answer = qa.generate_from_frames(_make_images(args.num_images), args.question)
    summary = {
        "answer": answer,
        "num_images": args.num_images,
        "ctr": {
            "budget": args.ctr_budget,
            "tau": args.ctr_tau,
            "tokens_before": qa._last_num_vision_tokens_before,
            "tokens_after": qa._last_num_vision_tokens_after,
            "vision_encode_ms": qa._last_ctr_vision_encode_ms,
            "compress_features_ms": qa._last_ctr_compress_features_ms,
        },
        "oqm": {
            "retrieval_max_tokens": args.oqm_retrieval_max_tokens,
            "bits": args.oqm_bits,
            "init_token_count_actual": qa._last_oqm_init_token_count,
            "prefill_ms": qa._last_oqm_prefill_ms,
            "store_ms": qa._last_oqm_store_ms,
            "window_reconstruct_ms": qa._last_oqm_window_reconstruct_ms,
            "retrieval_ms": qa._last_oqm_retrieval_ms,
            "reconstruct_ms": qa._last_oqm_reconstruct_ms,
            "query_prefill_ms": qa._last_oqm_query_prefill_ms,
            "decode_loop_ms": qa._last_oqm_decode_loop_ms,
            "full_seq_len": qa._last_oqm_full_seq_len,
            "reconstructed_seq_len": qa._last_oqm_reconstructed_seq_len,
            "cache_build": qa._last_oqm_cache_build,
            "storage_summary": qa._last_oqm_storage_summary,
        },
        "ttft_seconds": qa._last_ttft_seconds,
        "model_generate_seconds": qa._last_model_generate_seconds,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
