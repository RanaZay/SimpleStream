#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.streamingtom_oqm import OQMConfig, OnlineQuantizedMemory


def _max_mean_error(reference: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, float]:
    diff = (reference.float() - reconstructed.float()).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug StreamingTOM OQM with synthetic KV tensors.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head_dim", type=int, default=32)
    parser.add_argument("--init_tokens", type=int, default=14)
    parser.add_argument("--group_size", type=int, default=50)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--retrieval_max_tokens", type=int, default=100)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--disable_quantization", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    vision_tokens = int(args.group_size) * int(args.groups)

    config = OQMConfig(
        retrieval_max_tokens=args.retrieval_max_tokens,
        enable_quantization=not args.disable_quantization,
        quantization_bits=args.bits,
        group_size=args.group_size,
        init_token_count=args.init_tokens,
        sliding_window_size=args.retrieval_max_tokens,
    )
    oqm = OnlineQuantizedMemory(config)

    init_k = torch.randn(args.batch, args.heads, args.init_tokens, args.head_dim, device=device, dtype=dtype)
    init_v = torch.randn_like(init_k)
    vision_k = torch.randn(args.batch, args.heads, vision_tokens, args.head_dim, device=device, dtype=dtype)
    vision_v = torch.randn_like(vision_k)

    # Token-level keys simulate compressed visual-token features after CTR.
    token_level_keys = torch.randn(vision_tokens, args.head_dim, device=device, dtype=dtype)
    group_keys = token_level_keys.reshape(args.groups, args.group_size, args.head_dim).mean(dim=1)
    query_key = group_keys[1 if args.groups > 1 else 0] + 0.01 * torch.randn(args.head_dim, device=device)

    t0 = time.perf_counter()
    init_meta = oqm.store_system_prompt("debug_video", 0, init_k, init_v)
    store_meta = oqm.store_kv_cache("debug_video", 0, vision_k, vision_v, token_level_keys=token_level_keys)
    total_store_ms = (time.perf_counter() - t0) * 1000.0

    selected_groups, retrieval_meta = oqm.retrieve_group_indices(
        "debug_video",
        0,
        query_key,
        max_tokens=args.retrieval_max_tokens,
    )
    (selective_k, selective_v), selective_meta = oqm.get_selective_kv("debug_video", 0, selected_groups)
    (window_k, window_v), window_meta = oqm.get_windowed_kv("debug_video", 0)

    selected_token_indices = (
        selected_groups.to(device).unsqueeze(1) * args.group_size
        + torch.arange(args.group_size, device=device)
    ).flatten()
    reference_selective_k = torch.cat([init_k, vision_k.index_select(2, selected_token_indices)], dim=2)
    reference_selective_v = torch.cat([init_v, vision_v.index_select(2, selected_token_indices)], dim=2)

    summary = {
        "config": {
            "retrieval_max_tokens": config.retrieval_max_tokens,
            "enable_quantization": config.enable_quantization,
            "quantization_bits": config.quantization_bits,
            "group_size": config.group_size,
            "init_token_count": config.init_token_count,
            "sliding_window_size": config.sliding_window_size,
        },
        "input_shapes": {
            "init_kv": list(init_k.shape),
            "vision_kv": list(vision_k.shape),
            "token_level_keys": list(token_level_keys.shape),
        },
        "store": {
            "system_prompt": init_meta.as_dict(),
            "vision_kv": store_meta.as_dict(),
            "total_store_time_ms": total_store_ms,
        },
        "retrieval": retrieval_meta.as_dict(),
        "selective_reconstruct": {
            **selective_meta.as_dict(),
            "key_shape": list(selective_k.shape),
            "value_shape": list(selective_v.shape),
            "key_error": _max_mean_error(reference_selective_k, selective_k),
            "value_error": _max_mean_error(reference_selective_v, selective_v),
        },
        "windowed_reconstruct": {
            **window_meta.as_dict(),
            "key_shape": list(window_k.shape),
            "value_shape": list(window_v.shape),
        },
        "storage_summary": oqm.storage_summary("debug_video"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
