#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.recent_window_eval_minicpm_ctr import CTRMiniCPMQAModel
from lib.streamingtom_ctr import CTRConfig
from lib.streamingtom_oqm import OQMConfig, OnlineQuantizedMemory


def _make_images(num_images: int) -> list[Image.Image]:
    colors = [(128, 128, 128), (160, 120, 90), (90, 140, 180), (180, 170, 90)]
    return [
        Image.new("RGB", (224, 224), color=colors[index % len(colors)])
        for index in range(num_images)
    ]


def _past_layers(past_key_values: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return [
            (key, value)
            for key, value in zip(past_key_values.key_cache, past_key_values.value_cache)
            if isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor)
        ]
    if isinstance(past_key_values, (list, tuple)):
        layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in past_key_values:
            if isinstance(layer, (list, tuple)) and len(layer) >= 2:
                key, value = layer[0], layer[1]
                if isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor):
                    layers.append((key, value))
        return layers
    return []


def _shape(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "device": str(value.device),
        }
    return str(type(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe MiniCPM-V CTR prefill KV cache with OQM storage.")
    parser.add_argument("--model_path", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_images", type=int, default=2)
    parser.add_argument("--ctr_budget", type=int, default=50)
    parser.add_argument("--ctr_tau", type=float, default=0.9)
    parser.add_argument("--oqm_retrieval_max_tokens", type=int, default=50)
    parser.add_argument("--oqm_bits", type=int, default=4)
    parser.add_argument("--oqm_init_tokens", type=int, default=14)
    parser.add_argument("--layers_to_store", type=int, default=1)
    parser.add_argument("--question", default="Describe the images briefly.")
    args = parser.parse_args()

    os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "false")
    os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "1")

    qa = CTRMiniCPMQAModel(
        model_name=args.model_path,
        device=args.device,
        max_new_tokens=8,
        attn_implementation=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
        ctr_config=CTRConfig(
            token_budget=args.ctr_budget,
            similarity_threshold=args.ctr_tau,
        ),
    )

    model_inputs = qa.build_ctr_model_inputs(_make_images(args.num_images), args.question)
    forward_kwargs = {
        "input_ids": model_inputs["input_ids"],
        "inputs_embeds": model_inputs["inputs_embeds"],
        "attention_mask": model_inputs["attention_mask"],
        "use_cache": True,
        "return_dict": True,
    }
    t0 = time.perf_counter()
    try:
        outputs = qa.model(**forward_kwargs)
    except TypeError:
        forward_kwargs.pop("input_ids", None)
        outputs = qa.model(**forward_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000.0

    past_key_values = getattr(outputs, "past_key_values", None)
    layers = _past_layers(past_key_values)
    if not layers:
        raise RuntimeError(f"Could not extract tensor KV layers from {type(past_key_values)}")

    image_positions = torch.nonzero(model_inputs["image_mask"][0], as_tuple=False).flatten()
    if int(image_positions.numel()) != int(model_inputs["tokens_after"]):
        raise RuntimeError("Image-token mask does not match CTR-compressed token count")

    token_level_keys = model_inputs["inputs_embeds"][0, image_positions, :].detach()
    query_key = token_level_keys[: args.ctr_budget].float().mean(dim=0)
    oqm = OnlineQuantizedMemory(
        OQMConfig(
            retrieval_max_tokens=args.oqm_retrieval_max_tokens,
            enable_quantization=True,
            quantization_bits=args.oqm_bits,
            group_size=args.ctr_budget,
            init_token_count=args.oqm_init_tokens,
            sliding_window_size=args.oqm_retrieval_max_tokens,
        )
    )

    layer_summaries: list[dict[str, Any]] = []
    for layer_idx, (key_cache, value_cache) in enumerate(layers[: args.layers_to_store]):
        key_cache = key_cache.detach()
        value_cache = value_cache.detach()
        init_count = min(args.oqm_init_tokens, int(key_cache.shape[2]))
        init_k = key_cache[:, :, :init_count, :]
        init_v = value_cache[:, :, :init_count, :]
        vision_k = key_cache.index_select(2, image_positions.to(key_cache.device))
        vision_v = value_cache.index_select(2, image_positions.to(value_cache.device))

        if init_count != args.oqm_init_tokens:
            raise RuntimeError(
                f"Layer {layer_idx}: prompt has only {init_count} tokens, "
                f"cannot preserve init_token_count={args.oqm_init_tokens}"
            )
        if int(vision_k.shape[2]) % args.ctr_budget != 0:
            raise RuntimeError(
                f"Layer {layer_idx}: vision KV tokens {int(vision_k.shape[2])} "
                f"not divisible by CTR budget {args.ctr_budget}"
            )

        init_meta = oqm.store_system_prompt("debug_video", layer_idx, init_k, init_v)
        store_meta = oqm.store_kv_cache(
            "debug_video",
            layer_idx,
            vision_k,
            vision_v,
            token_level_keys=token_level_keys,
        )
        selected_groups, retrieval_meta = oqm.retrieve_group_indices(
            "debug_video",
            layer_idx,
            query_key=query_key,
            max_tokens=args.oqm_retrieval_max_tokens,
        )
        (selected_k, selected_v), reconstruct_meta = oqm.get_selective_kv(
            "debug_video",
            layer_idx,
            selected_groups,
        )
        layer_summaries.append(
            {
                "layer_idx": layer_idx,
                "source_key": _shape(key_cache),
                "source_value": _shape(value_cache),
                "init_store": init_meta.as_dict(),
                "vision_store": store_meta.as_dict(),
                "retrieval": retrieval_meta.as_dict(),
                "reconstruct": reconstruct_meta.as_dict(),
                "reconstructed_key": _shape(selected_k),
                "reconstructed_value": _shape(selected_v),
            }
        )

    summary = {
        "model_class": type(qa.model).__name__,
        "past_key_values_type": type(past_key_values).__name__,
        "num_layers_found": len(layers),
        "prefill_ms": prefill_ms,
        "prompt_length": int(model_inputs["prompt_length"]),
        "image_tokens_after_ctr": int(model_inputs["tokens_after"]),
        "ctr": {
            "budget": args.ctr_budget,
            "tau": args.ctr_tau,
            "tokens_before": model_inputs["tokens_before"],
            "tokens_after": model_inputs["tokens_after"],
            "vision_encode_ms": qa._last_ctr_vision_encode_ms,
            "compress_features_ms": qa._last_ctr_compress_features_ms,
        },
        "first_layer_shapes": {
            "key": _shape(layers[0][0]),
            "value": _shape(layers[0][1]),
        },
        "layers_stored": layer_summaries,
        "oqm_storage_summary": oqm.storage_summary("debug_video"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
