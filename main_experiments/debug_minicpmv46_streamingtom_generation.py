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
    def append_if_tensor(
        layers: list[tuple[torch.Tensor, torch.Tensor]],
        key: Any,
        value: Any,
    ) -> None:
        if isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor):
            if key.ndim == 4 and value.ndim == 4 and int(key.shape[2]) > 0:
                layers.append((key, value))

    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        for key, value in zip(past_key_values.key_cache, past_key_values.value_cache):
            append_if_tensor(layers, key, value)
        if layers:
            return layers

    cache_layers = getattr(past_key_values, "layers", None)
    if isinstance(cache_layers, (list, tuple)):
        layers = []
        for cache_layer in cache_layers:
            append_if_tensor(
                layers,
                getattr(cache_layer, "keys", None),
                getattr(cache_layer, "values", None),
            )
        if layers:
            return layers

    if isinstance(past_key_values, (list, tuple)):
        layers = []
        for layer in past_key_values:
            if isinstance(layer, (list, tuple)) and len(layer) >= 2:
                append_if_tensor(layers, layer[0], layer[1])
        if layers:
            return layers

    try:
        iterator = iter(past_key_values)
    except TypeError:
        return []
    layers = []
    for layer in iterator:
        if isinstance(layer, (list, tuple)) and len(layer) >= 2:
            append_if_tensor(layers, layer[0], layer[1])
    return layers


def _set_cache_layer_kv(cache_layer: Any, key: torch.Tensor, value: torch.Tensor) -> None:
    cache_layer.keys = key
    cache_layer.values = value
    cache_layer.is_initialized = True
    cache_layer.dtype = key.dtype
    cache_layer.device = key.device
    cumulative_length = getattr(cache_layer, "cumulative_length", None)
    if isinstance(cumulative_length, int):
        cache_layer.cumulative_length = int(key.shape[2])
    elif isinstance(cumulative_length, torch.Tensor):
        cumulative_length.zero_()
        cumulative_length.add_(int(key.shape[2]))
    if hasattr(cache_layer, "cumulative_length_int"):
        cache_layer.cumulative_length_int = int(key.shape[2])


def _dynamic_cache_from_layers(
    template_cache: Any,
    layers: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[Any, dict[str, Any]]:
    cache_layers = getattr(template_cache, "layers", None)
    if isinstance(cache_layers, (list, tuple)):
        attention_idx = 0
        for cache_layer in cache_layers:
            key = getattr(cache_layer, "keys", None)
            value = getattr(cache_layer, "values", None)
            if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                continue
            if key.ndim != 4 or value.ndim != 4 or int(key.shape[2]) <= 0:
                continue
            if attention_idx >= len(layers):
                break
            new_key, new_value = layers[attention_idx]
            _set_cache_layer_kv(cache_layer, new_key, new_value)
            attention_idx += 1
        if attention_idx != len(layers):
            raise RuntimeError(
                "Could not map all reconstructed attention layers back into the "
                f"template cache: replaced={attention_idx}, expected={len(layers)}"
            )
        return template_cache, {
            "format": "template_dynamic_cache",
            "template_type": type(template_cache).__name__,
            "attention_layers_replaced": attention_idx,
            "total_cache_layers": len(cache_layers),
        }

    try:
        from transformers.cache_utils import DynamicCache

        return DynamicCache([(key, value, None) for key, value in layers]), {
            "format": "generic_dynamic_cache",
            "attention_layers_replaced": len(layers),
            "total_cache_layers": len(layers),
        }
    except Exception:
        return tuple((key, value) for key, value in layers), {
            "format": "tuple_cache",
            "attention_layers_replaced": len(layers),
            "total_cache_layers": len(layers),
        }


def _reconstruct_prompt_order_cache(
    *,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    image_positions: torch.Tensor,
    selected_groups: torch.Tensor,
    selected_kv: tuple[torch.Tensor, torch.Tensor],
    group_size: int,
    init_token_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = int(key_cache.shape[2])
    prompt_device = key_cache.device
    image_positions = image_positions.to(prompt_device)
    selected_groups = selected_groups.to(prompt_device)

    image_token_offsets = (
        selected_groups.unsqueeze(1) * int(group_size)
        + torch.arange(int(group_size), device=prompt_device)
    ).flatten()
    selected_image_positions = image_positions.index_select(0, image_token_offsets)

    image_mask = torch.zeros(seq_len, device=prompt_device, dtype=torch.bool)
    image_mask[image_positions] = True
    non_image_positions = torch.nonzero(~image_mask, as_tuple=False).flatten()
    keep_positions = torch.sort(torch.cat([non_image_positions, selected_image_positions], dim=0)).values

    selected_k, selected_v = selected_kv
    selected_k = selected_k.to(prompt_device)
    selected_v = selected_v.to(prompt_device)
    selected_image_k = selected_k[:, :, int(init_token_count) :, :]
    selected_image_v = selected_v[:, :, int(init_token_count) :, :]
    selected_by_position = {
        int(pos): idx
        for idx, pos in enumerate(selected_image_positions.detach().cpu().tolist())
    }

    key_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    for pos in keep_positions.detach().cpu().tolist():
        if int(pos) in selected_by_position:
            local = selected_by_position[int(pos)]
            key_parts.append(selected_image_k[:, :, local : local + 1, :])
            value_parts.append(selected_image_v[:, :, local : local + 1, :])
        else:
            key_parts.append(key_cache[:, :, int(pos) : int(pos) + 1, :])
            value_parts.append(value_cache[:, :, int(pos) : int(pos) + 1, :])
    return torch.cat(key_parts, dim=2), torch.cat(value_parts, dim=2)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Try MiniCPM-V decoding from a CTR+OQM reconstructed cache.")
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

    qa = CTRMiniCPMQAModel(
        model_name=args.model_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        attn_implementation=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
        ctr_config=CTRConfig(
            token_budget=args.ctr_budget,
            similarity_threshold=args.ctr_tau,
        ),
    )

    model_inputs = qa.build_ctr_model_inputs(_make_images(args.num_images), args.question)
    prefill_t0 = time.perf_counter()
    forward_kwargs = {
        "input_ids": model_inputs["input_ids"],
        "inputs_embeds": model_inputs["inputs_embeds"],
        "attention_mask": model_inputs["attention_mask"],
        "use_cache": True,
        "return_dict": True,
    }
    try:
        prefill_outputs = qa.model(**forward_kwargs)
    except TypeError:
        forward_kwargs.pop("input_ids", None)
        prefill_outputs = qa.model(**forward_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - prefill_t0) * 1000.0

    source_layers = _past_layers(getattr(prefill_outputs, "past_key_values", None))
    if not source_layers:
        raise RuntimeError("No KV layers found after CTR prefill")

    image_positions = torch.nonzero(model_inputs["image_mask"][0], as_tuple=False).flatten()
    non_image_mask = ~model_inputs["image_mask"][0]
    query_key = model_inputs["inputs_embeds"][0, non_image_mask, :].detach().float().mean(dim=0)
    token_level_keys = model_inputs["inputs_embeds"][0, image_positions, :].detach()

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

    reconstructed_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    layer0_retrieval: dict[str, Any] | None = None
    layer0_reconstruct: dict[str, Any] | None = None
    store_ms = 0.0
    retrieval_ms = 0.0
    reconstruct_ms = 0.0

    for layer_idx, (key_cache, value_cache) in enumerate(source_layers):
        key_cache = key_cache.detach()
        value_cache = value_cache.detach()
        init_k = key_cache[:, :, : args.oqm_init_tokens, :]
        init_v = value_cache[:, :, : args.oqm_init_tokens, :]
        vision_k = key_cache.index_select(2, image_positions.to(key_cache.device))
        vision_v = value_cache.index_select(2, image_positions.to(value_cache.device))

        if int(vision_k.shape[2]) % args.ctr_budget != 0:
            raise RuntimeError(
                f"Layer {layer_idx}: vision tokens {int(vision_k.shape[2])} "
                f"not divisible by group_size={args.ctr_budget}"
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
            query_key,
            max_tokens=args.oqm_retrieval_max_tokens,
        )
        selected_kv, reconstruct_meta = oqm.get_selective_kv(
            "debug_video",
            layer_idx,
            selected_groups,
        )
        store_ms += init_meta.store_time_ms + store_meta.store_time_ms
        retrieval_ms += retrieval_meta.retrieval_time_ms
        reconstruct_ms += reconstruct_meta.reconstruct_time_ms
        if layer_idx == 0:
            layer0_retrieval = retrieval_meta.as_dict()
            layer0_reconstruct = reconstruct_meta.as_dict()
        reconstructed_layers.append(
            _reconstruct_prompt_order_cache(
                key_cache=key_cache,
                value_cache=value_cache,
                image_positions=image_positions,
                selected_groups=selected_groups,
                selected_kv=selected_kv,
                group_size=args.ctr_budget,
                init_token_count=args.oqm_init_tokens,
            )
        )

    reconstructed_cache, cache_build_metadata = _dynamic_cache_from_layers(
        getattr(prefill_outputs, "past_key_values", None),
        reconstructed_layers,
    )
    reconstructed_seq_len = int(reconstructed_layers[0][0].shape[2])
    full_seq_len = int(source_layers[0][0].shape[2])

    generated_tokens: list[torch.Tensor] = []
    next_token = torch.argmax(prefill_outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated_tokens.append(next_token)
    attention_mask = torch.ones(
        (1, reconstructed_seq_len + 1),
        dtype=model_inputs["attention_mask"].dtype,
        device=model_inputs["attention_mask"].device,
    )

    decode_t0 = time.perf_counter()
    cache = reconstructed_cache
    decode_error: str | None = None
    try:
        for _step in range(max(0, int(args.max_new_tokens) - 1)):
            outputs = qa.model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = getattr(outputs, "past_key_values", cache)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_tokens.append(next_token)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=1,
            )
    except Exception as exc:
        decode_error = f"{type(exc).__name__}: {exc}"
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - decode_t0) * 1000.0

    if generated_tokens:
        generated_ids = torch.cat(generated_tokens, dim=1)
        answer = qa.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
    else:
        answer = ""

    summary = {
        "decode_error": decode_error,
        "answer": answer,
        "num_layers": len(source_layers),
        "full_prefill_seq_len": full_seq_len,
        "reconstructed_seq_len": reconstructed_seq_len,
        "prompt_length": int(model_inputs["prompt_length"]),
        "image_tokens_after_ctr": int(model_inputs["tokens_after"]),
        "selected_visual_tokens_per_layer": int(args.oqm_retrieval_max_tokens),
        "cache_build": cache_build_metadata,
        "timing_ms": {
            "ctr_vision_encode": qa._last_ctr_vision_encode_ms,
            "ctr_compress_features": qa._last_ctr_compress_features_ms,
            "prefill": prefill_ms,
            "oqm_store_total": store_ms,
            "oqm_retrieval_total": retrieval_ms,
            "oqm_reconstruct_total": reconstruct_ms,
            "decode_loop": decode_ms,
        },
        "layer0_retrieval": layer0_retrieval,
        "layer0_reconstruct": layer0_reconstruct,
        "storage_summary": oqm.storage_summary("debug_video"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
