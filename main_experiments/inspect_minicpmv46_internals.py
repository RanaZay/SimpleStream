#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch
from PIL import Image


def _safe_shape(value: Any) -> str | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return str(tuple(int(dim) for dim in shape))
    except Exception:
        return str(shape)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect MiniCPM-V 4.6 model internals so the CTR integration can "
            "hook the real visual-token path instead of guessing module names."
        )
    )
    parser.add_argument("--model_path", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn_implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"))
    parser.add_argument("--max_modules", type=int, default=240)
    parser.add_argument("--show_all_modules", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "false")
    os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "1")

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map=args.device,
    )
    model.eval()

    interesting_terms = (
        "vision",
        "visual",
        "image",
        "vpm",
        "siglip",
        "resampler",
        "projector",
        "projection",
        "mlp1",
        "embed",
        "llm",
    )
    module_rows = []
    for name, module in model.named_modules():
        lower = name.lower()
        if args.show_all_modules or any(term in lower for term in interesting_terms):
            module_rows.append(
                {
                    "name": name,
                    "type": type(module).__name__,
                    "device": str(next(module.parameters(), torch.empty(0, device="cpu")).device)
                    if hasattr(module, "parameters")
                    else None,
                }
            )
        if not args.show_all_modules and len(module_rows) >= args.max_modules:
            break

    callable_names = []
    for attr in dir(model):
        if attr.startswith("_"):
            continue
        if any(term in attr.lower() for term in interesting_terms):
            value = getattr(model, attr, None)
            if callable(value):
                callable_names.append(attr)

    config = getattr(model, "config", None)
    config_bits = {}
    if config is not None:
        for attr in ("image_token_id", "video_token_id", "vision_config", "model_type", "architectures"):
            value = getattr(config, attr, None)
            if value is not None:
                config_bits[attr] = str(value)

    dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": dummy_image},
                {"type": "text", "text": "Describe the image briefly."},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_shapes = {key: _safe_shape(value) for key, value in inputs.items()}
    image_token_id = getattr(config, "image_token_id", None) if config is not None else None
    image_placeholder_tokens = None
    if image_token_id is not None and "input_ids" in inputs:
        image_placeholder_tokens = int((inputs["input_ids"] == int(image_token_id)).sum().item())

    summary = {
        "model_class": type(model).__name__,
        "processor_class": type(processor).__name__,
        "model_device": str(getattr(model, "device", "unknown")),
        "config": config_bits,
        "dummy_processor_input_shapes": input_shapes,
        "dummy_image_placeholder_tokens": image_placeholder_tokens,
        "callable_visual_api_candidates": sorted(callable_names),
        "modules": module_rows,
    }
    print(json.dumps(summary, indent=2))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
