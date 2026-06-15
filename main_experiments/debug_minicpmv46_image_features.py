#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
from typing import Any

import torch
from PIL import Image


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    return {
        "type": "tensor",
        "shape": tuple(int(dim) for dim in value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def _summarize(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _tensor_summary(value)
    if isinstance(value, dict):
        return {key: _summarize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summarize(item) for item in value]
    return {"type": type(value).__name__, "repr": repr(value)[:240]}


def _collect_feature_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        for key in ("pooler_output", "image_embeds", "image_features", "last_hidden_state", "hidden_states"):
            if key in value:
                result = _collect_feature_tensors(value[key])
                if result:
                    return result
        collected: list[torch.Tensor] = []
        for item in value.values():
            collected.extend(_collect_feature_tensors(item))
        return collected
    if isinstance(value, (list, tuple)):
        collected = []
        for item in value:
            collected.extend(_collect_feature_tensors(item))
        return collected
    return []


def _concat_feature_tensors(value: Any) -> torch.Tensor | None:
    tensors = _collect_feature_tensors(value)
    if not tensors:
        return None
    flattened = [tensor.reshape(-1, tensor.shape[-1]) for tensor in tensors]
    return torch.cat(flattened, dim=0)


def _make_images(num_images: int) -> list[Image.Image]:
    images = []
    colors = [(128, 128, 128), (160, 120, 90), (90, 140, 180), (180, 170, 90)]
    for idx in range(num_images):
        image = Image.new("RGB", (224, 224), color=colors[idx % len(colors)])
        images.append(image)
    return images


def _call_get_image_features(model: Any, inputs: Any, args: argparse.Namespace) -> tuple[str, Any]:
    device = getattr(model, "device", args.device)
    pixel_values = inputs["pixel_values"].to(device)
    target_sizes = inputs.get("target_sizes")
    if target_sizes is not None:
        target_sizes = target_sizes.to(device)

    base_kwargs: dict[str, Any] = {"pixel_values": pixel_values}
    if target_sizes is not None:
        base_kwargs["target_sizes"] = target_sizes

    variants: list[tuple[str, dict[str, Any]]] = [
        ("basic", dict(base_kwargs)),
        ("with_downsample", {**base_kwargs, "downsample_mode": args.downsample_mode}),
        (
            "with_downsample_and_slices",
            {
                **base_kwargs,
                "downsample_mode": args.downsample_mode,
                "max_slice_nums": args.max_slice_nums,
            },
        ),
    ]

    last_error: Exception | None = None
    for name, kwargs in variants:
        try:
            return name, model.get_image_features(**kwargs)
        except TypeError as exc:
            last_error = exc
            continue

    if target_sizes is not None:
        try:
            return "positional_pixel_target", model.get_image_features(pixel_values, target_sizes)
        except TypeError as exc:
            last_error = exc

    raise RuntimeError(f"Could not call get_image_features. Last error: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe MiniCPM-V 4.6 image-feature extraction for CTR integration.")
    parser.add_argument("--model_path", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--attn_implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"))
    parser.add_argument("--num_images", type=int, default=2)
    parser.add_argument("--downsample_mode", default=os.environ.get("MINICPM_DOWNSAMPLE_MODE", "16x"))
    parser.add_argument("--max_slice_nums", type=int, default=int(os.environ.get("MINICPM_MAX_SLICE_NUMS", "1")))
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

    content = [{"type": "image", "image": image} for image in _make_images(args.num_images)]
    content.append({"type": "text", "text": "Describe the images briefly."})
    messages = [{"role": "user", "content": content}]

    template_kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    processor_kwargs = {
        "downsample_mode": args.downsample_mode,
        "max_slice_nums": args.max_slice_nums,
        "use_image_id": False,
    }
    try:
        inputs = processor.apply_chat_template(messages, **template_kwargs, **processor_kwargs)
    except TypeError:
        inputs = processor.apply_chat_template(messages, **template_kwargs, processor_kwargs=processor_kwargs)

    image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
    image_placeholder_tokens = None
    if image_token_id is not None:
        image_placeholder_tokens = int((inputs["input_ids"] == int(image_token_id)).sum().item())

    with torch.inference_mode():
        call_variant, features = _call_get_image_features(model=model, inputs=inputs, args=args)

    feature_tensor = _concat_feature_tensors(features)
    feature_tokens = int(feature_tensor.reshape(-1, feature_tensor.shape[-1]).shape[0]) if feature_tensor is not None else None
    feature_dim = int(feature_tensor.shape[-1]) if feature_tensor is not None else None

    summary = {
        "model_class": type(model).__name__,
        "processor_class": type(processor).__name__,
        "get_image_features_signature": str(inspect.signature(model.get_image_features)),
        "call_variant": call_variant,
        "num_images": args.num_images,
        "downsample_mode": args.downsample_mode,
        "max_slice_nums": args.max_slice_nums,
        "input_shapes": {key: _summarize(value) for key, value in inputs.items()},
        "image_placeholder_tokens": image_placeholder_tokens,
        "features": _summarize(features),
        "flattened_feature_tokens": feature_tokens,
        "flattened_feature_dim": feature_dim,
        "placeholder_feature_token_match": (
            image_placeholder_tokens == feature_tokens
            if image_placeholder_tokens is not None and feature_tokens is not None
            else None
        ),
    }
    print(json.dumps(summary, indent=2))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
