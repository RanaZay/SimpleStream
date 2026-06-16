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
        "shape": [int(dim) for dim in value.shape],
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def _summarize(value: Any, depth: int = 0, max_depth: int = 4) -> Any:
    if isinstance(value, torch.Tensor):
        return _tensor_summary(value)
    if depth >= max_depth:
        return {"type": type(value).__name__, "repr": repr(value)[:180]}
    if isinstance(value, dict):
        return {str(key): _summarize(item, depth + 1, max_depth) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summarize(item, depth + 1, max_depth) for item in value[:8]]
    attrs = {}
    for key in ("attentions", "hidden_states", "last_hidden_state", "pooler_output"):
        if hasattr(value, key):
            attrs[key] = _summarize(getattr(value, key), depth + 1, max_depth)
    if attrs:
        attrs["type"] = type(value).__name__
        return attrs
    return {"type": type(value).__name__, "repr": repr(value)[:180]}


def _find_attention_like_tensors(value: Any, path: str = "root") -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, torch.Tensor):
        shape = [int(dim) for dim in value.shape]
        if len(shape) >= 3:
            found.append({"path": path, **_tensor_summary(value)})
        return found
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(_find_attention_like_tensors(item, f"{path}.{key}"))
        return found
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value[:64]):
            found.extend(_find_attention_like_tensors(item, f"{path}[{index}]"))
        return found
    for key in ("attentions", "attention", "attn_weights", "hidden_states", "last_hidden_state", "pooler_output"):
        if hasattr(value, key):
            found.extend(_find_attention_like_tensors(getattr(value, key), f"{path}.{key}"))
    return found


def _make_images(num_images: int) -> list[Image.Image]:
    colors = [(128, 128, 128), (160, 120, 90), (90, 140, 180), (180, 170, 90)]
    return [
        Image.new("RGB", (224, 224), color=colors[index % len(colors)])
        for index in range(num_images)
    ]


def _prepare_inputs(processor: Any, args: argparse.Namespace) -> Any:
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
        return processor.apply_chat_template(messages, **template_kwargs, **processor_kwargs)
    except TypeError:
        return processor.apply_chat_template(messages, **template_kwargs, processor_kwargs=processor_kwargs)


def _base_feature_kwargs(model: Any, inputs: Any, args: argparse.Namespace) -> dict[str, Any]:
    device = getattr(model, "device", args.device)
    kwargs: dict[str, Any] = {"pixel_values": inputs["pixel_values"].to(device)}
    target_sizes = inputs.get("target_sizes")
    if target_sizes is not None:
        kwargs["target_sizes"] = target_sizes.to(device)
    return kwargs


def _call_get_image_features_with_attentions(model: Any, inputs: Any, args: argparse.Namespace) -> dict[str, Any]:
    base = _base_feature_kwargs(model, inputs, args)
    variants: list[tuple[str, dict[str, Any]]] = [
        ("basic_output_attentions", {**base, "output_attentions": True, "return_dict": True}),
        (
            "downsample_output_attentions",
            {
                **base,
                "downsample_mode": args.downsample_mode,
                "max_slice_nums": args.max_slice_nums,
                "output_attentions": True,
                "return_dict": True,
            },
        ),
        ("basic_no_attention_arg", dict(base)),
    ]
    errors = []
    for name, kwargs in variants:
        try:
            with torch.inference_mode():
                output = model.get_image_features(**kwargs)
            return {
                "ok": True,
                "variant": name,
                "summary": _summarize(output),
                "attention_like_tensors": _find_attention_like_tensors(output),
            }
        except Exception as exc:
            errors.append({"variant": name, "error": f"{type(exc).__name__}: {exc}"})
    return {"ok": False, "errors": errors}


def _call_vision_tower_with_attentions(model: Any, inputs: Any, args: argparse.Namespace) -> dict[str, Any]:
    model_body = getattr(model, "model", None)
    vision_tower = getattr(model_body, "vision_tower", None)
    if vision_tower is None:
        return {"ok": False, "error": "model.model.vision_tower not found"}

    base = _base_feature_kwargs(model, inputs, args)
    variants: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = [
        ("kwargs_output_attentions", tuple(), {**base, "output_attentions": True, "return_dict": True}),
        ("pixel_only_output_attentions", (base["pixel_values"],), {"output_attentions": True, "return_dict": True}),
        ("pixel_target_output_attentions", (base["pixel_values"], base.get("target_sizes")), {"output_attentions": True, "return_dict": True})
        if "target_sizes" in base
        else ("pixel_only_output_attentions_dup", (base["pixel_values"],), {"output_attentions": True, "return_dict": True}),
    ]
    errors = []
    for name, positional, kwargs in variants:
        try:
            with torch.inference_mode():
                output = vision_tower(*positional, **kwargs)
            return {
                "ok": True,
                "variant": name,
                "signature": str(inspect.signature(vision_tower.forward)),
                "summary": _summarize(output),
                "attention_like_tensors": _find_attention_like_tensors(output),
            }
        except Exception as exc:
            errors.append({"variant": name, "error": f"{type(exc).__name__}: {exc}"})
    return {
        "ok": False,
        "signature": str(inspect.signature(vision_tower.forward)),
        "errors": errors,
    }


def _probe_attention_hooks(model: Any, inputs: Any, args: argparse.Namespace) -> dict[str, Any]:
    model_body = getattr(model, "model", None)
    vision_tower = getattr(model_body, "vision_tower", None)
    if vision_tower is None:
        return {"ok": False, "error": "model.model.vision_tower not found"}

    candidates = []
    for name, module in vision_tower.named_modules():
        lower_name = name.lower()
        lower_type = type(module).__name__.lower()
        if "attn" in lower_name or "attention" in lower_name or "attn" in lower_type or "attention" in lower_type:
            candidates.append((name, module))
    candidates = candidates[: args.max_hook_modules]

    captured: list[dict[str, Any]] = []

    def hook(name: str):
        def _inner(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            captured.append(
                {
                    "module": name,
                    "type": type(_module).__name__,
                    "output_summary": _summarize(output, max_depth=2),
                    "attention_like_tensors": _find_attention_like_tensors(output),
                }
            )

        return _inner

    handles = [module.register_forward_hook(hook(name)) for name, module in candidates]
    try:
        call_result = _call_get_image_features_with_attentions(model, inputs, args)
    finally:
        for handle in handles:
            handle.remove()

    return {
        "ok": True,
        "hooked_modules": [{"name": name, "type": type(module).__name__} for name, module in candidates],
        "captured": captured[: args.max_hook_records],
        "call_result_ok": call_result.get("ok"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether MiniCPM-V 4.6 exposes visual attention scores usable for StreamingTOM CTR."
    )
    parser.add_argument("--model_path", default="openbmb/MiniCPM-V-4.6")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--attn_implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "eager"))
    parser.add_argument("--num_images", type=int, default=2)
    parser.add_argument("--downsample_mode", default=os.environ.get("MINICPM_DOWNSAMPLE_MODE", "16x"))
    parser.add_argument("--max_slice_nums", type=int, default=int(os.environ.get("MINICPM_MAX_SLICE_NUMS", "1")))
    parser.add_argument("--max_hook_modules", type=int, default=12)
    parser.add_argument("--max_hook_records", type=int, default=12)
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

    inputs = _prepare_inputs(processor, args)
    image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
    image_placeholder_tokens = (
        int((inputs["input_ids"] == int(image_token_id)).sum().item()) if image_token_id is not None else None
    )

    if hasattr(model, "config"):
        model.config.output_attentions = True
    model_body = getattr(model, "model", None)
    vision_tower = getattr(model_body, "vision_tower", None)
    if vision_tower is not None and hasattr(vision_tower, "config"):
        vision_tower.config.output_attentions = True

    summary = {
        "model_class": type(model).__name__,
        "processor_class": type(processor).__name__,
        "attn_implementation": args.attn_implementation,
        "device": args.device,
        "num_images": args.num_images,
        "image_placeholder_tokens": image_placeholder_tokens,
        "get_image_features_signature": str(inspect.signature(model.get_image_features)),
        "get_image_features_probe": _call_get_image_features_with_attentions(model, inputs, args),
        "vision_tower_probe": _call_vision_tower_with_attentions(model, inputs, args),
        "hook_probe": _probe_attention_hooks(model, inputs, args),
    }

    def has_attention_scores(section: Any) -> bool:
        if isinstance(section, dict):
            tensors = section.get("attention_like_tensors")
            if isinstance(tensors, list):
                for tensor in tensors:
                    shape = tensor.get("shape", [])
                    if len(shape) >= 4 and shape[-1] == shape[-2]:
                        return True
            return any(has_attention_scores(value) for value in section.values())
        if isinstance(section, list):
            return any(has_attention_scores(item) for item in section)
        return False

    summary["likely_visual_attention_scores_available"] = has_attention_scores(summary)
    print(json.dumps(summary, indent=2))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
