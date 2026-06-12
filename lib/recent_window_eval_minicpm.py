from __future__ import annotations

import copy
import os
import re
import time
from contextlib import contextmanager
from typing import Any

import torch
from PIL import Image

from lib.recent_window_eval import (
    RecentWindowResult,
    build_ovo_prompt,
    decode_video_to_chunks_qwen,
    print_ovo_results,
)
from lib.cdas_sampler import CDASConfig, select_recent_frames_cdas


def _synchronize_gpu_devices() -> None:
    if not torch.cuda.is_available():
        return
    for device_index in _profile_cuda_device_indices():
        try:
            torch.cuda.synchronize(device_index)
        except Exception:
            continue


def _profile_cuda_device_indices() -> list[int]:
    if not torch.cuda.is_available():
        return []
    if os.environ.get("MINICPM_PROFILE_ALL_VISIBLE_GPUS", "").strip().lower() in {"1", "true", "yes", "on"}:
        return list(range(torch.cuda.device_count()))
    try:
        return [int(torch.cuda.current_device())]
    except Exception:
        return [0]


def _capture_gpu_memory() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False, "devices": [], "summary": {}}

    devices: list[dict[str, Any]] = []
    summary = {
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "max_allocated_mb": 0.0,
        "max_reserved_mb": 0.0,
    }
    for device_index in _profile_cuda_device_indices():
        try:
            props = torch.cuda.get_device_properties(device_index)
            allocated = float(torch.cuda.memory_allocated(device_index)) / (1024**2)
            reserved = float(torch.cuda.memory_reserved(device_index)) / (1024**2)
            max_allocated = float(torch.cuda.max_memory_allocated(device_index)) / (1024**2)
            max_reserved = float(torch.cuda.max_memory_reserved(device_index)) / (1024**2)
        except Exception:
            continue
        devices.append(
            {
                "index": device_index,
                "name": getattr(props, "name", ""),
                "total_mb": float(getattr(props, "total_memory", 0)) / (1024**2),
                "allocated_mb": allocated,
                "reserved_mb": reserved,
                "max_allocated_mb": max_allocated,
                "max_reserved_mb": max_reserved,
            }
        )
        summary["allocated_mb"] += allocated
        summary["reserved_mb"] += reserved
        summary["max_allocated_mb"] += max_allocated
        summary["max_reserved_mb"] += max_reserved

    return {"available": True, "devices": devices, "summary": summary}


def _reset_gpu_memory_peaks() -> dict[str, Any]:
    _synchronize_gpu_devices()
    before = _capture_gpu_memory()
    if torch.cuda.is_available():
        for device_index in _profile_cuda_device_indices():
            try:
                torch.cuda.reset_peak_memory_stats(device_index)
            except Exception:
                continue
    return before


class _GeneratedTokenTTFTStreamer:
    def __init__(self, start_time: float, prompt_length: int) -> None:
        self.start_time = start_time
        self.prompt_length = int(prompt_length)
        self.ttft_seconds: float | None = None

    def put(self, value: torch.Tensor) -> None:
        if self.ttft_seconds is not None:
            return
        if isinstance(value, torch.Tensor):
            # Transformers may stream the prompt once before generated tokens.
            # Ignore that prompt chunk and record only the first generated token.
            if value.numel() >= max(2, self.prompt_length):
                return
        self.ttft_seconds = time.perf_counter() - self.start_time

    def end(self) -> None:
        pass


def _build_profile(
    *,
    mode: str,
    decode_time: float,
    selection_time: float,
    generate_time: float,
    before_memory: dict[str, Any],
    after_memory: dict[str, Any],
    qa: "RecentWindowQAModel",
) -> dict[str, Any]:
    before_summary = before_memory.get("summary", {})
    after_summary = after_memory.get("summary", {})
    peak_allocated = float(after_summary.get("max_allocated_mb", 0.0))
    peak_reserved = float(after_summary.get("max_reserved_mb", 0.0))
    before_allocated = float(before_summary.get("allocated_mb", 0.0))
    before_reserved = float(before_summary.get("reserved_mb", 0.0))
    component_times = getattr(qa, "_last_component_times", {})
    component_forward_ms = component_times.get("forward_ms_by_component", {}) if isinstance(component_times, dict) else {}
    generation_forward_ms = component_times.get("generation_forward_ms", {}) if isinstance(component_times, dict) else {}
    generation_forward_calls = component_times.get("generation_forward_calls", {}) if isinstance(component_times, dict) else {}
    vision_preprocess_ms = getattr(qa, "_last_preprocess_seconds", 0.0) * 1000.0
    vision_hook_subtask_ms = float(component_times.get("vision_subtask_ms", 0.0)) if isinstance(component_times, dict) else 0.0
    non_vision_generate_ms = float(component_times.get("non_vision_generate_ms", 0.0)) if isinstance(component_times, dict) else 0.0
    model_generate_ms = getattr(qa, "_last_model_generate_seconds", 0.0) * 1000.0
    ttft_ms = (getattr(qa, "_last_ttft_seconds", 0.0) or 0.0) * 1000.0
    prefill_forward_ms = float(generation_forward_ms.get("prefill", 0.0))
    decode_forward_ms = float(generation_forward_ms.get("decode", 0.0))
    vision_tower_ms = float(component_forward_ms.get("vision_encoder", 0.0))
    projector_ms = float(component_forward_ms.get("vision_projector", 0.0))
    resampler_ms = float(component_forward_ms.get("vision_resampler", 0.0))
    prefill_kv_ms = max(0.0, prefill_forward_ms - vision_hook_subtask_ms)
    generate_tokens_ms = max(0.0, model_generate_ms - ttft_ms)
    streamingtom_timeline_ms = {
        "vision_subtask_components": {
            "vision_tower": vision_tower_ms,
            "projector": projector_ms,
            "compress_features": 0.0,
            "prefill_kv": prefill_kv_ms,
            "store_kv": 0.0,
        },
        "query_subtask_components": {
            "retrieval_forward": 0.0,
            "reconstruct_kv": 0.0,
            "generate_first_token": ttft_ms,
            "generate_tokens": generate_tokens_ms,
        },
        "notes": {
            "compress_features": "0 for the no-StreamingTOM baseline.",
            "store_kv": "0 for the no-StreamingTOM baseline; standard KV creation is included in prefill_kv.",
            "retrieval_forward": "0 for the no-StreamingTOM baseline.",
            "reconstruct_kv": "0 for the no-StreamingTOM baseline.",
            "generate_first_token": "Measured as TTFT from model.generate start to first generated token.",
            "prefill_kv": "Estimated as root prefill forward time minus detected vision module time.",
        },
    }
    profile = {
        "mode": mode,
        "decode_time_seconds": decode_time,
        "selection_time_seconds": selection_time,
        "generate_time_seconds": generate_time,
        "end_to_end_time_seconds": decode_time + selection_time + generate_time,
        "model_generate_time_seconds": getattr(qa, "_last_model_generate_seconds", 0.0),
        "preprocess_time_seconds": getattr(qa, "_last_preprocess_seconds", 0.0),
        "component_profile_enabled": getattr(qa, "profile_components", False),
        "component_times": component_times,
        "component_times_ms": {
            key: value * 1000.0
            for key, value in component_times.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        },
        "vision_preprocess_time_ms": vision_preprocess_ms,
        "vision_encoder_time_ms": vision_tower_ms,
        "vision_resampler_time_ms": resampler_ms,
        "vision_projector_time_ms": projector_ms,
        "vision_hook_subtask_time_ms": vision_hook_subtask_ms,
        "vision_total_frontend_time_ms": vision_preprocess_ms + vision_hook_subtask_ms,
        "non_vision_generate_time_ms": non_vision_generate_ms,
        "prefill_forward_time_ms": prefill_forward_ms,
        "decode_forward_time_ms": decode_forward_ms,
        "prefill_forward_calls": int(generation_forward_calls.get("prefill", 0)),
        "decode_forward_calls": int(generation_forward_calls.get("decode", 0)),
        "prefill_kv_time_ms": prefill_kv_ms,
        "generate_first_token_time_ms": ttft_ms,
        "generate_tokens_time_ms": generate_tokens_ms,
        "model_generate_time_ms": model_generate_ms,
        "ttft_seconds": getattr(qa, "_last_ttft_seconds", 0.0) or 0.0,
        "streamingtom_timeline_ms": streamingtom_timeline_ms,
        "st_vision_tower_ms": vision_tower_ms,
        "st_projector_ms": projector_ms,
        "st_compress_features_ms": 0.0,
        "st_prefill_kv_ms": prefill_kv_ms,
        "st_store_kv_ms": 0.0,
        "st_retrieval_forward_ms": 0.0,
        "st_reconstruct_kv_ms": 0.0,
        "st_generate_first_token_ms": ttft_ms,
        "st_generate_tokens_ms": generate_tokens_ms,
        "gpu_memory_before": before_memory,
        "gpu_memory_after": after_memory,
        "gpu_peak_allocated_mb": peak_allocated,
        "gpu_peak_reserved_mb": peak_reserved,
        "gpu_peak_extra_allocated_mb": max(0.0, peak_allocated - before_allocated),
        "gpu_peak_extra_reserved_mb": max(0.0, peak_reserved - before_reserved),
    }
    return profile


class _ModuleForwardProfiler:
    def __init__(self, model: Any, targets: dict[str, Any]) -> None:
        self.model = model
        self.targets = targets
        self.handles: list[Any] = []
        self.times: dict[str, float] = {name: 0.0 for name in targets}
        self.calls: dict[str, int] = {name: 0 for name in targets}
        self._starts: dict[int, float] = {}

    def __enter__(self) -> "_ModuleForwardProfiler":
        for label, module in self.targets.items():
            module_id = id(module)

            def pre_hook(_module: Any, _inputs: Any, *, _module_id: int = module_id) -> None:
                _synchronize_gpu_devices()
                self._starts[_module_id] = time.perf_counter()

            def post_hook(_module: Any, _inputs: Any, _output: Any, *, _label: str = label, _module_id: int = module_id) -> None:
                _synchronize_gpu_devices()
                start = self._starts.pop(_module_id, None)
                if start is None:
                    return
                self.times[_label] += time.perf_counter() - start
                self.calls[_label] += 1

            self.handles.append(module.register_forward_pre_hook(pre_hook))
            self.handles.append(module.register_forward_hook(post_hook))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class _GenerationForwardProfiler:
    def __init__(self, module: Any) -> None:
        self.module = module
        self.handles: list[Any] = []
        self.times = {"prefill": 0.0, "decode": 0.0}
        self.calls = {"prefill": 0, "decode": 0}
        self.sequence_lengths: list[int | None] = []
        self._stack: list[tuple[float, str, int | None]] = []
        self._prefill_seen = False

    @staticmethod
    def _tensor_seq_len(tensor: Any) -> int | None:
        if not isinstance(tensor, torch.Tensor):
            return None
        if tensor.ndim >= 2:
            return int(tensor.shape[1])
        if tensor.ndim == 1:
            return int(tensor.shape[0])
        return None

    @classmethod
    def _infer_seq_len(cls, args: tuple[Any, ...], kwargs: dict[str, Any]) -> int | None:
        for key in ("input_ids", "inputs_embeds", "decoder_input_ids"):
            seq_len = cls._tensor_seq_len(kwargs.get(key))
            if seq_len is not None:
                return seq_len
        for item in args:
            seq_len = cls._tensor_seq_len(item)
            if seq_len is not None:
                return seq_len
        return None

    def _classify(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[str, int | None]:
        seq_len = self._infer_seq_len(args, kwargs)
        past_key_values = kwargs.get("past_key_values")
        if not self._prefill_seen:
            if past_key_values is None or (seq_len is not None and seq_len > 1):
                self._prefill_seen = True
                return "prefill", seq_len
        return "decode", seq_len

    def _pre_hook_with_kwargs(self, _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        _synchronize_gpu_devices()
        kind, seq_len = self._classify(args, kwargs)
        self.sequence_lengths.append(seq_len)
        self._stack.append((time.perf_counter(), kind, seq_len))

    def _post_hook_with_kwargs(self, _module: Any, _args: tuple[Any, ...], _kwargs: dict[str, Any], _output: Any) -> None:
        _synchronize_gpu_devices()
        if not self._stack:
            return
        start, kind, _seq_len = self._stack.pop()
        self.times[kind] += time.perf_counter() - start
        self.calls[kind] += 1

    def _pre_hook_legacy(self, _module: Any, args: tuple[Any, ...]) -> None:
        self._pre_hook_with_kwargs(_module, args, {})

    def _post_hook_legacy(self, _module: Any, _args: tuple[Any, ...], _output: Any) -> None:
        self._post_hook_with_kwargs(_module, _args, {}, _output)

    def __enter__(self) -> "_GenerationForwardProfiler":
        try:
            self.handles.append(self.module.register_forward_pre_hook(self._pre_hook_with_kwargs, with_kwargs=True))
            self.handles.append(self.module.register_forward_hook(self._post_hook_with_kwargs, with_kwargs=True))
        except TypeError:
            self.handles.append(self.module.register_forward_pre_hook(self._pre_hook_legacy))
            self.handles.append(self.module.register_forward_hook(self._post_hook_legacy))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class RecentWindowQAModel:
    """MiniCPM-V compatibility wrapper for the SimpleStream recent-frame recipe.

    MiniCPM-V-4.6 exposes the standard HF image-text-to-text interface. We keep
    SimpleStream's frame selection exactly the same, then pass the selected
    frames through MiniCPM's official processor/model generation path.
    """

    def __init__(
        self,
        model_name: str,
        device: str | torch.device = "auto",
        max_new_tokens: int = 256,
        attn_implementation: str | None = None,
    ) -> None:
        # ROCm can segfault in Transformers' threaded model materialization
        # path when multiple eval ranks load MiniCPM at the same time.
        os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "false")
        os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "1")

        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_name = model_name
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.downsample_mode = os.environ.get("MINICPM_DOWNSAMPLE_MODE", "16x")
        self.max_slice_nums = int(os.environ.get("MINICPM_MAX_SLICE_NUMS", "1"))
        self._last_ttft_seconds: float = 0.0
        self._last_num_vision_tokens: int = 0
        self._last_num_vision_frames: int = 0
        self._last_downsample_mode: str = self.downsample_mode
        self._last_preprocess_seconds: float = 0.0
        self._last_model_generate_seconds: float = 0.0
        self.profile_components = os.environ.get("MINICPM_PROFILE_COMPONENTS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._last_component_times: dict[str, Any] = {}

        self.processor = AutoProcessor.from_pretrained(model_name)

        model_kwargs: dict[str, Any] = {
            "dtype": torch.bfloat16,
            "attn_implementation": attn_implementation or os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
        }
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = str(device)

        _saved_ws = os.environ.pop("WORLD_SIZE", None)
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
        finally:
            if _saved_ws is not None:
                os.environ["WORLD_SIZE"] = _saved_ws

        self.model.eval()
        self._component_profile_targets = self._find_component_profile_targets()

    def _find_component_profile_targets(self) -> dict[str, Any]:
        if not self.profile_components:
            return {}

        modules = list(self.model.named_modules())
        selected: dict[str, Any] = {}
        used_ids: set[int] = set()

        def pick(label: str, exact_names: tuple[str, ...], patterns: tuple[str, ...]) -> None:
            exact_lower = {item.lower() for item in exact_names}
            compiled = [re.compile(pattern) for pattern in patterns]
            for name, module in modules:
                lower_name = name.lower()
                short_name = lower_name.rsplit(".", 1)[-1]
                if short_name in exact_lower or lower_name in exact_lower or any(pattern.search(lower_name) for pattern in compiled):
                    module_id = id(module)
                    if module_id in used_ids:
                        return
                    selected[label] = module
                    used_ids.add(module_id)
                    return

        pick(
            "vision_encoder",
            ("vpm", "visual", "vision_model", "vision_tower", "image_encoder", "siglip"),
            (r"(^|\.)vpm$", r"(^|\.)visual$", r"vision_(model|tower|encoder)", r"siglip"),
        )
        pick(
            "vision_resampler",
            ("resampler", "perceiver_resampler"),
            (r"resampler", r"perceiver"),
        )
        pick(
            "vision_projector",
            ("projector", "mm_projector", "multi_modal_projector", "vision_projection", "image_projection", "mlp1"),
            (r"projector", r"projection", r"(^|\.)mlp1$"),
        )
        return selected

    @contextmanager
    def _profile_model_generate_components(self):
        if not self.profile_components:
            self._last_component_times = {
                "enabled": False,
                "targets_found": [],
            }
            yield
            return

        module_profiler = _ModuleForwardProfiler(self.model, self._component_profile_targets)
        generation_profiler = _GenerationForwardProfiler(self.model)
        with generation_profiler, module_profiler:
            yield
        total_component_time = sum(module_profiler.times.values())
        self._last_component_times = {
            "enabled": True,
            "targets_found": list(self._component_profile_targets),
            "forward_seconds_by_component": module_profiler.times,
            "forward_ms_by_component": {key: value * 1000.0 for key, value in module_profiler.times.items()},
            "forward_calls_by_component": module_profiler.calls,
            "generation_forward_seconds": generation_profiler.times,
            "generation_forward_ms": {key: value * 1000.0 for key, value in generation_profiler.times.items()},
            "generation_forward_calls": generation_profiler.calls,
            "generation_forward_sequence_lengths": generation_profiler.sequence_lengths,
            "vision_subtask_seconds": total_component_time,
            "vision_subtask_ms": total_component_time * 1000.0,
            "non_vision_generate_seconds": max(0.0, self._last_model_generate_seconds - total_component_time),
            "non_vision_generate_ms": max(0.0, self._last_model_generate_seconds - total_component_time) * 1000.0,
        }

    def _estimate_vision_tokens(self, inputs: Any) -> int:
        for key in ("image_grid_thw", "video_grid_thw"):
            grid = inputs.get(key) if hasattr(inputs, "get") else None
            if grid is not None:
                return int(grid.prod(dim=-1).sum().item())

        input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
        image_token_id = getattr(getattr(self.model, "config", None), "image_token_id", None)
        if input_ids is not None and image_token_id is not None:
            return int((input_ids == int(image_token_id)).sum().item())
        return 0

    @torch.inference_mode()
    def _generate_from_model_inputs(
        self,
        prompt_length: int,
        downsample_mode: str | None = None,
        **generate_kwargs: Any,
    ) -> str:
        effective_downsample_mode = downsample_mode or self.downsample_mode
        self._last_downsample_mode = effective_downsample_mode
        t0 = time.perf_counter()
        streamer = _GeneratedTokenTTFTStreamer(t0, prompt_length=prompt_length)
        with self._profile_model_generate_components():
            generated_ids = self.model.generate(
                **generate_kwargs,
                downsample_mode=effective_downsample_mode,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                streamer=streamer,
            )
        _synchronize_gpu_devices()
        self._last_model_generate_seconds = time.perf_counter() - t0
        component_times = self._last_component_times
        if isinstance(component_times, dict) and "vision_subtask_seconds" in component_times:
            non_vision = max(0.0, self._last_model_generate_seconds - float(component_times["vision_subtask_seconds"]))
            component_times["non_vision_generate_seconds"] = non_vision
            component_times["non_vision_generate_ms"] = non_vision * 1000.0
        self._last_ttft_seconds = (
            streamer.ttft_seconds
            if streamer.ttft_seconds is not None
            else (time.perf_counter() - t0)
        )

        trimmed = [
            generated_ids[0][prompt_length:]
            if generated_ids.shape[1] > prompt_length
            else generated_ids[0]
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    @torch.inference_mode()
    def generate_from_frames(
        self,
        frames: list[Image.Image],
        question: str,
        downsample_mode: str | None = None,
    ) -> str:
        effective_downsample_mode = downsample_mode or self.downsample_mode
        content = [{"type": "image", "image": frame} for frame in frames]
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]

        template_kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        processor_kwargs: dict[str, Any] = {
            "downsample_mode": effective_downsample_mode,
            "max_slice_nums": self.max_slice_nums,
            "use_image_id": False,
        }
        preprocess_t0 = time.perf_counter()
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                **template_kwargs,
                **processor_kwargs,
            )
        except TypeError:
            # Some processor versions route MiniCPM image/video options through
            # this nested argument instead of accepting them directly.
            inputs = self.processor.apply_chat_template(
                messages,
                **template_kwargs,
                processor_kwargs=processor_kwargs,
            )
        inputs = inputs.to(self.model.device)
        self._last_preprocess_seconds = time.perf_counter() - preprocess_t0

        self._last_num_vision_frames = len(frames)
        self._last_num_vision_tokens = self._estimate_vision_tokens(inputs)

        prompt_length = int(inputs["input_ids"].shape[1])
        return self._generate_from_model_inputs(
            prompt_length=prompt_length,
            downsample_mode=effective_downsample_mode,
            **inputs,
        )


def query_recent_window(
    qa: RecentWindowQAModel,
    video_path: str,
    prompt: str,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    video_start: float | None = None,
    video_end: float | None = None,
    cdas_config: CDASConfig | None = None,
) -> tuple[RecentWindowResult, str]:
    before_memory = _reset_gpu_memory_peaks()
    decode_t0 = time.perf_counter()
    chunks, decode_backend = decode_video_to_chunks_qwen(
        video_path=video_path,
        chunk_duration=chunk_duration,
        fps=fps,
        recent_frames_only=recent_frames_only,
        video_start=video_start,
        video_end=video_end,
    )
    decode_time = time.perf_counter() - decode_t0
    if not chunks:
        raise ValueError(f"No chunks decoded from video: {video_path}")

    selection_t0 = time.perf_counter()
    window_size = max(1, int(recent_frames_only))
    cdas_metadata: dict[str, Any] | None = None
    selected_downsample_mode: str | None = None
    if cdas_config is not None and cdas_config.enabled:
        selection = select_recent_frames_cdas(
            chunks=chunks,
            window_size=window_size,
            config=cdas_config,
            default_downsample_mode=qa.downsample_mode,
        )
        recent_chunks = []
        recent_frames = selection.frames
        final_chunk_ids = selection.final_chunk_ids
        selected_downsample_mode = selection.downsample_mode
        cdas_metadata = selection.metadata
    else:
        recent_chunks = chunks[-window_size:]
        recent_frames = [frame for chunk in recent_chunks for frame in chunk.frames]
        final_chunk_ids = [chunk.chunk_index for chunk in recent_chunks]
    selection_time = time.perf_counter() - selection_t0
    if not recent_frames:
        raise ValueError(f"No frames decoded from video: {video_path}")

    t0 = time.perf_counter()
    answer = qa.generate_from_frames(
        recent_frames,
        prompt,
        downsample_mode=selected_downsample_mode,
    )
    _synchronize_gpu_devices()
    generate_time = time.perf_counter() - t0
    ttft_seconds = getattr(qa, "_last_ttft_seconds", 0.0) or 0.0
    num_vision_tokens = getattr(qa, "_last_num_vision_tokens", 0) or 0
    num_frames = getattr(qa, "_last_num_vision_frames", 0) or len(recent_frames)
    _synchronize_gpu_devices()
    after_memory = _capture_gpu_memory()
    profile_metadata = _build_profile(
        mode="recent_window_cdas" if cdas_metadata is not None else "recent_window",
        decode_time=decode_time,
        selection_time=selection_time,
        generate_time=generate_time,
        before_memory=before_memory,
        after_memory=after_memory,
        qa=qa,
    )
    result = RecentWindowResult(
        answer=answer,
        final_chunk_ids=final_chunk_ids,
        generate_time=generate_time,
        ttft_seconds=ttft_seconds,
        num_vision_tokens=num_vision_tokens,
        num_vision_tokens_before=num_vision_tokens,
        num_vision_tokens_after=num_vision_tokens,
        num_frames=num_frames,
    )
    result.profile_metadata = profile_metadata
    if cdas_metadata is not None:
        cdas_metadata["actual_vision_tokens"] = num_vision_tokens
        cdas_metadata["actual_vision_frames"] = num_frames
        cdas_metadata["actual_downsample_mode"] = getattr(qa, "_last_downsample_mode", qa.downsample_mode)
        result.cdas_metadata = cdas_metadata
    return result, decode_backend


def query_all_frames(
    qa: RecentWindowQAModel,
    video_path: str,
    prompt: str,
    chunk_duration: float,
    fps: float,
    video_start: float | None = None,
    video_end: float | None = None,
) -> tuple[RecentWindowResult, str]:
    before_memory = _reset_gpu_memory_peaks()
    decode_t0 = time.perf_counter()
    saved_exact_recent = os.environ.pop("QWEN_EXACT_RECENT_DECODE", None)
    try:
        chunks, decode_backend = decode_video_to_chunks_qwen(
            video_path=video_path,
            chunk_duration=chunk_duration,
            fps=fps,
            recent_frames_only=None,
            video_start=video_start,
            video_end=video_end,
        )
    finally:
        if saved_exact_recent is not None:
            os.environ["QWEN_EXACT_RECENT_DECODE"] = saved_exact_recent
    decode_time = time.perf_counter() - decode_t0
    if not chunks:
        raise ValueError(f"No chunks decoded from video: {video_path}")

    selection_t0 = time.perf_counter()
    frames = [frame for chunk in chunks for frame in chunk.frames]
    final_chunk_ids = [chunk.chunk_index for chunk in chunks]
    frame_timestamps = [ts for chunk in chunks for ts in chunk.frame_timestamps]
    selection_time = time.perf_counter() - selection_t0
    if not frames:
        raise ValueError(f"No frames decoded from video: {video_path}")

    t0 = time.perf_counter()
    answer = qa.generate_from_frames(frames, prompt)
    _synchronize_gpu_devices()
    generate_time = time.perf_counter() - t0
    ttft_seconds = getattr(qa, "_last_ttft_seconds", 0.0) or 0.0
    num_vision_tokens = getattr(qa, "_last_num_vision_tokens", 0) or 0
    num_frames = getattr(qa, "_last_num_vision_frames", 0) or len(frames)
    _synchronize_gpu_devices()
    after_memory = _capture_gpu_memory()
    profile_metadata = _build_profile(
        mode="all_frames",
        decode_time=decode_time,
        selection_time=selection_time,
        generate_time=generate_time,
        before_memory=before_memory,
        after_memory=after_memory,
        qa=qa,
    )
    profile_metadata["decoded_chunks"] = len(chunks)
    profile_metadata["decoded_frames"] = len(frames)
    profile_metadata["video_start"] = video_start
    profile_metadata["video_end"] = video_end

    result = RecentWindowResult(
        answer=answer,
        final_chunk_ids=final_chunk_ids,
        generate_time=generate_time,
        ttft_seconds=ttft_seconds,
        num_vision_tokens=num_vision_tokens,
        num_vision_tokens_before=num_vision_tokens,
        num_vision_tokens_after=num_vision_tokens,
        num_frames=num_frames,
    )
    result.profile_metadata = profile_metadata
    result.full_frame_metadata = {
        "mode": "all_frames",
        "decoded_chunks": len(chunks),
        "decoded_frames": len(frames),
        "selected_frames": len(frames),
        "selected_chunk_ids": final_chunk_ids,
        "selected_timestamps": frame_timestamps,
        "downsample_mode": getattr(qa, "_last_downsample_mode", qa.downsample_mode),
        "video_start": video_start,
        "video_end": video_end,
    }
    return result, decode_backend


def _result_metadata(result: RecentWindowResult, decode_backend: str) -> dict[str, Any]:
    metadata = {
        "decode_backend": decode_backend,
        "final_chunk_ids": result.final_chunk_ids,
        "generate_time": result.generate_time,
        "ttft_seconds": result.ttft_seconds,
        "num_vision_tokens": result.num_vision_tokens,
        "num_vision_tokens_before": result.num_vision_tokens_before,
        "num_vision_tokens_after": result.num_vision_tokens_after,
        "num_frames": result.num_frames,
    }
    profile_metadata = getattr(result, "profile_metadata", None)
    if profile_metadata is not None:
        metadata["profile"] = profile_metadata
        metadata["decode_time"] = profile_metadata.get("decode_time_seconds")
        metadata["end_to_end_time"] = profile_metadata.get("end_to_end_time_seconds")
        metadata["model_generate_time"] = profile_metadata.get("model_generate_time_seconds")
        metadata["preprocess_time"] = profile_metadata.get("preprocess_time_seconds")
        metadata["vision_preprocess_time_ms"] = profile_metadata.get("vision_preprocess_time_ms")
        metadata["vision_encoder_time_ms"] = profile_metadata.get("vision_encoder_time_ms")
        metadata["vision_resampler_time_ms"] = profile_metadata.get("vision_resampler_time_ms")
        metadata["vision_projector_time_ms"] = profile_metadata.get("vision_projector_time_ms")
        metadata["vision_hook_subtask_time_ms"] = profile_metadata.get("vision_hook_subtask_time_ms")
        metadata["vision_total_frontend_time_ms"] = profile_metadata.get("vision_total_frontend_time_ms")
        metadata["non_vision_generate_time_ms"] = profile_metadata.get("non_vision_generate_time_ms")
        metadata["prefill_forward_time_ms"] = profile_metadata.get("prefill_forward_time_ms")
        metadata["decode_forward_time_ms"] = profile_metadata.get("decode_forward_time_ms")
        metadata["prefill_kv_time_ms"] = profile_metadata.get("prefill_kv_time_ms")
        metadata["generate_first_token_time_ms"] = profile_metadata.get("generate_first_token_time_ms")
        metadata["generate_tokens_time_ms"] = profile_metadata.get("generate_tokens_time_ms")
        metadata["streamingtom_timeline_ms"] = profile_metadata.get("streamingtom_timeline_ms")
        metadata["st_vision_tower_ms"] = profile_metadata.get("st_vision_tower_ms")
        metadata["st_projector_ms"] = profile_metadata.get("st_projector_ms")
        metadata["st_compress_features_ms"] = profile_metadata.get("st_compress_features_ms")
        metadata["st_prefill_kv_ms"] = profile_metadata.get("st_prefill_kv_ms")
        metadata["st_store_kv_ms"] = profile_metadata.get("st_store_kv_ms")
        metadata["st_retrieval_forward_ms"] = profile_metadata.get("st_retrieval_forward_ms")
        metadata["st_reconstruct_kv_ms"] = profile_metadata.get("st_reconstruct_kv_ms")
        metadata["st_generate_first_token_ms"] = profile_metadata.get("st_generate_first_token_ms")
        metadata["st_generate_tokens_ms"] = profile_metadata.get("st_generate_tokens_ms")
        metadata["component_profile_enabled"] = profile_metadata.get("component_profile_enabled")
        metadata["gpu_peak_allocated_mb"] = profile_metadata.get("gpu_peak_allocated_mb")
        metadata["gpu_peak_reserved_mb"] = profile_metadata.get("gpu_peak_reserved_mb")
        metadata["gpu_peak_extra_allocated_mb"] = profile_metadata.get("gpu_peak_extra_allocated_mb")
        metadata["gpu_peak_extra_reserved_mb"] = profile_metadata.get("gpu_peak_extra_reserved_mb")
    full_frame_metadata = getattr(result, "full_frame_metadata", None)
    if full_frame_metadata is not None:
        metadata["full_frames"] = full_frame_metadata
    cdas_metadata = getattr(result, "cdas_metadata", None)
    if cdas_metadata is not None:
        metadata["cdas"] = cdas_metadata
    return metadata


def evaluate_ovo_backward_realtime(
    anno: dict,
    chunked_dir: str,
    qa: RecentWindowQAModel,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    cdas_config: CDASConfig | None = None,
    frame_selection: str = "recent",
) -> dict:
    video_path = os.path.join(chunked_dir, f"{anno['id']}.mp4")
    response = None
    metadata: dict = {}
    if os.path.exists(video_path):
        prompt = build_ovo_prompt(anno["task"], anno)
        if frame_selection == "all":
            result, decode_backend = query_all_frames(
                qa=qa,
                video_path=video_path,
                prompt=prompt,
                chunk_duration=chunk_duration,
                fps=fps,
            )
        else:
            result, decode_backend = query_recent_window(
                qa=qa,
                video_path=video_path,
                prompt=prompt,
                chunk_duration=chunk_duration,
                fps=fps,
                recent_frames_only=recent_frames_only,
                cdas_config=cdas_config,
            )
        response = result.answer
        metadata = _result_metadata(result, decode_backend)
    return {
        "id": anno["id"],
        "video": anno["video"],
        "task": anno["task"],
        "question": anno["question"],
        "response": response,
        "ground_truth": chr(65 + anno["gt"]),
        **metadata,
    }


def evaluate_ovo_forward(
    anno: dict,
    chunked_dir: str,
    qa: RecentWindowQAModel,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    cdas_config: CDASConfig | None = None,
    frame_selection: str = "recent",
) -> dict:
    result_anno = copy.deepcopy(anno)
    for index, test_info in enumerate(result_anno["test_info"]):
        video_path = os.path.join(chunked_dir, f"{anno['id']}_{index}.mp4")
        if not os.path.exists(video_path):
            test_info["response"] = None
            continue
        prompt = build_ovo_prompt(anno["task"], anno, index=index)
        if frame_selection == "all":
            result, decode_backend = query_all_frames(
                qa=qa,
                video_path=video_path,
                prompt=prompt,
                chunk_duration=chunk_duration,
                fps=fps,
            )
        else:
            result, decode_backend = query_recent_window(
                qa=qa,
                video_path=video_path,
                prompt=prompt,
                chunk_duration=chunk_duration,
                fps=fps,
                recent_frames_only=recent_frames_only,
                cdas_config=cdas_config,
            )
        test_info["response"] = result.answer
        test_info.update(_result_metadata(result, decode_backend))
    return result_anno
