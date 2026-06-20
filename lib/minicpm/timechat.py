from __future__ import annotations

import os
import time
from typing import Any

import torch
from PIL import Image

from lib.cdas_sampler import CDASConfig, select_recent_frames_cdas
from lib.minicpm.baseline import (
    RecentWindowQAModel as _MiniCPMRecentWindowQAModel,
    _build_profile,
    _capture_gpu_memory,
    _reset_gpu_memory_peaks,
    _synchronize_gpu_devices,
)
from lib.shared.recent_window import RecentWindowResult, decode_video_to_chunks_qwen
from lib.timechat.dtd import DTDConfig, DTDFrameMetadata, DifferentialTokenDropper


class TimeChatMiniCPMQAModel(_MiniCPMRecentWindowQAModel):
    """MiniCPM-V 4.6 wrapper with TimeChat-Online style DTD.

    This follows TimeChat-Online's feature-level Differential Token Drop idea:
    compare spatially aligned tokens across consecutive frames, drop temporally
    redundant visual tokens, and preserve the remaining token order. We expose
    a fixed retention ratio so experiments can be run at 100%, 80%, and 40%
    VToken budgets.
    """

    def __init__(
        self,
        model_name: str,
        device: str | torch.device = "auto",
        max_new_tokens: int = 256,
        attn_implementation: str | None = None,
        dtd_config: DTDConfig | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
            attn_implementation=attn_implementation,
        )
        self.dtd_config = dtd_config or DTDConfig(
            retention_ratio=float(os.environ.get("MINICPM_TIMECHAT_RETENTION_RATIO", "0.8")),
            protect_first_frame=os.environ.get("MINICPM_TIMECHAT_PROTECT_FIRST_FRAME", "1").strip().lower()
            not in {"0", "false", "no", "off"},
        )
        self.dtd_dropper = DifferentialTokenDropper(self.dtd_config)
        self.image_token_id = int(getattr(self.model.config, "image_token_id"))
        self._last_num_vision_tokens_before: int = 0
        self._last_num_vision_tokens_after: int = 0
        self._last_timechat_compress_features_ms: float = 0.0
        self._last_timechat_vision_encode_ms: float = 0.0
        self._last_timechat_metadata: list[dict[str, Any]] = []

    def _infer_module_device(self, module: Any) -> torch.device:
        for parameter in module.parameters():
            return parameter.device
        for buffer in module.buffers():
            return buffer.device
        if hasattr(self.model, "device"):
            return torch.device(self.model.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_vision_device(self) -> torch.device:
        model_body = getattr(self.model, "model", None)
        vision_tower = getattr(model_body, "vision_tower", None)
        if vision_tower is not None:
            return self._infer_module_device(vision_tower)
        return self._infer_module_device(self.model)

    def _get_text_input_device(self) -> torch.device:
        return self._infer_module_device(self.model.get_input_embeddings())

    @staticmethod
    def _pooled_feature_list(features: Any) -> list[torch.Tensor]:
        pooler_output = getattr(features, "pooler_output", None)
        if pooler_output is None and isinstance(features, dict):
            pooler_output = features.get("pooler_output")
        if pooler_output is None:
            raise TypeError(
                "MiniCPM get_image_features did not return pooler_output; "
                f"got {type(features)}"
            )
        if isinstance(pooler_output, torch.Tensor):
            if pooler_output.ndim == 3:
                return [item for item in pooler_output]
            return [pooler_output]
        if isinstance(pooler_output, (list, tuple)) and all(isinstance(item, torch.Tensor) for item in pooler_output):
            return list(pooler_output)
        raise TypeError(f"Unexpected MiniCPM pooler_output type: {type(pooler_output)}")

    def _call_get_image_features(self, inputs: Any) -> list[torch.Tensor]:
        vision_device = self._get_vision_device()
        pixel_values = inputs["pixel_values"].to(vision_device)
        target_sizes = inputs.get("target_sizes")
        if target_sizes is not None:
            target_sizes = target_sizes.to(vision_device)

        variants: list[dict[str, Any]] = [
            {"pixel_values": pixel_values},
            {
                "pixel_values": pixel_values,
                "target_sizes": target_sizes,
            }
            if target_sizes is not None
            else {"pixel_values": pixel_values},
            {
                "pixel_values": pixel_values,
                "target_sizes": target_sizes,
                "downsample_mode": self._last_downsample_mode,
                "max_slice_nums": self.max_slice_nums,
            }
            if target_sizes is not None
            else {
                "pixel_values": pixel_values,
                "downsample_mode": self._last_downsample_mode,
                "max_slice_nums": self.max_slice_nums,
            },
        ]

        last_error: Exception | None = None
        for kwargs in variants:
            try:
                features = self.model.get_image_features(**kwargs)
                return self._pooled_feature_list(features)
            except TypeError as exc:
                last_error = exc
                continue
        raise RuntimeError(f"Could not call MiniCPM get_image_features: {last_error}")

    @staticmethod
    def _image_token_blocks(input_ids_1d: torch.Tensor, image_token_id: int) -> list[tuple[int, int]]:
        positions = torch.nonzero(input_ids_1d == int(image_token_id), as_tuple=False).flatten().tolist()
        if not positions:
            return []
        blocks: list[tuple[int, int]] = []
        start = positions[0]
        prev = positions[0]
        for pos in positions[1:]:
            if pos == prev + 1:
                prev = pos
                continue
            blocks.append((start, prev + 1))
            start = pos
            prev = pos
        blocks.append((start, prev + 1))
        return blocks

    @staticmethod
    def _replace_image_blocks(
        input_ids_1d: torch.Tensor,
        blocks: list[tuple[int, int]],
        replacement_lengths: list[int],
        image_token_id: int,
    ) -> torch.Tensor:
        if len(blocks) != len(replacement_lengths):
            raise ValueError(
                "image block count does not match compressed feature count: "
                f"blocks={len(blocks)} replacements={len(replacement_lengths)}"
            )
        pieces: list[torch.Tensor] = []
        cursor = 0
        for (start, end), length in zip(blocks, replacement_lengths):
            pieces.append(input_ids_1d[cursor:start])
            pieces.append(
                torch.full(
                    (int(length),),
                    int(image_token_id),
                    dtype=input_ids_1d.dtype,
                    device=input_ids_1d.device,
                )
            )
            cursor = end
        pieces.append(input_ids_1d[cursor:])
        return torch.cat(pieces, dim=0)

    def _compress_pooled_features(
        self,
        pooled_features: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[DTDFrameMetadata]]:
        return self.dtd_dropper.reduce_frames(pooled_features)

    @torch.inference_mode()
    def build_timechat_model_inputs(
        self,
        frames: list[Image.Image],
        question: str,
        downsample_mode: str | None = None,
    ) -> dict[str, Any]:
        effective_downsample_mode = downsample_mode or self.downsample_mode
        self._last_downsample_mode = effective_downsample_mode
        self._last_timechat_compress_features_ms = 0.0
        self._last_timechat_vision_encode_ms = 0.0
        self._last_timechat_metadata = []

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
            inputs = self.processor.apply_chat_template(
                messages,
                **template_kwargs,
                processor_kwargs=processor_kwargs,
            )
        self._last_preprocess_seconds = time.perf_counter() - preprocess_t0

        vision_t0 = time.perf_counter()
        pooled_features = self._call_get_image_features(inputs)
        _synchronize_gpu_devices()
        self._last_timechat_vision_encode_ms = (time.perf_counter() - vision_t0) * 1000.0

        before_tokens = sum(int(item.reshape(-1, item.shape[-1]).shape[0]) for item in pooled_features)
        compress_t0 = time.perf_counter()
        compressed_features, metadata = self._compress_pooled_features(pooled_features)
        _synchronize_gpu_devices()
        self._last_timechat_compress_features_ms = (time.perf_counter() - compress_t0) * 1000.0
        self._last_timechat_metadata = [item.as_dict() for item in metadata]

        compressed_lengths = [int(item.reshape(-1, item.shape[-1]).shape[0]) for item in compressed_features]
        after_tokens = sum(compressed_lengths)
        self._last_num_vision_frames = len(pooled_features)
        self._last_num_vision_tokens_before = before_tokens
        self._last_num_vision_tokens_after = after_tokens
        self._last_num_vision_tokens = after_tokens

        input_ids = inputs["input_ids"][0].to(self._get_text_input_device())
        blocks = self._image_token_blocks(input_ids, self.image_token_id)
        if len(blocks) != len(compressed_features):
            raise ValueError(
                "MiniCPM image token blocks do not align with pooled image features: "
                f"blocks={len(blocks)} features={len(compressed_features)}"
            )
        original_lengths = [end - start for start, end in blocks]
        if sum(original_lengths) != before_tokens:
            raise ValueError(
                "MiniCPM placeholder count does not match pooled image features: "
                f"placeholders={sum(original_lengths)} pooled={before_tokens}"
            )

        compressed_input_ids_1d = self._replace_image_blocks(
            input_ids_1d=input_ids,
            blocks=blocks,
            replacement_lengths=compressed_lengths,
            image_token_id=self.image_token_id,
        )
        compressed_input_ids = compressed_input_ids_1d.unsqueeze(0)
        attention_mask = torch.ones_like(compressed_input_ids)

        inputs_embeds = self.model.get_input_embeddings()(compressed_input_ids)
        image_mask = compressed_input_ids == self.image_token_id
        image_positions = int(image_mask.sum().item())
        if image_positions != after_tokens:
            raise ValueError(
                "Compressed prompt image tokens do not match DTD output: "
                f"prompt={image_positions} compressed={after_tokens}"
            )

        compressed_embeds = torch.cat(
            [item.reshape(-1, item.shape[-1]) for item in compressed_features],
            dim=0,
        ).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_expanded, compressed_embeds)

        return {
            "input_ids": compressed_input_ids,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "prompt_length": int(compressed_input_ids.shape[1]),
            "downsample_mode": effective_downsample_mode,
            "image_mask": image_mask,
            "compressed_features": compressed_features,
            "compressed_lengths": compressed_lengths,
            "tokens_before": before_tokens,
            "tokens_after": after_tokens,
            "timechat_metadata": self._last_timechat_metadata,
        }

    @torch.inference_mode()
    def generate_from_frames(
        self,
        frames: list[Image.Image],
        question: str,
        downsample_mode: str | None = None,
    ) -> str:
        model_inputs = self.build_timechat_model_inputs(
            frames=frames,
            question=question,
            downsample_mode=downsample_mode,
        )

        answer = self._generate_from_model_inputs(
            prompt_length=int(model_inputs["prompt_length"]),
            downsample_mode=model_inputs["downsample_mode"],
            input_ids=model_inputs["input_ids"],
            inputs_embeds=model_inputs["inputs_embeds"],
            attention_mask=model_inputs["attention_mask"],
        )

        component_times = self._last_component_times
        if isinstance(component_times, dict):
            component_times["timechat_enabled"] = True
            component_times["timechat_retention_ratio"] = float(self.dtd_config.retention_ratio)
            component_times["timechat_vision_encode_ms"] = self._last_timechat_vision_encode_ms
            component_times["timechat_compress_features_ms"] = self._last_timechat_compress_features_ms
            component_times["timechat_tokens_before"] = model_inputs["tokens_before"]
            component_times["timechat_tokens_after"] = model_inputs["tokens_after"]
            component_times["timechat_frames"] = len(model_inputs["compressed_features"])
        return answer


def _apply_timechat_profile_overrides(profile_metadata: dict[str, Any], qa: TimeChatMiniCPMQAModel) -> None:
    compress_ms = float(getattr(qa, "_last_timechat_compress_features_ms", 0.0))
    vision_ms = float(getattr(qa, "_last_timechat_vision_encode_ms", 0.0))
    tokens_before = int(getattr(qa, "_last_num_vision_tokens_before", 0))
    tokens_after = int(getattr(qa, "_last_num_vision_tokens_after", 0))
    profile_metadata["timechat"] = {
        "enabled": True,
        "method": "feature_cosine_dtd",
        "retention_ratio": float(qa.dtd_config.retention_ratio),
        "protect_first_frame": bool(qa.dtd_config.protect_first_frame),
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "dropped_tokens": max(0, tokens_before - tokens_after),
        "frames": int(getattr(qa, "_last_num_vision_frames", 0)),
        "compression_time_ms": compress_ms,
        "vision_encode_time_ms": vision_ms,
        "frame_metadata": getattr(qa, "_last_timechat_metadata", []),
    }
    profile_metadata["vision_encoder_time_ms"] = vision_ms
    profile_metadata["vision_hook_subtask_time_ms"] = vision_ms
    profile_metadata["vision_total_frontend_time_ms"] = (
        float(profile_metadata.get("vision_preprocess_time_ms", 0.0)) + vision_ms
    )
    profile_metadata["st_vision_tower_ms"] = vision_ms
    profile_metadata["st_compress_features_ms"] = compress_ms
    timeline = profile_metadata.get("streamingtom_timeline_ms")
    if isinstance(timeline, dict):
        vision_components = timeline.setdefault("vision_subtask_components", {})
        vision_components["vision_tower"] = vision_ms
        vision_components["compress_features"] = compress_ms
        notes = timeline.setdefault("notes", {})
        notes["compress_features"] = "Measured TimeChat-Online DTD feature-token drop time."


def query_recent_window(
    qa: TimeChatMiniCPMQAModel,
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
    num_tokens_before = getattr(qa, "_last_num_vision_tokens_before", 0) or 0
    num_tokens_after = getattr(qa, "_last_num_vision_tokens_after", 0) or 0
    num_frames = getattr(qa, "_last_num_vision_frames", 0) or len(recent_frames)
    _synchronize_gpu_devices()
    after_memory = _capture_gpu_memory()
    profile_metadata = _build_profile(
        mode="recent_window_timechat_cdas" if cdas_metadata is not None else "recent_window_timechat",
        decode_time=decode_time,
        selection_time=selection_time,
        generate_time=generate_time,
        before_memory=before_memory,
        after_memory=after_memory,
        qa=qa,
    )
    _apply_timechat_profile_overrides(profile_metadata, qa)
    result = RecentWindowResult(
        answer=answer,
        final_chunk_ids=final_chunk_ids,
        generate_time=generate_time,
        ttft_seconds=ttft_seconds,
        num_vision_tokens=num_tokens_after,
        num_vision_tokens_before=num_tokens_before,
        num_vision_tokens_after=num_tokens_after,
        num_frames=num_frames,
    )
    result.profile_metadata = profile_metadata
    if cdas_metadata is not None:
        cdas_metadata["actual_vision_tokens"] = num_tokens_after
        cdas_metadata["actual_vision_frames"] = num_frames
        cdas_metadata["actual_downsample_mode"] = getattr(qa, "_last_downsample_mode", qa.downsample_mode)
        result.cdas_metadata = cdas_metadata
    return result, decode_backend


def query_all_frames(
    qa: TimeChatMiniCPMQAModel,
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
    num_tokens_before = getattr(qa, "_last_num_vision_tokens_before", 0) or 0
    num_tokens_after = getattr(qa, "_last_num_vision_tokens_after", 0) or 0
    num_frames = getattr(qa, "_last_num_vision_frames", 0) or len(frames)
    _synchronize_gpu_devices()
    after_memory = _capture_gpu_memory()
    profile_metadata = _build_profile(
        mode="all_frames_timechat",
        decode_time=decode_time,
        selection_time=selection_time,
        generate_time=generate_time,
        before_memory=before_memory,
        after_memory=after_memory,
        qa=qa,
    )
    _apply_timechat_profile_overrides(profile_metadata, qa)
    profile_metadata["decoded_chunks"] = len(chunks)
    profile_metadata["decoded_frames"] = len(frames)
    profile_metadata["video_start"] = video_start
    profile_metadata["video_end"] = video_end

    result = RecentWindowResult(
        answer=answer,
        final_chunk_ids=final_chunk_ids,
        generate_time=generate_time,
        ttft_seconds=ttft_seconds,
        num_vision_tokens=num_tokens_after,
        num_vision_tokens_before=num_tokens_before,
        num_vision_tokens_after=num_tokens_after,
        num_frames=num_frames,
    )
    result.profile_metadata = profile_metadata
    result.full_frame_metadata = {
        "mode": "all_frames_timechat",
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

