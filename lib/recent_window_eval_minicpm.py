from __future__ import annotations

import copy
import os
import time
from typing import Any

import torch
from PIL import Image

from lib.recent_window_eval import (
    _TTFTStreamer,
    RecentWindowResult,
    build_ovo_prompt,
    decode_video_to_chunks_qwen,
    print_ovo_results,
)
from lib.cdas_sampler import CDASConfig, select_recent_frames_cdas


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

        self.processor = AutoProcessor.from_pretrained(model_name)

        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
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
        streamer = _TTFTStreamer(t0)
        generated_ids = self.model.generate(
            **generate_kwargs,
            downsample_mode=effective_downsample_mode,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            streamer=streamer,
        )
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
    chunks, decode_backend = decode_video_to_chunks_qwen(
        video_path=video_path,
        chunk_duration=chunk_duration,
        fps=fps,
        recent_frames_only=recent_frames_only,
        video_start=video_start,
        video_end=video_end,
    )
    if not chunks:
        raise ValueError(f"No chunks decoded from video: {video_path}")

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
    if not recent_frames:
        raise ValueError(f"No frames decoded from video: {video_path}")

    t0 = time.perf_counter()
    answer = qa.generate_from_frames(
        recent_frames,
        prompt,
        downsample_mode=selected_downsample_mode,
    )
    generate_time = time.perf_counter() - t0
    ttft_seconds = getattr(qa, "_last_ttft_seconds", 0.0) or 0.0
    num_vision_tokens = getattr(qa, "_last_num_vision_tokens", 0) or 0
    num_frames = getattr(qa, "_last_num_vision_frames", 0) or len(recent_frames)
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
    if cdas_metadata is not None:
        cdas_metadata["actual_vision_tokens"] = num_vision_tokens
        cdas_metadata["actual_vision_frames"] = num_frames
        cdas_metadata["actual_downsample_mode"] = getattr(qa, "_last_downsample_mode", qa.downsample_mode)
        result.cdas_metadata = cdas_metadata
    return result, decode_backend


def evaluate_ovo_backward_realtime(
    anno: dict,
    chunked_dir: str,
    qa: RecentWindowQAModel,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    cdas_config: CDASConfig | None = None,
) -> dict:
    video_path = os.path.join(chunked_dir, f"{anno['id']}.mp4")
    response = None
    metadata: dict = {}
    if os.path.exists(video_path):
        result, decode_backend = query_recent_window(
            qa=qa,
            video_path=video_path,
            prompt=build_ovo_prompt(anno["task"], anno),
            chunk_duration=chunk_duration,
            fps=fps,
            recent_frames_only=recent_frames_only,
            cdas_config=cdas_config,
        )
        response = result.answer
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
        cdas_metadata = getattr(result, "cdas_metadata", None)
        if cdas_metadata is not None:
            metadata["cdas"] = cdas_metadata
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
) -> dict:
    result_anno = copy.deepcopy(anno)
    for index, test_info in enumerate(result_anno["test_info"]):
        video_path = os.path.join(chunked_dir, f"{anno['id']}_{index}.mp4")
        if not os.path.exists(video_path):
            test_info["response"] = None
            continue
        result, decode_backend = query_recent_window(
            qa=qa,
            video_path=video_path,
            prompt=build_ovo_prompt(anno["task"], anno, index=index),
            chunk_duration=chunk_duration,
            fps=fps,
            recent_frames_only=recent_frames_only,
            cdas_config=cdas_config,
        )
        test_info["response"] = result.answer
        test_info["decode_backend"] = decode_backend
        test_info["final_chunk_ids"] = result.final_chunk_ids
        test_info["generate_time"] = result.generate_time
        test_info["ttft_seconds"] = result.ttft_seconds
        test_info["num_vision_tokens"] = result.num_vision_tokens
        test_info["num_vision_tokens_before"] = result.num_vision_tokens_before
        test_info["num_vision_tokens_after"] = result.num_vision_tokens_after
        test_info["num_frames"] = result.num_frames
        cdas_metadata = getattr(result, "cdas_metadata", None)
        if cdas_metadata is not None:
            test_info["cdas"] = cdas_metadata
    return result_anno
