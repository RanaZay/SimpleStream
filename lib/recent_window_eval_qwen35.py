from __future__ import annotations

import copy
import os
import time

import torch
from PIL import Image

from lib.recent_window_eval import (
    RecentWindowResult,
    build_ovo_prompt,
    decode_video_to_chunks_qwen,
)
from lib.recent_window_eval_qwen3 import (
    RecentWindowQAModel as _Qwen3RecentWindowQAModel,
    print_ovo_results,
)


class RecentWindowQAModel(_Qwen3RecentWindowQAModel):
    """Qwen3.5 compatibility wrapper for the SimpleStream recent-frame recipe.

    Qwen3.5 uses newer Transformers internals than the Qwen3-VL release used by
    SimpleStream. We therefore keep the same recent-frame selection, but generate
    through the official processor/model path instead of the Qwen3 cached-vision
    feature builder.
    """

    def __init__(
        self,
        model_name: str,
        device: str | torch.device = "auto",
        max_new_tokens: int = 256,
        attn_implementation: str | None = None,
        enable_thinking: bool | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
            attn_implementation=attn_implementation or os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
        )
        if enable_thinking is None:
            enable_thinking = os.environ.get("QWEN35_ENABLE_THINKING", "").strip().lower() in {"1", "true", "yes", "on"}
        self.enable_thinking = bool(enable_thinking)

    @torch.inference_mode()
    def generate_from_frames(self, frames: list[Image.Image], question: str) -> str:
        content = [{"type": "image", "image": frame} for frame in frames]
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]

        chat_template_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if self.enable_thinking:
            chat_template_kwargs["enable_thinking"] = True
        inputs = self.processor.apply_chat_template(messages, **chat_template_kwargs)
        inputs = inputs.to(self.model.device)

        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            grid_rows = image_grid_thw.to(self.model.device)
            self._last_num_vision_frames = int(grid_rows.shape[0])
            self._last_num_vision_tokens = int((grid_rows.prod(dim=-1) // (self.merge_size**2)).sum().item())
        else:
            self._last_num_vision_frames = len(frames)
            self._last_num_vision_tokens = 0

        prompt_length = int(inputs["input_ids"].shape[1])
        return self._generate_from_model_inputs(prompt_length=prompt_length, **inputs)


def query_recent_window(
    qa: RecentWindowQAModel,
    video_path: str,
    prompt: str,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    video_start: float | None = None,
    video_end: float | None = None,
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
    recent_chunks = chunks[-window_size:]
    recent_frames = [frame for chunk in recent_chunks for frame in chunk.frames]
    if not recent_frames:
        raise ValueError(f"No frames decoded from video: {video_path}")

    t0 = time.perf_counter()
    answer = qa.generate_from_frames(recent_frames, prompt)
    generate_time = time.perf_counter() - t0
    ttft_seconds = getattr(qa, "_last_ttft_seconds", 0.0) or 0.0
    num_vision_tokens = getattr(qa, "_last_num_vision_tokens", 0) or 0
    num_frames = getattr(qa, "_last_num_vision_frames", 0) or len(recent_frames)

    return (
        RecentWindowResult(
            answer=answer,
            final_chunk_ids=[chunk.chunk_index for chunk in recent_chunks],
            generate_time=generate_time,
            ttft_seconds=ttft_seconds,
            num_vision_tokens=num_vision_tokens,
            num_vision_tokens_before=num_vision_tokens,
            num_vision_tokens_after=num_vision_tokens,
            num_frames=num_frames,
        ),
        decode_backend,
    )


def evaluate_ovo_backward_realtime(
    anno: dict,
    chunked_dir: str,
    qa: RecentWindowQAModel,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
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
    return result_anno
