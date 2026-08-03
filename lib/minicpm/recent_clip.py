from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from lib.minicpm.baseline import (
    RecentWindowQAModel,
    _build_profile,
    _capture_gpu_memory,
    _reset_gpu_memory_peaks,
    _synchronize_gpu_devices,
)
from lib.shared.recent_window import RecentWindowResult


class RecentClipQAModel(RecentWindowQAModel):
    """MiniCPM wrapper that feeds a short video clip instead of sampled frames."""

    @torch.inference_mode()
    def generate_from_video_clip(
        self,
        clip_path: str,
        question: str,
        downsample_mode: str | None = None,
    ) -> str:
        effective_downsample_mode = downsample_mode or self.downsample_mode
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": clip_path},
                    {"type": "text", "text": question},
                ],
            }
        ]

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
                processor_kwargs=processor_kwargs,
            )
        except TypeError:
            inputs = self.processor.apply_chat_template(
                messages,
                **template_kwargs,
                **processor_kwargs,
            )
        inputs = inputs.to(self.model.device)
        self._last_preprocess_seconds = time.perf_counter() - preprocess_t0

        self._last_num_vision_frames = 0
        self._last_num_vision_tokens = self._estimate_vision_tokens(inputs)
        prompt_length = int(inputs["input_ids"].shape[1])
        return self._generate_from_model_inputs(
            prompt_length=prompt_length,
            downsample_mode=effective_downsample_mode,
            **inputs,
        )


def cut_recent_clip(
    *,
    video_path: str | Path,
    clip_path: str | Path,
    end_time_seconds: float,
    clip_seconds: float,
) -> tuple[float, float]:
    """Create a small MP4 containing [end-clip_seconds, end]."""

    video_path = Path(video_path)
    clip_path = Path(clip_path)
    start_time = max(0.0, float(end_time_seconds) - float(clip_seconds))
    duration = max(0.05, float(end_time_seconds) - start_time)
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_time:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.3f}",
        "-an",
        "-c:v",
        os.environ.get("RECENT_CLIP_FFMPEG_CODEC", "libx264"),
        "-preset",
        "ultrafast",
        "-crf",
        os.environ.get("RECENT_CLIP_FFMPEG_CRF", "23"),
        str(clip_path),
    ]
    subprocess.run(cmd, check=True)
    return start_time, start_time + duration


def query_recent_clip(
    qa: RecentClipQAModel,
    *,
    video_path: str,
    prompt: str,
    question_time_seconds: float,
    clip_seconds: float,
    keep_clip_path: str | Path | None = None,
) -> tuple[RecentWindowResult, str]:
    before_memory = _reset_gpu_memory_peaks()
    decode_t0 = time.perf_counter()
    if keep_clip_path is None:
        tmpdir_obj = tempfile.TemporaryDirectory(prefix="minicpm_recent_clip_")
        clip_path = Path(tmpdir_obj.name) / "clip.mp4"
    else:
        tmpdir_obj = None
        clip_path = Path(keep_clip_path)
    try:
        clip_start, clip_end = cut_recent_clip(
            video_path=video_path,
            clip_path=clip_path,
            end_time_seconds=question_time_seconds,
            clip_seconds=clip_seconds,
        )
        decode_time = time.perf_counter() - decode_t0

        selection_t0 = time.perf_counter()
        selection_time = time.perf_counter() - selection_t0

        t0 = time.perf_counter()
        answer = qa.generate_from_video_clip(str(clip_path), prompt)
        _synchronize_gpu_devices()
        generate_time = time.perf_counter() - t0
        ttft_seconds = getattr(qa, "_last_ttft_seconds", 0.0) or 0.0
        num_vision_tokens = getattr(qa, "_last_num_vision_tokens", 0) or 0
        num_frames = getattr(qa, "_last_num_vision_frames", 0) or 0
        _synchronize_gpu_devices()
        after_memory = _capture_gpu_memory()
        profile_metadata = _build_profile(
            mode="recent_clip",
            decode_time=decode_time,
            selection_time=selection_time,
            generate_time=generate_time,
            before_memory=before_memory,
            after_memory=after_memory,
            qa=qa,
        )
        profile_metadata["recent_clip"] = {
            "clip_seconds": float(clip_seconds),
            "clip_start_seconds": clip_start,
            "clip_end_seconds": clip_end,
            "clip_path": str(clip_path) if keep_clip_path is not None else "",
        }
        result = RecentWindowResult(
            answer=answer,
            final_chunk_ids=list(range(int(clip_start), int(clip_end) + 1)),
            generate_time=generate_time,
            ttft_seconds=ttft_seconds,
            num_vision_tokens=num_vision_tokens,
            num_vision_tokens_before=num_vision_tokens,
            num_vision_tokens_after=num_vision_tokens,
            num_frames=num_frames,
        )
        result.profile_metadata = profile_metadata
        result.recent_clip_metadata = profile_metadata["recent_clip"]
        return result, "ffmpeg_recent_clip"
    finally:
        if tmpdir_obj is not None:
            tmpdir_obj.cleanup()
