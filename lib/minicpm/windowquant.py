from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any

import torch

from lib.minicpm.baseline import (
    _build_profile,
    _capture_gpu_memory,
    _reset_gpu_memory_peaks,
    _synchronize_gpu_devices,
    query_all_frames as _baseline_query_all_frames,
    query_recent_window as _baseline_query_recent_window,
)
from lib.minicpm.ctr import CTRMiniCPMQAModel


@dataclass
class WindowQuantConfig:
    """WindowQuant-style mixed precision settings.

    The WindowQuant paper assigns mixed precision at the visual-window level
    using query-window similarity. This implementation keeps the same selection
    policy but uses fake quantization on MiniCPM visual embeddings, because the
    public repo currently does not include the paper's custom KV-cache kernels.
    """

    window_size: int = 16
    low_bits: int = 4
    high_bits: int = 8
    high_ratio: float = 0.25
    protect_recent_windows: int = 1
    symmetric: bool = True

    @classmethod
    def from_env(cls) -> "WindowQuantConfig":
        return cls(
            window_size=int(os.environ.get("MINICPM_WINDOWQUANT_WINDOW_SIZE", "16")),
            low_bits=int(os.environ.get("MINICPM_WINDOWQUANT_LOW_BITS", "4")),
            high_bits=int(os.environ.get("MINICPM_WINDOWQUANT_HIGH_BITS", "8")),
            high_ratio=float(os.environ.get("MINICPM_WINDOWQUANT_HIGH_RATIO", "0.25")),
            protect_recent_windows=int(os.environ.get("MINICPM_WINDOWQUANT_PROTECT_RECENT_WINDOWS", "1")),
            symmetric=os.environ.get("MINICPM_WINDOWQUANT_SYMMETRIC", "1").strip().lower()
            in {"1", "true", "yes", "on"},
        )

    def validate(self) -> None:
        if self.window_size < 1:
            raise ValueError("WindowQuant window_size must be >= 1")
        if self.low_bits < 2 or self.high_bits < 2:
            raise ValueError("WindowQuant bit-widths must be >= 2")
        if self.low_bits > self.high_bits:
            raise ValueError("WindowQuant low_bits must be <= high_bits")
        if not (0.0 <= self.high_ratio <= 1.0):
            raise ValueError("WindowQuant high_ratio must be in [0, 1]")
        if self.protect_recent_windows < 0:
            raise ValueError("WindowQuant protect_recent_windows must be >= 0")


def _fake_quantize_tensor(x: torch.Tensor, bits: int, symmetric: bool = True) -> torch.Tensor:
    """Quantize-dequantize a tensor without changing its dtype or device."""

    if bits >= 16 or x.numel() == 0:
        return x
    work = x.float()
    if symmetric:
        qmax = float((1 << (bits - 1)) - 1)
        scale = work.abs().amax().clamp_min(1e-6) / qmax
        quantized = torch.clamp(torch.round(work / scale), -qmax, qmax)
        return (quantized * scale).to(dtype=x.dtype)

    qmin = 0.0
    qmax = float((1 << bits) - 1)
    xmin = work.amin()
    xmax = work.amax()
    scale = (xmax - xmin).clamp_min(1e-6) / (qmax - qmin)
    zero_point = torch.round(qmin - xmin / scale).clamp(qmin, qmax)
    quantized = torch.clamp(torch.round(work / scale + zero_point), qmin, qmax)
    return ((quantized - zero_point) * scale).to(dtype=x.dtype)


class WindowQuantMiniCPMQAModel(CTRMiniCPMQAModel):
    """MiniCPM wrapper for WindowQuant-style visual-window mixed precision.

    This reuses the MiniCPM-compatible visual embedding interception path from
    the CTR wrapper, but does not reduce tokens. Instead, visual tokens are
    split into windows, scored against the question embedding, assigned high or
    low bit-widths, and fake-quantized before LLM prefill.
    """

    def __init__(
        self,
        model_name: str,
        device: str | torch.device = "auto",
        max_new_tokens: int = 256,
        attn_implementation: str | None = None,
        windowquant_config: WindowQuantConfig | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
            attn_implementation=attn_implementation,
        )
        self.windowquant_config = windowquant_config or WindowQuantConfig.from_env()
        self.windowquant_config.validate()
        self._current_windowquant_question = ""
        self._last_windowquant_ms = 0.0
        self._last_windowquant_metadata: dict[str, Any] = {}

    def _question_embedding(self, question: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return torch.zeros((1,), device=device, dtype=dtype)
        encoded = tokenizer(question, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(device)
        if input_ids.numel() == 0:
            return torch.zeros((1,), device=device, dtype=dtype)
        embedding_layer = self.model.get_input_embeddings()
        text_embeds = embedding_layer(input_ids).float()
        query = text_embeds.mean(dim=1).squeeze(0)
        return torch.nn.functional.normalize(query, dim=0).to(dtype=dtype)

    def _window_scores_for_frame(
        self,
        frame_features: torch.Tensor,
        query_embedding: torch.Tensor,
    ) -> list[float]:
        flat = frame_features.reshape(-1, frame_features.shape[-1])
        if flat.numel() == 0:
            return []
        scores: list[float] = []
        for start in range(0, flat.shape[0], self.windowquant_config.window_size):
            window = flat[start : start + self.windowquant_config.window_size].float()
            window_vec = torch.nn.functional.normalize(window.mean(dim=0), dim=0)
            score = torch.dot(window_vec, query_embedding.float()).item()
            scores.append(float(score))
        return scores

    def _assign_bits(self, scores: list[float]) -> list[int]:
        if not scores:
            return []
        cfg = self.windowquant_config
        num_windows = len(scores)
        high_count = max(1, int(math.ceil(num_windows * cfg.high_ratio))) if cfg.high_ratio > 0 else 0
        high_indices = set()
        if high_count > 0:
            ranked = sorted(range(num_windows), key=lambda idx: scores[idx], reverse=True)
            high_indices.update(ranked[:high_count])
        if cfg.protect_recent_windows:
            start = max(0, num_windows - cfg.protect_recent_windows)
            high_indices.update(range(start, num_windows))
        return [cfg.high_bits if idx in high_indices else cfg.low_bits for idx in range(num_windows)]

    def _quantize_frame_features(
        self,
        frame_features: torch.Tensor,
        query_embedding: torch.Tensor,
        frame_index: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        original_shape = frame_features.shape
        flat = frame_features.reshape(-1, frame_features.shape[-1])
        scores = self._window_scores_for_frame(frame_features, query_embedding)
        bit_assignments = self._assign_bits(scores)
        pieces: list[torch.Tensor] = []
        window_rows: list[dict[str, Any]] = []
        for window_index, start in enumerate(range(0, flat.shape[0], self.windowquant_config.window_size)):
            end = min(flat.shape[0], start + self.windowquant_config.window_size)
            bits = bit_assignments[window_index]
            window = flat[start:end]
            pieces.append(_fake_quantize_tensor(window, bits=bits, symmetric=self.windowquant_config.symmetric))
            window_rows.append(
                {
                    "frame_index": frame_index,
                    "window_index": window_index,
                    "token_start": int(start),
                    "token_end": int(end),
                    "similarity": float(scores[window_index]),
                    "bits": int(bits),
                }
            )
        quantized = torch.cat(pieces, dim=0).reshape(original_shape) if pieces else frame_features
        return quantized, {"windows": window_rows}

    def _compress_pooled_features(self, pooled_features: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[Any]]:
        quant_t0 = time.perf_counter()
        if not pooled_features:
            self._last_windowquant_metadata = {"enabled": True, "frames": 0, "windows": []}
            return [], []

        device = pooled_features[0].device
        dtype = pooled_features[0].dtype
        query_embedding = self._question_embedding(self._current_windowquant_question, device=device, dtype=dtype)

        quantized_features: list[torch.Tensor] = []
        windows: list[dict[str, Any]] = []
        for frame_index, frame_features in enumerate(pooled_features):
            quantized, metadata = self._quantize_frame_features(
                frame_features=frame_features,
                query_embedding=query_embedding,
                frame_index=frame_index,
            )
            quantized_features.append(quantized)
            windows.extend(metadata["windows"])

        _synchronize_gpu_devices()
        self._last_windowquant_ms = (time.perf_counter() - quant_t0) * 1000.0
        high_bits = int(self.windowquant_config.high_bits)
        low_bits = int(self.windowquant_config.low_bits)
        high_windows = sum(1 for row in windows if int(row["bits"]) == high_bits)
        low_windows = sum(1 for row in windows if int(row["bits"]) == low_bits)
        self._last_windowquant_metadata = {
            "enabled": True,
            "note": (
                "WindowQuant-style query-window mixed precision with fake "
                "quantization on MiniCPM visual embeddings; no custom KV-cache "
                "kernel is used in this prototype."
            ),
            "window_size": int(self.windowquant_config.window_size),
            "low_bits": low_bits,
            "high_bits": high_bits,
            "high_ratio": float(self.windowquant_config.high_ratio),
            "protect_recent_windows": int(self.windowquant_config.protect_recent_windows),
            "frames": len(pooled_features),
            "windows_total": len(windows),
            "windows_high_bits": high_windows,
            "windows_low_bits": low_windows,
            "quantize_time_ms": float(self._last_windowquant_ms),
            "windows": windows,
        }
        return quantized_features, []

    @torch.inference_mode()
    def build_ctr_model_inputs(self, frames: list[Any], question: str, downsample_mode: str | None = None) -> dict[str, Any]:
        self._current_windowquant_question = question
        model_inputs = super().build_ctr_model_inputs(
            frames=frames,
            question=question,
            downsample_mode=downsample_mode,
        )
        self._last_ctr_metadata = []
        return model_inputs

    @torch.inference_mode()
    def generate_from_frames(
        self,
        frames: list[Any],
        question: str,
        downsample_mode: str | None = None,
    ) -> str:
        answer = super().generate_from_frames(frames=frames, question=question, downsample_mode=downsample_mode)
        component_times = self._last_component_times
        if isinstance(component_times, dict):
            component_times["windowquant_enabled"] = True
            component_times["windowquant_ms"] = self._last_windowquant_ms
            component_times["windowquant"] = self._last_windowquant_metadata
            component_times["ctr_enabled"] = False
        return answer


def _apply_windowquant_profile(profile_metadata: dict[str, Any], qa: WindowQuantMiniCPMQAModel) -> None:
    metadata = dict(getattr(qa, "_last_windowquant_metadata", {}) or {})
    profile_metadata["windowquant"] = metadata
    profile_metadata["st_compress_features_ms"] = float(metadata.get("quantize_time_ms", 0.0))
    timeline = profile_metadata.get("streamingtom_timeline_ms")
    if isinstance(timeline, dict):
        vision_components = timeline.setdefault("vision_subtask_components", {})
        vision_components["compress_features"] = float(metadata.get("quantize_time_ms", 0.0))
        notes = timeline.setdefault("notes", {})
        notes["compress_features"] = (
            "WindowQuant-style mixed-precision fake quantization time for visual "
            "embedding windows."
        )


def query_recent_window(*args: Any, **kwargs: Any) -> tuple[Any, str]:
    result, decode_backend = _baseline_query_recent_window(*args, **kwargs)
    qa = args[0] if args else kwargs.get("qa")
    profile = getattr(result, "profile_metadata", None)
    if isinstance(profile, dict) and isinstance(qa, WindowQuantMiniCPMQAModel):
        _apply_windowquant_profile(profile, qa)
        result.num_vision_tokens_before = int(getattr(qa, "_last_num_vision_tokens_before", result.num_vision_tokens))
        result.num_vision_tokens_after = int(getattr(qa, "_last_num_vision_tokens_after", result.num_vision_tokens))
    return result, decode_backend


def query_all_frames(*args: Any, **kwargs: Any) -> tuple[Any, str]:
    result, decode_backend = _baseline_query_all_frames(*args, **kwargs)
    qa = args[0] if args else kwargs.get("qa")
    profile = getattr(result, "profile_metadata", None)
    if isinstance(profile, dict) and isinstance(qa, WindowQuantMiniCPMQAModel):
        _apply_windowquant_profile(profile, qa)
        result.num_vision_tokens_before = int(getattr(qa, "_last_num_vision_tokens_before", result.num_vision_tokens))
        result.num_vision_tokens_after = int(getattr(qa, "_last_num_vision_tokens_after", result.num_vision_tokens))
    return result, decode_backend
