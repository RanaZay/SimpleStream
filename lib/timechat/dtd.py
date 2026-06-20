from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DTDConfig:
    """Feature-level Differential Token Drop configuration.

    TimeChat-Online exposes a threshold argument. In relative mode, the
    threshold is the percentage of tokens to drop per frame. Our comparison
    table is phrased as visual-token retention, so this adapter converts
    retention into TimeChat's per-frame relative drop ratio:

    ``drop_ratio = 1 - retention_ratio``.
    """

    retention_ratio: float = 0.8
    protect_first_frame: bool = True
    score_method: str = "feature_cosine"

    def validate(self) -> None:
        if not 0.0 < float(self.retention_ratio) <= 1.0:
            raise ValueError("retention_ratio must be in (0, 1].")
        if self.score_method != "feature_cosine":
            raise ValueError("Only feature_cosine DTD is implemented for MiniCPM.")


@dataclass
class DTDFrameMetadata:
    frame_index: int
    input_tokens: int
    output_tokens: int
    dropped_tokens: int
    target_retention_ratio: float
    actual_retention_ratio: float
    has_previous_frame: bool
    mean_temporal_similarity: float | None
    min_temporal_similarity: float | None
    max_temporal_similarity: float | None
    compression_time_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "input_tokens": int(self.input_tokens),
            "output_tokens": int(self.output_tokens),
            "dropped_tokens": int(self.dropped_tokens),
            "target_retention_ratio": float(self.target_retention_ratio),
            "actual_retention_ratio": float(self.actual_retention_ratio),
            "has_previous_frame": bool(self.has_previous_frame),
            "mean_temporal_similarity": self.mean_temporal_similarity,
            "min_temporal_similarity": self.min_temporal_similarity,
            "max_temporal_similarity": self.max_temporal_similarity,
            "compression_time_ms": float(self.compression_time_ms),
        }


class DifferentialTokenDropper:
    """Feature-level DTD for frame-wise visual embeddings.

    For each frame after the first, aligned visual tokens are compared against
    the previous original frame using cosine similarity. In the same relative
    mode used by the official repo, the highest-similarity tokens are dropped
    independently per frame. Output tokens preserve the original frame/time
    order.
    """

    def __init__(self, config: DTDConfig | None = None) -> None:
        self.config = config or DTDConfig()
        self.config.validate()

    @staticmethod
    def _flatten_frame(frame_features: torch.Tensor) -> torch.Tensor:
        if frame_features.ndim < 2:
            raise ValueError(f"Expected frame features with at least 2 dims, got {tuple(frame_features.shape)}")
        return frame_features.reshape(-1, frame_features.shape[-1])

    @staticmethod
    def _stats(values: torch.Tensor) -> tuple[float | None, float | None, float | None]:
        if values.numel() == 0:
            return None, None, None
        values_float = values.detach().float()
        return (
            float(values_float.mean().item()),
            float(values_float.min().item()),
            float(values_float.max().item()),
        )

    def _score_frames(
        self,
        frames: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
        scores: list[torch.Tensor] = []
        stats: list[dict[str, Any]] = []
        previous: torch.Tensor | None = None
        for frame_index, frame in enumerate(frames):
            frame_flat = self._flatten_frame(frame)
            token_count = int(frame_flat.shape[0])
            if token_count == 0:
                scores.append(torch.empty(0, device=frame_flat.device, dtype=torch.float32))
                stats.append(
                    {
                        "has_previous_frame": previous is not None,
                        "mean": None,
                        "min": None,
                        "max": None,
                    }
                )
                previous = frame_flat.detach()
                continue

            if previous is None:
                if self.config.protect_first_frame:
                    score = torch.full((token_count,), float("inf"), device=frame_flat.device, dtype=torch.float32)
                else:
                    score = torch.ones(token_count, device=frame_flat.device, dtype=torch.float32)
                mean_sim, min_sim, max_sim = None, None, None
            else:
                aligned_tokens = min(int(previous.shape[0]), token_count)
                score = torch.full((token_count,), float("-inf"), device=frame_flat.device, dtype=torch.float32)
                if aligned_tokens > 0:
                    similarity = F.cosine_similarity(
                        previous[:aligned_tokens].detach().float(),
                        frame_flat[:aligned_tokens].detach().float(),
                        dim=1,
                    )
                    score[:aligned_tokens] = similarity
                    mean_sim, min_sim, max_sim = self._stats(similarity)
                else:
                    mean_sim, min_sim, max_sim = None, None, None

            scores.append(score)
            stats.append(
                {
                    "has_previous_frame": previous is not None,
                    "mean": mean_sim,
                    "min": min_sim,
                    "max": max_sim,
                }
            )
            previous = frame_flat.detach()
        return scores, stats

    @staticmethod
    def _select_keep_masks_per_frame(scores: list[torch.Tensor], drop_ratio: float) -> list[torch.Tensor]:
        """Match TimeChat-Online relative DTD: drop top-similar tokens per frame."""
        masks: list[torch.Tensor] = []
        for score in scores:
            token_count = int(score.numel())
            keep_mask = torch.ones(token_count, device=score.device, dtype=torch.bool)
            if token_count == 0:
                masks.append(keep_mask)
                continue
            if bool(torch.isinf(score).all().item()):
                # Official DTD keeps the first visual frame because there is no
                # previous frame to compare against.
                masks.append(keep_mask)
                continue
            k = int(token_count * float(drop_ratio))
            if k > 0:
                k = min(k, token_count - 1)
                if k > 0:
                    drop_indices = torch.topk(score, k=k, largest=True, sorted=False).indices
                    keep_mask[drop_indices] = False
            masks.append(keep_mask)
        return masks

    @torch.inference_mode()
    def reduce_frames(
        self,
        pooled_features: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[DTDFrameMetadata]]:
        if not pooled_features:
            return [], []

        reduce_t0 = time.perf_counter()
        frames = [self._flatten_frame(frame) for frame in pooled_features]

        if float(self.config.retention_ratio) >= 1.0:
            elapsed_per_frame_ms = ((time.perf_counter() - reduce_t0) * 1000.0) / max(1, len(frames))
            metadata: list[DTDFrameMetadata] = []
            for frame_index, frame in enumerate(frames):
                token_count = int(frame.shape[0])
                metadata.append(
                    DTDFrameMetadata(
                        frame_index=frame_index,
                        input_tokens=token_count,
                        output_tokens=token_count,
                        dropped_tokens=0,
                        target_retention_ratio=float(self.config.retention_ratio),
                        actual_retention_ratio=1.0,
                        has_previous_frame=frame_index > 0,
                        mean_temporal_similarity=None,
                        min_temporal_similarity=None,
                        max_temporal_similarity=None,
                        compression_time_ms=elapsed_per_frame_ms,
                    )
                )
            return frames, metadata

        scores, score_stats = self._score_frames(frames)
        drop_ratio = 1.0 - float(self.config.retention_ratio)
        keep_masks = self._select_keep_masks_per_frame(scores, drop_ratio)
        elapsed_per_frame_ms = ((time.perf_counter() - reduce_t0) * 1000.0) / max(1, len(frames))

        reduced_frames: list[torch.Tensor] = []
        metadata = []
        for frame_index, (frame, keep_mask, stats) in enumerate(zip(frames, keep_masks, score_stats)):
            if frame.numel() == 0:
                reduced = frame
            else:
                keep_mask = keep_mask.to(frame.device)
                if not bool(keep_mask.any().item()):
                    # Keep at least one token per non-empty frame so the prompt
                    # never contains an empty image block.
                    local_score = scores[frame_index]
                    keep_mask = torch.zeros_like(keep_mask, dtype=torch.bool)
                    keep_mask[int(torch.argmax(local_score).item())] = True
                reduced = frame[keep_mask]
            input_tokens = int(frame.shape[0])
            output_tokens = int(reduced.shape[0])
            reduced_frames.append(reduced)
            metadata.append(
                DTDFrameMetadata(
                    frame_index=frame_index,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    dropped_tokens=max(0, input_tokens - output_tokens),
                    target_retention_ratio=float(self.config.retention_ratio),
                    actual_retention_ratio=(output_tokens / input_tokens) if input_tokens else 0.0,
                    has_previous_frame=bool(stats["has_previous_frame"]),
                    mean_temporal_similarity=stats["mean"],
                    min_temporal_similarity=stats["min"],
                    max_temporal_similarity=stats["max"],
                    compression_time_ms=elapsed_per_frame_ms,
                )
            )

        return reduced_frames, metadata
