from __future__ import annotations

import time
from dataclasses import dataclass, field
from math import sqrt
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class CTRConfig:
    """Configuration for StreamingTOM-style Causal Temporal Reduction."""

    token_budget: int = 50
    similarity_threshold: float = 0.9
    dpc_k: int = 7
    merge_beta: float = 0.6
    saliency_mode: str = "norm"
    static_merge: str = "dpc"
    eps: float = 1e-6

    def validate(self) -> None:
        if self.token_budget < 1:
            raise ValueError(f"token_budget must be >= 1, got {self.token_budget}")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(
                "similarity_threshold must be in [0, 1], "
                f"got {self.similarity_threshold}"
            )
        if self.dpc_k < 1:
            raise ValueError(f"dpc_k must be >= 1, got {self.dpc_k}")
        if not 0.0 <= self.merge_beta <= 1.0:
            raise ValueError(f"merge_beta must be in [0, 1], got {self.merge_beta}")
        if self.saliency_mode not in {"norm", "uniform"}:
            raise ValueError(f"Unsupported saliency_mode: {self.saliency_mode}")
        if self.static_merge not in {"dpc", "mean"}:
            raise ValueError(f"Unsupported static_merge: {self.static_merge}")


@dataclass(slots=True)
class CTRFrameMetadata:
    frame_index: int
    input_tokens: int
    output_tokens: int
    token_budget: int
    static_tokens: int
    dynamic_tokens: int
    static_budget: int
    dynamic_budget: int
    has_previous_frame: bool
    compression_time_ms: float
    mean_temporal_similarity: float | None = None
    selected_dynamic_indices: list[int] = field(default_factory=list)
    selected_static_indices: list[int] = field(default_factory=list)
    merged_static_clusters: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "token_budget": self.token_budget,
            "static_tokens": self.static_tokens,
            "dynamic_tokens": self.dynamic_tokens,
            "static_budget": self.static_budget,
            "dynamic_budget": self.dynamic_budget,
            "has_previous_frame": self.has_previous_frame,
            "compression_time_ms": self.compression_time_ms,
            "mean_temporal_similarity": self.mean_temporal_similarity,
            "selected_dynamic_indices": self.selected_dynamic_indices,
            "selected_static_indices": self.selected_static_indices,
            "merged_static_clusters": self.merged_static_clusters,
        }


@dataclass(slots=True)
class CTRStreamOutput:
    tokens: torch.Tensor
    metadata: list[CTRFrameMetadata]

    @property
    def metadata_dicts(self) -> list[dict[str, Any]]:
        return [item.as_dict() for item in self.metadata]


class CausalTemporalReducer:
    """Token-level CTR module from StreamingTOM.

    The reducer is deliberately independent from MiniCPM internals. It expects
    per-frame visual tokens shaped ``[N, D]`` or a stream shaped ``[T, N, D]``.
    The MiniCPM integration point is after visual encoding/projecting and before
    LLM prefill.
    """

    def __init__(self, config: CTRConfig | None = None) -> None:
        self.config = config or CTRConfig()
        self.config.validate()
        self._previous_features: torch.Tensor | None = None
        self._frame_index = 0

    def reset(self) -> None:
        self._previous_features = None
        self._frame_index = 0

    @torch.no_grad()
    def reduce_stream(
        self,
        frame_tokens: torch.Tensor,
        saliency: torch.Tensor | None = None,
    ) -> CTRStreamOutput:
        """Reduce a token stream from ``[T, N, D]`` to approximately ``[T, G, D]``."""

        if frame_tokens.ndim != 3:
            raise ValueError(f"Expected frame_tokens [T, N, D], got {tuple(frame_tokens.shape)}")
        if saliency is not None and saliency.shape[:2] != frame_tokens.shape[:2]:
            raise ValueError(
                "saliency must have shape [T, N] matching frame_tokens, "
                f"got {tuple(saliency.shape)} for {tuple(frame_tokens.shape)}"
            )

        reduced_frames: list[torch.Tensor] = []
        metadata: list[CTRFrameMetadata] = []
        for frame_idx in range(int(frame_tokens.shape[0])):
            frame_saliency = saliency[frame_idx] if saliency is not None else None
            reduced, meta = self.reduce_frame(frame_tokens[frame_idx], frame_saliency)
            reduced_frames.append(reduced)
            metadata.append(meta)

        return CTRStreamOutput(tokens=torch.stack(reduced_frames, dim=0), metadata=metadata)

    @torch.no_grad()
    def reduce_frame(
        self,
        current_tokens: torch.Tensor,
        saliency: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, CTRFrameMetadata]:
        """Reduce one frame using only the previous frame as causal state."""

        t0 = time.perf_counter()
        if current_tokens.ndim != 2:
            raise ValueError(f"Expected current_tokens [N, D], got {tuple(current_tokens.shape)}")
        if saliency is not None and saliency.ndim != 1:
            raise ValueError(f"Expected saliency [N], got {tuple(saliency.shape)}")
        if saliency is not None and int(saliency.shape[0]) != int(current_tokens.shape[0]):
            raise ValueError(
                f"Saliency length {int(saliency.shape[0])} does not match "
                f"token count {int(current_tokens.shape[0])}"
            )

        n_tokens = int(current_tokens.shape[0])
        if n_tokens == 0:
            raise ValueError("current_tokens cannot be empty")

        budget = min(int(self.config.token_budget), n_tokens)
        scores = self._saliency(current_tokens, saliency)
        previous = self._previous_features
        has_previous = previous is not None and previous.shape == current_tokens.shape

        if not has_previous:
            selected = torch.sort(self._topk_indices(scores, budget)).values
            reduced = current_tokens.index_select(0, selected)
            self._previous_features = current_tokens.detach()
            meta = CTRFrameMetadata(
                frame_index=self._frame_index,
                input_tokens=n_tokens,
                output_tokens=int(reduced.shape[0]),
                token_budget=budget,
                static_tokens=0,
                dynamic_tokens=n_tokens,
                static_budget=0,
                dynamic_budget=budget,
                has_previous_frame=False,
                compression_time_ms=(time.perf_counter() - t0) * 1000.0,
                selected_dynamic_indices=selected.detach().cpu().tolist(),
            )
            self._frame_index += 1
            return reduced, meta

        similarity = F.cosine_similarity(current_tokens, previous, dim=-1, eps=self.config.eps)
        static_mask = similarity > float(self.config.similarity_threshold)
        dynamic_mask = ~static_mask
        static_indices = torch.nonzero(static_mask, as_tuple=False).flatten()
        dynamic_indices = torch.nonzero(dynamic_mask, as_tuple=False).flatten()

        static_budget, dynamic_budget = self._allocate_budget(
            budget=budget,
            static_count=int(static_indices.numel()),
            dynamic_count=int(dynamic_indices.numel()),
        )

        dynamic_selected = self._select_dynamic(
            dynamic_indices=dynamic_indices,
            saliency=scores,
            budget=dynamic_budget,
        )
        static_tokens = current_tokens.index_select(0, static_indices) if static_indices.numel() else current_tokens[:0]
        static_reduced, static_selected_local = self._merge_static(
            static_tokens=static_tokens,
            budget=static_budget,
        )
        if static_selected_local.numel() and static_indices.numel():
            static_selected = static_indices.index_select(0, static_selected_local)
        else:
            static_selected = static_indices[:0]

        feature_pieces: list[torch.Tensor] = []
        position_pieces: list[torch.Tensor] = []
        if static_reduced.numel():
            feature_pieces.append(static_reduced)
            position_pieces.append(static_selected)
        if dynamic_selected.numel():
            feature_pieces.append(current_tokens.index_select(0, dynamic_selected))
            position_pieces.append(dynamic_selected)

        if feature_pieces:
            all_features = torch.cat(feature_pieces, dim=0)
            all_positions = torch.cat(position_pieces, dim=0)
            order = torch.argsort(all_positions)
            reduced = all_features.index_select(0, order)
            selected_positions = all_positions.index_select(0, order)
        else:
            selected_positions = torch.sort(self._topk_indices(scores, budget)).values
            reduced = current_tokens.index_select(0, selected_positions)
        if int(reduced.shape[0]) < budget:
            reduced, selected_positions = self._fill_to_budget(
                current_tokens=current_tokens,
                reduced=reduced,
                selected_positions=selected_positions,
                saliency=scores,
                budget=budget,
            )

        self._previous_features = current_tokens.detach()
        meta = CTRFrameMetadata(
            frame_index=self._frame_index,
            input_tokens=n_tokens,
            output_tokens=int(reduced.shape[0]),
            token_budget=budget,
            static_tokens=int(static_indices.numel()),
            dynamic_tokens=int(dynamic_indices.numel()),
            static_budget=static_budget,
            dynamic_budget=dynamic_budget,
            has_previous_frame=True,
            compression_time_ms=(time.perf_counter() - t0) * 1000.0,
            mean_temporal_similarity=float(similarity.mean().detach().cpu().item()),
            selected_dynamic_indices=torch.sort(dynamic_selected).values.detach().cpu().tolist(),
            selected_static_indices=torch.sort(static_selected).values.detach().cpu().tolist(),
            merged_static_clusters=static_budget if int(static_indices.numel()) > static_budget else 0,
        )
        self._frame_index += 1
        return reduced[:budget], meta

    def _saliency(self, tokens: torch.Tensor, saliency: torch.Tensor | None) -> torch.Tensor:
        if saliency is not None:
            return saliency.to(device=tokens.device, dtype=torch.float32)
        if self.config.saliency_mode == "uniform":
            return torch.ones(int(tokens.shape[0]), device=tokens.device, dtype=torch.float32)
        return tokens.float().norm(dim=-1)

    @staticmethod
    def _topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
        k = min(max(int(k), 0), int(scores.shape[0]))
        if k == 0:
            return torch.empty(0, device=scores.device, dtype=torch.long)
        return torch.topk(scores.float(), k=k, largest=True, sorted=False).indices

    @staticmethod
    def _allocate_budget(budget: int, static_count: int, dynamic_count: int) -> tuple[int, int]:
        total = int(static_count) + int(dynamic_count)
        if total <= 0:
            return 0, 0
        if static_count <= 0:
            return 0, min(budget, dynamic_count)
        if dynamic_count <= 0:
            return min(budget, static_count), 0

        static_budget = int(budget * static_count / total)
        dynamic_budget = int(budget) - static_budget
        static_budget = min(static_budget, static_count)
        dynamic_budget = min(dynamic_budget, dynamic_count)

        remaining = int(budget) - static_budget - dynamic_budget
        if remaining > 0:
            static_room = max(0, static_count - static_budget)
            add_static = min(remaining, static_room)
            static_budget += add_static
            remaining -= add_static
        if remaining > 0:
            dynamic_room = max(0, dynamic_count - dynamic_budget)
            dynamic_budget += min(remaining, dynamic_room)
        return static_budget, dynamic_budget

    def _select_dynamic(self, dynamic_indices: torch.Tensor, saliency: torch.Tensor, budget: int) -> torch.Tensor:
        if budget <= 0 or dynamic_indices.numel() == 0:
            return dynamic_indices[:0]
        dynamic_scores = saliency.index_select(0, dynamic_indices)
        local = self._topk_indices(dynamic_scores, int(budget))
        return dynamic_indices.index_select(0, local)

    def _merge_static(self, static_tokens: torch.Tensor, budget: int) -> tuple[torch.Tensor, torch.Tensor]:
        static_count = int(static_tokens.shape[0])
        if budget <= 0 or static_count == 0:
            return static_tokens[:0], torch.empty(0, device=static_tokens.device, dtype=torch.long)
        if static_count <= budget:
            indices = torch.arange(static_count, device=static_tokens.device, dtype=torch.long)
            return static_tokens, indices
        if self.config.static_merge == "mean":
            return self._merge_static_by_mean(static_tokens, budget)
        return self._merge_static_by_dpc(static_tokens, budget)

    def _merge_static_by_mean(self, static_tokens: torch.Tensor, budget: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunks = torch.tensor_split(static_tokens, int(budget), dim=0)
        merged = torch.stack([chunk.mean(dim=0) for chunk in chunks], dim=0)
        indices = torch.linspace(
            0,
            int(static_tokens.shape[0]) - 1,
            steps=int(budget),
            device=static_tokens.device,
        ).round().long()
        return merged.to(dtype=static_tokens.dtype), indices

    def _merge_static_by_dpc(self, static_tokens: torch.Tensor, budget: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len, embed_dim = static_tokens.shape
        num_clusters = min(int(budget), int(seq_len))
        if num_clusters <= 0:
            return static_tokens[:0], torch.empty(0, device=static_tokens.device, dtype=torch.long)
        if num_clusters == seq_len:
            indices = torch.arange(seq_len, device=static_tokens.device, dtype=torch.long)
            return static_tokens, indices

        k_neighbors = min(int(self.config.dpc_k), seq_len - 1)
        dist_matrix = torch.cdist(static_tokens.float(), static_tokens.float()) / sqrt(float(embed_dim))
        nearest_dists, _ = torch.topk(dist_matrix, k_neighbors, dim=-1, largest=False)
        density = (-(nearest_dists**2).mean(dim=-1)).exp()
        density = density + torch.rand_like(density) * 1e-6
        higher_density_mask = density[:, None] > density[None, :]
        dist_to_higher = torch.where(higher_density_mask, dist_matrix, dist_matrix.max()).min(dim=-1)[0]
        scores = density * dist_to_higher
        centers = torch.sort(self._topk_indices(scores, num_clusters)).values

        all_indices = torch.arange(seq_len, device=static_tokens.device)
        non_center_mask = ~torch.isin(all_indices, centers)
        non_center_indices = all_indices[non_center_mask]
        if non_center_indices.numel() == 0:
            return static_tokens.index_select(0, centers), centers

        non_center_features = static_tokens.index_select(0, non_center_indices)
        dist_to_centers = dist_matrix.index_select(0, non_center_indices).index_select(1, centers)
        assignments = torch.argmin(dist_to_centers, dim=1)
        merged = []
        for cluster_idx, center_idx in enumerate(centers):
            member_mask = assignments == cluster_idx
            if member_mask.any():
                cluster_mean = non_center_features[member_mask].mean(dim=0)
                merged.append(
                    float(self.config.merge_beta) * static_tokens[center_idx]
                    + (1.0 - float(self.config.merge_beta)) * cluster_mean
                )
            else:
                merged.append(static_tokens[center_idx])
        return torch.stack(merged, dim=0).to(dtype=static_tokens.dtype), centers

    def _fill_to_budget(
        self,
        *,
        current_tokens: torch.Tensor,
        reduced: torch.Tensor,
        selected_positions: torch.Tensor,
        saliency: torch.Tensor,
        budget: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        needed = int(budget) - int(reduced.shape[0])
        if needed <= 0:
            return reduced, selected_positions
        mask = torch.ones(int(current_tokens.shape[0]), device=current_tokens.device, dtype=torch.bool)
        if selected_positions.numel():
            mask[selected_positions.long()] = False
        candidate_indices = torch.nonzero(mask, as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            return reduced, selected_positions
        candidate_scores = saliency.index_select(0, candidate_indices)
        local = self._topk_indices(candidate_scores, needed)
        fill_indices = candidate_indices.index_select(0, local)
        all_positions = torch.cat([selected_positions, fill_indices], dim=0)
        all_features = torch.cat([reduced, current_tokens.index_select(0, fill_indices)], dim=0)
        order = torch.argsort(all_positions)
        return all_features.index_select(0, order), all_positions.index_select(0, order)
