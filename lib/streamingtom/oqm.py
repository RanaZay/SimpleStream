from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class OQMConfig:
    """Configuration for StreamingTOM-style Online Quantized Memory."""

    retrieval_max_tokens: int = 12544
    enable_quantization: bool = True
    quantization_bits: int = 4
    group_size: int = 50
    init_token_count: int = 14
    sliding_window_size: int = 4800
    eps: float = 1e-8

    def validate(self) -> None:
        if self.retrieval_max_tokens < 1:
            raise ValueError("retrieval_max_tokens must be positive")
        if self.quantization_bits not in {2, 4}:
            raise ValueError("quantization_bits must be 2 or 4")
        if self.group_size < 1:
            raise ValueError("group_size must be positive")
        if self.init_token_count < 0:
            raise ValueError("init_token_count must be non-negative")
        if self.sliding_window_size < 1:
            raise ValueError("sliding_window_size must be positive")
        pack_size = 8 // int(self.quantization_bits)
        if int(self.group_size) % pack_size != 0:
            raise ValueError(
                f"group_size={self.group_size} must be divisible by pack_size={pack_size}"
            )


@dataclass(slots=True)
class OQMStoreMetadata:
    video_id: str
    layer_idx: int
    stored_tokens: int
    stored_groups: int
    store_time_ms: float
    quantized: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "layer_idx": self.layer_idx,
            "stored_tokens": self.stored_tokens,
            "stored_groups": self.stored_groups,
            "store_time_ms": self.store_time_ms,
            "quantized": self.quantized,
        }


@dataclass(slots=True)
class OQMRetrievalMetadata:
    video_id: str
    layer_idx: int
    available_groups: int
    selected_groups: list[int]
    retrieved_tokens: int
    retrieval_time_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "layer_idx": self.layer_idx,
            "available_groups": self.available_groups,
            "selected_groups": self.selected_groups,
            "retrieved_tokens": self.retrieved_tokens,
            "retrieval_time_ms": self.retrieval_time_ms,
        }


@dataclass(slots=True)
class OQMReconstructMetadata:
    video_id: str
    layer_idx: int
    output_tokens: int
    reconstruct_time_ms: float
    mode: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "layer_idx": self.layer_idx,
            "output_tokens": self.output_tokens,
            "reconstruct_time_ms": self.reconstruct_time_ms,
            "mode": self.mode,
        }


class OnlineQuantizedMemory:
    """Standalone OQM memory used by StreamingTOM.

    The class stores layer KV tensors shaped ``[B, H, S, D]``. System/prompt
    tokens are kept in their original dtype. Vision tokens are stored in
    fixed-size groups, optionally quantized to 4-bit or 2-bit values. Retrieval
    operates at the group level and reconstructs only the selected groups.

    This file is intentionally independent from MiniCPM internals. The MiniCPM
    StreamingTOM wrapper feeds it each online prefill chunk's newly produced KV
    tensors and token group keys, then uses ``get_windowed_kv`` during visual
    streaming and ``get_selective_kv`` during query answering.
    """

    def __init__(self, config: OQMConfig | None = None) -> None:
        self.config = config or OQMConfig()
        self.config.validate()
        self.quantization_levels = (1 << int(self.config.quantization_bits)) - 1
        self.pack_size = 8 // int(self.config.quantization_bits)
        self.kv_cache_storage: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
        self.quantized_storage: dict[str, dict[int, dict[str, Any]]] = {}
        self.group_keys: dict[str, dict[int, torch.Tensor]] = {}
        self.stored_tokens_count: dict[str, dict[int, int]] = {}
        self.original_dtype: dict[str, dict[int, torch.dtype]] = {}

    def clear(self, video_id: str | None = None) -> None:
        storages = [
            self.kv_cache_storage,
            self.quantized_storage,
            self.group_keys,
            self.stored_tokens_count,
            self.original_dtype,
        ]
        if video_id is None:
            for storage in storages:
                storage.clear()
            return
        for storage in storages:
            storage.pop(video_id, None)

    def store_system_prompt(
        self,
        video_id: str,
        layer_idx: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> OQMStoreMetadata:
        t0 = time.perf_counter()
        self._validate_kv(key_cache, value_cache)
        if int(key_cache.shape[2]) != int(self.config.init_token_count):
            raise ValueError(
                "system prompt length does not match init_token_count: "
                f"{int(key_cache.shape[2])} vs {self.config.init_token_count}"
            )
        self._ensure_layer(video_id, layer_idx)
        self.original_dtype.setdefault(video_id, {})[layer_idx] = key_cache.dtype
        if self.config.enable_quantization:
            self.quantized_storage[video_id][layer_idx]["init_tokens"] = (
                key_cache.detach(),
                value_cache.detach(),
            )
        else:
            self.kv_cache_storage.setdefault(video_id, {})[layer_idx] = (
                key_cache.detach(),
                value_cache.detach(),
            )
        self.stored_tokens_count.setdefault(video_id, {})[layer_idx] = int(key_cache.shape[2])
        return OQMStoreMetadata(
            video_id=video_id,
            layer_idx=layer_idx,
            stored_tokens=int(key_cache.shape[2]),
            stored_groups=0,
            store_time_ms=(time.perf_counter() - t0) * 1000.0,
            quantized=bool(self.config.enable_quantization),
        )

    def store_kv_cache(
        self,
        video_id: str,
        layer_idx: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        token_level_keys: torch.Tensor | None = None,
    ) -> OQMStoreMetadata:
        t0 = time.perf_counter()
        self._validate_kv(key_cache, value_cache)
        new_tokens = int(key_cache.shape[2])
        if new_tokens <= 0:
            raise ValueError("Cannot store empty KV cache")
        if new_tokens % int(self.config.group_size) != 0:
            raise ValueError(
                f"new token count {new_tokens} must be divisible by group_size={self.config.group_size}"
            )

        self._ensure_layer(video_id, layer_idx)
        existing = self.stored_tokens_count.setdefault(video_id, {}).get(layer_idx, 0)
        if existing < int(self.config.init_token_count):
            raise ValueError("store_system_prompt must be called before store_kv_cache")
        self.original_dtype.setdefault(video_id, {})[layer_idx] = key_cache.dtype

        if self.config.enable_quantization:
            storage = self.quantized_storage[video_id][layer_idx]
            storage["key_blocks"].append(self._quantize_tensor(key_cache.detach()))
            storage["value_blocks"].append(self._quantize_tensor(value_cache.detach()))
        else:
            cache = self.kv_cache_storage.setdefault(video_id, {})
            if layer_idx in cache:
                old_k, old_v = cache[layer_idx]
                cache[layer_idx] = (
                    torch.cat([old_k, key_cache.detach()], dim=2),
                    torch.cat([old_v, value_cache.detach()], dim=2),
                )
            else:
                cache[layer_idx] = (key_cache.detach(), value_cache.detach())

        if token_level_keys is None:
            token_level_keys = self._token_keys_from_kv(key_cache)
        self.store_token_keys_as_groups(video_id, layer_idx, token_level_keys)
        self.stored_tokens_count[video_id][layer_idx] = existing + new_tokens
        return OQMStoreMetadata(
            video_id=video_id,
            layer_idx=layer_idx,
            stored_tokens=new_tokens,
            stored_groups=new_tokens // int(self.config.group_size),
            store_time_ms=(time.perf_counter() - t0) * 1000.0,
            quantized=bool(self.config.enable_quantization),
        )

    def store_token_keys_as_groups(
        self,
        video_id: str,
        layer_idx: int,
        token_level_keys: torch.Tensor,
    ) -> torch.Tensor:
        if token_level_keys.ndim != 2:
            raise ValueError(f"token_level_keys must be [S, D], got {tuple(token_level_keys.shape)}")
        num_tokens = int(token_level_keys.shape[0])
        if num_tokens % int(self.config.group_size) != 0:
            raise ValueError(
                f"token key count {num_tokens} must be divisible by group_size={self.config.group_size}"
            )
        grouped = token_level_keys.reshape(num_tokens // int(self.config.group_size), int(self.config.group_size), -1)
        new_keys = grouped.float().mean(dim=1).detach()
        self.group_keys.setdefault(video_id, {})
        if layer_idx in self.group_keys[video_id]:
            old = self.group_keys[video_id][layer_idx]
            if int(old.shape[1]) != int(new_keys.shape[1]):
                raise ValueError(f"group key dimension changed: {old.shape[1]} vs {new_keys.shape[1]}")
            self.group_keys[video_id][layer_idx] = torch.cat([old, new_keys.to(old.device)], dim=0)
        else:
            self.group_keys[video_id][layer_idx] = new_keys
        return new_keys

    def retrieve_group_indices(
        self,
        video_id: str,
        layer_idx: int,
        query_key: torch.Tensor,
        max_tokens: int | None = None,
        top_k: int | None = None,
    ) -> tuple[torch.Tensor, OQMRetrievalMetadata]:
        t0 = time.perf_counter()
        group_keys = self.get_group_keys(video_id, layer_idx)
        if group_keys is None or int(group_keys.shape[0]) == 0:
            empty = torch.empty(0, dtype=torch.long, device=query_key.device)
            return empty, OQMRetrievalMetadata(video_id, layer_idx, 0, [], 0, 0.0)

        query = query_key.float()
        if query.ndim == 1:
            query = query.unsqueeze(0)
        if query.ndim != 2:
            raise ValueError(f"query_key must be [D] or [Q, D], got {tuple(query_key.shape)}")
        if int(query.shape[-1]) != int(group_keys.shape[-1]):
            raise ValueError(f"query/group key dim mismatch: {query.shape[-1]} vs {group_keys.shape[-1]}")

        max_tokens = int(max_tokens or self.config.retrieval_max_tokens)
        max_groups = max(1, max_tokens // int(self.config.group_size))
        if top_k is not None:
            max_groups = min(max_groups, max(1, int(top_k)))
        max_groups = min(max_groups, int(group_keys.shape[0]))

        group_keys_device = group_keys.to(query.device)
        query_norm = F.normalize(query, dim=-1, eps=self.config.eps)
        key_norm = F.normalize(group_keys_device.float(), dim=-1, eps=self.config.eps)
        scores = query_norm @ key_norm.transpose(0, 1)
        scores = scores.max(dim=0).values
        selected = torch.topk(scores, k=max_groups, largest=True, sorted=False).indices
        selected = torch.sort(selected).values
        metadata = OQMRetrievalMetadata(
            video_id=video_id,
            layer_idx=layer_idx,
            available_groups=int(group_keys.shape[0]),
            selected_groups=selected.detach().cpu().tolist(),
            retrieved_tokens=int(selected.numel()) * int(self.config.group_size),
            retrieval_time_ms=(time.perf_counter() - t0) * 1000.0,
        )
        return selected, metadata

    def get_selective_kv(
        self,
        video_id: str,
        layer_idx: int,
        selected_vision_group_indices: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], OQMReconstructMetadata]:
        t0 = time.perf_counter()
        if selected_vision_group_indices.ndim != 1:
            raise ValueError("selected_vision_group_indices must be 1D")
        selected = torch.sort(selected_vision_group_indices.long()).values
        if selected.numel() == 0:
            raise ValueError("At least one group must be selected")

        total_vision_groups = self._total_vision_groups(video_id, layer_idx)
        if int(selected.min().item()) < 0 or int(selected.max().item()) >= total_vision_groups:
            raise ValueError("selected group index out of range")

        if self.config.enable_quantization:
            kv = self._reconstruct_selective_quantized(video_id, layer_idx, selected)
        else:
            kv = self._slice_selective_unquantized(video_id, layer_idx, selected)
        metadata = OQMReconstructMetadata(
            video_id=video_id,
            layer_idx=layer_idx,
            output_tokens=int(kv[0].shape[2]),
            reconstruct_time_ms=(time.perf_counter() - t0) * 1000.0,
            mode="selective",
        )
        return kv, metadata

    def get_windowed_kv(
        self,
        video_id: str,
        layer_idx: int,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], OQMReconstructMetadata]:
        t0 = time.perf_counter()
        total_groups = self._total_vision_groups(video_id, layer_idx)
        groups_needed = min(total_groups, int(self.config.sliding_window_size) // int(self.config.group_size))
        if groups_needed <= 0:
            raise ValueError("No vision KV groups are available")
        selected = torch.arange(total_groups - groups_needed, total_groups, dtype=torch.long)
        if self.config.enable_quantization:
            kv = self._reconstruct_selective_quantized(video_id, layer_idx, selected)
        else:
            kv = self._slice_selective_unquantized(video_id, layer_idx, selected)
        metadata = OQMReconstructMetadata(
            video_id=video_id,
            layer_idx=layer_idx,
            output_tokens=int(kv[0].shape[2]),
            reconstruct_time_ms=(time.perf_counter() - t0) * 1000.0,
            mode="windowed",
        )
        return kv, metadata

    def get_group_keys(self, video_id: str, layer_idx: int) -> torch.Tensor | None:
        return self.group_keys.get(video_id, {}).get(layer_idx)

    def storage_summary(self, video_id: str | None = None) -> dict[str, Any]:
        video_ids = [video_id] if video_id is not None else sorted(
            set(self.stored_tokens_count) | set(self.quantized_storage) | set(self.kv_cache_storage)
        )
        total_original = 0
        total_stored = 0
        layers: list[dict[str, Any]] = []
        for vid in video_ids:
            if vid is None:
                continue
            for layer_idx, token_count in self.stored_tokens_count.get(vid, {}).items():
                original_bytes = self._original_layer_bytes(vid, layer_idx)
                stored_bytes = self._stored_layer_bytes(vid, layer_idx)
                total_original += original_bytes
                total_stored += stored_bytes
                layers.append(
                    {
                        "video_id": vid,
                        "layer_idx": layer_idx,
                        "stored_tokens": token_count,
                        "vision_groups": self._total_vision_groups(vid, layer_idx),
                        "original_bytes_estimated": original_bytes,
                        "stored_bytes_estimated": stored_bytes,
                        "compression_ratio_estimated": (original_bytes / stored_bytes) if stored_bytes else None,
                    }
                )
        return {
            "quantized": bool(self.config.enable_quantization),
            "quantization_bits": int(self.config.quantization_bits),
            "group_size": int(self.config.group_size),
            "layers": layers,
            "total_original_bytes_estimated": total_original,
            "total_stored_bytes_estimated": total_stored,
            "compression_ratio_estimated": (total_original / total_stored) if total_stored else None,
        }

    def _ensure_layer(self, video_id: str, layer_idx: int) -> None:
        self.stored_tokens_count.setdefault(video_id, {})
        self.original_dtype.setdefault(video_id, {})
        if self.config.enable_quantization:
            self.quantized_storage.setdefault(video_id, {}).setdefault(
                layer_idx,
                {
                    "init_tokens": None,
                    "key_blocks": [],
                    "value_blocks": [],
                },
            )
        else:
            self.kv_cache_storage.setdefault(video_id, {})

    @staticmethod
    def _validate_kv(key_cache: torch.Tensor, value_cache: torch.Tensor) -> None:
        if key_cache.shape != value_cache.shape:
            raise ValueError(f"K/V shape mismatch: {tuple(key_cache.shape)} vs {tuple(value_cache.shape)}")
        if key_cache.ndim != 4:
            raise ValueError(f"KV tensors must be [B, H, S, D], got {tuple(key_cache.shape)}")

    def _token_keys_from_kv(self, key_cache: torch.Tensor) -> torch.Tensor:
        # Fallback retrieval key when semantic token features are unavailable.
        return key_cache.detach().float().mean(dim=(0, 1))

    def _pack_nbit(self, tensor: torch.Tensor) -> torch.Tensor:
        *batch_dims, width = tensor.shape
        if width % self.pack_size != 0:
            raise ValueError(f"Cannot pack width={width} with pack_size={self.pack_size}")
        tensor = tensor.reshape(*batch_dims, width // self.pack_size, self.pack_size)
        shifts = torch.arange(self.pack_size, device=tensor.device, dtype=torch.uint8) * int(self.config.quantization_bits)
        return ((tensor.to(torch.uint8) << shifts).sum(dim=-1)).to(torch.uint8)

    def _unpack_nbit(self, packed: torch.Tensor) -> torch.Tensor:
        mask = (1 << int(self.config.quantization_bits)) - 1
        shifts = torch.arange(self.pack_size, device=packed.device, dtype=torch.uint8) * int(self.config.quantization_bits)
        expanded = packed.unsqueeze(-1)
        unpacked = (expanded >> shifts) & mask
        return unpacked.reshape(*packed.shape[:-1], packed.shape[-1] * self.pack_size)

    def _quantize_tensor(self, tensor: torch.Tensor) -> dict[str, Any]:
        batch, heads, tokens, dim = tensor.shape
        if tokens % int(self.config.group_size) != 0:
            raise ValueError("token count must align with group_size")
        groups = tokens // int(self.config.group_size)
        flat = tensor.permute(0, 1, 3, 2).contiguous()
        grouped = flat.reshape(batch * heads * dim, groups, int(self.config.group_size))
        mins = grouped.min(dim=-1).values
        maxs = grouped.max(dim=-1).values
        scales = (maxs - mins).clamp_min(self.config.eps) / float(self.quantization_levels)
        quantized = ((grouped - mins.unsqueeze(-1)) / scales.unsqueeze(-1)).clamp(
            0,
            self.quantization_levels,
        ).round().to(torch.uint8)
        quantized = quantized.reshape(batch, heads, dim, tokens)
        packed = self._pack_nbit(quantized).permute(0, 1, 3, 2).contiguous()
        scales = scales.reshape(batch, heads, dim, groups).permute(0, 1, 3, 2).contiguous()
        mins = mins.reshape(batch, heads, dim, groups).permute(0, 1, 3, 2).contiguous()
        return {
            "packed": packed,
            "scales": scales,
            "mins": mins,
            "original_tokens": tokens,
            "dtype": tensor.dtype,
        }

    def _dequantize_tensor(self, entry: dict[str, Any], target_dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        packed = entry["packed"].to(device)
        scales = entry["scales"].to(device)
        mins = entry["mins"].to(device)
        original_tokens = int(entry["original_tokens"])
        batch, heads, _, dim = packed.shape
        groups = original_tokens // int(self.config.group_size)
        packed = packed.permute(0, 1, 3, 2).contiguous()
        unpacked = self._unpack_nbit(packed)
        unpacked = unpacked.reshape(batch * heads * dim, groups, int(self.config.group_size)).to(scales.dtype)
        scales_flat = scales.permute(0, 1, 3, 2).reshape(batch * heads * dim, groups)
        mins_flat = mins.permute(0, 1, 3, 2).reshape(batch * heads * dim, groups)
        dequant = unpacked * scales_flat.unsqueeze(-1) + mins_flat.unsqueeze(-1)
        dequant = dequant.reshape(batch, heads, dim, original_tokens).permute(0, 1, 3, 2).contiguous()
        return dequant.to(dtype=target_dtype)

    def _reconstruct_quantized_vision(self, video_id: str, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        storage = self.quantized_storage[video_id][layer_idx]
        target_dtype = self.original_dtype.get(video_id, {}).get(layer_idx, torch.float16)
        if storage["init_tokens"] is not None:
            device = storage["init_tokens"][0].device
        elif storage["key_blocks"]:
            device = storage["key_blocks"][0]["packed"].device
        else:
            raise ValueError("No quantized KV blocks available")
        keys = [self._dequantize_tensor(entry, target_dtype, device) for entry in storage["key_blocks"]]
        values = [self._dequantize_tensor(entry, target_dtype, device) for entry in storage["value_blocks"]]
        if not keys:
            raise ValueError("No quantized vision KV blocks available")
        return torch.cat(keys, dim=2), torch.cat(values, dim=2)

    def _reconstruct_selective_quantized(
        self,
        video_id: str,
        layer_idx: int,
        selected_groups: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        storage = self.quantized_storage[video_id][layer_idx]
        if storage["init_tokens"] is None:
            raise ValueError("system prompt KV is missing")
        init_k, init_v = storage["init_tokens"]
        target_dtype = self.original_dtype.get(video_id, {}).get(layer_idx, init_k.dtype)
        device = selected_groups.device
        selected = torch.sort(selected_groups.to(device=device, dtype=torch.long)).values

        key_parts: list[torch.Tensor] = []
        value_parts: list[torch.Tensor] = []
        group_cursor = 0
        for key_entry, value_entry in zip(storage["key_blocks"], storage["value_blocks"]):
            block_groups = int(key_entry["original_tokens"]) // int(self.config.group_size)
            block_start = group_cursor
            block_end = group_cursor + block_groups
            in_block = (selected >= block_start) & (selected < block_end)
            if in_block.any():
                local_groups = selected[in_block] - block_start
                key_parts.append(self._dequantize_selected_groups(key_entry, local_groups, target_dtype, device))
                value_parts.append(self._dequantize_selected_groups(value_entry, local_groups, target_dtype, device))
            group_cursor = block_end

        if not key_parts:
            raise ValueError("No selected quantized groups reconstructed")
        selected_k = torch.cat(key_parts, dim=2)
        selected_v = torch.cat(value_parts, dim=2)
        return torch.cat([init_k.to(selected_k.device), selected_k], dim=2), torch.cat(
            [init_v.to(selected_v.device), selected_v],
            dim=2,
        )

    def _dequantize_selected_groups(
        self,
        entry: dict[str, Any],
        group_indices: torch.Tensor,
        target_dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if group_indices.ndim != 1:
            raise ValueError("group_indices must be 1D")
        if group_indices.numel() == 0:
            packed = entry["packed"]
            return torch.empty(
                packed.shape[0],
                packed.shape[1],
                0,
                packed.shape[-1],
                dtype=target_dtype,
                device=device,
            )

        packed = entry["packed"].to(device)
        scales = entry["scales"].to(device)
        mins = entry["mins"].to(device)
        batch, heads, _packed_tokens, dim = packed.shape
        packed_group_size = int(self.config.group_size) // int(self.pack_size)
        group_indices = group_indices.to(device=device, dtype=torch.long)
        packed_indices = (
            group_indices.unsqueeze(1) * packed_group_size
            + torch.arange(packed_group_size, device=device)
        ).flatten()
        selected_packed = packed.index_select(2, packed_indices)
        selected_packed = selected_packed.reshape(batch, heads, int(group_indices.numel()), packed_group_size, dim)
        packed_for_unpack = selected_packed.permute(0, 1, 2, 4, 3).reshape(
            batch * heads * int(group_indices.numel()) * dim,
            packed_group_size,
        )
        unpacked = self._unpack_nbit(packed_for_unpack)
        unpacked = unpacked.reshape(
            batch,
            heads,
            int(group_indices.numel()),
            dim,
            int(self.config.group_size),
        ).permute(0, 1, 2, 4, 3)
        selected_scales = scales.index_select(2, group_indices)
        selected_mins = mins.index_select(2, group_indices)
        dequant = unpacked.to(selected_scales.dtype) * selected_scales.unsqueeze(3) + selected_mins.unsqueeze(3)
        return dequant.reshape(batch, heads, int(group_indices.numel()) * int(self.config.group_size), dim).to(
            dtype=target_dtype
        )

    def _slice_selective_unquantized(
        self,
        video_id: str,
        layer_idx: int,
        selected_groups: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_cache, value_cache = self.kv_cache_storage[video_id][layer_idx]
        init_count = int(self.config.init_token_count)
        init_indices = torch.arange(init_count, device=key_cache.device)
        group_indices = selected_groups.to(key_cache.device)
        vision_indices = (
            init_count
            + group_indices.unsqueeze(1) * int(self.config.group_size)
            + torch.arange(int(self.config.group_size), device=key_cache.device)
        ).flatten()
        indices = torch.cat([init_indices, vision_indices])
        return key_cache.index_select(2, indices), value_cache.index_select(2, indices)

    def _total_vision_groups(self, video_id: str, layer_idx: int) -> int:
        total = self.stored_tokens_count.get(video_id, {}).get(layer_idx, 0)
        vision_tokens = max(0, int(total) - int(self.config.init_token_count))
        return vision_tokens // int(self.config.group_size)

    def _original_layer_bytes(self, video_id: str, layer_idx: int) -> int:
        dtype = self.original_dtype.get(video_id, {}).get(layer_idx, torch.float16)
        dtype_size = torch.empty((), dtype=dtype).element_size()
        if self.config.enable_quantization:
            storage = self.quantized_storage.get(video_id, {}).get(layer_idx)
            if not storage:
                return 0
            sample = None
            if storage["init_tokens"] is not None:
                sample = storage["init_tokens"][0]
            elif storage["key_blocks"]:
                sample = storage["key_blocks"][0]["packed"]
            if sample is None:
                return 0
            batch, heads, _, dim = sample.shape
            total_tokens = self.stored_tokens_count.get(video_id, {}).get(layer_idx, 0)
            return int(batch * heads * total_tokens * dim * dtype_size * 2)
        key_cache, value_cache = self.kv_cache_storage.get(video_id, {}).get(layer_idx, (None, None))
        if key_cache is None or value_cache is None:
            return 0
        return int(key_cache.numel() * key_cache.element_size() + value_cache.numel() * value_cache.element_size())

    def _stored_layer_bytes(self, video_id: str, layer_idx: int) -> int:
        if not self.config.enable_quantization:
            return self._original_layer_bytes(video_id, layer_idx)
        storage = self.quantized_storage.get(video_id, {}).get(layer_idx)
        if not storage:
            return 0
        total = 0
        if storage["init_tokens"] is not None:
            init_k, init_v = storage["init_tokens"]
            total += init_k.numel() * init_k.element_size()
            total += init_v.numel() * init_v.element_size()
        for entry in storage["key_blocks"] + storage["value_blocks"]:
            for key in ("packed", "scales", "mins"):
                tensor = entry[key]
                total += tensor.numel() * tensor.element_size()
        return int(total)
