from __future__ import annotations

import os
import time
from dataclasses import replace as dataclass_replace
from typing import Any

import torch
from PIL import Image

from lib.cdas_sampler import CDASConfig
from lib.recent_window_eval import RecentWindowResult
from lib.recent_window_eval_minicpm import _synchronize_gpu_devices
from lib.recent_window_eval_minicpm_ctr import (
    CTRMiniCPMQAModel,
    query_all_frames as _query_all_frames_ctr,
    query_recent_window as _query_recent_window_ctr,
)
from lib.streamingtom_ctr import CTRConfig
from lib.streamingtom_oqm import OQMConfig, OnlineQuantizedMemory


def _past_layers(past_key_values: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    def append_if_tensor(
        layers: list[tuple[torch.Tensor, torch.Tensor]],
        key: Any,
        value: Any,
    ) -> None:
        if isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor):
            if key.ndim == 4 and value.ndim == 4 and int(key.shape[2]) > 0:
                layers.append((key, value))

    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        for key, value in zip(past_key_values.key_cache, past_key_values.value_cache):
            append_if_tensor(layers, key, value)
        if layers:
            return layers

    cache_layers = getattr(past_key_values, "layers", None)
    if isinstance(cache_layers, (list, tuple)):
        layers = []
        for cache_layer in cache_layers:
            append_if_tensor(
                layers,
                getattr(cache_layer, "keys", None),
                getattr(cache_layer, "values", None),
            )
        if layers:
            return layers

    if isinstance(past_key_values, (list, tuple)):
        layers = []
        for layer in past_key_values:
            if isinstance(layer, (list, tuple)) and len(layer) >= 2:
                append_if_tensor(layers, layer[0], layer[1])
        if layers:
            return layers

    try:
        iterator = iter(past_key_values)
    except TypeError:
        return []
    layers = []
    for layer in iterator:
        if isinstance(layer, (list, tuple)) and len(layer) >= 2:
            append_if_tensor(layers, layer[0], layer[1])
    return layers


def _set_cache_layer_kv(cache_layer: Any, key: torch.Tensor, value: torch.Tensor) -> None:
    cache_layer.keys = key
    cache_layer.values = value
    cache_layer.is_initialized = True
    cache_layer.dtype = key.dtype
    cache_layer.device = key.device
    cumulative_length = getattr(cache_layer, "cumulative_length", None)
    if isinstance(cumulative_length, int):
        cache_layer.cumulative_length = int(key.shape[2])
    elif isinstance(cumulative_length, torch.Tensor):
        cumulative_length.zero_()
        cumulative_length.add_(int(key.shape[2]))
    if hasattr(cache_layer, "cumulative_length_int"):
        cache_layer.cumulative_length_int = int(key.shape[2])


def _cache_from_template(
    template_cache: Any,
    layers: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[Any, dict[str, Any]]:
    cache_layers = getattr(template_cache, "layers", None)
    if isinstance(cache_layers, (list, tuple)):
        attention_idx = 0
        for cache_layer in cache_layers:
            key = getattr(cache_layer, "keys", None)
            value = getattr(cache_layer, "values", None)
            if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                continue
            if key.ndim != 4 or value.ndim != 4 or int(key.shape[2]) <= 0:
                continue
            if attention_idx >= len(layers):
                break
            new_key, new_value = layers[attention_idx]
            _set_cache_layer_kv(cache_layer, new_key, new_value)
            attention_idx += 1
        if attention_idx != len(layers):
            raise RuntimeError(
                "Could not map all OQM-reconstructed attention layers into MiniCPM cache: "
                f"replaced={attention_idx}, expected={len(layers)}"
            )
        return template_cache, {
            "format": "template_dynamic_cache",
            "template_type": type(template_cache).__name__,
            "attention_layers_replaced": attention_idx,
            "total_cache_layers": len(cache_layers),
        }

    try:
        from transformers.cache_utils import DynamicCache

        return DynamicCache([(key, value, None) for key, value in layers]), {
            "format": "generic_dynamic_cache",
            "attention_layers_replaced": len(layers),
            "total_cache_layers": len(layers),
        }
    except Exception:
        return tuple((key, value) for key, value in layers), {
            "format": "tuple_cache",
            "attention_layers_replaced": len(layers),
            "total_cache_layers": len(layers),
        }


def _reconstruct_prompt_order_cache(
    *,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    image_positions: torch.Tensor,
    selected_groups: torch.Tensor,
    selected_kv: tuple[torch.Tensor, torch.Tensor],
    group_size: int,
    init_token_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = int(key_cache.shape[2])
    prompt_device = key_cache.device
    image_positions = image_positions.to(prompt_device)
    selected_groups = selected_groups.to(prompt_device)
    image_token_offsets = (
        selected_groups.unsqueeze(1) * int(group_size)
        + torch.arange(int(group_size), device=prompt_device)
    ).flatten()
    selected_image_positions = image_positions.index_select(0, image_token_offsets)

    image_mask = torch.zeros(seq_len, device=prompt_device, dtype=torch.bool)
    image_mask[image_positions] = True
    non_image_positions = torch.nonzero(~image_mask, as_tuple=False).flatten()
    keep_positions = torch.sort(torch.cat([non_image_positions, selected_image_positions], dim=0)).values

    selected_k, selected_v = selected_kv
    selected_k = selected_k.to(prompt_device)
    selected_v = selected_v.to(prompt_device)
    selected_image_k = selected_k[:, :, int(init_token_count) :, :]
    selected_image_v = selected_v[:, :, int(init_token_count) :, :]
    selected_by_position = {
        int(pos): idx
        for idx, pos in enumerate(selected_image_positions.detach().cpu().tolist())
    }

    key_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    for pos in keep_positions.detach().cpu().tolist():
        if int(pos) in selected_by_position:
            local = selected_by_position[int(pos)]
            key_parts.append(selected_image_k[:, :, local : local + 1, :])
            value_parts.append(selected_image_v[:, :, local : local + 1, :])
        else:
            key_parts.append(key_cache[:, :, int(pos) : int(pos) + 1, :])
            value_parts.append(value_cache[:, :, int(pos) : int(pos) + 1, :])
    return torch.cat(key_parts, dim=2), torch.cat(value_parts, dim=2)


class StreamingTOMMiniCPMQAModel(CTRMiniCPMQAModel):
    """MiniCPM-V 4.6 wrapper with CTR + OQM.

    This is the paper-aligned StreamingTOM path kept separate from the baseline
    and CTR-only wrappers. It follows the paper lifecycle:

    1. apply CTR before LLM prefill,
    2. prefill compressed visual tokens online in chunks,
    3. store only the newly produced visual KV in OQM,
    4. retrieve/reconstruct OQM groups for the question, and
    5. decode from the retrieved cache.

    It intentionally avoids the earlier prototype behavior of building a full
    prompt KV cache first and quantizing it afterwards.
    """

    def __init__(
        self,
        model_name: str,
        device: str | torch.device = "auto",
        max_new_tokens: int = 256,
        attn_implementation: str | None = None,
        ctr_config: CTRConfig | None = None,
        oqm_config: OQMConfig | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
            attn_implementation=attn_implementation,
            ctr_config=ctr_config,
        )
        self.oqm_config = oqm_config or OQMConfig(
            retrieval_max_tokens=int(os.environ.get("MINICPM_OQM_RETRIEVAL_MAX_TOKENS", "12544")),
            enable_quantization=os.environ.get("MINICPM_OQM_ENABLE_QUANTIZATION", "1").strip().lower()
            not in {"0", "false", "no", "off"},
            quantization_bits=int(os.environ.get("MINICPM_OQM_QUANTIZATION_BITS", "4")),
            group_size=int(os.environ.get("MINICPM_OQM_GROUP_SIZE", str(self.ctr_config.token_budget))),
            init_token_count=int(os.environ.get("MINICPM_OQM_INIT_TOKEN_COUNT", "14")),
            sliding_window_size=int(os.environ.get("MINICPM_OQM_SLIDING_WINDOW_SIZE", "4800")),
        )
        if int(self.oqm_config.group_size) != int(self.ctr_config.token_budget):
            raise ValueError(
                "StreamingTOM expects OQM group_size to match CTR token budget: "
                f"group_size={self.oqm_config.group_size}, G={self.ctr_config.token_budget}"
            )
        self._last_oqm_store_ms = 0.0
        self._last_oqm_retrieval_ms = 0.0
        self._last_oqm_reconstruct_ms = 0.0
        self._last_oqm_decode_loop_ms = 0.0
        self._last_oqm_prefill_ms = 0.0
        self._last_oqm_storage_summary: dict[str, Any] = {}
        self._last_oqm_cache_build: dict[str, Any] = {}
        self._last_oqm_layer0_retrieval: dict[str, Any] | None = None
        self._last_oqm_layer0_reconstruct: dict[str, Any] | None = None
        self._last_oqm_full_seq_len = 0
        self._last_oqm_reconstructed_seq_len = 0
        self.streaming_encoder_batch_size = int(os.environ.get("MINICPM_STREAMING_ENCODER_BATCH_SIZE", "32"))
        self._last_oqm_window_reconstruct_ms = 0.0
        self._last_oqm_query_prefill_ms = 0.0
        self._last_oqm_init_token_count = int(self.oqm_config.init_token_count)

    def _eos_token_ids(self) -> set[int]:
        generation_config = getattr(self.model, "generation_config", None)
        candidates = [
            getattr(generation_config, "eos_token_id", None),
            getattr(self.processor, "eos_token_id", None),
            getattr(getattr(self.processor, "tokenizer", None), "eos_token_id", None),
        ]
        output: set[int] = set()
        for item in candidates:
            if item is None:
                continue
            if isinstance(item, (list, tuple, set)):
                output.update(int(value) for value in item if value is not None)
            else:
                output.add(int(item))
        return output

    def _effective_oqm_config(self, init_token_count: int) -> OQMConfig:
        # MiniCPM's chat template does not always have the paper's exact 14
        # initial text tokens. Preserve the actual MiniCPM prefix as the
        # unquantized OQM init cache, which is the same role those tokens play
        # in StreamingTOM.
        return dataclass_replace(self.oqm_config, init_token_count=int(init_token_count))

    def _split_ctr_prompt(self, model_inputs: dict[str, Any]) -> dict[str, Any]:
        input_ids = model_inputs["input_ids"]
        inputs_embeds = model_inputs["inputs_embeds"]
        image_mask = model_inputs["image_mask"][0]
        blocks = self._image_token_blocks(input_ids[0], self.image_token_id)
        if not blocks:
            raise RuntimeError("StreamingTOM active path needs at least one image-token block")

        first_image_start = int(blocks[0][0])
        last_image_end = int(blocks[-1][1])
        prefix_ids = input_ids[:, :first_image_start]
        prefix_embeds = inputs_embeds[:, :first_image_start, :]
        query_ids = input_ids[:, last_image_end:]
        query_embeds = inputs_embeds[:, last_image_end:, :]
        if int(prefix_ids.shape[1]) <= 0:
            raise RuntimeError("Could not find MiniCPM text prefix before visual tokens")
        if int(query_ids.shape[1]) <= 0:
            raise RuntimeError("Could not find MiniCPM query/generation suffix after visual tokens")

        image_positions = torch.nonzero(image_mask, as_tuple=False).flatten()
        vision_embeds = inputs_embeds[:, image_positions, :]
        expected_tokens = int(model_inputs["tokens_after"])
        if int(vision_embeds.shape[1]) != expected_tokens:
            raise RuntimeError(
                "Compressed prompt image tokens do not match CTR output: "
                f"prompt={int(vision_embeds.shape[1])}, ctr={expected_tokens}"
            )

        return {
            "prefix_ids": prefix_ids,
            "prefix_embeds": prefix_embeds,
            "query_ids": query_ids,
            "query_embeds": query_embeds,
            "vision_embeds": vision_embeds,
        }

    def _forward_embeds(
        self,
        *,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor,
        past_key_values: Any | None = None,
        past_length: int = 0,
    ) -> Any:
        attention_mask = torch.ones(
            (inputs_embeds.shape[0], int(past_length) + int(inputs_embeds.shape[1])),
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        kwargs: dict[str, Any] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "use_cache": True,
            "return_dict": True,
        }
        if input_ids is not None:
            kwargs["input_ids"] = input_ids
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values

        try:
            return self.model(**kwargs)
        except TypeError:
            kwargs.pop("input_ids", None)
            return self.model(**kwargs)

    def _build_cache_from_layers(
        self,
        template_cache: Any,
        layers: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> Any:
        cache, _ = _cache_from_template(template_cache, layers)
        return cache

    def _windowed_layers_or_init(
        self,
        *,
        oqm: OnlineQuantizedMemory,
        video_id: str,
        init_layers: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], float]:
        layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        reconstruct_ms = 0.0
        for layer_idx, init_layer in enumerate(init_layers):
            if oqm.get_group_keys(video_id, layer_idx) is None:
                layers.append(init_layer)
                continue
            kv, metadata = oqm.get_windowed_kv(video_id, layer_idx)
            layers.append(kv)
            reconstruct_ms += metadata.reconstruct_time_ms
        return layers, reconstruct_ms

    def _store_new_vision_layers(
        self,
        *,
        oqm: OnlineQuantizedMemory,
        video_id: str,
        source_layers: list[tuple[torch.Tensor, torch.Tensor]],
        past_length: int,
        token_level_keys: torch.Tensor,
        init_layers: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> float:
        store_ms = 0.0
        for layer_idx, (key_cache, value_cache) in enumerate(source_layers):
            if oqm.get_group_keys(video_id, layer_idx) is None:
                init_k, init_v = init_layers[layer_idx]
                init_meta = oqm.store_system_prompt(video_id, layer_idx, init_k, init_v)
                store_ms += init_meta.store_time_ms

            new_k = key_cache[:, :, int(past_length) :, :].detach()
            new_v = value_cache[:, :, int(past_length) :, :].detach()
            if int(new_k.shape[2]) <= 0:
                raise RuntimeError("StreamingTOM active prefill produced no new visual KV")
            store_meta = oqm.store_kv_cache(
                video_id,
                layer_idx,
                new_k,
                new_v,
                token_level_keys=token_level_keys,
            )
            store_ms += store_meta.store_time_ms
        return store_ms

    @staticmethod
    def _mean_query_key(query_embeds: torch.Tensor) -> torch.Tensor:
        return query_embeds[0].detach().float().mean(dim=0)

    @torch.inference_mode()
    def generate_from_frames(
        self,
        frames: list[Image.Image],
        question: str,
        downsample_mode: str | None = None,
    ) -> str:
        self._last_oqm_store_ms = 0.0
        self._last_oqm_retrieval_ms = 0.0
        self._last_oqm_reconstruct_ms = 0.0
        self._last_oqm_decode_loop_ms = 0.0
        self._last_oqm_prefill_ms = 0.0
        self._last_oqm_storage_summary = {}
        self._last_oqm_cache_build = {}
        self._last_oqm_layer0_retrieval = None
        self._last_oqm_layer0_reconstruct = None
        self._last_oqm_window_reconstruct_ms = 0.0
        self._last_oqm_query_prefill_ms = 0.0

        model_inputs = self.build_ctr_model_inputs(
            frames=frames,
            question=question,
            downsample_mode=downsample_mode,
        )
        split = self._split_ctr_prompt(model_inputs)
        oqm_config = self._effective_oqm_config(int(split["prefix_ids"].shape[1]))
        self._last_oqm_init_token_count = int(oqm_config.init_token_count)
        oqm = OnlineQuantizedMemory(oqm_config)
        video_id = "video"

        model_generate_t0 = time.perf_counter()

        init_t0 = time.perf_counter()
        init_outputs = self._forward_embeds(
            input_ids=split["prefix_ids"],
            inputs_embeds=split["prefix_embeds"],
        )
        _synchronize_gpu_devices()
        init_prefill_ms = (time.perf_counter() - init_t0) * 1000.0
        init_cache = getattr(init_outputs, "past_key_values", None)
        init_layers = _past_layers(init_cache)
        if not init_layers:
            raise RuntimeError("No MiniCPM KV cache layers found after StreamingTOM init prefill")

        current_cache_template = init_cache
        vision_embeds = split["vision_embeds"]
        flattened_features = [item.reshape(-1, item.shape[-1]) for item in model_inputs["compressed_features"]]
        batch_size = max(1, int(self.streaming_encoder_batch_size))
        prefill_ms = init_prefill_ms

        for start in range(0, len(flattened_features), batch_size):
            frame_batch = flattened_features[start : start + batch_size]
            batch_tokens = torch.cat(frame_batch, dim=0).unsqueeze(0)
            batch_tokens = batch_tokens.to(device=vision_embeds.device, dtype=vision_embeds.dtype)
            token_level_keys = batch_tokens[0].detach()
            if int(batch_tokens.shape[1]) % int(oqm_config.group_size) != 0:
                raise RuntimeError(
                    f"Vision batch token count {int(batch_tokens.shape[1])} is not divisible "
                    f"by OQM group_size={oqm_config.group_size}"
                )

            window_layers, window_ms = self._windowed_layers_or_init(
                oqm=oqm,
                video_id=video_id,
                init_layers=init_layers,
            )
            self._last_oqm_window_reconstruct_ms += window_ms
            past_length = int(window_layers[0][0].shape[2])
            past_cache = self._build_cache_from_layers(current_cache_template, window_layers)

            batch_t0 = time.perf_counter()
            batch_outputs = self._forward_embeds(
                input_ids=None,
                inputs_embeds=batch_tokens,
                past_key_values=past_cache,
                past_length=past_length,
            )
            _synchronize_gpu_devices()
            prefill_ms += (time.perf_counter() - batch_t0) * 1000.0
            batch_layers = _past_layers(getattr(batch_outputs, "past_key_values", None))
            if not batch_layers:
                raise RuntimeError("No MiniCPM KV cache layers found after StreamingTOM visual prefill")
            self._last_oqm_store_ms += self._store_new_vision_layers(
                oqm=oqm,
                video_id=video_id,
                source_layers=batch_layers,
                past_length=past_length,
                token_level_keys=token_level_keys,
                init_layers=init_layers,
            )
            current_cache_template = getattr(batch_outputs, "past_key_values", current_cache_template)

        self._last_oqm_prefill_ms = prefill_ms

        query_key = self._mean_query_key(split["query_embeds"])
        reconstructed_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(len(init_layers)):
            selected_groups, retrieval_meta = oqm.retrieve_group_indices(
                video_id,
                layer_idx,
                query_key,
                max_tokens=int(oqm_config.retrieval_max_tokens),
            )
            selected_kv, reconstruct_meta = oqm.get_selective_kv(video_id, layer_idx, selected_groups)
            self._last_oqm_retrieval_ms += retrieval_meta.retrieval_time_ms
            self._last_oqm_reconstruct_ms += reconstruct_meta.reconstruct_time_ms
            if layer_idx == 0:
                self._last_oqm_layer0_retrieval = retrieval_meta.as_dict()
                self._last_oqm_layer0_reconstruct = reconstruct_meta.as_dict()
            reconstructed_layers.append(selected_kv)

        reconstructed_cache, cache_build = _cache_from_template(
            current_cache_template,
            reconstructed_layers,
        )
        self._last_oqm_cache_build = cache_build
        self._last_oqm_storage_summary = oqm.storage_summary(video_id)
        self._last_oqm_full_seq_len = int(oqm.stored_tokens_count.get(video_id, {}).get(0, 0))
        self._last_oqm_reconstructed_seq_len = int(reconstructed_layers[0][0].shape[2])

        query_t0 = time.perf_counter()
        query_outputs = self._forward_embeds(
            input_ids=split["query_ids"],
            inputs_embeds=split["query_embeds"],
            past_key_values=reconstructed_cache,
            past_length=self._last_oqm_reconstructed_seq_len,
        )
        _synchronize_gpu_devices()
        self._last_oqm_query_prefill_ms = (time.perf_counter() - query_t0) * 1000.0

        generated_tokens: list[torch.Tensor] = []
        next_token = torch.argmax(query_outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens.append(next_token)
        eos_token_ids = self._eos_token_ids()
        attention_mask = torch.ones(
            (1, self._last_oqm_reconstructed_seq_len + int(split["query_embeds"].shape[1]) + 1),
            dtype=torch.long,
            device=split["query_embeds"].device,
        )

        decode_t0 = time.perf_counter()
        cache = getattr(query_outputs, "past_key_values", reconstructed_cache)
        for _step in range(max(0, int(self.max_new_tokens) - 1)):
            if eos_token_ids and int(next_token.item()) in eos_token_ids:
                break
            outputs = self.model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = getattr(outputs, "past_key_values", cache)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_tokens.append(next_token)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=1,
            )
        _synchronize_gpu_devices()
        self._last_oqm_decode_loop_ms = (time.perf_counter() - decode_t0) * 1000.0
        self._last_model_generate_seconds = time.perf_counter() - model_generate_t0
        self._last_ttft_seconds = (
            + self._last_oqm_retrieval_ms
            + self._last_oqm_reconstruct_ms
            + self._last_oqm_query_prefill_ms
        ) / 1000.0

        generated_ids = torch.cat(generated_tokens, dim=1)
        answer = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        self._last_component_times = {
            "enabled": False,
            "streamingtom_manual_generation": True,
            "ctr_enabled": True,
            "ctr_vision_encode_ms": self._last_ctr_vision_encode_ms,
            "ctr_compress_features_ms": self._last_ctr_compress_features_ms,
            "ctr_tokens_before": model_inputs["tokens_before"],
            "ctr_tokens_after": model_inputs["tokens_after"],
            "ctr_frames": len(model_inputs["compressed_features"]),
            "oqm_enabled": True,
            "oqm_prefill_ms": self._last_oqm_prefill_ms,
            "oqm_store_ms": self._last_oqm_store_ms,
            "oqm_window_reconstruct_ms": self._last_oqm_window_reconstruct_ms,
            "oqm_retrieval_ms": self._last_oqm_retrieval_ms,
            "oqm_reconstruct_ms": self._last_oqm_reconstruct_ms,
            "oqm_query_prefill_ms": self._last_oqm_query_prefill_ms,
            "oqm_decode_loop_ms": self._last_oqm_decode_loop_ms,
            "oqm_init_token_count": self._last_oqm_init_token_count,
            "oqm_cache_build": self._last_oqm_cache_build,
        }
        return answer


def _apply_streamingtom_profile_overrides(
    profile_metadata: dict[str, Any],
    qa: StreamingTOMMiniCPMQAModel,
) -> None:
    oqm = {
        "enabled": True,
        "retrieval_max_tokens": int(qa.oqm_config.retrieval_max_tokens),
        "quantization_bits": int(qa.oqm_config.quantization_bits),
        "group_size": int(qa.oqm_config.group_size),
        "init_token_count": int(getattr(qa, "_last_oqm_init_token_count", qa.oqm_config.init_token_count)),
        "prefill_ms": float(getattr(qa, "_last_oqm_prefill_ms", 0.0)),
        "store_ms": float(getattr(qa, "_last_oqm_store_ms", 0.0)),
        "window_reconstruct_ms": float(getattr(qa, "_last_oqm_window_reconstruct_ms", 0.0)),
        "retrieval_ms": float(getattr(qa, "_last_oqm_retrieval_ms", 0.0)),
        "reconstruct_ms": float(getattr(qa, "_last_oqm_reconstruct_ms", 0.0)),
        "query_prefill_ms": float(getattr(qa, "_last_oqm_query_prefill_ms", 0.0)),
        "decode_loop_ms": float(getattr(qa, "_last_oqm_decode_loop_ms", 0.0)),
        "full_seq_len": int(getattr(qa, "_last_oqm_full_seq_len", 0)),
        "reconstructed_seq_len": int(getattr(qa, "_last_oqm_reconstructed_seq_len", 0)),
        "cache_build": getattr(qa, "_last_oqm_cache_build", {}),
        "layer0_retrieval": getattr(qa, "_last_oqm_layer0_retrieval", None),
        "layer0_reconstruct": getattr(qa, "_last_oqm_layer0_reconstruct", None),
        "storage_summary": getattr(qa, "_last_oqm_storage_summary", {}),
    }
    profile_metadata["oqm"] = oqm
    profile_metadata["mode"] = str(profile_metadata.get("mode", "")).replace("_ctr", "_streamingtom")
    profile_metadata["prefill_kv_time_ms"] = oqm["prefill_ms"]
    profile_metadata["st_prefill_kv_ms"] = oqm["prefill_ms"]
    profile_metadata["st_store_kv_ms"] = oqm["store_ms"]
    profile_metadata["st_retrieval_forward_ms"] = oqm["retrieval_ms"]
    profile_metadata["st_reconstruct_kv_ms"] = oqm["reconstruct_ms"]
    profile_metadata["st_generate_first_token_ms"] = oqm["query_prefill_ms"]
    profile_metadata["st_generate_tokens_ms"] = oqm["decode_loop_ms"]
    profile_metadata["ttft_seconds"] = (
        oqm["retrieval_ms"] + oqm["reconstruct_ms"] + oqm["query_prefill_ms"]
    ) / 1000.0
    timeline = profile_metadata.get("streamingtom_timeline_ms")
    if isinstance(timeline, dict):
        vision_components = timeline.setdefault("vision_subtask_components", {})
        query_components = timeline.setdefault("query_subtask_components", {})
        vision_components["prefill_kv"] = oqm["prefill_ms"]
        vision_components["store_kv"] = oqm["store_ms"]
        query_components["retrieval_forward"] = oqm["retrieval_ms"]
        query_components["reconstruct_kv"] = oqm["reconstruct_ms"]
        query_components["generate_first_token"] = profile_metadata["st_generate_first_token_ms"]
        query_components["generate_tokens"] = oqm["decode_loop_ms"]
        notes = timeline.setdefault("notes", {})
        notes["prefill_kv"] = "Measured online visual prefill over CTR-compressed chunks."
        notes["store_kv"] = "Measured OQM 4-bit KV storage time."
        notes["retrieval_forward"] = "Measured OQM group retrieval time."
        notes["reconstruct_kv"] = "Measured OQM selective KV reconstruction time."
        notes["generate_first_token"] = "Measured query suffix prefill with retrieved OQM cache."


def query_recent_window(
    qa: StreamingTOMMiniCPMQAModel,
    video_path: str,
    prompt: str,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    video_start: float | None = None,
    video_end: float | None = None,
    cdas_config: CDASConfig | None = None,
) -> tuple[RecentWindowResult, str]:
    result, backend = _query_recent_window_ctr(
        qa=qa,
        video_path=video_path,
        prompt=prompt,
        chunk_duration=chunk_duration,
        fps=fps,
        recent_frames_only=recent_frames_only,
        video_start=video_start,
        video_end=video_end,
        cdas_config=cdas_config,
    )
    _apply_streamingtom_profile_overrides(result.profile_metadata, qa)
    return result, backend


def query_all_frames(
    qa: StreamingTOMMiniCPMQAModel,
    video_path: str,
    prompt: str,
    chunk_duration: float,
    fps: float,
    video_start: float | None = None,
    video_end: float | None = None,
) -> tuple[RecentWindowResult, str]:
    result, backend = _query_all_frames_ctr(
        qa=qa,
        video_path=video_path,
        prompt=prompt,
        chunk_duration=chunk_duration,
        fps=fps,
        video_start=video_start,
        video_end=video_end,
    )
    _apply_streamingtom_profile_overrides(result.profile_metadata, qa)
    if result.full_frame_metadata is not None:
        result.full_frame_metadata["mode"] = "all_frames_streamingtom"
    return result, backend
