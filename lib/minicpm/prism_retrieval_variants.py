from __future__ import annotations

import math
import os
import time
from typing import Any

import torch
from PIL import Image

from lib.minicpm import progressive_sufficiency as psm
from lib.minicpm.baseline import RecentWindowQAModel, _synchronize_gpu_devices


VALID_RETRIEVAL_VARIANTS = {
    "current",
    "clip_question",
    "clip_question_options",
    "clip_mmr",
}


def chunk_id(chunk: Any) -> int:
    return int(getattr(chunk, "chunk_index"))


def chunk_bounds(chunk: Any) -> tuple[float, float]:
    timestamps = getattr(chunk, "frame_timestamps", None) or []
    numeric = [float(ts) for ts in timestamps if isinstance(ts, (int, float)) and math.isfinite(float(ts))]
    if numeric:
        return min(numeric), max(numeric)
    start = getattr(chunk, "start_time", None)
    end = getattr(chunk, "end_time", None)
    start_f = float(start) if isinstance(start, (int, float)) and math.isfinite(float(start)) else float(chunk_id(chunk))
    end_f = float(end) if isinstance(end, (int, float)) and math.isfinite(float(end)) else start_f
    return (end_f, start_f) if end_f < start_f else (start_f, end_f)


def representative_frame(chunk: Any) -> Image.Image:
    frames = list(getattr(chunk, "frames", []) or [])
    if not frames:
        raise ValueError(f"Chunk {chunk_id(chunk)} has no frames")
    if len(frames) == 1:
        return frames[0]
    timestamps = getattr(chunk, "frame_timestamps", None) or []
    numeric = [float(ts) for ts in timestamps if isinstance(ts, (int, float)) and math.isfinite(float(ts))]
    if len(numeric) != len(frames):
        return frames[len(frames) // 2]
    center = 0.5 * (min(numeric) + max(numeric))
    index = min(range(len(frames)), key=lambda item: abs(float(numeric[item]) - center))
    return frames[index]


def _clip_device(qa: RecentWindowQAModel) -> str:
    return os.environ.get("MINICPM_PSM_CLIP_DEVICE", "").strip() or str(qa.model.device)


def _clip_model_name() -> str:
    return os.environ.get("MINICPM_PSM_CLIP_MODEL", "openai/clip-vit-base-patch32")


def _get_embedding_cache(qa: RecentWindowQAModel) -> dict[str, Any]:
    cache = getattr(qa, "_prism_retrieval_variant_cache", None)
    if cache is None:
        scorer = psm._get_clip_scorer(qa)
        cache = {
            "model_name": _clip_model_name(),
            "device": _clip_device(qa),
            "processor": scorer.processor,
            "model": scorer.model,
            "text": {},
        }
        qa._prism_retrieval_variant_cache = cache
    return cache


def _text_embeddings(qa: RecentWindowQAModel, texts: list[str]) -> torch.Tensor:
    cache = _get_embedding_cache(qa)
    processor = cache["processor"]
    model = cache["model"]
    device = cache["device"]
    embeds: list[torch.Tensor] = []
    missing = [text for text in texts if text not in cache["text"]]
    if missing:
        torch_mod = torch
        with torch_mod.inference_mode():
            inputs = processor(
                text=missing,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            output = _as_feature_tensor(model.get_text_features(**inputs)).float()
            output = torch_mod.nn.functional.normalize(output, dim=-1)
            for text, vector in zip(missing, output.detach().cpu()):
                cache["text"][text] = vector
    for text in texts:
        embeds.append(cache["text"][text])
    return torch.stack(embeds, dim=0).to(device)


def _image_embeddings(qa: RecentWindowQAModel, frames: list[Image.Image]) -> torch.Tensor:
    if not frames:
        return torch.empty((0, 0), device=_clip_device(qa))
    cache = _get_embedding_cache(qa)
    processor = cache["processor"]
    model = cache["model"]
    device = cache["device"]
    with torch.inference_mode():
        inputs = processor(images=frames, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        output = _as_feature_tensor(model.get_image_features(**inputs)).float()
        output = torch.nn.functional.normalize(output, dim=-1)
    return output


def _as_feature_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Could not extract CLIP feature tensor from {type(output).__name__}")


def _option_queries(prompt: str) -> list[tuple[str, str]]:
    question = psm._question_text(prompt)
    options = psm._extract_mcq_options(prompt)
    return [(option["letter"], f"{question} {option['text']}".strip()) for option in options]


def _temporal_separated(candidate: dict[str, Any], selected: list[dict[str, Any]], min_gap_seconds: float) -> bool:
    if not selected:
        return True
    c_start, c_end = candidate["start_time_seconds"], candidate["end_time_seconds"]
    for item in selected:
        s_start, s_end = item["start_time_seconds"], item["end_time_seconds"]
        if max(c_start, s_start) <= min(c_end, s_end):
            return False
        gap = min(abs(c_start - s_end), abs(s_start - c_end))
        if gap < min_gap_seconds:
            return False
    return True


def _temporal_non_overlapping(candidate: dict[str, Any], selected: list[dict[str, Any]]) -> bool:
    if not selected:
        return True
    c_start, c_end = candidate["start_time_seconds"], candidate["end_time_seconds"]
    for item in selected:
        s_start, s_end = item["start_time_seconds"], item["end_time_seconds"]
        if max(c_start, s_start) <= min(c_end, s_end):
            return False
    return True


def _quality_stats(queue: list[dict[str, Any]]) -> dict[str, Any]:
    if not queue:
        return {}
    top3 = queue[:3]
    distances = [
        float(item["candidate_temporal_distance_seconds"])
        for item in queue
        if isinstance(item.get("candidate_temporal_distance_seconds"), (int, float))
    ]
    return {
        "mean_top1_relevance": float(queue[0].get("retrieval_relevance", queue[0].get("semantic_score", 0.0))),
        "mean_top3_relevance": float(
            sum(float(item.get("retrieval_relevance", item.get("semantic_score", 0.0))) for item in top3) / len(top3)
        ),
        "mean_temporal_distance_seconds": float(sum(distances) / len(distances)) if distances else None,
    }


def rank_candidates(
    qa: RecentWindowQAModel,
    older_chunks: list[Any],
    prompt: str,
    config: Any,
    candidate_pool: int,
    min_temporal_gap: int = 2,
    variant: str = "current",
    mmr_lambda: float = 0.80,
    recent_start_time: float | None = None,
) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
    if variant not in VALID_RETRIEVAL_VARIANTS:
        raise ValueError(f"Unknown retrieval variant {variant!r}; expected one of {sorted(VALID_RETRIEVAL_VARIANTS)}")
    if variant == "current":
        queue, elapsed = psm._rank_candidates(
            qa,
            older_chunks,
            prompt,
            config,
            candidate_pool=candidate_pool,
            min_temporal_gap=min_temporal_gap,
        )
        for item in queue:
            start, end = chunk_bounds(item["chunk"])
            item["start_time_seconds"] = start
            item["end_time_seconds"] = end
            item["retrieval_variant"] = "current"
            item["retrieval_relevance"] = item.get("semantic_score")
            if recent_start_time is not None:
                item["candidate_temporal_distance_seconds"] = float(recent_start_time - end)
                item["history_temporal_violation"] = bool(end >= float(recent_start_time))
        return queue, elapsed, {"retrieval_variant": "current", **_quality_stats(queue)}

    ranking_t0 = time.perf_counter()
    from lib.minicpm import adaptive as adaptive_mod

    bank = adaptive_mod._build_online_memory_bank(older_chunks, config)
    chunks = [entry["chunk"] for entry in bank]
    if not chunks:
        _synchronize_gpu_devices()
        elapsed_ms = (time.perf_counter() - ranking_t0) * 1000.0
        return [], elapsed_ms, {
            "retrieval_variant": variant,
            "candidate_count": 0,
            "queue_count": 0,
            "mmr_lambda": float(mmr_lambda) if variant == "clip_mmr" else None,
            "temporal_gap_fallback_count": 0,
        }
    frames = [representative_frame(chunk) for chunk in chunks]
    image_embeds = _image_embeddings(qa, frames)
    question = psm._question_text(prompt)

    if variant == "clip_question":
        text_embeds = _text_embeddings(qa, [question])
        relevance_tensor = image_embeds @ text_embeds[0].unsqueeze(-1)
        relevance = [float(value) for value in relevance_tensor.squeeze(-1).detach().cpu().tolist()]
        best_supported_options = [None for _ in chunks]
    else:
        queries = _option_queries(prompt)
        if not queries:
            queries = [("?", question)]
        text_embeds = _text_embeddings(qa, [query for _letter, query in queries])
        scores = image_embeds @ text_embeds.T
        best_indices = torch.argmax(scores, dim=1).detach().cpu().tolist()
        relevance = [float(value) for value in torch.max(scores, dim=1).values.detach().cpu().tolist()]
        best_supported_options = [queries[int(index)][0] for index in best_indices]

    candidates: list[dict[str, Any]] = []
    for index, entry in enumerate(bank):
        start, end = chunk_bounds(entry["chunk"])
        item = {
            "bank_index": index,
            "chunk": entry["chunk"],
            "chunk_id": int(entry["chunk_id"]),
            "timestamp": float(adaptive_mod._chunk_timestamp(entry["chunk"])),
            "semantic_score": float(psm._normalize_clip_support(relevance[index])),
            "semantic_score_raw": float(relevance[index]),
            "retrieval_relevance": float(relevance[index]),
            "event_score": float(entry["event_change_norm"]),
            "detail_score": float(0.50 * float(entry["contrast_norm"]) + 0.50 * float(entry["text_detail_norm"])),
            "cue_score": 0.0,
            "start_time_seconds": float(start),
            "end_time_seconds": float(end),
            "candidate_temporal_distance_seconds": float(recent_start_time - end) if recent_start_time is not None else None,
            "history_temporal_violation": bool(recent_start_time is not None and end >= float(recent_start_time)),
            "retrieval_variant": variant,
            "best_supported_option": best_supported_options[index],
        }
        candidates.append(item)

    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    temporal_gap_fallback_count = 0
    if variant in {"clip_question", "clip_question_options"}:
        while remaining and len(selected) < candidate_pool:
            eligible = [item for item in remaining if _temporal_separated(item, selected, float(min_temporal_gap))]
            if not eligible:
                temporal_gap_fallback_count += 1
                eligible = remaining
            best = max(eligible, key=lambda item: (float(item["retrieval_relevance"]), -int(item["chunk_id"])))
            best["diversity_score"] = 1.0
            best["total_score"] = float(best["retrieval_relevance"])
            selected.append(best)
            remaining = [item for item in remaining if int(item["chunk_id"]) != int(best["chunk_id"])]
    else:
        while remaining and len(selected) < candidate_pool:
            best = None
            eligible = [item for item in remaining if _temporal_non_overlapping(item, selected)]
            if not eligible:
                break
            for item in eligible:
                if selected:
                    selected_indices = [int(chosen["bank_index"]) for chosen in selected]
                    sims = image_embeds[int(item["bank_index"])] @ image_embeds[selected_indices].T
                    redundancy = float(torch.max(sims).detach().cpu().item())
                else:
                    redundancy = 0.0
                mmr = float(mmr_lambda) * float(item["retrieval_relevance"]) - (1.0 - float(mmr_lambda)) * redundancy
                candidate = {**item, "visual_redundancy": redundancy, "diversity_score": 1.0 - redundancy, "total_score": mmr}
                if best is None or (float(candidate["total_score"]), -int(candidate["chunk_id"])) > (
                    float(best["total_score"]),
                    -int(best["chunk_id"]),
                ):
                    best = candidate
            if best is None:
                break
            selected.append(best)
            remaining = [item for item in remaining if int(item["chunk_id"]) != int(best["chunk_id"])]

    _synchronize_gpu_devices()
    elapsed_ms = (time.perf_counter() - ranking_t0) * 1000.0
    stats = {
        "retrieval_variant": variant,
        "candidate_count": len(candidates),
        "queue_count": len(selected),
        "mmr_lambda": float(mmr_lambda) if variant == "clip_mmr" else None,
        "temporal_gap_fallback_count": temporal_gap_fallback_count,
        **_quality_stats(selected),
    }
    if variant == "clip_question_options":
        counts: dict[str, int] = {}
        for item in selected:
            letter = str(item.get("best_supported_option"))
            counts[letter] = counts.get(letter, 0) + 1
        stats["best_supported_option_distribution"] = counts
    return selected, elapsed_ms, stats
