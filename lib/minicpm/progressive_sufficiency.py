from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from lib.minicpm.baseline import RecentWindowQAModel, _synchronize_gpu_devices
from lib.minicpm.referential_memory import AnswerGroundedFrameScorer


_PSM_OPTION_RE = re.compile(
    r"([A-E])[\.)]\s*(.+?)(?=(?:\s*;?\s*[A-E][\.)]\s)|$)",
    re.IGNORECASE | re.DOTALL,
)
_PSM_HISTORY_INSTRUCTION = (
    "When historical frames are present, they appear before the six recent frames.\n\n"
)
_PSM_MICROCLIP_INSTRUCTION = (
    "A short historical event clip appears before the six recent frames.\n"
    "The historical clip frames are ordered from earlier to later.\n\n"
)


@dataclass
class ProgressiveSufficiencySelection:
    frames: list[Image.Image]
    final_chunk_ids: list[int]
    metadata: dict[str, Any]
    answer_prompt: str
    downsample_mode: str | None = None


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_offsets(name: str, default: str) -> list[float]:
    raw = os.environ.get(name, default)
    offsets: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        offsets.append(float(item))
    return offsets or [0.0]


def _extract_mcq_options(prompt: str) -> list[dict[str, str]]:
    match = re.search(r"\bOptions:\s*", prompt, flags=re.IGNORECASE)
    if match is None:
        return []
    option_text = prompt[match.end() :]
    option_text = re.split(
        r"\n\s*(?:Only give|Answer with|Respond with)",
        option_text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    options: list[dict[str, str]] = []
    for item in _PSM_OPTION_RE.finditer(option_text.strip()):
        letter = item.group(1).upper()
        text = item.group(2).strip().rstrip(";").strip()
        if text and letter not in {entry["letter"] for entry in options}:
            options.append({"letter": letter, "text": text})
    return options if len(options) >= 2 else []


def _question_text(prompt: str) -> str:
    text = re.split(r"\n\s*Options:\s*", prompt, maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.sub(
        r"^You are an advanced video question-answering AI assistant\..*?Question:\s*",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return text.strip()


def _get_clip_scorer(qa: RecentWindowQAModel) -> AnswerGroundedFrameScorer:
    scorer = getattr(qa, "_progressive_sufficiency_clip_scorer", None)
    if scorer is None:
        model_name = os.environ.get("MINICPM_PSM_CLIP_MODEL", "openai/clip-vit-base-patch32")
        device = os.environ.get("MINICPM_PSM_CLIP_DEVICE", "").strip() or str(qa.model.device)
        scorer = AnswerGroundedFrameScorer(model_name=model_name, device=device)
        qa._progressive_sufficiency_clip_scorer = scorer
    return scorer


def _score_chunks_with_clip(
    scorer: AnswerGroundedFrameScorer,
    text: str,
    chunks: list[Any],
) -> list[float]:
    flat_frames: list[Image.Image] = []
    owners: list[int] = []
    for index, chunk in enumerate(chunks):
        for frame in chunk.frames:
            flat_frames.append(frame)
            owners.append(index)
    if not flat_frames:
        return [0.0 for _ in chunks]
    frame_scores = scorer.score(text, flat_frames)
    scores = [-1.0 for _ in chunks]
    for owner, score in zip(owners, frame_scores):
        scores[owner] = max(scores[owner], float(score))
    return [max(-1.0, score) for score in scores]


def _normalize_clip_support(raw_score: float) -> float:
    return float(np.clip((float(raw_score) - 0.15) / 0.30, 0.0, 1.0))


def _top_margin(values: dict[str, float]) -> float | None:
    ordered = sorted(values.values(), reverse=True)
    if len(ordered) < 2:
        return None
    return float(ordered[0] - ordered[1])


def _argmax(values: dict[str, float]) -> str | None:
    if not values:
        return None
    return max(values, key=lambda key: (float(values[key]), -ord(str(key)[0])))


class _OptionEvidenceGainScorer:
    def __init__(
        self,
        scorer: AnswerGroundedFrameScorer,
        prompt: str,
        options: list[dict[str, str]],
    ) -> None:
        self.scorer = scorer
        self.question = _question_text(prompt)
        self.option_queries = {
            option["letter"]: f"{self.question} {option['text']}".strip()
            for option in options
        }
        self._score_cache: dict[tuple[str, int], float] = {}

    def _chunk_score(self, letter: str, chunk: Any) -> float:
        chunk_id = int(chunk.chunk_index)
        key = (letter, chunk_id)
        if key not in self._score_cache:
            scores = self.scorer.score(self.option_queries[letter], list(chunk.frames))
            self._score_cache[key] = float(max(scores) if scores else -1.0)
        return self._score_cache[key]

    def compute(
        self,
        *,
        context_chunks: list[Any],
        unused_candidates: list[dict[str, Any]],
        predicted_option: str,
        recent_ids: set[int],
        selected_memory_ids: set[int],
        heg_threshold: float,
    ) -> dict[str, Any]:
        unused_chunks = [candidate["chunk"] for candidate in unused_candidates]
        unused_ids = [int(chunk.chunk_index) for chunk in unused_chunks]
        assert len(unused_ids) == len(set(unused_ids))
        assert not (set(unused_ids) & recent_ids)
        assert not (set(unused_ids) & selected_memory_ids)

        current_support: dict[str, float] = {}
        best_historical_support: dict[str, float] = {}
        best_historical_chunk_id: dict[str, int | None] = {}
        for letter in self.option_queries:
            current_scores = [self._chunk_score(letter, chunk) for chunk in context_chunks]
            historical_scores = [(int(chunk.chunk_index), self._chunk_score(letter, chunk)) for chunk in unused_chunks]
            current_support[letter] = float(max(current_scores) if current_scores else -1.0)
            if historical_scores:
                best_chunk_id, best_score = max(historical_scores, key=lambda item: (item[1], -item[0]))
                best_historical_chunk_id[letter] = int(best_chunk_id)
                best_historical_support[letter] = float(best_score)
            else:
                best_historical_chunk_id[letter] = None
                best_historical_support[letter] = -1.0

        evidence_gain = {
            letter: float(best_historical_support[letter] - current_support[letter])
            for letter in current_support
        }
        alternatives = {letter: value for letter, value in evidence_gain.items() if letter != predicted_option}
        best_alternative = _argmax(alternatives)
        historical_option = _argmax(best_historical_support)
        current_option = _argmax(current_support)
        heg_alternative = alternatives[best_alternative] if best_alternative else None
        heg_current = evidence_gain.get(predicted_option)
        return {
            "heg_threshold": float(heg_threshold),
            "current_support_by_option": current_support,
            "best_historical_support_by_option": best_historical_support,
            "best_historical_chunk_id_by_option": best_historical_chunk_id,
            "evidence_gain_by_option": evidence_gain,
            "heg_current": heg_current,
            "heg_alternative": heg_alternative,
            "best_alternative_option": best_alternative,
            "current_option_from_clip": current_option,
            "historical_option_from_clip": historical_option,
            "historical_option_margin": _top_margin(best_historical_support),
            "evidence_conflict": bool(historical_option and current_option and historical_option != current_option),
            "unused_historical_candidate_ids": unused_ids,
        }


def _rank_candidates(
    qa: RecentWindowQAModel,
    older_chunks: list[Any],
    prompt: str,
    config: Any,
    candidate_pool: int,
    min_temporal_gap: int,
) -> tuple[list[dict[str, Any]], float]:
    from lib.minicpm import adaptive as adaptive_mod

    ranking_t0 = time.perf_counter()
    bank = adaptive_mod._build_online_memory_bank(older_chunks, config)
    semantic_query = adaptive_mod._extract_semantic_query(prompt)
    cue_scores = [adaptive_mod._semantic_proxy_score(entry, semantic_query) for entry in bank]
    scorer = _get_clip_scorer(qa)
    semantic_scores_raw = _score_chunks_with_clip(scorer, _question_text(prompt), older_chunks)
    semantic_scores = [_normalize_clip_support(score) for score in semantic_scores_raw]

    candidates: list[dict[str, Any]] = []
    for index, entry in enumerate(bank):
        candidates.append(
            {
                "bank_index": index,
                "chunk": entry["chunk"],
                "chunk_id": int(entry["chunk_id"]),
                "timestamp": float(adaptive_mod._chunk_timestamp(entry["chunk"])),
                "semantic_score": float(semantic_scores[index]),
                "semantic_score_raw": float(semantic_scores_raw[index]),
                "event_score": float(entry["event_change_norm"]),
                "detail_score": float(
                    0.50 * float(entry["contrast_norm"]) + 0.50 * float(entry["text_detail_norm"])
                ),
                "cue_score": float(cue_scores[index]),
            }
        )

    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    while remaining and len(selected) < candidate_pool:
        eligible = remaining
        if selected:
            separated = [
                item
                for item in remaining
                if all(
                    abs(int(item["chunk_id"]) - int(chosen["chunk_id"])) >= min_temporal_gap
                    for chosen in selected
                )
            ]
            if separated:
                eligible = separated

        best: dict[str, Any] | None = None
        for item in eligible:
            if not selected:
                diversity = 1.0
            else:
                distance = min(
                    abs(int(item["chunk_id"]) - int(chosen["chunk_id"])) for chosen in selected
                )
                diversity = min(1.0, distance / max(1.0, float(min_temporal_gap)))
            total = (
                0.55 * float(item["semantic_score"])
                + 0.15 * float(item["event_score"])
                + 0.10 * float(item["detail_score"])
                + 0.10 * float(diversity)
                + 0.10 * float(item["cue_score"])
            )
            candidate = {**item, "diversity_score": float(diversity), "total_score": float(total)}
            if best is None or (
                float(candidate["total_score"]), -int(candidate["chunk_id"])
            ) > (float(best["total_score"]), -int(best["chunk_id"])):
                best = candidate
        if best is None:
            break
        selected.append(best)
        remaining = [item for item in remaining if int(item["chunk_id"]) != int(best["chunk_id"])]

    _synchronize_gpu_devices()
    return selected, (time.perf_counter() - ranking_t0) * 1000.0


def _build_minicpm_inputs(
    qa: RecentWindowQAModel,
    frames: list[Image.Image],
    prompt: str,
    assistant_text: str | None = None,
) -> Any:
    content = [{"type": "image", "image": frame} for frame in frames]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    if assistant_text is not None:
        messages.append({"role": "assistant", "content": assistant_text})
    template_kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": assistant_text is None,
        "return_dict": True,
        "return_tensors": "pt",
    }
    processor_kwargs: dict[str, Any] = {
        "downsample_mode": qa.downsample_mode,
        "max_slice_nums": qa.max_slice_nums,
        "use_image_id": False,
    }
    try:
        inputs = qa.processor.apply_chat_template(
            messages,
            **template_kwargs,
            processor_kwargs=processor_kwargs,
        )
    except TypeError:
        inputs = qa.processor.apply_chat_template(
            messages,
            **template_kwargs,
            **processor_kwargs,
        )
    return inputs.to(qa.model.device)


def _sequence_option_logits(
    qa: RecentWindowQAModel,
    frames: list[Image.Image],
    prompt: str,
    options: list[dict[str, str]],
) -> tuple[torch.Tensor, float]:
    """Fallback to teacher-forced label likelihood when labels are not one token."""

    base_ids = _build_minicpm_inputs(qa, frames, prompt).input_ids[0]
    scores: list[torch.Tensor] = []
    forward_t0 = time.perf_counter()
    for option in options:
        inputs = _build_minicpm_inputs(qa, frames, prompt, assistant_text=option["letter"])
        input_ids = inputs.input_ids[0]
        prefix = 0
        limit = min(int(base_ids.numel()), int(input_ids.numel()))
        while prefix < limit and int(base_ids[prefix]) == int(input_ids[prefix]):
            prefix += 1
        if prefix < 1 or prefix >= int(input_ids.numel()):
            raise RuntimeError(
                "Could not isolate assistant option tokens for sequence-likelihood scoring"
            )
        outputs = qa.model(**inputs, use_cache=False, return_dict=True)
        log_probs = torch.log_softmax(outputs.logits[0].float(), dim=-1)
        positions = torch.arange(prefix - 1, int(input_ids.numel()) - 1, device=log_probs.device)
        targets = input_ids[prefix:].to(log_probs.device)
        scores.append(log_probs[positions, targets].sum())
    _synchronize_gpu_devices()
    return torch.stack(scores), (time.perf_counter() - forward_t0) * 1000.0


def _minimal_decode_option_logits(
    qa: RecentWindowQAModel,
    frames: list[Image.Image],
    prompt: str,
    options: list[dict[str, str]],
    fallback_error: Exception,
) -> tuple[torch.Tensor, float, str]:
    inputs = _build_minicpm_inputs(qa, frames, prompt)
    forward_t0 = time.perf_counter()
    generated = qa.model.generate(
        **inputs,
        downsample_mode=qa.downsample_mode,
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=getattr(qa.processor.tokenizer, "eos_token_id", None),
    )
    _synchronize_gpu_devices()
    forward_ms = (time.perf_counter() - forward_t0) * 1000.0
    prompt_len = int(inputs.input_ids.shape[1])
    decoded = qa.processor.tokenizer.decode(
        generated[0][prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip().upper()
    match = re.search(r"\b([A-E])\b", decoded)
    letters = [item["letter"] for item in options]
    predicted_letter = match.group(1) if match and match.group(1) in letters else letters[0]
    logits = torch.full((len(options),), -20.0, dtype=torch.float32, device=qa.model.device)
    logits[letters.index(predicted_letter)] = 20.0
    return logits, forward_ms, repr(fallback_error)


@torch.inference_mode()
def _score_options(
    qa: RecentWindowQAModel,
    frames: list[Image.Image],
    prompt: str,
    options: list[dict[str, str]],
) -> dict[str, Any]:
    tokenizer = qa.processor.tokenizer
    token_ids: dict[str, int] = {}
    direct_logits_valid = True
    for option in options:
        ids = tokenizer.encode(option["letter"], add_special_tokens=False)
        if len(ids) != 1:
            direct_logits_valid = False
        elif option["letter"] in token_ids or int(ids[0]) in token_ids.values():
            direct_logits_valid = False
        else:
            token_ids[option["letter"]] = int(ids[0])
    letters = [item["letter"] for item in options]
    if direct_logits_valid:
        inputs = _build_minicpm_inputs(qa, frames, prompt)
        forward_t0 = time.perf_counter()
        outputs = qa.model(**inputs, use_cache=False, return_dict=True)
        _synchronize_gpu_devices()
        forward_ms = (time.perf_counter() - forward_t0) * 1000.0
        logits = outputs.logits[0, -1].float()
        selected_logits = torch.stack([logits[token_ids[letter]] for letter in letters])
        scoring_mechanism = "direct_option_logits"
        fallback_error = None
    else:
        try:
            selected_logits, forward_ms = _sequence_option_logits(qa, frames, prompt, options)
            scoring_mechanism = "sequence_log_likelihood"
            fallback_error = None
        except Exception as exc:
            selected_logits, forward_ms, fallback_error = _minimal_decode_option_logits(
                qa, frames, prompt, options, exc
            )
            scoring_mechanism = "minimal_one_token_decode_fallback"
    probabilities_tensor = torch.softmax(selected_logits, dim=0)
    probabilities = {
        letter: float(probabilities_tensor[index].item()) for index, letter in enumerate(letters)
    }
    ordered = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))
    top_probability = ordered[0][1]
    second_probability = ordered[1][1] if len(ordered) > 1 else 0.0
    entropy = -sum(probability * math.log(probability) for probability in probabilities.values() if probability > 0)
    normalized_entropy = entropy / max(math.log(len(probabilities)), 1e-8)
    predicted_letter = ordered[0][0]
    predicted_text = next(item["text"] for item in options if item["letter"] == predicted_letter)
    result = {
        "predicted_option": predicted_letter,
        "predicted_answer_text": predicted_text,
        "option_probabilities": probabilities,
        "answer_margin": float(top_probability - second_probability),
        "normalized_entropy": float(normalized_entropy),
        "entropy_confidence": float(1.0 - normalized_entropy),
        "option_token_ids": token_ids,
        "option_scoring_mechanism": scoring_mechanism,
        "option_forward_ms": float(forward_ms),
    }
    if fallback_error is not None:
        result["option_scoring_fallback_error"] = fallback_error
    return result


def _evaluate_sufficiency(
    qa: RecentWindowQAModel,
    context_chunks: list[Any],
    prompt: str,
    options: list[dict[str, str]],
    scorer: AnswerGroundedFrameScorer,
    margin_weight: float,
    entropy_weight: float,
    visual_support_weight: float,
) -> tuple[dict[str, Any], float]:
    iteration_t0 = time.perf_counter()
    frames = [frame for chunk in context_chunks for frame in chunk.frames]
    option_score = _score_options(qa, frames, prompt, options)
    support_text = f"{_question_text(prompt)} Answer: {option_score['predicted_answer_text']}"
    support_scores = scorer.score(support_text, frames)
    visual_support_raw = max(support_scores) if support_scores else -1.0
    visual_support_norm = _normalize_clip_support(visual_support_raw)
    sufficiency = (
        margin_weight * float(option_score["answer_margin"])
        + entropy_weight * float(option_score["entropy_confidence"])
        + visual_support_weight * visual_support_norm
    )
    _synchronize_gpu_devices()
    elapsed_ms = (time.perf_counter() - iteration_t0) * 1000.0
    return {
        **option_score,
        "visual_support_raw": float(visual_support_raw),
        "visual_support_norm": visual_support_norm,
        "sufficiency": float(sufficiency),
        "sufficiency_ms": float(elapsed_ms),
    }, elapsed_ms


def _candidate_metadata(candidate: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        key: candidate[key]
        for key in (
            "chunk_id",
            "timestamp",
            "start_time_seconds",
            "end_time_seconds",
            "candidate_temporal_distance_seconds",
            "history_temporal_violation",
            "semantic_score",
            "event_score",
            "detail_score",
            "cue_score",
            "diversity_score",
            "total_score",
        )
    }
    if "semantic_score_raw" in candidate:
        metadata["semantic_score_raw"] = candidate["semantic_score_raw"]
    for key in (
        "retrieval_variant",
        "retrieval_relevance",
        "best_supported_option",
        "visual_redundancy",
    ):
        if key in candidate:
            metadata[key] = candidate[key]
    return metadata


def _chunk_anchor_time(chunk: Any) -> float:
    start, end = _chunk_temporal_bounds(chunk)
    timestamps = getattr(chunk, "frame_timestamps", None) or []
    numeric_timestamps = [
        float(value)
        for value in timestamps
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    if numeric_timestamps:
        return float(numeric_timestamps[len(numeric_timestamps) // 2])
    return float(0.5 * (start + end))


def _build_temporal_microclip(
    *,
    anchor_chunk: Any,
    historical_chunks: list[Any],
    recent_start_time: float | None,
    offsets: list[float],
    temporal_epsilon: float,
    max_frames: int = 3,
) -> tuple[list[Any], list[float], list[float]]:
    if recent_start_time is None:
        return [anchor_chunk], list(offsets), [_chunk_anchor_time(anchor_chunk)]
    valid_chunks = [
        chunk
        for chunk in historical_chunks
        if _chunk_temporal_bounds(chunk)[1] < float(recent_start_time) - temporal_epsilon
    ]
    if not valid_chunks:
        return [], list(offsets), []

    anchor_time = _chunk_anchor_time(anchor_chunk)
    selected_by_id: dict[int, Any] = {}
    for offset in offsets:
        target_time = anchor_time + float(offset)
        eligible = [
            chunk
            for chunk in valid_chunks
            if int(chunk.chunk_index) not in selected_by_id
            and _chunk_temporal_bounds(chunk)[1] < float(recent_start_time) - temporal_epsilon
        ]
        if not eligible:
            break
        closest = min(
            eligible,
            key=lambda chunk: (
                abs(_chunk_anchor_time(chunk) - target_time),
                abs(int(chunk.chunk_index) - int(anchor_chunk.chunk_index)),
                int(chunk.chunk_index),
            ),
        )
        selected_by_id[int(closest.chunk_index)] = closest
        if len(selected_by_id) >= max_frames:
            break

    selected = sorted(selected_by_id.values(), key=_chunk_sort_key)
    selected_timestamps = [_chunk_anchor_time(chunk) for chunk in selected]
    return selected, list(offsets), selected_timestamps


def _state_record(
    *,
    state: str,
    state_index: int,
    context_chunks: list[Any],
    memory_chunks: list[Any],
    score: dict[str, Any],
    previous_sufficiency: float | None,
) -> dict[str, Any]:
    sufficiency = float(score["sufficiency"])
    return {
        "iteration": state_index,
        "state": state,
        "context_chunk_ids": [int(chunk.chunk_index) for chunk in context_chunks],
        "memory_chunk_ids": [int(chunk.chunk_index) for chunk in memory_chunks],
        "predicted_option": score["predicted_option"],
        "option_probabilities": score["option_probabilities"],
        "option_scoring_mechanism": score["option_scoring_mechanism"],
        "answer_margin": score["answer_margin"],
        "normalized_entropy": score["normalized_entropy"],
        "entropy_confidence": score["entropy_confidence"],
        "visual_support_raw": score["visual_support_raw"],
        "visual_support_norm": score["visual_support_norm"],
        "sufficiency": sufficiency,
        "gain_vs_previous": None if previous_sufficiency is None else sufficiency - previous_sufficiency,
        "sufficiency_ms": score["sufficiency_ms"],
        "option_forward_ms": score["option_forward_ms"],
    }


def _mode_name(
    enable_heg: bool,
    enable_conservative_gate: bool,
    retrieval_variant: str = "current",
    enable_evidence_override: bool = False,
    enable_candidate_override: bool = False,
    enable_candidate_override_protected_rollback: bool = False,
    enable_candidate_override_guarded_rollback: bool = False,
    enable_p3_low_suff_disagree: bool = False,
) -> str:
    if retrieval_variant == "clip_mmr" and enable_p3_low_suff_disagree:
        return "progressive_sufficiency_memory_clip_mmr_p3_low_suff_disagree"
    if retrieval_variant == "clip_mmr" and enable_candidate_override_guarded_rollback:
        return "progressive_sufficiency_memory_clip_mmr_candidate_override_guarded_rollback"
    if retrieval_variant == "clip_mmr" and enable_candidate_override_protected_rollback:
        return "progressive_sufficiency_memory_clip_mmr_candidate_override_protected_rollback"
    if retrieval_variant == "clip_mmr" and enable_candidate_override:
        return "progressive_sufficiency_memory_clip_mmr_candidate_override"
    if retrieval_variant == "clip_mmr" and enable_evidence_override:
        return "progressive_sufficiency_memory_clip_mmr_evidence_override"
    if retrieval_variant == "clip_question_options":
        return "progressive_sufficiency_memory_clip_question_options"
    if retrieval_variant == "clip_mmr":
        return "progressive_sufficiency_memory_clip_mmr"
    if enable_conservative_gate:
        return "progressive_sufficiency_memory_conservative_gate"
    if enable_heg:
        return "progressive_sufficiency_memory_heg"
    return "progressive_sufficiency_memory"


def _chunk_temporal_bounds(chunk: Any) -> tuple[float, float]:
    timestamps = getattr(chunk, "frame_timestamps", None) or []
    numeric_timestamps = [
        float(value)
        for value in timestamps
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    if numeric_timestamps:
        return min(numeric_timestamps), max(numeric_timestamps)
    start = getattr(chunk, "start_time", None)
    end = getattr(chunk, "end_time", None)
    if isinstance(start, (int, float)) and math.isfinite(float(start)):
        start_time = float(start)
    else:
        start_time = float(getattr(chunk, "chunk_index", 0))
    if isinstance(end, (int, float)) and math.isfinite(float(end)):
        end_time = float(end)
    else:
        end_time = start_time
    if end_time < start_time:
        return end_time, start_time
    return start_time, end_time


def _context_temporal_bounds(chunks: list[Any]) -> tuple[float | None, float | None]:
    if not chunks:
        return None, None
    bounds = [_chunk_temporal_bounds(chunk) for chunk in chunks]
    return min(start for start, _end in bounds), max(end for _start, end in bounds)


def _chunk_sort_key(chunk: Any) -> tuple[float, float, int]:
    start, end = _chunk_temporal_bounds(chunk)
    return start, end, int(getattr(chunk, "chunk_index", 0))


def _candidate_distance_from_recent_seconds(candidate: dict[str, Any] | None) -> float | None:
    if candidate is None:
        return None
    distance = candidate.get("candidate_temporal_distance_seconds")
    if isinstance(distance, (int, float)) and math.isfinite(float(distance)):
        return float(distance)
    return None


def _conservative_gate_decision(
    *,
    current_sufficiency: float,
    unused_candidates: list[dict[str, Any]],
    tau_low: float,
    tau_high: float,
    candidate_threshold: float,
    temporal_distance_threshold_seconds: float,
) -> dict[str, Any]:
    best_candidate = unused_candidates[0] if unused_candidates else None
    best_candidate_score = (
        float(best_candidate["total_score"])
        if best_candidate is not None and isinstance(best_candidate.get("total_score"), (int, float))
        else None
    )
    best_candidate_distance = _candidate_distance_from_recent_seconds(best_candidate)
    if current_sufficiency < tau_low:
        return {
            "retrieve": bool(unused_candidates),
            "reason": "low_sufficiency" if unused_candidates else "low_sufficiency_no_candidates",
            "best_candidate_score": best_candidate_score,
            "best_candidate_temporal_distance_seconds": best_candidate_distance,
            "best_candidate_chunk_id": int(best_candidate["chunk_id"]) if best_candidate is not None else None,
        }
    if current_sufficiency >= tau_high:
        return {
            "retrieve": False,
            "reason": "high_sufficiency",
            "best_candidate_score": best_candidate_score,
            "best_candidate_temporal_distance_seconds": best_candidate_distance,
            "best_candidate_chunk_id": int(best_candidate["chunk_id"]) if best_candidate is not None else None,
        }
    candidate_ok = best_candidate_score is not None and best_candidate_score > candidate_threshold
    temporal_ok = (
        best_candidate_distance is not None
        and best_candidate_distance > temporal_distance_threshold_seconds
    )
    retrieve = bool(unused_candidates) and candidate_ok and temporal_ok
    if retrieve:
        reason = "ambiguous_strong_temporal_candidate"
    elif not unused_candidates:
        reason = "ambiguous_no_candidates"
    elif not candidate_ok and not temporal_ok:
        reason = "ambiguous_weak_near_candidate"
    elif not candidate_ok:
        reason = "ambiguous_weak_candidate"
    else:
        reason = "ambiguous_near_candidate"
    return {
        "retrieve": retrieve,
        "reason": reason,
        "best_candidate_score": best_candidate_score,
        "best_candidate_temporal_distance_seconds": best_candidate_distance,
        "best_candidate_chunk_id": int(best_candidate["chunk_id"]) if best_candidate is not None else None,
    }


def _print_trace(metadata: dict[str, Any]) -> None:
    if os.environ.get("MINICPM_PSM_PRINT_TRACE", "1").strip().lower() in {"0", "false", "no", "off"}:
        return
    print(
        f"[PSM] stop={metadata['stop_reason']} memory={metadata['memory_chunk_ids']} "
        f"final_sufficiency={metadata['final_sufficiency']}",
        flush=True,
    )
    for item in metadata["iterations"]:
        print(
            "[PSM] "
            f"iter={item['iteration']} added={item.get('added_chunk_id')} "
            f"context={item['context_chunk_ids']} pred={item.get('predicted_option')} "
            f"margin={item.get('answer_margin')} entropy={item.get('normalized_entropy')} "
            f"support={item.get('visual_support_norm')} suff={item.get('sufficiency')} "
            f"gain={item.get('gain_vs_previous')}",
            flush=True,
        )


def select_progressive_sufficiency_memory_microclip(
    qa: RecentWindowQAModel,
    chunks: list[Any],
    prompt: str,
    config: Any,
    recent_chunks: list[Any] | None = None,
    recent_frames: list[Image.Image] | None = None,
    recent_chunk_ids: list[int] | None = None,
    recent_downsample_mode: str | None = None,
    baseline_recent_metadata: dict[str, Any] | None = None,
) -> ProgressiveSufficiencySelection:
    """PRISM micro-clip mode: one semantic anchor, then local temporal expansion.

    This is intentionally isolated from select_progressive_sufficiency_memory so
    the existing PRISM/HEG/conservative-gate modes keep their behavior.
    """

    recent_window = 6
    history_search_chunks = _env_int("MINICPM_PSM_HISTORY_SEARCH_CHUNKS", 64)
    candidate_pool = _env_int("MINICPM_PSM_HISTORY_CANDIDATE_POOL", 12)
    min_temporal_gap = _env_int("MINICPM_PSM_MIN_TEMPORAL_GAP", 2)
    sufficiency_threshold = _env_float("MINICPM_PSM_SUFFICIENCY_THRESHOLD", 0.62)
    margin_weight = _env_float("MINICPM_PSM_MARGIN_WEIGHT", 0.50)
    entropy_weight = _env_float("MINICPM_PSM_ENTROPY_WEIGHT", 0.20)
    visual_support_weight = _env_float("MINICPM_PSM_VISUAL_SUPPORT_WEIGHT", 0.30)
    temporal_epsilon = _env_float("MINICPM_PSM_TEMPORAL_EPSILON_SECONDS", 1e-6)
    microclip_offsets = _env_offsets("MINICPM_PSM_MICROCLIP_OFFSETS", "-1,0,1")
    variant = os.environ.get("MINICPM_PSM_MICROCLIP_VARIANT", "temporal_microclip").strip()
    valid_variants = {"anchor_only", "temporal_microclip", "sparse_history_3"}
    if variant not in valid_variants:
        raise ValueError(
            f"Unknown MINICPM_PSM_MICROCLIP_VARIANT={variant!r}; expected one of {sorted(valid_variants)}"
        )

    if recent_chunks is None:
        recent_chunks = list(chunks[-recent_window:])
    else:
        recent_chunks = list(recent_chunks)
    if recent_frames is None:
        recent_frames = [frame for chunk in recent_chunks for frame in chunk.frames]
    else:
        recent_frames = list(recent_frames)
    if recent_chunk_ids is None:
        recent_ids = [int(chunk.chunk_index) for chunk in recent_chunks]
    else:
        recent_ids = [int(value) for value in recent_chunk_ids]

    recent_start_time, recent_end_time = _context_temporal_bounds(recent_chunks)
    if recent_start_time is None:
        all_older_chunks: list[Any] = []
    else:
        all_older_chunks = [
            chunk
            for chunk in chunks
            if _chunk_temporal_bounds(chunk)[1] < float(recent_start_time) - temporal_epsilon
        ]
    older_chunks = all_older_chunks[-history_search_chunks:] if history_search_chunks > 0 else all_older_chunks
    older_chunk_bounds = {
        int(chunk.chunk_index): _chunk_temporal_bounds(chunk)
        for chunk in older_chunks
    }
    options = _extract_mcq_options(prompt)
    mode_name = "progressive_sufficiency_memory_microclip"

    if not options:
        final_ids = list(recent_ids)
        metadata = {
            "mode": mode_name,
            "microclip_variant": variant,
            "recent_chunk_ids": recent_ids,
            "baseline_recent_equivalence": {
                "enabled": recent_chunk_ids is not None,
                "source": "select_recent_window_frames",
                "prompt_equal_to_final": True,
                "final_equals_recent": True,
                "downsample_mode": recent_downsample_mode,
                "baseline_recent": baseline_recent_metadata or {},
            },
            "history_search_start": int(older_chunks[0].chunk_index) if older_chunks else None,
            "history_search_end": int(older_chunks[-1].chunk_index) if older_chunks else None,
            "recent_start_time_seconds": recent_start_time,
            "recent_end_time_seconds": recent_end_time,
            "history_candidate_start_time_seconds": [],
            "history_candidate_end_time_seconds": [],
            "history_temporal_violation_count": 0,
            "candidate_queue": [],
            "iterations": [],
            "microclip_requested_offsets": microclip_offsets,
            "microclip_selected_chunk_ids": [],
            "microclip_selected_timestamps": [],
            "microclip_num_frames": 0,
            "memory_triggered": False,
            "memory_chunk_ids": [],
            "num_memory_frames": 0,
            "final_selected_chunk_ids": final_ids,
            "final_sufficiency": None,
            "final_state": "recent_only",
            "prompt_variant": "recent_only",
            "stop_reason": "unsupported_non_mcq_recent_only",
            "history_candidate_ranking_ms": 0.0,
            "sufficiency_iterations_ms": [],
            "total_sufficiency_ms": 0.0,
            "final_generation_ms": None,
            "num_sufficiency_iterations": 0,
            "num_extra_frames": 0,
            "all_history_strictly_before_recent": True,
        }
        _validate_metadata(metadata)
        _print_trace(metadata)
        return ProgressiveSufficiencySelection(
            frames=recent_frames,
            final_chunk_ids=final_ids,
            metadata=metadata,
            answer_prompt=prompt,
            downsample_mode=recent_downsample_mode,
        )

    candidate_queue, ranking_ms = _rank_candidates(
        qa,
        older_chunks,
        prompt,
        config,
        candidate_pool=max(0, candidate_pool),
        min_temporal_gap=max(1, min_temporal_gap),
    )
    for candidate in candidate_queue:
        candidate_chunk_id = int(candidate["chunk_id"])
        start_time, end_time = older_chunk_bounds.get(candidate_chunk_id, _chunk_temporal_bounds(candidate["chunk"]))
        candidate["start_time_seconds"] = float(start_time)
        candidate["end_time_seconds"] = float(end_time)
        candidate["candidate_temporal_distance_seconds"] = (
            float(recent_start_time - end_time)
            if recent_start_time is not None
            else None
        )
        candidate["history_temporal_violation"] = bool(
            recent_start_time is not None and end_time >= float(recent_start_time)
        )
    temporal_violations = [
        candidate
        for candidate in candidate_queue
        if bool(candidate.get("history_temporal_violation"))
    ]
    assert_temporal_alignment = os.environ.get(
        "MINICPM_PSM_ASSERT_TEMPORAL_ALIGNMENT",
        "1",
    ).strip().lower() not in {"0", "false", "no", "off"}
    if temporal_violations and assert_temporal_alignment:
        first = temporal_violations[0]
        raise AssertionError(
            "PRISM microclip history temporal alignment violation: "
            f"candidate chunk_id={first.get('chunk_id')} end={first.get('end_time_seconds')} "
            f"is not strictly before recent_start={recent_start_time}."
        )

    scorer = _get_clip_scorer(qa)
    iterations: list[dict[str, Any]] = []
    iteration_times: list[float] = []
    best_state = "recent_only"
    best_memory: list[Any] = []
    best_sufficiency = -float("inf")
    stop_reason = "candidate_queue_exhausted"

    def evaluate_state(state: str, memory_chunks: list[Any]) -> dict[str, Any]:
        nonlocal best_memory, best_state, best_sufficiency
        chronological_memory = sorted(memory_chunks, key=_chunk_sort_key)
        context_chunks = [*chronological_memory, *recent_chunks]
        previous = iterations[-1]["sufficiency"] if iterations else None
        score, elapsed_ms = _evaluate_sufficiency(
            qa,
            context_chunks,
            prompt,
            options,
            scorer,
            margin_weight,
            entropy_weight,
            visual_support_weight,
        )
        iteration_times.append(float(elapsed_ms))
        record = _state_record(
            state=state,
            state_index=len(iterations),
            context_chunks=context_chunks,
            memory_chunks=chronological_memory,
            score=score,
            previous_sufficiency=previous,
        )
        iterations.append(record)
        if float(record["sufficiency"]) > best_sufficiency:
            best_sufficiency = float(record["sufficiency"])
            best_memory = list(chronological_memory)
            best_state = state
        return record

    state0 = evaluate_state("recent_only", [])
    anchor_candidate = candidate_queue[0] if candidate_queue else None
    anchor_chunk = anchor_candidate["chunk"] if anchor_candidate is not None else None
    state1: dict[str, Any] | None = None
    state2: dict[str, Any] | None = None
    microclip_chunks: list[Any] = []
    microclip_selected_timestamps: list[float] = []

    if float(state0["sufficiency"]) >= sufficiency_threshold:
        stop_reason = "sufficient_recent_only"
    elif anchor_chunk is None:
        stop_reason = "low_sufficiency_no_candidates"
    else:
        state1 = evaluate_state("anchor", [anchor_chunk])
        if variant == "anchor_only":
            stop_reason = (
                "sufficient_anchor"
                if float(state1["sufficiency"]) >= sufficiency_threshold
                else "anchor_only_variant_stop"
            )
        elif float(state1["sufficiency"]) >= sufficiency_threshold:
            stop_reason = "sufficient_anchor"
        else:
            if variant == "sparse_history_3":
                sparse_chunks = [candidate["chunk"] for candidate in candidate_queue[:3]]
                state2 = evaluate_state("sparse_history_3", sparse_chunks)
                stop_reason = (
                    "sufficient_sparse_history_3"
                    if float(state2["sufficiency"]) >= sufficiency_threshold
                    else "sparse_history_3_variant_stop"
                )
            else:
                microclip_chunks, _requested_offsets, microclip_selected_timestamps = _build_temporal_microclip(
                    anchor_chunk=anchor_chunk,
                    historical_chunks=older_chunks,
                    recent_start_time=recent_start_time,
                    offsets=microclip_offsets,
                    temporal_epsilon=temporal_epsilon,
                    max_frames=3,
                )
                state2 = evaluate_state("microclip", microclip_chunks)
                stop_reason = (
                    "sufficient_microclip"
                    if float(state2["sufficiency"]) >= sufficiency_threshold
                    else "microclip_variant_stop"
                )

    best_memory = sorted(best_memory, key=_chunk_sort_key)
    memory_ids = [int(chunk.chunk_index) for chunk in best_memory]
    final_chunks = [*best_memory, *recent_chunks]
    final_ids = [int(chunk.chunk_index) for chunk in final_chunks]
    final_historical_frames = sum(len(chunk.frames) for chunk in best_memory)
    if best_memory:
        memory_start, memory_end = _context_temporal_bounds(best_memory)
        temporal_span = float(memory_end - memory_start) if memory_start is not None and memory_end is not None else 0.0
        all_history_before_recent = bool(
            recent_start_time is not None
            and all(_chunk_temporal_bounds(chunk)[1] < float(recent_start_time) - temporal_epsilon for chunk in best_memory)
        )
    else:
        temporal_span = 0.0
        all_history_before_recent = True

    if best_state == "microclip":
        prompt_variant = "temporal_microclip"
        answer_prompt = f"{_PSM_MICROCLIP_INSTRUCTION}{prompt}"
    elif memory_ids:
        prompt_variant = "historical_anchor"
        answer_prompt = f"{_PSM_HISTORY_INSTRUCTION}{prompt}"
    else:
        prompt_variant = "recent_only"
        answer_prompt = prompt

    anchor_metadata = _candidate_metadata(anchor_candidate) if anchor_candidate is not None else {}
    metadata = {
        "mode": mode_name,
        "microclip_variant": variant,
        "config": {
            "recent_window": recent_window,
            "history_search_chunks": history_search_chunks,
            "history_candidate_pool": candidate_pool,
            "min_temporal_gap": min_temporal_gap,
            "sufficiency_threshold": sufficiency_threshold,
            "microclip_offsets": microclip_offsets,
            "microclip_variant": variant,
            "margin_weight": margin_weight,
            "entropy_weight": entropy_weight,
            "visual_support_weight": visual_support_weight,
            "visual_support_raw_low": 0.15,
            "visual_support_raw_span": 0.30,
        },
        "recent_chunk_ids": recent_ids,
        "recent_start_time_seconds": recent_start_time,
        "recent_end_time_seconds": recent_end_time,
        "baseline_recent_equivalence": {
            "enabled": recent_chunk_ids is not None,
            "source": "select_recent_window_frames",
            "prompt_equal_to_final": not bool(memory_ids),
            "final_equals_recent": final_ids == recent_ids,
            "downsample_mode": recent_downsample_mode,
            "baseline_recent": baseline_recent_metadata or {},
        },
        "history_search_start": int(older_chunks[0].chunk_index) if older_chunks else None,
        "history_search_end": int(older_chunks[-1].chunk_index) if older_chunks else None,
        "history_candidate_start_time_seconds": [
            float(_chunk_temporal_bounds(chunk)[0]) for chunk in older_chunks
        ],
        "history_candidate_end_time_seconds": [
            float(_chunk_temporal_bounds(chunk)[1]) for chunk in older_chunks
        ],
        "history_temporal_violation_count": len(temporal_violations),
        "candidate_queue": [_candidate_metadata(candidate) for candidate in candidate_queue],
        "iterations": iterations,
        "candidate_anchor_id": anchor_metadata.get("chunk_id"),
        "candidate_anchor_timestamp": anchor_metadata.get("timestamp"),
        "candidate_total_score": anchor_metadata.get("total_score"),
        "candidate_semantic_score": anchor_metadata.get("semantic_score"),
        "microclip_requested_offsets": microclip_offsets,
        "microclip_selected_chunk_ids": [int(chunk.chunk_index) for chunk in microclip_chunks],
        "microclip_selected_timestamps": microclip_selected_timestamps,
        "microclip_num_frames": sum(len(chunk.frames) for chunk in microclip_chunks),
        "state_0_sufficiency": state0.get("sufficiency"),
        "state_1_sufficiency": state1.get("sufficiency") if state1 else None,
        "state_2_sufficiency": state2.get("sufficiency") if state2 else None,
        "state_0_prediction": state0.get("predicted_option"),
        "state_1_prediction": state1.get("predicted_option") if state1 else None,
        "state_2_prediction": state2.get("predicted_option") if state2 else None,
        "state_1_gain": state1.get("gain_vs_previous") if state1 else None,
        "state_2_gain": state2.get("gain_vs_previous") if state2 else None,
        "final_state": best_state,
        "final_historical_frames": final_historical_frames,
        "temporal_span_seconds": temporal_span,
        "all_history_strictly_before_recent": all_history_before_recent,
        "prompt_variant": prompt_variant,
        "memory_triggered": bool(memory_ids),
        "memory_chunk_ids": memory_ids,
        "num_memory_frames": len(memory_ids),
        "final_selected_chunk_ids": final_ids,
        "final_sufficiency": float(best_sufficiency),
        "stop_reason": stop_reason,
        "history_candidate_ranking_ms": float(ranking_ms),
        "sufficiency_iterations_ms": iteration_times,
        "total_sufficiency_ms": float(sum(iteration_times)),
        "final_generation_ms": None,
        "num_sufficiency_iterations": len(iterations),
        "num_extra_frames": len(memory_ids),
    }
    _validate_metadata(metadata)
    if not all_history_before_recent:
        raise AssertionError(
            "PRISM microclip selected non-historical frames: "
            f"memory_ids={memory_ids} recent_start={recent_start_time}"
        )
    _print_trace(metadata)
    memory_frames = [frame for chunk in best_memory for frame in chunk.frames]
    return ProgressiveSufficiencySelection(
        frames=[*memory_frames, *recent_frames],
        final_chunk_ids=final_ids,
        metadata=metadata,
        answer_prompt=answer_prompt,
        downsample_mode=recent_downsample_mode,
    )


def select_progressive_sufficiency_memory(
    qa: RecentWindowQAModel,
    chunks: list[Any],
    prompt: str,
    config: Any,
    recent_chunks: list[Any] | None = None,
    recent_frames: list[Image.Image] | None = None,
    recent_chunk_ids: list[int] | None = None,
    recent_downsample_mode: str | None = None,
    baseline_recent_metadata: dict[str, Any] | None = None,
    enable_heg: bool = False,
    enable_conservative_gate: bool = False,
    retrieval_variant: str = "current",
    enable_evidence_override: bool = False,
    enable_candidate_override: bool = False,
    enable_candidate_override_protected_rollback: bool = False,
    enable_candidate_override_guarded_rollback: bool = False,
    enable_p3_low_suff_disagree: bool = False,
) -> ProgressiveSufficiencySelection:
    recent_window = 6
    history_search_chunks = _env_int("MINICPM_PSM_HISTORY_SEARCH_CHUNKS", 64)
    candidate_pool = _env_int("MINICPM_PSM_HISTORY_CANDIDATE_POOL", 12)
    max_memory_frames = min(3, _env_int("MINICPM_PSM_MAX_MEMORY_FRAMES", 3))
    min_temporal_gap = _env_int("MINICPM_PSM_MIN_TEMPORAL_GAP", 2)
    sufficiency_threshold = _env_float("MINICPM_PSM_SUFFICIENCY_THRESHOLD", 0.62)
    conservative_tau_low = _env_float("MINICPM_PSM_GATE_TAU_LOW", sufficiency_threshold)
    conservative_tau_high = _env_float("MINICPM_PSM_GATE_TAU_HIGH", 0.74)
    conservative_candidate_threshold = _env_float("MINICPM_PSM_GATE_CANDIDATE_THRESHOLD", 0.535)
    if "MINICPM_PSM_GATE_TEMPORAL_DISTANCE_SECONDS" in os.environ:
        conservative_temporal_distance_seconds = _env_float(
            "MINICPM_PSM_GATE_TEMPORAL_DISTANCE_SECONDS",
            10.0,
        )
        conservative_temporal_distance_source = "MINICPM_PSM_GATE_TEMPORAL_DISTANCE_SECONDS"
    else:
        conservative_temporal_distance_seconds = _env_float(
            "MINICPM_PSM_GATE_TEMPORAL_DISTANCE_THRESHOLD",
            10.0,
        )
        conservative_temporal_distance_source = "MINICPM_PSM_GATE_TEMPORAL_DISTANCE_THRESHOLD_compat_seconds"
    heg_threshold = _env_float("ADAPTIVE_HEG_THRESHOLD", 0.10)
    if "ADAPTIVE_HEG_THRESHOLD" not in os.environ:
        heg_threshold = _env_float("MINICPM_PSM_HEG_THRESHOLD", 0.10)
    min_evidence_gain = _env_float("MINICPM_PSM_MIN_EVIDENCE_GAIN", 0.035)
    negative_gain_tolerance = _env_float("MINICPM_PSM_NEGATIVE_GAIN_TOLERANCE", 0.02)
    margin_weight = _env_float("MINICPM_PSM_MARGIN_WEIGHT", 0.50)
    entropy_weight = _env_float("MINICPM_PSM_ENTROPY_WEIGHT", 0.20)
    visual_support_weight = _env_float("MINICPM_PSM_VISUAL_SUPPORT_WEIGHT", 0.30)
    evidence_override_gamma = _env_float("MINICPM_PSM_EVIDENCE_OVERRIDE_GAMMA", 0.30)
    evidence_override_min_margin = _env_float("MINICPM_PSM_EVIDENCE_OVERRIDE_MIN_MARGIN", 0.10)
    clip_override_threshold = _env_float("MINICPM_PSM_CLIP_OVERRIDE_THRESHOLD", 0.2995)

    if recent_chunks is None:
        recent_chunks = list(chunks[-recent_window:])
    else:
        recent_chunks = list(recent_chunks)
    if recent_frames is None:
        recent_frames = [frame for chunk in recent_chunks for frame in chunk.frames]
    else:
        recent_frames = list(recent_frames)
    if recent_chunk_ids is None:
        recent_ids = [int(chunk.chunk_index) for chunk in recent_chunks]
    else:
        recent_ids = [int(value) for value in recent_chunk_ids]

    recent_id_set = set(recent_ids)
    recent_start_time, recent_end_time = _context_temporal_bounds(recent_chunks)
    temporal_epsilon = _env_float("MINICPM_PSM_TEMPORAL_EPSILON_SECONDS", 1e-6)
    if recent_start_time is None:
        all_older_chunks = []
    else:
        all_older_chunks = [
            chunk
            for chunk in chunks
            if _chunk_temporal_bounds(chunk)[1] < float(recent_start_time) - temporal_epsilon
        ]
    older_chunks = all_older_chunks[-history_search_chunks:] if history_search_chunks > 0 else all_older_chunks
    older_chunk_bounds = {
        int(chunk.chunk_index): _chunk_temporal_bounds(chunk)
        for chunk in older_chunks
    }
    options = _extract_mcq_options(prompt)
    retrieval_variant = str(retrieval_variant or "current")
    mode_name = _mode_name(
        enable_heg=enable_heg,
        enable_conservative_gate=enable_conservative_gate,
        retrieval_variant=retrieval_variant,
        enable_evidence_override=enable_evidence_override,
        enable_candidate_override=enable_candidate_override,
        enable_candidate_override_protected_rollback=enable_candidate_override_protected_rollback,
        enable_candidate_override_guarded_rollback=enable_candidate_override_guarded_rollback,
        enable_p3_low_suff_disagree=enable_p3_low_suff_disagree,
    )
    candidate_override_like = bool(
        enable_candidate_override
        or enable_candidate_override_protected_rollback
        or enable_candidate_override_guarded_rollback
        or enable_p3_low_suff_disagree
    )

    if not options:
        final_ids = list(recent_ids)
        metadata = {
            "mode": mode_name,
            "recent_chunk_ids": recent_ids,
            "baseline_recent_equivalence": {
                "enabled": recent_chunk_ids is not None,
                "source": "select_recent_window_frames",
                "prompt_equal_to_final": True,
                "final_equals_recent": True,
                "downsample_mode": recent_downsample_mode,
                "baseline_recent": baseline_recent_metadata or {},
            },
            "history_search_start": int(older_chunks[0].chunk_index) if older_chunks else None,
            "history_search_end": int(older_chunks[-1].chunk_index) if older_chunks else None,
            "recent_start_time_seconds": recent_start_time,
            "recent_end_time_seconds": recent_end_time,
            "history_candidate_start_time_seconds": [],
            "history_candidate_end_time_seconds": [],
            "history_temporal_violation_count": 0,
            "candidate_queue": [],
            "iterations": [],
            "memory_triggered": False,
            "memory_chunk_ids": [],
            "num_memory_frames": 0,
            "final_selected_chunk_ids": final_ids,
            "final_sufficiency": None,
            "stop_reason": "unsupported_non_mcq_recent_only",
            "history_candidate_ranking_ms": 0.0,
            "sufficiency_iterations_ms": [],
            "total_sufficiency_ms": 0.0,
            "final_generation_ms": None,
            "num_sufficiency_iterations": 0,
            "num_extra_frames": 0,
        }
        _validate_metadata(metadata)
        _print_trace(metadata)
        return ProgressiveSufficiencySelection(
            frames=recent_frames,
            final_chunk_ids=final_ids,
            metadata=metadata,
            answer_prompt=prompt,
            downsample_mode=recent_downsample_mode,
        )

    if retrieval_variant == "current":
        candidate_queue, ranking_ms = _rank_candidates(
            qa,
            older_chunks,
            prompt,
            config,
            candidate_pool=max(0, candidate_pool),
            min_temporal_gap=max(1, min_temporal_gap),
        )
    else:
        from lib.minicpm.prism_retrieval_variants import rank_candidates

        candidate_queue, ranking_ms, retrieval_stats = rank_candidates(
            qa=qa,
            older_chunks=older_chunks,
            prompt=prompt,
            config=config,
            candidate_pool=max(0, candidate_pool),
            min_temporal_gap=max(1, min_temporal_gap),
            variant=retrieval_variant,
            mmr_lambda=_env_float("MINICPM_PSM_MMR_LAMBDA", 0.80),
            recent_start_time=recent_start_time,
        )
        baseline_recent_metadata = {
            **(baseline_recent_metadata or {}),
            "retrieval_variant_stats": retrieval_stats,
        }
    for candidate in candidate_queue:
        chunk_id = int(candidate["chunk_id"])
        start_time, end_time = older_chunk_bounds.get(chunk_id, _chunk_temporal_bounds(candidate["chunk"]))
        candidate["start_time_seconds"] = float(start_time)
        candidate["end_time_seconds"] = float(end_time)
        candidate["candidate_temporal_distance_seconds"] = (
            float(recent_start_time - end_time)
            if recent_start_time is not None
            else None
        )
        candidate["history_temporal_violation"] = bool(
            recent_start_time is not None and end_time >= float(recent_start_time)
        )
    temporal_violations = [
        candidate
        for candidate in candidate_queue
        if bool(candidate.get("history_temporal_violation"))
    ]
    assert_temporal_alignment = os.environ.get(
        "MINICPM_PSM_ASSERT_TEMPORAL_ALIGNMENT",
        "1",
    ).strip().lower() not in {"0", "false", "no", "off"}
    if temporal_violations and assert_temporal_alignment:
        first = temporal_violations[0]
        raise AssertionError(
            "PRISM history temporal alignment violation: "
            f"candidate chunk_id={first.get('chunk_id')} end={first.get('end_time_seconds')} "
            f"is not strictly before recent_start={recent_start_time}. "
            "History candidates must be selected by absolute video time, not cross-decode chunk IDs."
        )
    scorer = _get_clip_scorer(qa)
    heg_scorer = _OptionEvidenceGainScorer(scorer, prompt, options) if enable_heg else None
    selected_memory: list[Any] = []
    best_memory: list[Any] = []
    iterations: list[dict[str, Any]] = []
    iteration_times: list[float] = []
    best_sufficiency = -float("inf")
    previous_sufficiency: float | None = None
    stop_reason = "candidate_queue_exhausted"
    override_triggered = False
    override_k0_prediction: str | None = None
    override_k0_sufficiency: float | None = None
    override_protected_memory: list[Any] | None = None

    for iteration_index in range(max_memory_frames + 1):
        added_chunk_id = None
        if iteration_index > 0:
            if iteration_index - 1 >= len(candidate_queue):
                stop_reason = "candidate_queue_exhausted"
                break
            candidate = candidate_queue[iteration_index - 1]
            selected_memory.append(candidate["chunk"])
            added_chunk_id = int(candidate["chunk_id"])

        chronological_memory = sorted(selected_memory, key=_chunk_sort_key)
        context_chunks = [*chronological_memory, *recent_chunks]
        score, elapsed_ms = _evaluate_sufficiency(
            qa,
            context_chunks,
            prompt,
            options,
            scorer,
            margin_weight,
            entropy_weight,
            visual_support_weight,
        )
        iteration_times.append(float(elapsed_ms))
        current_sufficiency = float(score["sufficiency"])
        gain = None if previous_sufficiency is None else current_sufficiency - previous_sufficiency
        if current_sufficiency > best_sufficiency:
            best_sufficiency = current_sufficiency
            best_memory = list(chronological_memory)
        selected_memory_ids = {int(chunk.chunk_index) for chunk in chronological_memory}
        unused_candidates = [
            candidate
            for candidate in candidate_queue
            if int(candidate["chunk_id"]) not in selected_memory_ids and int(candidate["chunk_id"]) not in recent_id_set
        ]
        heg_metadata: dict[str, Any] = {}
        if heg_scorer is not None:
            heg_metadata = heg_scorer.compute(
                context_chunks=context_chunks,
                unused_candidates=unused_candidates,
                predicted_option=str(score["predicted_option"]),
                recent_ids=recent_id_set,
                selected_memory_ids=selected_memory_ids,
                heg_threshold=heg_threshold,
            )
        low_sufficiency_trigger = current_sufficiency < sufficiency_threshold
        top1_candidate = unused_candidates[0] if unused_candidates else None
        top1_relevance = (
            float(top1_candidate.get("retrieval_relevance", top1_candidate.get("semantic_score")))
            if top1_candidate is not None
            and isinstance(top1_candidate.get("retrieval_relevance", top1_candidate.get("semantic_score")), (int, float))
            else None
        )
        strong_candidate_override = bool(
            enable_evidence_override
            and iteration_index == 0
            and not low_sufficiency_trigger
            and top1_relevance is not None
            and top1_relevance >= evidence_override_gamma
            and bool(unused_candidates)
        )
        top1_best_supported_option = top1_candidate.get("best_supported_option") if top1_candidate else None
        retrieval_disagreement = bool(
            candidate_override_like
            and top1_best_supported_option is not None
            and str(top1_best_supported_option) != str(score["predicted_option"])
        )
        strong_candidate_disagreement_override = bool(
            candidate_override_like
            and iteration_index == 0
            and top1_relevance is not None
            and top1_relevance >= clip_override_threshold
            and retrieval_disagreement
            and bool(unused_candidates)
        )
        conservative_gate: dict[str, Any] = {}
        if enable_conservative_gate:
            conservative_gate = _conservative_gate_decision(
                current_sufficiency=current_sufficiency,
                unused_candidates=unused_candidates,
                tau_low=conservative_tau_low,
                tau_high=conservative_tau_high,
                candidate_threshold=conservative_candidate_threshold,
                temporal_distance_threshold_seconds=conservative_temporal_distance_seconds,
            )
        heg_alternative = heg_metadata.get("heg_alternative")
        historical_gain_trigger = (
            enable_heg
            and isinstance(heg_alternative, (int, float))
            and float(heg_alternative) > heg_threshold
            and bool(unused_candidates)
        )
        if low_sufficiency_trigger and historical_gain_trigger:
            trigger_reason = "low_sufficiency_and_historical_gain"
        elif low_sufficiency_trigger and strong_candidate_disagreement_override:
            trigger_reason = "low_sufficiency_and_strong_candidate_disagreement_override"
        elif enable_p3_low_suff_disagree and low_sufficiency_trigger:
            trigger_reason = "low_sufficiency_no_strong_candidate_disagreement"
        elif low_sufficiency_trigger:
            trigger_reason = "low_sufficiency"
        elif historical_gain_trigger:
            trigger_reason = "historical_evidence_gain"
        elif strong_candidate_override:
            trigger_reason = "strong_candidate_override"
        elif strong_candidate_disagreement_override:
            trigger_reason = "strong_candidate_disagreement_override"
        else:
            trigger_reason = "none"

        iteration_record = {
            "iteration": iteration_index,
            "context_chunk_ids": [int(chunk.chunk_index) for chunk in context_chunks],
            "added_chunk_id": added_chunk_id,
            "predicted_option": score["predicted_option"],
            "option_probabilities": score["option_probabilities"],
            "option_scoring_mechanism": score["option_scoring_mechanism"],
            "answer_margin": score["answer_margin"],
            "normalized_entropy": score["normalized_entropy"],
            "entropy_confidence": score["entropy_confidence"],
            "visual_support_raw": score["visual_support_raw"],
            "visual_support_norm": score["visual_support_norm"],
            "sufficiency": current_sufficiency,
            "gain_vs_previous": gain,
            "sufficiency_ms": score["sufficiency_ms"],
            "option_forward_ms": score["option_forward_ms"],
            "retrieval_trigger_reason": trigger_reason,
            "top1_unused_candidate_chunk_id": top1_candidate.get("chunk_id") if top1_candidate else None,
            "top1_unused_candidate_relevance": top1_relevance,
            "top1_unused_candidate_total_score": top1_candidate.get("total_score") if top1_candidate else None,
            "top1_unused_candidate_best_supported_option": top1_best_supported_option,
            "top1_unused_candidate_temporal_distance_seconds": (
                top1_candidate.get("candidate_temporal_distance_seconds") if top1_candidate else None
            ),
            "evidence_override_enabled": bool(enable_evidence_override),
            "evidence_override_gamma": float(evidence_override_gamma) if enable_evidence_override else None,
            "strong_candidate_override": bool(strong_candidate_override),
            "candidate_override_enabled": bool(candidate_override_like),
            "candidate_override_protected_rollback_enabled": bool(enable_candidate_override_protected_rollback),
            "candidate_override_guarded_rollback_enabled": bool(enable_candidate_override_guarded_rollback),
            "p3_low_suff_disagree_enabled": bool(enable_p3_low_suff_disagree),
            "clip_override_threshold": float(clip_override_threshold) if candidate_override_like else None,
            "retrieval_disagreement": bool(retrieval_disagreement),
            "strong_candidate_disagreement_override": bool(strong_candidate_disagreement_override),
        }
        if enable_heg:
            iteration_record["retrieval_trigger_reason"] = trigger_reason
            iteration_record.update(heg_metadata)
        if enable_conservative_gate:
            iteration_record["retrieval_trigger_reason"] = str(conservative_gate.get("reason", "none"))
            iteration_record["conservative_gate"] = {
                "retrieve": bool(conservative_gate.get("retrieve")),
                "reason": str(conservative_gate.get("reason", "none")),
                "tau_low": float(conservative_tau_low),
                "tau_high": float(conservative_tau_high),
                "candidate_threshold": float(conservative_candidate_threshold),
                "temporal_distance_threshold_seconds": float(conservative_temporal_distance_seconds),
                "temporal_distance_threshold_source": conservative_temporal_distance_source,
                "best_unused_candidate_chunk_id": conservative_gate.get("best_candidate_chunk_id"),
                "best_unused_candidate_total_score": conservative_gate.get("best_candidate_score"),
                "best_unused_candidate_temporal_distance_seconds": conservative_gate.get(
                    "best_candidate_temporal_distance_seconds"
                ),
            }
        iterations.append(iteration_record)

        if override_triggered and iteration_index == 1:
            k1_prediction = str(score["predicted_option"])
            answer_changed = bool(override_k0_prediction is not None and k1_prediction != override_k0_prediction)
            confidence_collapsed = bool(float(score["answer_margin"]) < evidence_override_min_margin)
            guarded_rollback_blocked = bool(
                enable_candidate_override_guarded_rollback
                and answer_changed
                and not confidence_collapsed
                and override_k0_sufficiency is not None
                and override_k0_sufficiency >= sufficiency_threshold
                and float(score["sufficiency"]) < float(override_k0_sufficiency)
            )
            iteration_record["answer_changed_after_strong_candidate"] = answer_changed
            iteration_record["evidence_override_confidence_collapsed"] = confidence_collapsed
            iteration_record["evidence_override_min_margin"] = float(evidence_override_min_margin)
            iteration_record["candidate_override_guarded_rollback_blocked"] = guarded_rollback_blocked
            if answer_changed and not confidence_collapsed and not guarded_rollback_blocked:
                override_protected_memory = list(chronological_memory)
                stop_reason = "strong_candidate_override_answer_changed_keep_k1"
                break
            if guarded_rollback_blocked:
                stop_reason = "candidate_override_guarded_rollback_blocked"
                break
            if confidence_collapsed:
                stop_reason = "strong_candidate_override_confidence_collapsed"
                break
            stop_reason = "strong_candidate_override_no_answer_change"
            break

        if iteration_index > 0 and gain is not None:
            if gain < -negative_gain_tolerance:
                stop_reason = "retrieval_harmed_sufficiency"
                break
            if 0.0 <= gain < min_evidence_gain:
                stop_reason = "low_marginal_gain"
                break
        if enable_conservative_gate:
            if not bool(conservative_gate.get("retrieve")):
                stop_reason = str(conservative_gate.get("reason", "conservative_gate_stop"))
                break
            if iteration_index >= max_memory_frames:
                stop_reason = "max_memory_frames_reached"
                break
            if not unused_candidates:
                stop_reason = "candidate_queue_exhausted"
                break
        elif enable_heg:
            if not low_sufficiency_trigger and not historical_gain_trigger:
                stop_reason = "sufficient_no_historical_advantage"
                break
            if iteration_index >= max_memory_frames:
                stop_reason = "max_memory_frames_reached"
                break
            if not unused_candidates:
                stop_reason = "candidate_queue_exhausted"
                break
        elif enable_evidence_override and iteration_index == 0:
            if strong_candidate_override:
                override_triggered = True
                override_k0_prediction = str(score["predicted_option"])
                override_k0_sufficiency = current_sufficiency
            elif current_sufficiency >= sufficiency_threshold:
                stop_reason = "sufficient_evidence"
                break
        elif enable_p3_low_suff_disagree and iteration_index == 0:
            if low_sufficiency_trigger and strong_candidate_disagreement_override:
                override_triggered = True
                override_k0_prediction = str(score["predicted_option"])
                override_k0_sufficiency = current_sufficiency
            else:
                stop_reason = "p3_no_low_suff_strong_disagreement"
                break
        elif candidate_override_like and iteration_index == 0:
            if (
                (enable_candidate_override_protected_rollback or enable_candidate_override_guarded_rollback)
                and strong_candidate_disagreement_override
            ):
                override_triggered = True
                override_k0_prediction = str(score["predicted_option"])
                override_k0_sufficiency = current_sufficiency
            if not low_sufficiency_trigger and not strong_candidate_disagreement_override:
                stop_reason = "sufficient_evidence"
                break
        elif current_sufficiency >= sufficiency_threshold:
            stop_reason = "sufficient_evidence"
            break
        if iteration_index >= max_memory_frames:
            stop_reason = "max_memory_frames_reached"
            break
        previous_sufficiency = current_sufficiency

    if override_protected_memory is not None:
        best_memory = list(override_protected_memory)
        best_sufficiency = float(iterations[-1].get("sufficiency", best_sufficiency))
    best_memory = sorted(best_memory, key=_chunk_sort_key)
    memory_ids = [int(chunk.chunk_index) for chunk in best_memory]
    final_chunks = [*best_memory, *recent_chunks]
    final_ids = [int(chunk.chunk_index) for chunk in final_chunks]
    metadata = {
        "mode": mode_name,
        "config": {
            "recent_window": recent_window,
            "history_search_chunks": history_search_chunks,
            "history_candidate_pool": candidate_pool,
            "max_memory_frames": max_memory_frames,
            "min_temporal_gap": min_temporal_gap,
            "sufficiency_threshold": sufficiency_threshold,
            "heg_enabled": bool(enable_heg),
            "heg_threshold": float(heg_threshold) if enable_heg else None,
            "conservative_gate_enabled": bool(enable_conservative_gate),
            "conservative_tau_low": float(conservative_tau_low) if enable_conservative_gate else None,
            "conservative_tau_high": float(conservative_tau_high) if enable_conservative_gate else None,
            "conservative_candidate_threshold": (
                float(conservative_candidate_threshold) if enable_conservative_gate else None
            ),
            "conservative_temporal_distance_seconds": (
                float(conservative_temporal_distance_seconds) if enable_conservative_gate else None
            ),
            "conservative_temporal_distance_source": (
                conservative_temporal_distance_source if enable_conservative_gate else None
            ),
            "min_evidence_gain": min_evidence_gain,
            "negative_gain_tolerance": negative_gain_tolerance,
            "margin_weight": margin_weight,
            "entropy_weight": entropy_weight,
            "visual_support_weight": visual_support_weight,
            "visual_support_raw_low": 0.15,
            "visual_support_raw_span": 0.30,
            "retrieval_variant": retrieval_variant,
            "mmr_lambda": _env_float("MINICPM_PSM_MMR_LAMBDA", 0.80)
            if retrieval_variant == "clip_mmr"
            else None,
            "evidence_override_enabled": bool(enable_evidence_override),
            "evidence_override_gamma": float(evidence_override_gamma) if enable_evidence_override else None,
            "evidence_override_min_margin": (
                float(evidence_override_min_margin)
                if (
                    enable_evidence_override
                    or enable_candidate_override_protected_rollback
                    or enable_candidate_override_guarded_rollback
                    or enable_p3_low_suff_disagree
                )
                else None
            ),
            "candidate_override_enabled": bool(candidate_override_like),
            "candidate_override_protected_rollback_enabled": bool(enable_candidate_override_protected_rollback),
            "candidate_override_guarded_rollback_enabled": bool(enable_candidate_override_guarded_rollback),
            "p3_low_suff_disagree_enabled": bool(enable_p3_low_suff_disagree),
            "clip_override_threshold": float(clip_override_threshold) if candidate_override_like else None,
        },
        "recent_chunk_ids": recent_ids,
        "recent_start_time_seconds": recent_start_time,
        "recent_end_time_seconds": recent_end_time,
        "baseline_recent_equivalence": {
            "enabled": recent_chunk_ids is not None,
            "source": "select_recent_window_frames",
            "prompt_equal_to_final": not bool(memory_ids),
            "final_equals_recent": final_ids == recent_ids,
            "downsample_mode": recent_downsample_mode,
            "baseline_recent": baseline_recent_metadata or {},
        },
        "history_search_start": int(older_chunks[0].chunk_index) if older_chunks else None,
        "history_search_end": int(older_chunks[-1].chunk_index) if older_chunks else None,
        "history_candidate_start_time_seconds": [
            float(_chunk_temporal_bounds(chunk)[0]) for chunk in older_chunks
        ],
        "history_candidate_end_time_seconds": [
            float(_chunk_temporal_bounds(chunk)[1]) for chunk in older_chunks
        ],
        "history_temporal_violation_count": len(temporal_violations),
        "candidate_queue": [_candidate_metadata(candidate) for candidate in candidate_queue],
        "iterations": iterations,
        "memory_triggered": bool(memory_ids),
        "memory_chunk_ids": memory_ids,
        "num_memory_frames": len(memory_ids),
        "final_selected_chunk_ids": final_ids,
        "final_sufficiency": float(best_sufficiency),
        "stop_reason": stop_reason,
        "history_candidate_ranking_ms": float(ranking_ms),
        "sufficiency_iterations_ms": iteration_times,
        "total_sufficiency_ms": float(sum(iteration_times)),
        "final_generation_ms": None,
        "num_sufficiency_iterations": len(iterations),
        "num_extra_frames": len(memory_ids),
        "evidence_override_triggered": bool(override_triggered),
        "evidence_override_k0_prediction": override_k0_prediction,
        "evidence_override_k0_sufficiency": override_k0_sufficiency,
        "answer_changed_after_strong_candidate": bool(override_protected_memory is not None),
        "candidate_override_enabled": bool(candidate_override_like),
        "candidate_override_protected_rollback_enabled": bool(enable_candidate_override_protected_rollback),
        "candidate_override_guarded_rollback_enabled": bool(enable_candidate_override_guarded_rollback),
        "p3_low_suff_disagree_enabled": bool(enable_p3_low_suff_disagree),
    }
    _validate_metadata(metadata)
    _print_trace(metadata)
    answer_prompt = f"{_PSM_HISTORY_INSTRUCTION}{prompt}" if memory_ids else prompt
    memory_frames = [frame for chunk in best_memory for frame in chunk.frames]
    return ProgressiveSufficiencySelection(
        frames=[*memory_frames, *recent_frames],
        final_chunk_ids=final_ids,
        metadata=metadata,
        answer_prompt=answer_prompt,
        downsample_mode=recent_downsample_mode,
    )


def _validate_metadata(metadata: dict[str, Any]) -> None:
    recent_ids = [int(value) for value in metadata["recent_chunk_ids"]]
    memory_ids = [int(value) for value in metadata["memory_chunk_ids"]]
    final_ids = [int(value) for value in metadata["final_selected_chunk_ids"]]
    assert bool(metadata["memory_triggered"]) == bool(memory_ids)
    assert len(memory_ids) <= 3
    assert len(memory_ids) == len(set(memory_ids))
    assert not (set(recent_ids) & set(memory_ids))
    assert final_ids == [*memory_ids, *recent_ids]
    assert len(final_ids) == len(set(final_ids))
    recent_start = metadata.get("recent_start_time_seconds")
    for candidate in metadata.get("candidate_queue", []):
        if (
            isinstance(candidate, dict)
            and isinstance(candidate.get("end_time_seconds"), (int, float))
            and isinstance(recent_start, (int, float))
        ):
            assert float(candidate["end_time_seconds"]) < float(recent_start), (
                "PRISM candidate is not strictly historical: "
                f"chunk_id={candidate.get('chunk_id')} end={candidate.get('end_time_seconds')} "
                f"recent_start={recent_start}"
            )
    if metadata.get("mode") == "progressive_sufficiency_memory_heg":
        for item in metadata.get("iterations", []):
            unused = {int(value) for value in item.get("unused_historical_candidate_ids", [])}
            context = {int(value) for value in item.get("context_chunk_ids", [])}
            assert not (unused & set(recent_ids))
            assert not (unused & context)
            gains = item.get("evidence_gain_by_option", {})
            current = item.get("current_support_by_option", {})
            historical = item.get("best_historical_support_by_option", {})
            assert set(gains) == set(current) == set(historical)
