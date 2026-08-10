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


@dataclass
class ProgressiveSufficiencySelection:
    frames: list[Image.Image]
    final_chunk_ids: list[int]
    metadata: dict[str, Any]
    answer_prompt: str


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


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
    return metadata


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


def select_progressive_sufficiency_memory(
    qa: RecentWindowQAModel,
    chunks: list[Any],
    prompt: str,
    config: Any,
) -> ProgressiveSufficiencySelection:
    recent_window = 6
    history_search_chunks = _env_int("MINICPM_PSM_HISTORY_SEARCH_CHUNKS", 64)
    candidate_pool = _env_int("MINICPM_PSM_HISTORY_CANDIDATE_POOL", 12)
    max_memory_frames = min(3, _env_int("MINICPM_PSM_MAX_MEMORY_FRAMES", 3))
    min_temporal_gap = _env_int("MINICPM_PSM_MIN_TEMPORAL_GAP", 2)
    sufficiency_threshold = _env_float("MINICPM_PSM_SUFFICIENCY_THRESHOLD", 0.62)
    min_evidence_gain = _env_float("MINICPM_PSM_MIN_EVIDENCE_GAIN", 0.035)
    negative_gain_tolerance = _env_float("MINICPM_PSM_NEGATIVE_GAIN_TOLERANCE", 0.02)
    margin_weight = _env_float("MINICPM_PSM_MARGIN_WEIGHT", 0.50)
    entropy_weight = _env_float("MINICPM_PSM_ENTROPY_WEIGHT", 0.20)
    visual_support_weight = _env_float("MINICPM_PSM_VISUAL_SUPPORT_WEIGHT", 0.30)

    recent_chunks = list(chunks[-recent_window:])
    all_older_chunks = list(chunks[: max(0, len(chunks) - recent_window)])
    older_chunks = all_older_chunks[-history_search_chunks:] if history_search_chunks > 0 else all_older_chunks
    recent_ids = [int(chunk.chunk_index) for chunk in recent_chunks]
    options = _extract_mcq_options(prompt)

    if not options:
        final_ids = list(recent_ids)
        metadata = {
            "mode": "progressive_sufficiency_memory",
            "recent_chunk_ids": recent_ids,
            "history_search_start": int(older_chunks[0].chunk_index) if older_chunks else None,
            "history_search_end": int(older_chunks[-1].chunk_index) if older_chunks else None,
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
            frames=[frame for chunk in recent_chunks for frame in chunk.frames],
            final_chunk_ids=final_ids,
            metadata=metadata,
            answer_prompt=prompt,
        )

    candidate_queue, ranking_ms = _rank_candidates(
        qa,
        older_chunks,
        prompt,
        config,
        candidate_pool=max(0, candidate_pool),
        min_temporal_gap=max(1, min_temporal_gap),
    )
    scorer = _get_clip_scorer(qa)
    selected_memory: list[Any] = []
    best_memory: list[Any] = []
    iterations: list[dict[str, Any]] = []
    iteration_times: list[float] = []
    best_sufficiency = -float("inf")
    previous_sufficiency: float | None = None
    stop_reason = "candidate_queue_exhausted"

    for iteration_index in range(max_memory_frames + 1):
        added_chunk_id = None
        if iteration_index > 0:
            if iteration_index - 1 >= len(candidate_queue):
                stop_reason = "candidate_queue_exhausted"
                break
            candidate = candidate_queue[iteration_index - 1]
            selected_memory.append(candidate["chunk"])
            added_chunk_id = int(candidate["chunk_id"])

        chronological_memory = sorted(selected_memory, key=lambda chunk: int(chunk.chunk_index))
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
        iterations.append(
            {
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
            }
        )

        if iteration_index > 0 and gain is not None:
            if gain < -negative_gain_tolerance:
                stop_reason = "retrieval_harmed_sufficiency"
                break
            if 0.0 <= gain < min_evidence_gain:
                stop_reason = "low_marginal_gain"
                break
        if current_sufficiency >= sufficiency_threshold:
            stop_reason = "sufficient_evidence"
            break
        if iteration_index >= max_memory_frames:
            stop_reason = "max_memory_frames_reached"
            break
        previous_sufficiency = current_sufficiency

    best_memory = sorted(best_memory, key=lambda chunk: int(chunk.chunk_index))
    memory_ids = [int(chunk.chunk_index) for chunk in best_memory]
    final_chunks = [*best_memory, *recent_chunks]
    final_ids = [int(chunk.chunk_index) for chunk in final_chunks]
    metadata = {
        "mode": "progressive_sufficiency_memory",
        "config": {
            "recent_window": recent_window,
            "history_search_chunks": history_search_chunks,
            "history_candidate_pool": candidate_pool,
            "max_memory_frames": max_memory_frames,
            "min_temporal_gap": min_temporal_gap,
            "sufficiency_threshold": sufficiency_threshold,
            "min_evidence_gain": min_evidence_gain,
            "negative_gain_tolerance": negative_gain_tolerance,
            "margin_weight": margin_weight,
            "entropy_weight": entropy_weight,
            "visual_support_weight": visual_support_weight,
            "visual_support_raw_low": 0.15,
            "visual_support_raw_span": 0.30,
        },
        "recent_chunk_ids": recent_ids,
        "history_search_start": int(older_chunks[0].chunk_index) if older_chunks else None,
        "history_search_end": int(older_chunks[-1].chunk_index) if older_chunks else None,
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
    }
    _validate_metadata(metadata)
    _print_trace(metadata)
    answer_prompt = f"{_PSM_HISTORY_INSTRUCTION}{prompt}" if memory_ids else prompt
    return ProgressiveSufficiencySelection(
        frames=[frame for chunk in final_chunks for frame in chunk.frames],
        final_chunk_ids=final_ids,
        metadata=metadata,
        answer_prompt=answer_prompt,
    )


def _validate_metadata(metadata: dict[str, Any]) -> None:
    recent_ids = [int(value) for value in metadata["recent_chunk_ids"]]
    memory_ids = [int(value) for value in metadata["memory_chunk_ids"]]
    final_ids = [int(value) for value in metadata["final_selected_chunk_ids"]]
    assert bool(metadata["memory_triggered"]) == bool(memory_ids)
    assert len(memory_ids) <= 3
    assert not (set(recent_ids) & set(memory_ids))
    assert final_ids == [*sorted(memory_ids), *recent_ids]
    assert len(final_ids) == len(set(final_ids))
