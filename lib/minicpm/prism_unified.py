"""Format-agnostic PRISM: one controller for multiple-choice and open-ended questions.

Same structure as progressive_arbitration (exact Recent-6, temporal pre-filter, CLIP-MMR, admission,
progressive arbitration up to B_mem = 3, prefix rollback), but no signal depends on answer options:

  retrieval     r_i = cos(E_v(h_i), E_t(q)) on the question text alone, then MMR.
  state K_j     one greedy generation; the answer a_j is the generated text.
  M_j           mean top-1 minus top-2 token probability over the answer's content tokens.
  E_j           1 - mean normalized top-k token entropy over the same tokens.
  V_j           normalized max_f cos(E_v(f), E_t(q + " Answer: " + a_j)), as before.
  S_j           alpha M_j + beta E_j + gamma V_j.
  admission     S_0 < tau_s, or r_1 >= tau_r and h_1 supports a_0 less than the recent frames do.
  answer change normalized text differs; answers of two or more words must also fall below a CLIP
                text-similarity threshold, so paraphrases do not count as changes.
  consistency   a close candidate (< d_close) that changed the answer must support the new answer at
                least as much as the old one.

K_0 uses the same inputs and greedy decoding as the Recent-6 baseline, so with no accepted memory the
answer is identical to Recent-6. The answer of the deepest accepted state is the final answer; no
separate final generation is run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
import os
import re
import time

from lib.minicpm.progressive_arbitration import (
    CONFIG as BASE, DEFINITION, SYSTEM_DEFINITION, ClipCache, image_key, initial_stats, prefilter, sync, timed,
)

MODE = "prism_unified_format_agnostic_exact_recent"


@dataclass(frozen=True)
class Config:
    recent: int = BASE.recent
    budget: int = BASE.budget
    horizon: int = BASE.horizon
    pool: int = BASE.pool
    alpha: float = BASE.alpha
    beta: float = BASE.beta
    gamma: float = BASE.gamma
    tau_s: float = BASE.tau_s
    tau_r: float = BASE.tau_r
    tau_m: float = BASE.tau_m
    delta_s: float = BASE.delta_s
    mmr_lambda: float = BASE.mmr_lambda
    d_min: float = BASE.d_min
    d_max: float = BASE.d_max
    d_close: float = BASE.d_close
    gain: float = BASE.gain
    entropy_topk: int = 10
    change_similarity: float = 0.92
    disagree_margin: float = 0.0


def config_from_env():
    """Every field can be overridden as PRISM_U_<FIELD> for threshold calibration."""
    values = {}
    for f in fields(Config):
        raw = os.environ.get('PRISM_U_' + f.name.upper())
        if raw is not None:
            values[f.name] = type(f.default)(raw)
    return Config(**values)


def question_text(prompt):
    """The question alone: text after the last "Question:", before options or answer instructions."""
    starts = list(re.finditer(r'Question:\s*', prompt, flags=re.IGNORECASE))
    text = prompt[starts[-1].end():] if starts else prompt
    text = re.split(r'\n\s*(?:Options:|Only give|Answer with|Respond with|Give a concise)', text, maxsplit=1,
                    flags=re.IGNORECASE)[0]
    return text.strip() or prompt.strip()


def normalize_answer(text):
    text = re.sub(r'^\s*(?:the\s+)?answer\s*(?:is)?\s*[:\-]?\s*', '', str(text), flags=re.IGNORECASE)
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return ' '.join(w for w in text.split() if w not in {'a', 'an', 'the'})


def token_confidence(scores, token_ids, special_ids, topk):
    """Mean top-2 margin and mean normalized top-k entropy confidence over content tokens."""
    import torch
    margins, confidences = [], []
    for step, token in zip(scores, token_ids):
        if int(token) in special_ids:
            continue
        probs = torch.softmax(step[0].float(), dim=-1)
        top = torch.topk(probs, k=max(2, topk)).values
        margins.append(float(top[0] - top[1]))
        local = top / top.sum()
        entropy = -float((local * torch.log(local.clamp_min(1e-12))).sum())
        confidences.append(1.0 - entropy / math.log(len(local)))
    if not margins:
        return 0.0, 0.0, 0
    return sum(margins) / len(margins), sum(confidences) / len(confidences), len(margins)


def generate_state(qa, frames, prompt, config):
    """Greedy answer plus per-token scores; identical inputs and decoding to the Recent-6 baseline."""
    from lib.minicpm.baseline import _GeneratedTokenTTFTStreamer, _synchronize_gpu_devices
    from lib.minicpm.progressive_sufficiency import _build_minicpm_inputs
    preprocess_start = time.perf_counter()
    inputs = _build_minicpm_inputs(qa, frames, prompt)
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000
    prompt_length = int(inputs['input_ids'].shape[1])
    tokenizer = qa.processor.tokenizer
    pad = getattr(getattr(qa.model, 'generation_config', None), 'pad_token_id', None)
    pad = pad if pad is not None else (tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                                       else tokenizer.eos_token_id)
    start = time.perf_counter()
    streamer = _GeneratedTokenTTFTStreamer(start, prompt_length=prompt_length)
    with qa._profile_model_generate_components():
        out = qa.model.generate(**inputs, downsample_mode=qa.downsample_mode, max_new_tokens=qa.max_new_tokens,
                                do_sample=False, pad_token_id=pad, streamer=streamer,
                                output_scores=True, return_dict_in_generate=True)
    _synchronize_gpu_devices()
    generate_ms = (time.perf_counter() - start) * 1000
    sequence = out.sequences[0]
    if sequence.numel() > prompt_length:
        sequence = sequence[prompt_length:]
    answer = qa.processor.batch_decode([sequence], skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)[0].strip()
    margin, confidence, content_tokens = token_confidence(out.scores, sequence[:len(out.scores)].tolist(),
                                                          set(tokenizer.all_special_ids), config.entropy_topk)
    ttft_ms = (streamer.ttft_seconds * 1000) if streamer.ttft_seconds is not None else generate_ms
    qa._last_preprocess_seconds = preprocess_ms / 1000
    qa._last_model_generate_seconds = generate_ms / 1000
    qa._last_ttft_seconds = ttft_ms / 1000
    qa._last_generated_tokens = int(sequence.numel())
    qa._last_num_vision_frames = len(frames)
    qa._last_num_vision_tokens = qa._estimate_vision_tokens(inputs)
    return dict(answer=answer, answer_margin=margin, entropy_confidence=confidence, content_tokens=content_tokens,
                generated_tokens=int(sequence.numel()), generate_ms=generate_ms, ttft_ms=ttft_ms,
                preprocess_ms=preprocess_ms, num_vision_tokens=qa._last_num_vision_tokens)


def score_state(qa, frames, prompt, question, clip, stats, depth, config, generator=generate_state):
    from lib.minicpm.progressive_sufficiency import _normalize_clip_support
    key = f'k{depth}'
    with timed(stats, key + '_option_forward_ms'):  # Key kept for the shared latency summaries.
        state = generator(qa, frames, prompt, config)
    stats[key + '_model_forward_only_ms'] = state['generate_ms']
    stats[key + '_forward_calls'] = 1
    with timed(stats, key + '_visual_support_ms'):
        images = clip.image_vectors(frames, {}, prefix='support')
        text = clip.text_vectors([f"{question} Answer: {state['answer']}"])[0]
        state['support_raw'] = float((images @ text).max().item())
    with timed(stats, key + '_sufficiency_compute_ms'):
        state['visual_support_norm'] = _normalize_clip_support(state['support_raw'])
        state['sufficiency'] = (config.alpha * state['answer_margin'] + config.beta * state['entropy_confidence']
                                + config.gamma * state['visual_support_norm'])
        if not all(math.isfinite(float(state[k])) for k in ('sufficiency', 'answer_margin', 'entropy_confidence')):
            raise ValueError('Non-finite sufficiency state')
    stats[key + '_total_ms'] = sum(stats[key + s] for s in ('_option_forward_ms', '_visual_support_ms',
                                                           '_sufficiency_compute_ms'))
    return state


def support(clip, candidate, question, answer):
    """CLIP support of one historical frame for "question + answer"."""
    text = clip.text_vectors([f"{question} Answer: {answer}"])[0]
    return float(candidate['image_vector'] @ text)


def rank(eligible, question, clip, stats, config):
    if not eligible:
        return []
    images = clip.image_vectors([c['frame'] for c in eligible], stats)
    with timed(stats, 'clip_text_encode_ms'):
        text = clip.text_vectors([question])[0]
    with timed(stats, 'clip_relevance_ms'):
        relevance = (images @ text).detach().cpu().tolist()
        for i, (item, value) in enumerate(zip(eligible, relevance)):
            item.update(index=i, relevance=float(value), image_vector=images[i])
    with timed(stats, 'mmr_ranking_ms'):
        similarities = (images @ images.T).detach().cpu().tolist()
        selected, remaining = [], list(eligible)
        while remaining and len(selected) < config.pool:
            valid = [c for c in remaining if all(c['end'] < s['start'] or s['end'] < c['start'] for s in selected)]
            if not valid:
                break
            for c in valid:
                redundancy = max((similarities[c['index']][s['index']] for s in selected), default=0.)
                c['mmr_score'] = config.mmr_lambda * c['relevance'] - (1 - config.mmr_lambda) * redundancy
            best = max(valid, key=lambda c: (c['mmr_score'], -c['source_id']))
            selected.append(best)
            remaining.remove(best)
    stats.update(num_candidates_input=len(eligible), num_candidates_output=len(selected),
                 candidate_pool_size=config.pool, num_options=0, num_text_embeddings=1)
    return selected


def answers_differ(a, b, clip, config):
    x, y = normalize_answer(a), normalize_answer(b)
    if x == y:
        return False
    if min(len(x.split()), len(y.split())) < 2:
        return True  # Letters, numbers and one-word answers: any textual difference is a change.
    vectors = clip.text_vectors([x, y])
    return float(vectors[0] @ vectors[1]) < config.change_similarity


def arbitrate(previous, current, candidate, depth, question, clip, config):
    change = answers_differ(previous['answer'], current['answer'], clip, config)
    margin = current['answer_margin'] >= config.tau_m
    drop = previous['sufficiency'] - current['sufficiency'] <= config.delta_s
    consistent, new_support, old_support = True, None, None
    if change and candidate['distance'] < config.d_close:
        new_support = support(clip, candidate, question, current['answer'])
        old_support = support(clip, candidate, question, previous['answer'])
        consistent = new_support >= old_support
    if depth == 1:
        progress = change or current['sufficiency'] > previous['sufficiency'] + config.gain
    else:
        progress = (current['sufficiency'] >= previous['sufficiency'] + config.gain
                    or (change and current['answer_margin'] >= previous['answer_margin']))
    return dict(accepted=bool(margin and drop and consistent and progress), answer_change=change,
                margin_safe=margin, sufficiency_safe=drop, temporal_consistency=consistent,
                candidate_support_new=new_support, candidate_support_old=old_support, progress_condition=progress)


def _public(state):
    return {k: v for k, v in state.items() if k not in ('image_vector',)}


def select(qa, chunks, recent, prompt, clip, stats, config, scorer=score_state):
    recent_count = len(recent.frames)
    if not 1 <= recent_count <= config.recent:
        raise ValueError(f'Recent context must contain 1..{config.recent} available frames, got {recent_count}')
    if len({f.size for f in recent.frames}) != 1:
        raise ValueError('Recent frame geometry differs; refusing to alter the exact baseline backbone')
    original_hashes = [image_key(f) for f in recent.frames]
    question = question_text(prompt)
    eligible = prefilter(chunks, recent, stats)
    previous = scorer(qa, recent.frames, prompt, question, clip, stats, 0, config)
    final_state = previous
    iterations = [dict(depth=0, **previous)]
    candidates = rank(eligible, question, clip, stats, config)
    with timed(stats, 'candidate_admission_ms'):
        disagree = bool(candidates) and (support(clip, candidates[0], question, previous['answer'])
                                         < previous['support_raw'] - config.disagree_margin)
        admitted = bool(candidates and (previous['sufficiency'] < config.tau_s
                                        or (candidates[0]['relevance'] >= config.tau_r and disagree)))
    stop = 'not_admitted' if not admitted else 'budget_or_queue_exhausted'
    accepted = []
    for depth, candidate in enumerate(candidates[:config.budget] if admitted else [], 1):
        with timed(stats, 'memory_frame_normalization_ms'):
            frame = candidate['frame']
            if frame.size != recent.frames[0].size:
                frame = frame.resize(recent.frames[0].size)
            candidate['normalized_frame'] = frame
        with timed(stats, 'final_context_assembly_ms'):
            trial = sorted([*accepted, candidate], key=lambda c: (c['end'], c['start']))
            frames = [c['normalized_frame'] for c in trial] + list(recent.frames)
        current = scorer(qa, frames, prompt, question, clip, stats, depth, config)
        with timed(stats, f'arbitration_k{depth}_ms'):
            decision = arbitrate(previous, current, candidate, depth, question, clip, config)
            if decision['accepted']:
                accepted, previous, final_state = trial, current, current
        iterations.append(dict(depth=depth, **current, arbitration=decision,
                               candidate_hash=candidate['hash'], candidate_distance=candidate['distance']))
        if not decision['accepted']:
            stop = 'arbitration_rejected'
            break
    frames = [c['normalized_frame'] for c in accepted] + list(recent.frames)
    assert [image_key(f) for f in frames[-recent_count:]] == original_hashes
    assert len(accepted) <= config.budget and len(frames) <= config.recent + config.budget
    stats.update(selected_memory_depth=len(accepted), max_depth_evaluated=max(i['depth'] for i in iterations),
                 num_sufficiency_iterations=len(iterations), final_frame_count=len(frames),
                 num_option_scoring_forwards=len(iterations))
    metadata = dict(config=asdict(config), mode=MODE, stop_reason=stop, iterations=iterations,
                    question_text=question, format_agnostic=True, admission_disagree=disagree,
                    selected_memory_depth=len(accepted), memory_triggered=len(iterations) > 1,
                    memory_accepted=bool(accepted), num_memory_frames=len(accepted),
                    recent_frame_hashes=original_hashes, recent_hashes_unchanged=True, prompt_unchanged=True,
                    recent_frame_count=recent_count, short_recent_context=recent_count < config.recent,
                    arbitration_supported=True, memory_distances=[c['distance'] for c in accepted],
                    recent_chunk_ids=list(recent.final_chunk_ids),
                    memory_source_ids=[c['source_id'] for c in accepted], memory_hashes=[c['hash'] for c in accepted],
                    candidate_queue=[{k: v for k, v in c.items() if k not in ('frame', 'chunk', 'normalized_frame',
                                                                                'image_vector')} for c in candidates],
                    baseline_recent_equivalence=dict(baseline_recent=recent.cdas_metadata))
    return frames, accepted, metadata, final_state


def aggregate_timing(stats):
    """Selection excludes the accepted state's own generation, which is reported as the final generation."""
    stats['retrieval_total_ms'] = sum(stats[k] for k in ('temporal_prefilter_ms', 'clip_image_cache_lookup_ms',
        'clip_image_encode_uncached_ms', 'clip_text_encode_ms', 'clip_relevance_ms', 'mmr_ranking_ms'))
    stats['vlm_selection_total_ms'] = (sum(stats.get(f'k{j}_total_ms') or 0 for j in range(4))
                                       - stats['final_generation_ms'])
    stats['arbitration_total_ms'] = sum(stats.get(f'arbitration_k{j}_ms') or 0 for j in range(1, 4))
    stats['decision_logic_total_ms'] = stats['candidate_admission_ms'] + stats['arbitration_total_ms']
    stats['context_total_ms'] = stats['final_context_assembly_ms'] + stats['memory_frame_normalization_ms']
    stats['PRISM_selection_latency_ms'] = sum(stats[k] for k in
        ('retrieval_total_ms', 'vlm_selection_total_ms', 'decision_logic_total_ms', 'context_total_ms'))
    stats['PRISM_end_to_end_latency_ms'] = stats['PRISM_selection_latency_ms'] + stats['final_generation_ms']
    stats['latency_accounting_residual_ms'] = stats['PRISM_algorithmic_latency_ms'] - stats['PRISM_end_to_end_latency_ms']
    stats['latency_accounting_error_ms'] = abs(stats['latency_accounting_residual_ms'])
    stats['latency_accounting_tolerance_ms'] = max(50., .02 * stats['PRISM_algorithmic_latency_ms'])
    stats['latency_accounting_valid'] = stats['latency_accounting_error_ms'] <= stats['latency_accounting_tolerance_ms']


def query(qa, video_path, prompt, chunk_duration, fps, recent_frames_only, video_start=None, video_end=None,
          cdas_config=None):
    from lib.minicpm import baseline as base
    from lib.shared.recent_window import decode_video_to_chunks_qwen
    from main_experiments.minicpm_v46.streamingbench.eval_prism_exact_recent_dist import select_exact_current_recent_frames
    if chunk_duration != 1.0 or fps != 1.0:
        raise ValueError('Frozen arbitration configuration requires chunk_duration=1 and fps=1')
    config = getattr(qa, '_prism_unified_config', None) or config_from_env()
    qa._prism_unified_config = config
    clip = getattr(qa, '_progressive_arbitration_clip', None)
    if clip is None:
        clip = qa._progressive_arbitration_clip = ClipCache(qa)  # Model loading is outside the timers.
    if not getattr(qa, '_prism_unified_warm', False):
        from PIL import Image
        import torch
        warm = [Image.new('RGB', (224, 224), (i * 20, 40, 60)) for i in range(6)]
        with torch.inference_mode():
            generate_state(qa, warm, 'Question: What color is the image? Give a concise direct answer.', config)
            clip.begin('__warmup__')
            clip.image_vectors(warm, {})
            clip.text_vectors(['warmup'])
        sync()
        qa._prism_unified_warm = True
    sync()
    system_start = time.perf_counter()
    stats = initial_stats()
    saved = os.environ.pop('QWEN_EXACT_RECENT_DECODE', None)
    try:
        with timed(stats, 'broad_history_decode_ms'):
            chunks, backend = decode_video_to_chunks_qwen(video_path, chunk_duration, fps, config.horizon + 6,
                                                          video_start=video_start, video_end=video_end)
        start = max(0., video_end - 6.) if video_end is not None else video_start
        with timed(stats, 'recent6_decode_ms'):
            recent = select_exact_current_recent_frames(qa, video_path, chunk_duration, fps, 6,
                                                        video_start=start, video_end=video_end)
    finally:
        if saved is not None:
            os.environ['QWEN_EXACT_RECENT_DECODE'] = saved
    stats['video_io_ms'] = None
    stats['total_visual_preparation_ms'] = (time.perf_counter() - system_start) * 1000
    before = base._reset_gpu_memory_peaks()
    sync()
    algorithm_start = time.perf_counter()
    with timed(stats, 'clip_image_cache_lookup_ms'):
        clip.begin(os.path.realpath(video_path))
    import torch
    with torch.inference_mode():
        frames, accepted, metadata, final = select(qa, chunks, recent, prompt, clip, stats, config)
    stats['PRISM_algorithmic_latency_ms'] = (time.perf_counter() - algorithm_start) * 1000
    stats['full_system_latency_ms'] = (time.perf_counter() - system_start) * 1000
    stats['final_wrapper_ms'] = 0.0  # The accepted state's answer is final; no extra generation.
    stats['final_generation_ms'] = final['generate_ms']
    stats['final_preprocess_ms'] = final['preprocess_ms']
    stats['final_ttft_ms'] = final['ttft_ms']
    stats['final_decode_after_first_token_ms'] = max(0., final['generate_ms'] - final['ttft_ms'])
    stats['final_num_generated_tokens'] = final['generated_tokens']
    stats['num_vision_tokens'] = final['num_vision_tokens']
    stats['request_to_answer_first_token_ms'] = stats['PRISM_algorithmic_latency_ms']  # Released after arbitration.
    aggregate_timing(stats)
    metadata.update(timing=stats, latency_definition=DEFINITION, full_system_definition=SYSTEM_DEFINITION,
                    warmup_enabled=True, baseline_control=False, final_answer_depth=len(accepted))
    profile = base._build_profile(mode=MODE, decode_time=stats['total_visual_preparation_ms'] / 1000,
                                  selection_time=stats['PRISM_selection_latency_ms'] / 1000,
                                  generate_time=stats['final_generation_ms'] / 1000, before_memory=before,
                                  after_memory=base._capture_gpu_memory(), qa=qa)
    profile['adaptive'] = metadata
    stats['gpu_peak_allocated_mb'] = profile['gpu_peak_allocated_mb']
    stats['gpu_peak_reserved_mb'] = profile['gpu_peak_reserved_mb']
    profile['progressive_arbitration'] = stats
    profile['end_to_end_time_seconds'] = stats['full_system_latency_ms'] / 1000
    ids = [-j - 1 for j in range(len(accepted))] + list(recent.final_chunk_ids)
    metadata['final_selected_chunk_ids'] = ids
    metadata['memory_chunk_ids'] = ids[:len(accepted)]
    result = base.RecentWindowResult(answer=final['answer'], final_chunk_ids=ids,
                                     generate_time=stats['final_generation_ms'] / 1000,
                                     ttft_seconds=stats['final_ttft_ms'] / 1000,
                                     num_vision_tokens=final['num_vision_tokens'],
                                     num_vision_tokens_before=final['num_vision_tokens'],
                                     num_vision_tokens_after=final['num_vision_tokens'], num_frames=len(frames))
    result.profile_metadata = profile
    result.adaptive_metadata = metadata
    return result, backend
