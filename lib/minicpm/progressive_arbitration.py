"""Frozen, opt-in progressive evidence arbitration. Existing PSM modes are unchanged.

G = [S0 < tau_s or (r1 >= tau_r and y_h1 != y0)].
A_j = margin_safe * drop_safe * temporal_consistent * progress_j.
Accepted states are a prefix: reject j -> return j-1, never erase that prefix.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import math
import os
import time
from types import SimpleNamespace

MODE = "progressive_sufficiency_memory_clip_mmr_progressive_arbitration_exact_recent"
BASELINE_MODE = "progressive_arbitration_exact_recent6_control"
DEFINITION = ("PRISM algorithmic latency excludes raw video I/O and video/frame decoding. "
              "Timing begins once exact Recent-6 and historical candidate frames with temporal metadata are available.")
SYSTEM_DEFINITION = "Full-system latency is reported separately and includes visual preparation/decode where measurable."


@dataclass(frozen=True)
class Config:
    recent: int = 6
    budget: int = 3
    horizon: int = 64
    pool: int = 12
    alpha: float = 0.50
    beta: float = 0.20
    gamma: float = 0.30
    tau_s: float = 0.62
    tau_r: float = 0.2995
    tau_m: float = 0.60
    delta_s: float = 0.08
    mmr_lambda: float = 0.80
    d_min: float = 3.0
    d_max: float = 30.0
    d_close: float = 10.0
    gain: float = 0.035
    clip_cache_size: int = 512


CONFIG = Config()


def sync():
    from lib.minicpm.baseline import _synchronize_gpu_devices
    _synchronize_gpu_devices()


@contextmanager
def timed(stats, key):
    sync()
    start = time.perf_counter()
    try:
        yield
    finally:
        sync()
        stats[key] = (stats.get(key) or 0.0) + (time.perf_counter() - start) * 1000


def image_key(image):
    digest = hashlib.sha256()
    digest.update(str((image.mode, image.size)).encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def arbitrate(previous, current, candidate, depth, config=CONFIG):
    change = current['predicted_option'] != previous['predicted_option']
    margin = current['answer_margin'] >= config.tau_m
    drop = previous['sufficiency'] - current['sufficiency'] <= config.delta_s
    supported = candidate.get('best_supported_option')
    meaningful = supported is not None
    consistent = not (meaningful and candidate['distance'] < config.d_close
                      and current['predicted_option'] != supported)
    if depth == 1:
        progress = change or current['sufficiency'] > previous['sufficiency'] + config.gain
    else:
        progress = (current['sufficiency'] >= previous['sufficiency'] + config.gain
                    or (change and current['answer_margin'] >= previous['answer_margin']))
    return dict(accepted=bool(margin and drop and consistent and progress),
                answer_change=change, margin_safe=margin, sufficiency_safe=drop,
                temporal_consistency=consistent, consistency_meaningful=meaningful,
                progress_condition=progress)


class ClipCache:
    """Bounded, per-video image cache. Hash includes geometry; no option scores cached."""
    def __init__(self, qa):
        from lib.minicpm.progressive_sufficiency import _get_clip_scorer
        self.scorer = _get_clip_scorer(qa)
        self.video = None
        self.images = OrderedDict()
        self.texts = OrderedDict()

    def begin(self, video):
        if video != self.video:
            self.images.clear()
            self.texts.clear()
            self.video = video

    def image_vectors(self, frames, stats, prefix='clip'):
        import torch
        from lib.minicpm.prism_retrieval_variants import _as_feature_tensor
        with timed(stats, prefix + '_image_cache_lookup_ms'):
            keys = [image_key(f) for f in frames]
            unique = dict(zip(keys, frames))
            missing = [k for k in unique if k not in self.images]
            stats[prefix + '_cache_hits'] = stats.get(prefix + '_cache_hits', 0) + sum(k in self.images for k in keys)
            stats[prefix + '_cache_misses'] = stats.get(prefix + '_cache_misses', 0) + len(missing)
        with timed(stats, prefix + '_image_encode_uncached_ms'):
            if missing:
                with torch.inference_mode():
                    inp = self.scorer.processor(images=[unique[k] for k in missing], return_tensors='pt')
                    inp = {k:v.to(self.scorer.device) for k,v in inp.items()}
                    vec = _as_feature_tensor(self.scorer.model.get_image_features(**inp)).float()
                    vec = torch.nn.functional.normalize(vec, dim=-1)
                for key, value in zip(missing, vec):
                    self.images[key] = value.detach()
            result = torch.stack([self.images[k] for k in keys])
            for k in keys:
                self.images.move_to_end(k)
            while len(self.images) > CONFIG.clip_cache_size:
                self.images.popitem(last=False)
        stats[prefix + '_image_encode_ms'] = (stats[prefix + '_image_cache_lookup_ms']
                                               + stats[prefix + '_image_encode_uncached_ms'])
        stats['num_images_encoded'] = stats.get('num_images_encoded', 0) + len(missing)
        return result

    def text_vectors(self, texts):
        import torch
        from lib.minicpm.prism_retrieval_variants import _as_feature_tensor
        missing = list(dict.fromkeys(t for t in texts if t not in self.texts))
        if missing:
            with torch.inference_mode():
                inp = self.scorer.processor(text=missing, padding=True, truncation=True, return_tensors='pt')
                inp = {k:v.to(self.scorer.device) for k,v in inp.items()}
                vec = _as_feature_tensor(self.scorer.model.get_text_features(**inp)).float()
                vec = torch.nn.functional.normalize(vec, dim=-1)
            self.texts.update(zip(missing, vec.detach()))
        result = torch.stack([self.texts[t] for t in texts])
        while len(self.texts) > CONFIG.clip_cache_size:
            self.texts.popitem(last=False)
        return result


def prefilter(chunks, recent, stats):
    from lib.minicpm.progressive_sufficiency import _chunk_temporal_bounds
    with timed(stats, 'temporal_prefilter_ms'):
        start = min(_chunk_temporal_bounds(c)[0] for c in recent.selected_chunks)
        eligible = []
        seen = {image_key(f) for f in recent.frames}
        seen_ids = set()
        for chunk in chunks:
            low, high = _chunk_temporal_bounds(chunk)
            if not (math.isfinite(low) and math.isfinite(high)):
                raise ValueError('Historical timestamps must be finite')
            distance = start - high
            if high < start and CONFIG.d_min <= distance <= CONFIG.d_max:
                # A candidate is exactly one historical frame, never an entire multi-frame chunk.
                from lib.minicpm.prism_retrieval_variants import representative_frame
                frame = representative_frame(chunk)
                key = image_key(frame)
                if key in seen or int(chunk.chunk_index) in seen_ids:
                    continue
                seen.add(key)
                seen_ids.add(int(chunk.chunk_index))
                eligible.append(dict(chunk=chunk, frame=frame, hash=key, start=low, end=high,
                                     distance=distance, source_id=int(chunk.chunk_index)))
        eligible.sort(key=lambda c:(c['end'], c['start'], c['source_id']))
        after_band = len(eligible)
        eligible = eligible[-CONFIG.horizon:]
    stats.update(history_before_prefilter_count=len(chunks), history_after_prefilter_count=len(eligible),
                 history_after_band_count=after_band, eligible_history_count=len(eligible),
                 temporal_prefilter_reduction_ratio=1-len(eligible)/max(len(chunks),1),
                 history_prefilter_ms=stats['temporal_prefilter_ms'])
    return eligible


def rank(eligible, options, prompt, clip, stats):
    from lib.minicpm.progressive_sufficiency import _question_text
    if not eligible:
        return []
    images = clip.image_vectors([c['frame'] for c in eligible], stats)
    with timed(stats, 'clip_text_encode_ms'):
        texts = clip.text_vectors([f"{_question_text(prompt)} {o['text']}" for o in options])
    with timed(stats, 'clip_relevance_ms'):
        scores = (images @ texts.T).detach().cpu().tolist()
        for i, (item, values) in enumerate(zip(eligible, scores)):
            best = max(range(len(values)), key=values.__getitem__)
            item.update(index=i, relevance=values[best], best_supported_option=options[best]['letter'])
    with timed(stats, 'mmr_ranking_ms'):
        similarities = (images @ images.T).detach().cpu().tolist()
        selected = []
        remaining = list(eligible)
        while remaining and len(selected) < CONFIG.pool:
            valid = [c for c in remaining if all(c['end'] < s['start'] or s['end'] < c['start'] for s in selected)]
            if not valid:
                break
            for c in valid:
                redundancy = max((similarities[c['index']][s['index']] for s in selected), default=0.)
                c['mmr_score'] = CONFIG.mmr_lambda*c['relevance'] - (1-CONFIG.mmr_lambda)*redundancy
            best = max(valid, key=lambda c:(c['mmr_score'], -c['source_id']))
            selected.append(best)
            remaining.remove(best)
        assert all(CONFIG.d_min <= c['distance'] <= CONFIG.d_max and c['distance'] > 0 for c in selected)
    stats.update(num_candidates_input=len(eligible), num_candidates_output=len(selected),
                 candidate_pool_size=CONFIG.pool, num_options=len(options), num_text_embeddings=len(options))
    return selected


def score_context(qa, frames, prompt, options, clip, stats, depth):
    import torch
    from lib.minicpm.progressive_sufficiency import _score_options, _question_text, _normalize_clip_support
    key = f'k{depth}'
    calls = [0]
    def hook(*_):
        calls[0] += 1
    handle = qa.model.register_forward_pre_hook(hook)
    try:
        # Includes option input preparation; raw model-forward time is logged separately.
        with timed(stats, key + '_option_forward_ms'), torch.inference_mode():
            state = _score_options(qa, frames, prompt, options)
    finally:
        handle.remove()
    stats[key + '_model_forward_only_ms'] = state['option_forward_ms']
    stats[key + '_forward_calls'] = calls[0]
    with timed(stats, key + '_visual_support_ms'):
        # These CLIP operations are charged to this state, not retrieval a second time.
        local = {}
        images = clip.image_vectors(frames, local, prefix='support')
        texts = clip.text_vectors([f"{_question_text(prompt)} Answer: {state['predicted_answer_text']}"])
        support = float((images @ texts[0]).max().item())
    with timed(stats, key + '_sufficiency_compute_ms'):
        state['visual_support_norm'] = _normalize_clip_support(support)
        state['sufficiency'] = (CONFIG.alpha*state['answer_margin'] + CONFIG.beta*state['entropy_confidence']
                                + CONFIG.gamma*state['visual_support_norm'])
        if not all(math.isfinite(float(state[k])) for k in ('sufficiency','answer_margin','entropy_confidence')):
            raise ValueError('Non-finite sufficiency state')
    stats[key + '_total_ms'] = sum(stats[key+s] for s in ('_option_forward_ms','_visual_support_ms','_sufficiency_compute_ms'))
    state['support_cache'] = local
    return state


def select(qa, chunks, recent, prompt, clip, stats, control=False, scorer=score_context):
    from lib.minicpm.progressive_sufficiency import _extract_mcq_options
    assert len(recent.frames) == CONFIG.recent, 'Exact Recent-6 requires six distinct selected frame records'
    if len({f.size for f in recent.frames}) != 1:
        raise ValueError('Recent frame geometry differs; refusing to alter the exact baseline backbone')
    original_hashes = [image_key(f) for f in recent.frames]
    options = _extract_mcq_options(prompt)
    # No synthesized choices or label-dependent handling for open-ended tasks.
    eligible = prefilter(chunks, recent, stats) if options and not control else []
    accepted, iterations, candidates = [], [], []
    stop = 'baseline_control' if control else 'no_mcq_options_exact_recent_fallback'
    if options and not control:
        previous = scorer(qa, recent.frames, prompt, options, clip, stats, 0)
        iterations.append(dict(depth=0, **previous))
        candidates = rank(eligible, options, prompt, clip, stats)
        with timed(stats, 'candidate_admission_ms'):
            admitted = bool(candidates and (previous['sufficiency'] < CONFIG.tau_s or
                (candidates[0]['relevance'] >= CONFIG.tau_r and candidates[0]['best_supported_option'] != previous['predicted_option'])))
        stop = 'not_admitted' if not admitted else 'budget_or_queue_exhausted'
        for depth, candidate in enumerate(candidates[:CONFIG.budget] if admitted else [], 1):
            with timed(stats, 'memory_frame_normalization_ms'):
                frame = candidate['frame']
                if frame.size != recent.frames[0].size:
                    frame = frame.resize(recent.frames[0].size)
                candidate['normalized_frame'] = frame
            with timed(stats, 'final_context_assembly_ms'):
                trial = sorted([*accepted, candidate], key=lambda c:(c['end'], c['start']))
                frames = [c['normalized_frame'] for c in trial] + list(recent.frames)
            current = scorer(qa, frames, prompt, options, clip, stats, depth)
            with timed(stats, f'arbitration_k{depth}_ms'):
                decision = arbitrate(previous, current, candidate, depth)
                if decision['accepted']:
                    accepted = trial
                    previous = current
            iterations.append(dict(depth=depth, **current, arbitration=decision,
                                   candidate_hash=candidate['hash'], candidate_distance=candidate['distance']))
            if not decision['accepted']:
                stop = 'arbitration_rejected'
                break
    with timed(stats, 'final_context_assembly_ms'):
        frames = [c['normalized_frame'] for c in accepted] + list(recent.frames)
        assert [image_key(f) for f in frames[-6:]] == original_hashes
        assert len(accepted) <= 3 and len(frames) <= 9
        assert len({c['hash'] for c in accepted}) == len(accepted)
        assert all(c['hash'] not in original_hashes for c in accepted)
        assert all(c['distance'] > 0 for c in accepted)
    stats.update(selected_memory_depth=len(accepted), max_depth_evaluated=max((i['depth'] for i in iterations),default=0),
                 num_sufficiency_iterations=len(iterations), final_frame_count=len(frames),
                 num_option_scoring_forwards=sum(stats.get(f'k{j}_forward_calls') or 0 for j in range(4)))
    metadata = dict(config=asdict(CONFIG), mode=BASELINE_MODE if control else MODE,
                    stop_reason=stop, iterations=iterations, selected_memory_depth=len(accepted),
                    memory_triggered=bool(len(iterations)>1), memory_accepted=bool(accepted),
                    num_memory_frames=len(accepted), recent_frame_hashes=original_hashes,
                    recent_hashes_unchanged=True, temporal_violations=0, prompt_unchanged=True,
                    arbitration_supported=bool(options), memory_distances=[c['distance'] for c in accepted],
                    recent_chunk_ids=list(recent.final_chunk_ids),
                    memory_source_ids=[c['source_id'] for c in accepted],
                    memory_hashes=[c['hash'] for c in accepted],
                    candidate_queue=[{k:v for k,v in c.items() if k not in ('frame','chunk','normalized_frame')} for c in candidates],
                    baseline_recent_equivalence=dict(baseline_recent=recent.cdas_metadata))
    return frames, accepted, metadata


def initial_stats():
    stats = {k:0. for k in ('temporal_prefilter_ms','clip_image_cache_lookup_ms','clip_image_encode_uncached_ms',
             'clip_image_encode_ms','clip_text_encode_ms','clip_relevance_ms','mmr_ranking_ms',
             'candidate_admission_ms','final_context_assembly_ms','memory_frame_normalization_ms')}
    stats.update(clip_cache_hits=0, clip_cache_misses=0, num_images_encoded=0, num_options=0,
                 num_candidates_input=0, num_candidates_output=0, num_text_embeddings=0,
                 history_before_prefilter_count=0, history_after_prefilter_count=0)
    for j in range(4):
        for suffix in ('option_forward_ms','visual_support_ms','sufficiency_compute_ms','total_ms','forward_calls'):
            stats[f'k{j}_{suffix}'] = None
        if j:
            stats[f'arbitration_k{j}_ms'] = None
    return stats


def aggregate_timing(stats):
    stats['retrieval_total_ms'] = sum(stats[k] for k in ('temporal_prefilter_ms','clip_image_cache_lookup_ms',
        'clip_image_encode_uncached_ms','clip_text_encode_ms','clip_relevance_ms','mmr_ranking_ms'))
    stats['vlm_selection_total_ms'] = sum(stats.get(f'k{j}_total_ms') or 0 for j in range(4))
    stats['arbitration_total_ms'] = sum(stats.get(f'arbitration_k{j}_ms') or 0 for j in range(1,4))
    stats['decision_logic_total_ms'] = stats['candidate_admission_ms'] + stats['arbitration_total_ms']
    stats['context_total_ms'] = stats['final_context_assembly_ms'] + stats['memory_frame_normalization_ms']
    stats['PRISM_selection_latency_ms'] = sum(stats[k] for k in
        ('retrieval_total_ms','vlm_selection_total_ms','decision_logic_total_ms','context_total_ms'))
    stats['PRISM_end_to_end_latency_ms'] = stats['PRISM_selection_latency_ms'] + stats['final_generation_ms']
    stats['latency_accounting_error_ms'] = abs(stats['PRISM_algorithmic_latency_ms'] - stats['PRISM_end_to_end_latency_ms'])
    # Direct wall timer validates the independently aggregated children, not a tautology.
    stats['latency_accounting_tolerance_ms'] = max(50., .02*stats['PRISM_algorithmic_latency_ms'])
    stats['latency_accounting_valid'] = stats['latency_accounting_error_ms'] <= stats['latency_accounting_tolerance_ms']
    if not stats['latency_accounting_valid']:
        raise RuntimeError(f"Unaccounted algorithmic latency: {stats['latency_accounting_error_ms']:.3f} ms")


def query(qa, video_path, prompt, chunk_duration, fps, recent_frames_only, video_start=None, video_end=None, cdas_config=None):
    from lib.minicpm import baseline as base
    from lib.shared.recent_window import decode_video_to_chunks_qwen
    from main_experiments.minicpm_v46.streamingbench.eval_prism_exact_recent_dist import select_exact_current_recent_frames
    control = os.environ.get('MINICPM_ADAPTIVE_MODE') == BASELINE_MODE
    if chunk_duration != 1.0 or fps != 1.0:
        raise ValueError('Frozen arbitration configuration requires chunk_duration=1 and fps=1')
    clip = getattr(qa, '_progressive_arbitration_clip', None)
    if clip is None:
        clip = ClipCache(qa)  # Model loading is outside both per-query timers.
        qa._progressive_arbitration_clip = clip
    if not getattr(qa, '_progressive_arbitration_warm', False):
        from PIL import Image
        from lib.minicpm.progressive_sufficiency import _score_options
        warm = [Image.new('RGB',(224,224),(i*20,40,60)) for i in range(6)]
        warm_prompt = 'Question: Select an option.\nA. one\nB. two\nAnswer with only the option letter.'
        import torch
        with torch.inference_mode():
            _score_options(qa,warm,warm_prompt,[dict(letter='A',text='one'),dict(letter='B',text='two')])
            clip.begin('__warmup__')
            clip.image_vectors(warm,{})
            clip.text_vectors(['warmup'])
            qa.generate_from_frames(warm,warm_prompt)
        sync()
        qa._progressive_arbitration_warm = True
    sync()
    system_start = time.perf_counter()
    stats = initial_stats()
    # Preserve caller bounds; never silently clamp out-of-range annotations.
    saved = os.environ.pop('QWEN_EXACT_RECENT_DECODE',None)
    try:
        if control:
            chunks, backend = [], 'exact_recent_control'
            stats['broad_history_decode_ms'] = 0.0
        else:
            with timed(stats,'broad_history_decode_ms'):
                chunks, backend = decode_video_to_chunks_qwen(video_path,chunk_duration,fps,CONFIG.horizon+6,
                                                              video_start=video_start,video_end=video_end)
        start = max(0.,video_end-6.) if video_end is not None else video_start
        with timed(stats,'recent6_decode_ms'):
            recent = select_exact_current_recent_frames(qa,video_path,chunk_duration,fps,6,video_start=start,video_end=video_end)
    finally:
        if saved is not None:
            os.environ['QWEN_EXACT_RECENT_DECODE'] = saved
    stats['video_io_ms'] = None  # Backend fuses container I/O and decode; not fabricated as zero.
    stats['total_visual_preparation_ms'] = (time.perf_counter()-system_start)*1000
    before = base._reset_gpu_memory_peaks()
    sync()
    algorithm_start = time.perf_counter()
    with timed(stats, 'clip_image_cache_lookup_ms'):
        clip.begin(os.path.realpath(video_path))
    from lib.minicpm.progressive_sufficiency import select_progressive_arbitration
    frames, accepted, metadata = select_progressive_arbitration(qa,chunks,recent,prompt,clip,stats,control=control)
    with timed(stats,'final_wrapper_ms'):
        answer = qa.generate_from_frames(frames,prompt,downsample_mode=recent.downsample_mode)
    stats['PRISM_algorithmic_latency_ms'] = (time.perf_counter()-algorithm_start)*1000
    stats['full_system_latency_ms'] = (time.perf_counter()-system_start)*1000
    stats['final_generation_ms'] = qa._last_model_generate_seconds*1000
    stats['final_preprocess_ms'] = qa._last_preprocess_seconds*1000
    # Final model input preparation belongs to context construction, not video preparation.
    stats['final_context_assembly_ms'] += stats['final_wrapper_ms']-stats['final_generation_ms']
    stats['final_ttft_ms'] = qa._last_ttft_seconds*1000
    stats['final_decode_after_first_token_ms'] = max(0.,stats['final_generation_ms']-stats['final_ttft_ms'])
    stats['final_num_generated_tokens'] = getattr(qa,'_last_generated_tokens',None)
    stats['num_vision_tokens'] = qa._last_num_vision_tokens
    aggregate_timing(stats)
    metadata.update(timing=stats, latency_definition=DEFINITION, full_system_definition=SYSTEM_DEFINITION,
                    warmup_enabled=True, warmup_calls=2, baseline_control=control)
    profile = base._build_profile(mode=metadata['mode'],decode_time=stats['total_visual_preparation_ms']/1000,
        selection_time=(stats['PRISM_algorithmic_latency_ms']-stats['final_wrapper_ms'])/1000,
        generate_time=stats['final_wrapper_ms']/1000,before_memory=before,after_memory=base._capture_gpu_memory(),qa=qa)
    profile['adaptive'] = metadata
    stats['gpu_peak_allocated_mb'] = profile['gpu_peak_allocated_mb']
    stats['gpu_peak_reserved_mb'] = profile['gpu_peak_reserved_mb']
    profile['progressive_arbitration'] = stats
    profile['end_to_end_time_seconds'] = stats['full_system_latency_ms']/1000
    ids = [-j-1 for j in range(len(accepted))] + list(recent.final_chunk_ids)
    metadata['final_selected_chunk_ids'] = ids  # Synthetic memory IDs; source identity stored separately.
    metadata['memory_chunk_ids'] = ids[:len(accepted)]
    tokens = qa._last_num_vision_tokens
    result = base.RecentWindowResult(answer=answer,final_chunk_ids=ids,generate_time=stats['final_wrapper_ms']/1000,
        ttft_seconds=qa._last_ttft_seconds,num_vision_tokens=tokens,num_vision_tokens_before=tokens,
        num_vision_tokens_after=tokens,num_frames=len(frames))
    result.profile_metadata = profile
    result.adaptive_metadata = metadata
    return result, backend
