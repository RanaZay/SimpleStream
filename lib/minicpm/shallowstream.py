"""ShallowStream (arXiv 2609.02780, github.com/CURRENTF/ShallowStream@0975506) ported to frozen MiniCPM-V-4.6.

Method, following the release's Qwen3-VL StreamingBench config:
  1. Index: every 1 fps frame passes through the shallow LLM layers only; each unit of 2 frames is stored as
     the L2-normalized mean of its shallow keys (pre-RoPE, post k_norm), flattened over KV heads.
  2. Gate (task_gate_mode=query_choice_logits, prompt retrieval_ab_v1): a text-only forward compares the
     next-token logits of "A" (retrieve older memory) and "B" (latest segment suffices); retrieve if A >= B.
  3. Real-time route: answer from the last 2 units. Retrieval route: score history units by the dot product of
     the question's shallow query vector (query heads averaged per KV group) with unit keys, keep the top 8,
     add 1 earlier neighbour at stride 2, and append the 6 most recent units.
  4. Answer: one normal full-depth MiniCPM generation over the selected frames in temporal order.

MiniCPM-specific changes (porting limitations):
  * MiniCPM-V-4.6's text model is hybrid (3 linear-attention layers per full-attention layer), so only
    full-attention layers have per-token keys. The shallow depth is up to and including the first
    full-attention layer (layers 0-3); the release uses layers 0-4 of Qwen3-VL-8B.
  * Frames are indexed in independent 64-frame batches (the release streams with a 128-frame local window).
  * The release's OVO config gates with a hidden-state threshold calibrated on Qwen3-VL-8B; it does not transfer,
    so the query_choice_logits gate is used on every benchmark.
  * MiniCPM answers "B" for nearly every routing prompt (logit(A) - logit(B) is about -17 to -19), so the release's
    default threshold of 0 would never retrieve. The threshold (the release's task_gate_query_choice_threshold) is
    set to the best split of the release's own 10 labeled routing questions (QUERY_CHOICE_FEWSHOT10), never on
    benchmark data, and can be overridden with SHALLOWSTREAM_GATE_THRESHOLD.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import os
import time

from lib.minicpm.progressive_arbitration import image_key, sync, timed

MODE = "shallowstream_minicpm_port"
SOURCE = "arXiv 2609.02780; github.com/CURRENTF/ShallowStream@0975506"
DEFINITION = ("ShallowStream algorithmic latency runs from question arrival (shallow index already built) to the "
              "final answer: query shallow pass, gate, retrieval scoring and the final MiniCPM generation. Shallow "
              "indexing of streamed frames is reported separately as streaming upkeep.")
SYSTEM_DEFINITION = "Full-system latency adds video decoding and shallow indexing of the whole history."
ROUTER_PROMPT = (
    "You are deciding whether answering a video question requires retrieving older "
    "video memory. The system already has the latest video segment immediately "
    "available. Do not answer the video question itself.\n\n"
    "<video_question>\n{question}\n</video_question>\n\n"
    "Choose exactly one routing option:\n"
    "A. Answering reliably requires inspecting earlier video memory.\n"
    "B. The latest video segment is sufficient; older video memory is unnecessary.\n\n"
    "Output only A or B.\n"
    "Routing answer:"
)


@dataclass(frozen=True)
class Config:
    frames_per_unit: int = 2
    recent_units: int = 6
    realtime_recent_units: int = 2
    topk_units: int = 8
    expand_prev_units: int = 1
    expand_prev_stride: int = 2
    index_batch_frames: int = 64
    gate_threshold: float | None = None  # None: calibrate on CALIBRATION_QUESTIONS.


CONFIG = Config()

# Release task_gate.QUERY_CHOICE_FEWSHOT10 (A: retrieve older memory, B: latest segment suffices).
CALIBRATION_QUESTIONS = (
    ("What object is the person holding in the latest visible moment?", "B"),
    ("Where did the person leave the keys near the beginning of the video?", "A"),
    ("What word is currently displayed on the screen?", "B"),
    ("How many times did the bell ring over the course of the video?", "A"),
    ("Which item did the cook pick up first?", "A"),
    ("What color is the vehicle visible in the latest scene?", "B"),
    ("What was inside the box before it was emptied?", "A"),
    ("What action is the person performing right now?", "B"),
    ("Which object is immediately to the left of the cup in the latest frame?", "B"),
    ("How has the room changed compared with its appearance earlier in the video?", "A"),
)


def best_split(scored):
    """Threshold maximizing routing accuracy on (score, label) pairs; midpoint between adjacent scores."""
    values = sorted({s for s, _ in scored})
    candidates = [values[0] - 1.0] + [(a + b) / 2 for a, b in zip(values, values[1:])] + [values[-1] + 1.0]
    def accuracy(t):
        return sum((s >= t) == (label == 'A') for s, label in scored)
    best = max(accuracy(t) for t in candidates)
    ties = [t for t in candidates if accuracy(t) == best]
    return ties[len(ties) // 2], best / len(scored)


def gate_threshold(qa, config=CONFIG):
    """Calibrated once per model instance; recorded in every result."""
    cached = getattr(qa, '_shallowstream_gate_calibration', None)
    if cached is None:
        override = os.environ.get('SHALLOWSTREAM_GATE_THRESHOLD')
        if override is not None:
            cached = dict(threshold=float(override), source='SHALLOWSTREAM_GATE_THRESHOLD')
        elif config.gate_threshold is not None:
            cached = dict(threshold=float(config.gate_threshold), source='config')
        else:
            scored = [(score_route(qa, q), label) for q, label in CALIBRATION_QUESTIONS]
            threshold, accuracy = best_split(scored)
            cached = dict(threshold=threshold, source='release QUERY_CHOICE_FEWSHOT10 best split',
                          calibration_accuracy=accuracy, calibration_scores=scored)
        qa._shallowstream_gate_calibration = cached
    return cached


def shallow_depth(model):
    """Layers up to and including the first full-attention layer, and that layer's index."""
    types = list(model.config.text_config.layer_types)
    first = types.index('full_attention')
    return first + 1, first


@contextmanager
def shallow_capture(model):
    """Run only the shallow layers and capture the index layer's normalized Q and K (pre-RoPE)."""
    import torch
    language = model.model.language_model
    depth, index_layer = shallow_depth(model)
    attention = language.layers[index_layer].self_attn
    captured = {}
    hooks = [attention.q_norm.register_forward_hook(lambda m, i, o: captured.__setitem__('q', o.detach())),
             attention.k_norm.register_forward_hook(lambda m, i, o: captured.__setitem__('k', o.detach()))]
    full = language.layers
    language.layers = torch.nn.ModuleList(list(full)[:depth])
    try:
        yield captured
    finally:
        language.layers = full
        for hook in hooks:
            hook.remove()


def visual_features(qa, inputs, count):
    """Per-frame MiniCPM visual tokens (vision encoder + merger), as the model computes them in forward()."""
    if count == 0:
        return []
    output = qa.model.model.get_image_features(inputs['pixel_values'][:1], inputs['target_sizes'],
                                               downsample_mode=qa.downsample_mode)
    features = [f.detach() for f in output.pooler_output]
    if len(features) != count:
        raise RuntimeError(f'Expected {count} frame feature blocks, got {len(features)}')
    return features


def embeds_with_features(qa, inputs, features):
    """Text embeddings with the image placeholders filled by cached visual features (forward()'s masked_scatter)."""
    import torch
    ids = inputs['input_ids']
    embeds = qa.model.model.get_input_embeddings()(ids)
    if not features:
        return embeds
    values = torch.cat(features, dim=0).to(device=embeds.device, dtype=embeds.dtype)
    mask = (ids == qa.model.config.image_token_id).unsqueeze(-1).expand_as(embeds)
    if int(mask.sum()) != values.numel():
        raise RuntimeError('Cached visual features do not match the image placeholders')
    return embeds.masked_scatter(mask, values)


def _forward_shallow(qa, inputs, features):
    qa.model.model.language_model(inputs_embeds=embeds_with_features(qa, inputs, features),
                                  attention_mask=inputs.get('attention_mask'), use_cache=False)


def generate_cached(qa, frames, features, prompt):
    """qa.generate_from_frames, but reusing visual features cached at indexing (the release stores visual_embeds)."""
    from lib.minicpm.progressive_sufficiency import _build_minicpm_inputs
    start = time.perf_counter()
    inputs = _build_minicpm_inputs(qa, frames, prompt)
    embeds = embeds_with_features(qa, inputs, features)
    qa._last_preprocess_seconds = time.perf_counter() - start
    qa._last_num_vision_frames = len(frames)
    qa._last_num_vision_tokens = qa._estimate_vision_tokens(inputs)
    return qa._generate_from_model_inputs(prompt_length=int(inputs['input_ids'].shape[1]), input_ids=inputs['input_ids'],
                                          attention_mask=inputs['attention_mask'], inputs_embeds=embeds)


def frame_token_runs(input_ids, image_token_id, frame_count):
    """Token positions of each frame's visual tokens (one contiguous run per frame)."""
    ids = input_ids.tolist()
    runs, current = [], []
    for position, token in enumerate(ids):
        if token == image_token_id:
            current.append(position)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    if len(runs) == frame_count:
        return runs
    flat = [p for run in runs for p in run]
    if frame_count and len(flat) % frame_count == 0:
        size = len(flat) // frame_count
        return [flat[i * size:(i + 1) * size] for i in range(frame_count)]
    raise RuntimeError(f'Cannot split {len(flat)} visual tokens into {frame_count} frames')


def key_vector(key, positions):
    """Release _normalized_key_vector: mean over tokens, flatten KV heads, L2-normalize. key: [seq, heads, dim]."""
    import torch
    return torch.nn.functional.normalize(key[positions].float().mean(dim=0).reshape(-1), dim=0)


def query_vector(query, positions, kv_heads):
    """Release _normalized_query_vector: average query heads within each KV group, then as key_vector."""
    import torch
    selected = query[positions].float()  # [tokens, q_heads, dim]
    tokens, heads, dim = selected.shape
    selected = selected.view(tokens, kv_heads, heads // kv_heads, dim).mean(dim=2)
    return torch.nn.functional.normalize(selected.mean(dim=0).reshape(-1), dim=0)


def index_frames(qa, frames, stats, config=CONFIG):
    """Shallow keys for every frame in independent batches.

    Returns per frame (token sum [heads*dim], token count), and the frame's visual features for reuse.
    """
    from lib.minicpm.progressive_sufficiency import _build_minicpm_inputs
    image_token_id = qa.model.config.image_token_id
    vectors, features = [], []
    with shallow_capture(qa.model) as captured:
        for start in range(0, len(frames), config.index_batch_frames):
            batch = frames[start:start + config.index_batch_frames]
            inputs = _build_minicpm_inputs(qa, batch, ' ')
            batch_features = visual_features(qa, inputs, len(batch))
            features += batch_features
            _forward_shallow(qa, inputs, batch_features)
            key = captured['k'][0]
            runs = frame_token_runs(inputs['input_ids'][0], image_token_id, len(batch))
            vectors += [(key[run].float().sum(dim=0).reshape(-1), len(run)) for run in runs]
    stats['shallow_index_batches'] = (len(frames) + config.index_batch_frames - 1) // config.index_batch_frames
    return vectors, features


def index_incremental(qa, video, frames, stamps, stats, config=CONFIG):
    """Index each streamed frame once per video, as frames arrive in deployment (same policy as the OASIS port).

    Frames already indexed for this video (matched by timestamp) are reused; only new frames run through the
    shallow layers. The state resets when the video changes or time goes backwards.
    """
    state = getattr(qa, '_shallowstream_stream', None)
    keys = [round(t, 3) for t in stamps]
    if state is None or state['video'] != video or (state['cache'] and keys and keys[-1] < max(state['cache'])):
        state = qa._shallowstream_stream = dict(video=video, cache={})
    cache = state['cache']
    new = [i for i, k in enumerate(keys) if k not in cache]
    if new:
        vectors, features = index_frames(qa, [frames[i] for i in new], stats, config)
        for i, vector, feature in zip(new, vectors, features):
            cache[keys[i]] = (vector, feature)
    stats['shallow_index_new_frames'] = len(new)
    stats['shallow_index_reused_frames'] = len(keys) - len(new)
    return [cache[k][0] for k in keys], [cache[k][1] for k in keys], len(keys) - len(new)


def question_vector(qa, recent_frames, recent_features, prompt):
    """The question's shallow query vector, with the recent frames as video context (release: full prompt)."""
    from lib.minicpm.progressive_sufficiency import _build_minicpm_inputs
    import torch
    inputs = _build_minicpm_inputs(qa, recent_frames, prompt)
    with shallow_capture(qa.model) as captured:
        _forward_shallow(qa, inputs, recent_features)
    ids = inputs['input_ids'][0]
    visual = torch.nonzero(ids == qa.model.config.image_token_id).flatten()
    start = int(visual[-1].item()) + 1 if visual.numel() else 0
    positions = list(range(start, int(ids.numel())))
    return query_vector(captured['q'][0], positions, int(captured['k'].shape[2]))


def score_route(qa, question):
    """logit("A") - logit("B") after the release's retrieval_ab_v1 routing prompt."""
    import torch
    from lib.minicpm.progressive_sufficiency import _build_minicpm_inputs
    inputs = _build_minicpm_inputs(qa, [], ROUTER_PROMPT.format(question=question))
    with torch.inference_mode():
        logits = qa.model(**inputs, use_cache=False, return_dict=True).logits[0, -1].float()
    tokenizer = qa.processor.tokenizer
    ids = [tokenizer.encode(label, add_special_tokens=False) for label in ('A', 'B')]
    if any(len(i) != 1 for i in ids) or ids[0] == ids[1]:
        raise RuntimeError(f'Routing labels must be distinct single tokens, got {ids}')
    return float(logits[ids[0][0]] - logits[ids[1][0]])


def gate(qa, question, config=CONFIG):
    """query_choice_logits gate: retrieve when logit(A) - logit(B) >= threshold."""
    calibration = gate_threshold(qa, config)
    score = score_route(qa, question)
    return dict(score=score, retrieve=score >= calibration['threshold'], threshold=calibration['threshold'],
                threshold_source=calibration['source'])


def units_of(count, size):
    """Unit u covers frames [u*size, (u+1)*size); the last unit may be shorter."""
    return [list(range(u * size, min((u + 1) * size, count))) for u in range((count + size - 1) // size)]


def select_units(unit_vectors, query, retrieve, config=CONFIG):
    """Release _select_memory with unit granularity and temporal_neighbors expansion."""
    total = len(unit_vectors)
    if not retrieve:
        return list(range(max(0, total - config.realtime_recent_units), total)), [], []
    recent_start = max(0, total - config.recent_units)
    scored = sorted(((float(unit_vectors[u] @ query), u) for u in range(recent_start)), reverse=True)
    seeds = [u for _score, u in scored[:config.topk_units]]
    expanded = set(seeds)
    for u in seeds:
        for distance in range(1, config.expand_prev_units + 1):
            neighbour = u - distance * config.expand_prev_stride
            if 0 <= neighbour < recent_start:
                expanded.add(neighbour)
    selected = sorted(expanded | set(range(recent_start, total)))
    return selected, seeds, scored


def query(qa, video_path, prompt, chunk_duration, fps, recent_frames_only, video_start=None, video_end=None,
          cdas_config=None):
    import torch
    from lib.minicpm import baseline as base
    from lib.minicpm.prism_unified import question_text
    from lib.shared.recent_window import decode_video_to_chunks_qwen
    if chunk_duration != 1.0 or fps != 1.0:
        raise ValueError('ShallowStream port streams at 1 fps with 1 s chunks')
    config = CONFIG
    if not getattr(qa, '_shallowstream_warm', False):
        from PIL import Image
        warm = [Image.new('RGB', (224, 224), (i * 20, 40, 60)) for i in range(4)]
        with torch.inference_mode():
            _vectors, warm_features = index_frames(qa, warm, {})
            question_vector(qa, warm[-2:], warm_features[-2:], 'Question: What is shown?')
            gate(qa, 'What is shown?')  # Also calibrates the gate threshold once.
            generate_cached(qa, warm, warm_features, 'Question: What is shown?')
        sync()
        qa._shallowstream_warm = True
    sync()
    system_start = time.perf_counter()
    stats = {}
    saved = os.environ.pop('QWEN_EXACT_RECENT_DECODE', None)
    try:
        with timed(stats, 'broad_history_decode_ms'):
            chunks, backend = decode_video_to_chunks_qwen(video_path, chunk_duration, fps, None,
                                                          video_start=video_start, video_end=video_end)
    finally:
        if saved is not None:
            os.environ['QWEN_EXACT_RECENT_DECODE'] = saved
    frames, stamps = [], []
    for chunk in chunks:
        for frame, stamp in zip(chunk.frames, chunk.frame_timestamps):
            frames.append(frame)
            stamps.append(float(stamp))
    if not frames:
        raise ValueError(f'No frames decoded for {video_path}')
    stats['total_visual_preparation_ms'] = stats['broad_history_decode_ms']
    stats['video_io_ms'] = None
    units = units_of(len(frames), config.frames_per_unit)
    before = base._reset_gpu_memory_peaks()
    with torch.inference_mode():
        with timed(stats, 'memory_maintenance_total_ms'):  # Streaming upkeep: in deployment, done as frames arrive.
            frame_vectors, features, reused = index_incremental(qa, os.path.realpath(video_path), frames, stamps,
                                                                stats, config)
            # Release key_vec: mean over all of the unit's visual tokens, then L2-normalize.
            unit_vectors = [torch.nn.functional.normalize(sum(frame_vectors[f][0] for f in unit)
                                                          / sum(frame_vectors[f][1] for f in unit), dim=0)
                            for unit in units]
        sync()
        algorithm_start = time.perf_counter()
        question = question_text(prompt)
        with timed(stats, 'gate_ms'):
            decision = gate(qa, question, config)
        recent_ids = [f for u in units[-config.recent_units:] for f in u]
        with timed(stats, 'query_vector_ms'):
            q = (question_vector(qa, [frames[f] for f in recent_ids], [features[f] for f in recent_ids], prompt)
                 if decision['retrieve'] else None)
        with timed(stats, 'retrieval_total_ms'):
            selected, seeds, scored = select_units(unit_vectors, q, decision['retrieve'], config)
            frame_ids = [f for u in selected for f in units[u]]
        with timed(stats, 'final_wrapper_ms'):
            answer = generate_cached(qa, [frames[f] for f in frame_ids], [features[f] for f in frame_ids], prompt)
    stats['PRISM_algorithmic_latency_ms'] = (time.perf_counter() - algorithm_start) * 1000
    stats['full_system_latency_ms'] = (time.perf_counter() - system_start) * 1000
    stats['final_generation_ms'] = qa._last_model_generate_seconds * 1000
    stats['final_ttft_ms'] = qa._last_ttft_seconds * 1000
    stats['final_preprocess_ms'] = qa._last_preprocess_seconds * 1000
    stats['context_total_ms'] = stats['final_wrapper_ms'] - stats['final_generation_ms']
    stats['vlm_selection_total_ms'] = stats['gate_ms'] + stats['query_vector_ms']
    stats['decision_logic_total_ms'] = 0.0
    stats['PRISM_selection_latency_ms'] = (stats['vlm_selection_total_ms'] + stats['retrieval_total_ms']
                                           + stats['context_total_ms'])
    stats['PRISM_end_to_end_latency_ms'] = stats['PRISM_selection_latency_ms'] + stats['final_generation_ms']
    stats['latency_accounting_residual_ms'] = stats['PRISM_algorithmic_latency_ms'] - stats['PRISM_end_to_end_latency_ms']
    stats['request_to_answer_first_token_ms'] = stats['PRISM_selection_latency_ms'] + stats['final_ttft_ms']
    stats['num_vision_tokens'] = qa._last_num_vision_tokens
    stats['final_frame_count'] = len(frame_ids)
    metadata = dict(mode=MODE, method_source=SOURCE, config=asdict(config), gate=decision,
                    gate_calibration={k: v for k, v in gate_threshold(qa, config).items() if k != 'calibration_scores'},
                    shallow_depth=shallow_depth(qa.model)[0], index_layer=shallow_depth(qa.model)[1],
                    history_frames=len(frames), history_units=len(units), reused_indexed_frames=reused, selected_units=selected,
                    retrieved_seed_units=seeds, unit_scores_top=[(round(s, 4), u) for s, u in scored[:16]],
                    selected_timestamps=[stamps[f] for f in frame_ids], memory_triggered=decision['retrieve'],
                    selected_memory_depth=len(seeds), timing=stats, latency_definition=DEFINITION,
                    full_system_definition=SYSTEM_DEFINITION, baseline_control=False,
                    recent_frame_hashes=None)
    profile = base._build_profile(mode=MODE, decode_time=stats['total_visual_preparation_ms'] / 1000,
                                  selection_time=stats['PRISM_selection_latency_ms'] / 1000,
                                  generate_time=stats['final_generation_ms'] / 1000, before_memory=before,
                                  after_memory=base._capture_gpu_memory(), qa=qa)
    profile['adaptive'] = metadata
    stats['gpu_peak_allocated_mb'] = profile['gpu_peak_allocated_mb']
    stats['gpu_peak_reserved_mb'] = profile['gpu_peak_reserved_mb']
    profile['progressive_arbitration'] = stats
    profile['end_to_end_time_seconds'] = stats['full_system_latency_ms'] / 1000
    result = base.RecentWindowResult(answer=answer, final_chunk_ids=[int(stamps[f]) for f in frame_ids],
                                     generate_time=stats['final_generation_ms'] / 1000,
                                     ttft_seconds=stats['final_ttft_ms'] / 1000,
                                     num_vision_tokens=qa._last_num_vision_tokens,
                                     num_vision_tokens_before=qa._last_num_vision_tokens,
                                     num_vision_tokens_after=qa._last_num_vision_tokens, num_frames=len(frame_ids))
    result.profile_metadata = profile
    result.adaptive_metadata = metadata
    return result, backend
