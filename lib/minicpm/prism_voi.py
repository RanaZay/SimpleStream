"""PRISM-VoI building blocks: format-agnostic semantic entropy of a frozen VLM's answer (design: docs/prism_voi_design.md).

The question format only sets the answer hypotheses Omega(q):
  multiple-choice  Omega = the options; p(option | C) from the option-letter logits (one forward); SE = option entropy.
  open-ended       Omega = greedy answer + M sampled answers under the reference context; p(answer | C) from
                   length-normalised teacher-forced likelihood; answers with the same meaning share a cluster.
SE(C) = -sum_k p(c_k | C) log p(c_k | C) over meaning clusters c_k (Kuhn et al., ICLR 2023).
"""
from __future__ import annotations

import copy
import math
import time


def option_distribution(qa, frames, prompt, options):
    """{letter: probability} over the MCQ options under context `frames` (one forward)."""
    from lib.minicpm.progressive_sufficiency import _score_options
    import torch
    with torch.inference_mode():
        state = _score_options(qa, frames, prompt, options)
    return dict(state['option_probabilities']), state['predicted_option']


def answer_logprob(qa, frames, prompt, answer):
    """Mean per-token log-probability of `answer` as the assistant reply (teacher forcing, one forward)."""
    import torch
    from lib.minicpm.progressive_sufficiency import _build_minicpm_inputs
    base = _build_minicpm_inputs(qa, frames, prompt).input_ids[0]
    inputs = _build_minicpm_inputs(qa, frames, prompt, assistant_text=answer)
    ids = inputs.input_ids[0]
    prefix = 0
    limit = min(int(base.numel()), int(ids.numel()))
    while prefix < limit and int(base[prefix]) == int(ids[prefix]):
        prefix += 1
    # Locate the answer's own tokens after the shared prefix (the template may add e.g. an empty think block first).
    start, end = _answer_span(qa, ids, prefix, answer)
    with torch.inference_mode():
        logits = qa.model(**inputs, use_cache=False, return_dict=True).logits[0].float()
    logp = torch.log_softmax(logits[start - 1:end - 1], dim=-1)
    target = ids[start:end].to(logp.device)
    return float(logp.gather(1, target[:, None]).mean())


def _answer_span(qa, ids, prefix, answer):
    """Token positions [start, end) of `answer` in a templated sequence, after the prompt's shared prefix."""
    seq = ids.tolist()
    for variant in (answer, ' ' + answer, answer.strip()):
        target = qa.processor.tokenizer.encode(variant, add_special_tokens=False)
        for j in range(max(prefix, 1), len(seq) - len(target) + 1):
            if target and seq[j:j + len(target)] == target:
                return j, j + len(target)
    raise RuntimeError(f'Could not locate the answer tokens of {answer!r} in the templated sequence')


def answer_logprobs(qa, frames, prompt, answers):
    """answer_logprob for several answers under one context, encoding the frames and prompt once.

    PRISM_VOI_PREFIX_CACHE=1 computes the prompt's key/value cache once and continues each answer from its own copy.
    It is off by default: on MiniCPM-V-4.6 the cached scores differ from one forward per answer by 0.1-0.2 nats
    (job 354395), so every answer gets its own forward. PRISM_VOI_CHECK_PREFIX=1 also scores each answer the
    single-forward way, prints the difference and fails on a real mismatch.
    """
    import torch
    from lib.minicpm.progressive_sufficiency import _build_minicpm_inputs
    answers = list(dict.fromkeys(answers))
    if len(answers) < 2 or os.environ.get('PRISM_VOI_PREFIX_CACHE', '').strip().lower() not in ('1', 'true', 'yes'):
        return {a: answer_logprob(qa, frames, prompt, a) for a in answers}
    base = _build_minicpm_inputs(qa, frames, prompt)
    base_ids = base.input_ids[0]
    full = {a: _build_minicpm_inputs(qa, frames, prompt, assistant_text=a).input_ids[0] for a in answers}
    spans, shared = {}, int(base_ids.numel())
    for a, ids in full.items():
        prefix = 0
        limit = min(int(base_ids.numel()), int(ids.numel()))
        while prefix < limit and int(base_ids[prefix]) == int(ids[prefix]):
            prefix += 1
        spans[a] = _answer_span(qa, ids, prefix, a)
        shared = min(shared, prefix, spans[a][0] - 1)
    if shared < 1:
        return {a: answer_logprob(qa, frames, prompt, a) for a in answers}
    keep = {k: v[:, :shared] if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == base_ids.numel() else v
            for k, v in base.items()}
    with torch.inference_mode():
        prefix_cache = qa.model(**keep, use_cache=True, return_dict=True).past_key_values
        out = {}
        device = base_ids.device
        for a in answers:
            start, end = spans[a]
            # A copy per answer: crop() is not supported by every cache layer type (e.g. MiniCPM-V-4.6's).
            cache = copy.deepcopy(prefix_cache)
            tail = full[a][shared:end].to(device)[None]
            mask = torch.ones(1, shared + tail.shape[1], dtype=torch.long, device=device)
            logits = qa.model(input_ids=tail, past_key_values=cache, attention_mask=mask, use_cache=True,
                              return_dict=True).logits[0].float()
            logp = torch.log_softmax(logits[start - 1 - shared:end - 1 - shared], dim=-1)
            out[a] = float(logp.gather(1, full[a][start:end].to(logp.device)[:, None]).mean())
            del cache, logits, logp
    if os.environ.get('PRISM_VOI_CHECK_PREFIX', '').strip().lower() in ('1', 'true', 'yes'):
        for a in answers:  # bf16 kernels differ slightly between the two paths; a real bug is far larger.
            reference = answer_logprob(qa, frames, prompt, a)
            print(f'[prefix-check] {abs(reference - out[a]):.5f} cached={out[a]:.4f} single={reference:.4f}', flush=True)
            if abs(reference - out[a]) > 0.1:
                raise AssertionError(f'Prefix-cached log-probability {out[a]:.4f} != {reference:.4f} for {a!r}')
    return out


def sample_answers(qa, frames, prompt, m, max_new_tokens=32, temperature=1.0):
    """Greedy answer plus m sampled answers under `frames`."""
    import torch
    from lib.minicpm.progressive_sufficiency import _build_minicpm_inputs
    greedy = qa.generate_from_frames(frames, prompt)
    if m <= 0:
        return greedy, []
    inputs = _build_minicpm_inputs(qa, frames, prompt)
    length = int(inputs['input_ids'].shape[1])
    tokenizer = qa.processor.tokenizer
    from lib.minicpm.baseline import generation_options
    with torch.inference_mode():
        out = qa.model.generate(**inputs, **generation_options(qa), max_new_tokens=max_new_tokens,
                                do_sample=True, temperature=temperature, top_p=1.0, num_return_sequences=m,
                                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    samples = []
    for row in out:
        row = row[length:] if row.numel() > length else row
        samples.append(qa.processor.batch_decode([row], skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)[0].strip())
    return greedy, samples


EQUIVALENCE_PROMPT = (
    "Question: {question}\n"
    "Answer 1: {a}\n"
    "Answer 2: {b}\n"
    "Do Answer 1 and Answer 2 give the same answer to the question? Reply with only yes or no."
)


def same_meaning(qa, question, a, b):
    """The frozen VLM (text only) judges whether two answers to `question` mean the same: P(yes) > P(no)."""
    import torch
    from lib.minicpm.progressive_sufficiency import _build_minicpm_inputs
    tokenizer = qa.processor.tokenizer
    yes = tokenizer.encode('yes', add_special_tokens=False)[0]
    no = tokenizer.encode('no', add_special_tokens=False)[0]
    votes = []
    for x, y in ((a, b), (b, a)):  # Both orders, as bidirectional entailment requires.
        inputs = _build_minicpm_inputs(qa, [], EQUIVALENCE_PROMPT.format(question=question, a=x, b=y))
        with torch.inference_mode():
            logits = qa.model(**inputs, use_cache=False, return_dict=True).logits[0, -1].float()
        votes.append(bool(logits[yes] > logits[no]))
    return all(votes)


def cluster_answers(answers, qa=None, question=None):
    """Meaning clusters: identical after normalisation, then merged when the VLM judges them equivalent."""
    from lib.minicpm.prism_unified import normalize_answer
    groups = {}
    for answer in answers:
        groups.setdefault(normalize_answer(answer) or answer.strip().lower(), []).append(answer)
    groups = list(groups.values())
    if qa is None or question is None:
        return groups
    clusters = []
    for group in groups:
        for cluster in clusters:
            if same_meaning(qa, question, cluster[0], group[0]):
                cluster.extend(group)
                break
        else:
            clusters.append(list(group))
    return clusters


def entropy(probabilities):
    return -sum(p * math.log(p) for p in probabilities if p > 0)


def semantic_entropy_open(qa, frames, prompt, clusters):
    """SE over meaning clusters under `frames`; each cluster scored by its members' normalised likelihoods."""
    weights = []
    for members in clusters:
        weights.append(sum(math.exp(answer_logprob(qa, frames, prompt, a)) for a in dict.fromkeys(members)))
    total = sum(weights) or 1.0
    probs = [w / total for w in weights]
    return entropy(probs), probs


def semantic_entropy_mcq(probabilities):
    return entropy(list(probabilities.values()))


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - start) * 1000


# --------------------------------------------------------------------------------------------------- controller
from dataclasses import asdict, dataclass, fields, replace  # noqa: E402
import os  # noqa: E402

MODE = "prism_voi_semantic_entropy"
DEFINITION = ("PRISM-VoI algorithmic latency runs from question arrival to the final answer: memory ranking, "
              "semantic-entropy evaluation of each tried context, and the final generation. Video decoding is excluded.")


@dataclass(frozen=True)
class Config:
    lam: float = 0.1          # nats of semantic entropy one added frame must remove (the accuracy-compute trade-off)
    max_frames: int = 3       # compute cap: memory frames tried per question
    reach: float = 30.0       # memory store reach before the recent window, seconds (storage budget)
    min_gap: float = 3.0      # memory frames start this far before the recent window (avoids near-duplicates)
    recent: int = 6
    samples: int = 5          # sampled answers for open-ended semantic entropy (estimation budget)
    horizon: int = 64         # frames kept in the memory store
    trace: bool = False       # evaluate every candidate context so any lambda can be replayed offline
    ranking: str = 'clip'     # candidate order: 'clip' (relevance + diversity), 'recency' (nearest first), 'random'
    stop: str = 'first'       # 'first': stop at the first frame that does not pay; 'scan': skip it and try the next
    pool: int = 0             # memory frames tried per question (0 = max_frames); 'scan' keeps at most max_frames
    signals: bool = False     # trace mode: also record competing decision signals for offline replay
    method: str = 'voi'       # 'voi' (PRISM-VoI); reimplemented baselines: 'wrws' (WRWS), 'streamscout' (zero-shot)


def config_from_env():
    """Every field can be overridden as PRISM_VOI_<FIELD>, e.g. PRISM_VOI_LAM=0.3."""
    values = {}
    for f in fields(Config):
        raw = os.environ.get('PRISM_VOI_' + f.name.upper())
        if raw is not None:
            values[f.name] = raw.strip().lower() in ('1', 'true', 'yes') if isinstance(f.default, bool) else type(f.default)(raw)
    return Config(**values)


def memory_candidates(chunks, recent, config):
    """One representative frame per streamed second in [start(R) - reach, start(R) - min_gap], deduplicated."""
    from lib.minicpm.progressive_arbitration import image_key
    from lib.minicpm.progressive_sufficiency import _chunk_temporal_bounds
    from lib.minicpm.prism_retrieval_variants import representative_frame
    start = min(_chunk_temporal_bounds(c)[0] for c in recent.selected_chunks)
    seen = {image_key(f) for f in recent.frames}
    out, ids = [], set()
    for chunk in chunks:
        low, high = _chunk_temporal_bounds(chunk)
        distance = start - high
        if high < start and config.min_gap <= distance <= config.reach and int(chunk.chunk_index) not in ids:
            frame = representative_frame(chunk)
            key = image_key(frame)
            if key in seen:
                continue
            seen.add(key)
            ids.add(int(chunk.chunk_index))
            out.append(dict(chunk=chunk, frame=frame, hash=key, start=low, end=high, distance=distance,
                            source_id=int(chunk.chunk_index)))
    out.sort(key=lambda c: (c['end'], c['start'], c['source_id']))
    return out[-config.horizon:]


class Evaluator:
    """Semantic entropy of the answer under a context; one hypothesis set per question, shared by all contexts."""

    def __init__(self, qa, prompt, question, options, config):
        self.qa, self.prompt, self.question, self.options, self.config = qa, prompt, question, options, config
        self.clusters, self.greedy, self.scores, self.calls = None, {}, {}, 0

    def answer(self, key, frames):
        """Greedy answer under a context (cached); also the final answer when that context is chosen."""
        if key not in self.greedy:
            start = time.perf_counter()
            text = self.qa.generate_from_frames(frames, self.prompt)
            self.greedy[key] = dict(text=text, ms=(time.perf_counter() - start) * 1000,
                                    generate_ms=self.qa._last_model_generate_seconds * 1000,
                                    ttft_ms=self.qa._last_ttft_seconds * 1000)
        return self.greedy[key]['text']

    def se(self, key, frames):
        if self.options:
            probs, pred = option_distribution(self.qa, frames, self.prompt, self.options)
            self.calls += 1
            return semantic_entropy_mcq(probs), dict(prediction=pred, probabilities=probs)
        if self.clusters is None:  # Hypotheses come from the reference context R (key 0).
            greedy = self.answer(key, frames)
            _greedy, sampled = sample_answers(self.qa, frames, self.prompt, self.config.samples)
            self.clusters = cluster_answers([greedy] + sampled, self.qa, self.question)
        else:
            new = self.answer(key, frames)  # An answer that only appears under added evidence joins the hypotheses.
            if all(new not in members for members in self.clusters):
                self.clusters = cluster_answers([a for c in self.clusters for a in c] + [new], self.qa, self.question)
        missing = [a for c in self.clusters for a in dict.fromkeys(c) if (key, a) not in self.scores]
        for a, value in answer_logprobs(self.qa, frames, self.prompt, missing).items():
            self.scores[(key, a)] = value
            self.calls += 1
        weights = [sum(math.exp(self.scores[(key, a)]) for a in dict.fromkeys(members)) for members in self.clusters]
        norm = sum(weights) or 1.0
        probs = [w / norm for w in weights]
        return entropy(probs), dict(clusters=[len(c) for c in self.clusters], probabilities=probs)


def order_candidates(eligible, question, clip, stats, config):
    """The memory frames to try, best first; the non-CLIP orders are ablations of the ranking."""
    from lib.minicpm import prism_unified as pu
    count = max(config.pool or config.max_frames, 1)
    if config.ranking == 'clip':
        rank_config = replace(pu.config_from_env(), pool=count)
        return pu.rank(eligible, question, clip, stats, rank_config)[:count]
    if config.ranking == 'recency':
        return sorted(eligible, key=lambda c: -c['end'])[:count]
    if config.ranking == 'random':  # Seeded by the question, so reruns pick the same frames.
        import random
        import zlib
        return random.Random(zlib.crc32(question.encode())).sample(eligible, min(count, len(eligible)))
    raise ValueError(f'Unknown PRISM_VOI_RANKING {config.ranking!r}')


def select(qa, chunks, recent, prompt, clip, stats, config):
    """Greedy value-of-information selection: keep a memory frame only if it removes more than lam nats of SE."""
    from lib.minicpm import prism_unified as pu
    from lib.minicpm.progressive_arbitration import extract_options, timed as ptimed
    question = pu.question_text(prompt)
    options = extract_options(prompt)
    ev = Evaluator(qa, prompt, question, options, config)
    contexts = {0: list(recent.frames)}
    start = time.perf_counter()
    with ptimed(stats, 'reference_uncertainty_ms'):
        se0, info0 = ev.se(0, contexts[0])
    trace = [dict(depth=0, se=se0, ms=(time.perf_counter() - start) * 1000, **info0)]
    accepted, current, se_current, stop = [], 0, se0, 'confident'
    retrieval_entropy = None
    chain, chain_se = [], se0  # Trace mode: every candidate is added cumulatively, the decision is replayed after.
    if se0 > config.lam or config.trace:  # Otherwise no single frame could pay for itself.
        with ptimed(stats, 'retrieval_total_ms'):
            eligible = memory_candidates(chunks() if callable(chunks) else chunks, recent, config)
            candidates = order_candidates(eligible, question, clip, stats, config)
        stop = 'no_candidates' if not candidates else 'cap_reached'
        base_depth = 0
        for j, cand in enumerate(candidates, 1):
            start = time.perf_counter()
            frame = cand['frame'] if cand['frame'].size == recent.frames[0].size else cand['frame'].resize(recent.frames[0].size)
            previous = chain if config.trace else accepted
            trial = sorted(previous + [(cand['end'], frame, cand)], key=lambda x: x[0])
            contexts[j] = [f for _t, f, _c in trial] + list(recent.frames)
            if not options:  # The hypothesis set may grow under the new context; re-score the current one on it.
                ev.answer(j, contexts[j])
            with ptimed(stats, 'context_uncertainty_ms'):
                se_trial, info = ev.se(j, contexts[j])
                if not options:
                    chain_se, _ = ev.se(j - 1 if config.trace else current, contexts[j - 1 if config.trace else current])
                    se_current = chain_se
            reference = chain_se if config.trace else se_current
            gain = reference - se_trial
            ok = gain > config.lam
            trace.append(dict(depth=j, se=se_trial, gain=gain, accepted=ok, distance=cand['distance'], time=cand['end'],
                              ms=(time.perf_counter() - start) * 1000, **info))
            if config.trace:
                chain, chain_se = trial, se_trial
                if ok and base_depth == j - 1:  # Replay the greedy rule: accept while every step so far passed.
                    accepted, current, se_current, base_depth = trial, j, se_trial, j
                continue
            if not ok:
                if config.stop == 'scan':  # A lower-ranked frame may still pay; this one is left out.
                    continue
                stop = 'gain_below_lambda'
                break
            accepted, current, se_current = trial, j, se_trial
            if len(accepted) >= config.max_frames:
                break
        if config.trace and current < len(candidates):
            stop = 'gain_below_lambda'
        if config.signals and config.trace:  # WRWS's retrieval uncertainty over the same memory and R (offline gate).
            alpha, _fused = wrws_fusion(*_pool_scores(eligible, recent, question, clip, {}))
            retrieval_entropy = alpha
    if not options:  # Record each tried context's own answer for offline replay at any lambda.
        for row in trace:
            row['answer'] = ev.greedy.get(row['depth'], {}).get('text')
            row['answer_logprob'] = ev.scores.get((row['depth'], row['answer']))  # token-confidence signal
    final_frames = contexts[current]
    if options or current not in ev.greedy:
        start = time.perf_counter()
        answer = qa.generate_from_frames(final_frames, prompt)
        final = dict(text=answer, ms=(time.perf_counter() - start) * 1000,
                     generate_ms=qa._last_model_generate_seconds * 1000, ttft_ms=qa._last_ttft_seconds * 1000)
    else:
        final = ev.greedy[current]  # Already generated while evaluating that context.
    meta = dict(mode=MODE, config=asdict(config), format='mcq' if options else 'open', question_text=question,
                stop_reason=stop, trace=trace, se_reference=se0, se_final=se_current,
                selected_memory_depth=len(accepted), memory_triggered=len(trace) > 1, memory_accepted=bool(accepted),
                memory_distances=[c['distance'] for _t, _f, c in accepted],
                memory_times=[c['end'] for _t, _f, c in accepted], vlm_scoring_calls=ev.calls,
                retrieval_entropy=retrieval_entropy,
                recent_frame_hashes=None, baseline_control=False, _contexts=contexts)
    return final_frames, final, meta


def query(qa, video_path, prompt, chunk_duration, fps, recent_frames_only, video_start=None, video_end=None,
          cdas_config=None):
    from lib.minicpm import baseline as base
    from lib.minicpm.progressive_arbitration import ClipCache, sync, timed as ptimed
    from lib.shared.recent_window import decode_video_to_chunks_qwen
    from main_experiments.minicpm_v46.streamingbench.eval_prism_exact_recent_dist import select_exact_current_recent_frames
    if chunk_duration != 1.0 or fps != 1.0:
        raise ValueError('PRISM-VoI streams at 1 fps with 1 s chunks')
    config = getattr(qa, '_prism_voi_config', None) or config_from_env()
    qa._prism_voi_config = config
    clip = getattr(qa, '_progressive_arbitration_clip', None)
    if clip is None:
        clip = qa._progressive_arbitration_clip = ClipCache(qa)
    if not getattr(qa, '_prism_voi_warm', False):  # Untimed: kernels for scoring, sampling and generation.
        from PIL import Image
        import torch
        warm = [Image.new('RGB', (224, 224), (i * 20, 40, 60)) for i in range(config.recent)]
        with torch.inference_mode():
            option_distribution(qa, warm, 'Question: Pick one.\nOptions:\nA. one\nB. two\nOnly give the letter.',
                                [dict(letter='A', text='one'), dict(letter='B', text='two')])
            sample_answers(qa, warm, 'Question: What is shown? Give a concise direct answer.', 1, max_new_tokens=4)
            answer_logprob(qa, warm, 'Question: What is shown? Give a concise direct answer.', 'a square')
            clip.begin('__warmup__')
            clip.image_vectors(warm, {})
            clip.text_vectors(['warmup'])
        sync()
        qa._prism_voi_warm = True
    sync()
    system_start = time.perf_counter()
    stats = {}
    saved = os.environ.pop('QWEN_EXACT_RECENT_DECODE', None)
    try:
        end = video_end
        if end is None:
            from lib.minicpm.oasis_memory import video_duration
            end = video_duration(video_path)
        with ptimed(stats, 'recent6_decode_ms'):
            # Same R as the Recent-6 baseline: without a question time the baseline anchors on the decoded video's end.
            recent_start = max(0.0, float(video_end) - config.recent) if video_end is not None else video_start
            recent = select_exact_current_recent_frames(qa, video_path, 1.0, 1.0, config.recent,
                                                        video_start=recent_start, video_end=video_end, tail_decode=True)
    finally:
        if saved is not None:
            os.environ['QWEN_EXACT_RECENT_DECODE'] = saved
    stats['total_visual_preparation_ms'] = (time.perf_counter() - system_start) * 1000
    stats['video_io_ms'] = None
    history = {}

    def history_chunks():
        # Decoded only when retrieval runs (a stream would hold these frames already); timed as visual preparation.
        if 'chunks' not in history:
            start = time.perf_counter()
            decode_start = max(0.0, float(end) - (config.reach + config.recent + 1.0), float(video_start or 0.0))
            history['chunks'], history['backend'] = decode_video_to_chunks_qwen(
                video_path, 1.0, 1.0, None, video_start=decode_start or None, video_end=video_end)
            history['ms'] = (time.perf_counter() - start) * 1000
        return history['chunks']

    before = base._reset_gpu_memory_peaks()
    sync()
    algorithm_start = time.perf_counter()
    clip.begin(os.path.realpath(video_path))
    import torch
    with torch.inference_mode():
        if config.method == 'streamscout':
            from lib.minicpm.streamscout import select as chooser
        else:
            chooser = {'voi': select, 'wrws': wrws_select}[config.method]
        frames, final, meta = chooser(qa, history_chunks, recent, prompt, clip, stats, config)
    sync()
    decode_ms = history.get('ms', 0.0)  # Moved from the algorithm's time to visual preparation.
    stats['broad_history_decode_ms'] = decode_ms
    stats['retrieval_total_ms'] = stats.get('retrieval_total_ms', 0.0) - decode_ms
    stats['total_visual_preparation_ms'] += decode_ms
    upkeep_ms = stats.get('stream_upkeep_ms', 0.0)  # Work a stream does as it arrives (StreamScout's captions).
    stats['PRISM_algorithmic_latency_ms'] = (time.perf_counter() - algorithm_start) * 1000 - decode_ms - upkeep_ms
    stats['full_system_latency_ms'] = (time.perf_counter() - system_start) * 1000
    stats['final_generation_ms'] = final['generate_ms']
    stats['final_ttft_ms'] = final['ttft_ms']
    stats.setdefault('retrieval_total_ms', 0.0)
    stats['vlm_selection_total_ms'] = stats.get('reference_uncertainty_ms', 0.0) + stats.get('context_uncertainty_ms', 0.0)
    stats['decision_logic_total_ms'] = 0.0
    stats['context_total_ms'] = max(0.0, final['ms'] - final['generate_ms'])
    stats['PRISM_end_to_end_latency_ms'] = stats['PRISM_algorithmic_latency_ms']
    stats['PRISM_selection_latency_ms'] = stats['PRISM_algorithmic_latency_ms'] - final['generate_ms']
    stats['request_to_answer_first_token_ms'] = stats['PRISM_selection_latency_ms'] + final['ttft_ms']
    stats['latency_accounting_residual_ms'] = 0.0
    stats['memory_maintenance_total_ms'] = upkeep_ms
    stats['final_frame_count'] = len(frames)
    stats['num_vision_tokens'] = qa._last_num_vision_tokens
    meta.update(timing=stats, latency_definition=DEFINITION, recent_chunk_ids=list(recent.final_chunk_ids),
                recent_times=list((getattr(recent, 'cdas_metadata', None) or {}).get('selected_timestamps') or []))
    profile = base._build_profile(mode=meta['mode'], decode_time=stats['total_visual_preparation_ms'] / 1000,
                                  selection_time=stats['PRISM_selection_latency_ms'] / 1000,
                                  generate_time=final['generate_ms'] / 1000, before_memory=before,
                                  after_memory=base._capture_gpu_memory(), qa=qa)
    profile['adaptive'] = meta
    stats['gpu_peak_allocated_mb'] = profile['gpu_peak_allocated_mb']
    stats['gpu_peak_reserved_mb'] = profile['gpu_peak_reserved_mb']
    profile['progressive_arbitration'] = stats
    profile['end_to_end_time_seconds'] = stats['full_system_latency_ms'] / 1000
    result = base.RecentWindowResult(answer=final['text'], final_chunk_ids=list(recent.final_chunk_ids),
                                     generate_time=final['generate_ms'] / 1000, ttft_seconds=final['ttft_ms'] / 1000,
                                     num_vision_tokens=qa._last_num_vision_tokens,
                                     num_vision_tokens_before=qa._last_num_vision_tokens,
                                     num_vision_tokens_after=qa._last_num_vision_tokens, num_frames=len(frames))
    result.profile_metadata = profile
    result.adaptive_metadata = meta
    contexts = meta.pop('_contexts')
    if config.trace and config.signals:  # After all timing: StreamScout-style self-assessment of every traced context.
        for row in meta['trace']:
            row['p_insufficient'] = insufficiency(qa, contexts[row['depth']], prompt)
    if config.trace and meta['format'] == 'mcq':
        # After all timing: each traced context's own generated answer, so a lambda replay scores exactly what that
        # lambda would have answered (the trace's `prediction` is the option-logit argmax used for SE).
        for row in meta['trace']:
            row['answer'] = (final['text'] if row['depth'] == meta['selected_memory_depth']
                             else qa.generate_from_frames(contexts[row['depth']], prompt))
    return result, history.get('backend', 'history_not_decoded')


# --------------------------------------------------------------------------------------- competing decision signals
SUFFICIENCY_PROMPT = ("{prompt}\n\nDo not answer yet. Do the frames shown contain enough visual evidence to answer "
                      "this question? Reply with only yes or no.")


def insufficiency(qa, frames, prompt):
    """StreamScout-style self-assessment: the frozen VLM's P(no) to "is the evidence enough?" (yes/no renormalised)."""
    import torch
    from lib.minicpm.progressive_sufficiency import _build_minicpm_inputs
    tokenizer = qa.processor.tokenizer
    yes = tokenizer.encode('yes', add_special_tokens=False)[0]
    no = tokenizer.encode('no', add_special_tokens=False)[0]
    inputs = _build_minicpm_inputs(qa, frames, SUFFICIENCY_PROMPT.format(prompt=prompt))
    with torch.inference_mode():
        logits = qa.model(**inputs, use_cache=False, return_dict=True).logits[0, -1].float()
    pair = torch.softmax(torch.stack([logits[yes], logits[no]]), dim=0)
    return float(pair[1])


def wrws_fusion(similarities, times, eps=1e-6):
    """WRWS (Eqs. 2-7): normalised entropy alpha of the calibrated relevance distribution, and the fused scores
    F_i = (1 - alpha) * calibrated relevance + alpha * linear recency prior."""
    import statistics
    n = len(similarities)
    if n == 0:
        return 0.0, []
    if n == 1:
        return 0.0, [1.0]
    mu, sigma = statistics.fmean(similarities), statistics.pstdev(similarities)
    cal = [1 / (1 + math.exp(-(s - mu) / (sigma + eps))) for s in similarities]
    ordered = sorted(cal)

    def quantile(q):  # Linear interpolation, as numpy's default.
        pos = q * (n - 1)
        low = int(math.floor(pos))
        high = min(low + 1, n - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)
    iqr, std = quantile(0.75) - quantile(0.25), statistics.pstdev(cal)
    tau = iqr if iqr > eps else std if std > eps else None
    if tau is None:
        probs = [1 / n] * n
    else:
        top = max(cal)
        weights = [math.exp((c - top) / tau) for c in cal]
        total = sum(weights)
        probs = [w / total for w in weights]
    alpha = entropy(probs) / math.log(n)
    rank = {i: r for r, i in enumerate(sorted(range(n), key=lambda i: times[i]))}
    recency = [rank[i] / (n - 1) for i in range(n)]
    return alpha, [(1 - alpha) * c + alpha * r for c, r in zip(cal, recency)]


def _pool_scores(eligible, recent, question, clip, stats):
    """Relevance of every frame WRWS chooses from (memory candidates and R) to the question, with its time."""
    from lib.minicpm.progressive_sufficiency import _chunk_temporal_bounds
    pool = [(c['frame'], c['end']) for c in eligible]
    pool += [(f, _chunk_temporal_bounds(c)[1]) for f, c in zip(recent.frames, recent.selected_chunks)]
    images = clip.image_vectors([f for f, _t in pool], stats)
    text = clip.text_vectors([question])[0]
    return (images @ text).detach().float().cpu().tolist(), [t for _f, t in pool]


def wrws_select(qa, chunks, recent, prompt, clip, stats, config):
    """WRWS reimplemented on our backbone: always answer from the top `recent` frames of history + R by the fused
    score (no retrieve/stay gate and no check of the chosen frames by the VLM). Run with a long reach, min_gap 0."""
    from lib.minicpm import prism_unified as pu
    from lib.minicpm.progressive_arbitration import extract_options, timed as ptimed
    question = pu.question_text(prompt)
    with ptimed(stats, 'retrieval_total_ms'):
        eligible = memory_candidates(chunks() if callable(chunks) else chunks, recent, config)
        similarities, times = _pool_scores(eligible, recent, question, clip, stats)
        alpha, fused = wrws_fusion(similarities, times)
        frames = [c['frame'] for c in eligible] + list(recent.frames)
        keep = sorted(sorted(range(len(frames)), key=lambda i: -fused[i])[:config.recent], key=lambda i: times[i])
        size = recent.frames[0].size
        final_frames = [frames[i] if frames[i].size == size else frames[i].resize(size) for i in keep]
    start = time.perf_counter()
    answer = qa.generate_from_frames(final_frames, prompt)
    final = dict(text=answer, ms=(time.perf_counter() - start) * 1000,
                 generate_ms=qa._last_model_generate_seconds * 1000, ttft_ms=qa._last_ttft_seconds * 1000)
    n_recent = len(recent.frames)
    from_memory = [i for i in keep if i < len(frames) - n_recent]
    meta = dict(mode='wrws_reimplementation', config=asdict(config), format='mcq' if extract_options(prompt) else 'open',
                question_text=question, retrieval_entropy=alpha, pool_size=len(frames),
                selected_memory_depth=len(from_memory), memory_triggered=bool(from_memory),
                memory_accepted=bool(from_memory), memory_times=[times[i] for i in from_memory],
                selected_times=[times[i] for i in keep], trace=[], vlm_scoring_calls=0, recent_frame_hashes=None,
                baseline_control=False, _contexts={0: final_frames})
    return final_frames, final, meta
