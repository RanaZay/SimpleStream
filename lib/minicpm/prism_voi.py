"""PRISM-VoI building blocks: format-agnostic semantic entropy of a frozen VLM's answer (design: docs/prism_voi_design.md).

The question format only sets the answer hypotheses Omega(q):
  multiple-choice  Omega = the options; p(option | C) from the option-letter logits (one forward); SE = option entropy.
  open-ended       Omega = greedy answer + M sampled answers under the reference context; p(answer | C) from
                   length-normalised teacher-forced likelihood; answers with the same meaning share a cluster.
SE(C) = -sum_k p(c_k | C) log p(c_k | C) over meaning clusters c_k (Kuhn et al., ICLR 2023).
"""
from __future__ import annotations

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
    seq = ids.tolist()
    start = end = None
    for variant in (answer, ' ' + answer, answer.strip()):
        target_ids = qa.processor.tokenizer.encode(variant, add_special_tokens=False)
        for j in range(max(prefix, 1), len(seq) - len(target_ids) + 1):
            if target_ids and seq[j:j + len(target_ids)] == target_ids:
                start, end = j, j + len(target_ids)
                break
        if start is not None:
            break
    if start is None:
        raise RuntimeError(f'Could not locate the answer tokens of {answer!r} in the templated sequence')
    with torch.inference_mode():
        logits = qa.model(**inputs, use_cache=False, return_dict=True).logits[0].float()
    logp = torch.log_softmax(logits[start - 1:end - 1], dim=-1)
    target = ids[start:end].to(logp.device)
    return float(logp.gather(1, target[:, None]).mean())


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
    with torch.inference_mode():
        out = qa.model.generate(**inputs, downsample_mode=qa.downsample_mode, max_new_tokens=max_new_tokens,
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


def config_from_env():
    """Every field can be overridden as PRISM_VOI_<FIELD>, e.g. PRISM_VOI_LAM=0.3."""
    values = {}
    for f in fields(Config):
        raw = os.environ.get('PRISM_VOI_' + f.name.upper())
        if raw is not None:
            values[f.name] = type(f.default)(raw)
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
        weights = []
        for members in self.clusters:
            total = 0.0
            for a in dict.fromkeys(members):
                if (key, a) not in self.scores:
                    self.scores[(key, a)] = answer_logprob(self.qa, frames, self.prompt, a)
                    self.calls += 1
                total += math.exp(self.scores[(key, a)])
            weights.append(total)
        norm = sum(weights) or 1.0
        probs = [w / norm for w in weights]
        return entropy(probs), dict(clusters=[len(c) for c in self.clusters], probabilities=probs)


def select(qa, chunks, recent, prompt, clip, stats, config):
    """Greedy value-of-information selection: keep a memory frame only if it removes more than lam nats of SE."""
    from lib.minicpm import prism_unified as pu
    from lib.minicpm.progressive_arbitration import extract_options, timed as ptimed
    question = pu.question_text(prompt)
    options = extract_options(prompt)
    ev = Evaluator(qa, prompt, question, options, config)
    contexts = {0: list(recent.frames)}
    with ptimed(stats, 'reference_uncertainty_ms'):
        se0, info0 = ev.se(0, contexts[0])
    trace = [dict(depth=0, se=se0, **info0)]
    accepted, current, se_current, stop = [], 0, se0, 'confident'
    if se0 > config.lam:  # Otherwise no single frame could pay for itself.
        with ptimed(stats, 'retrieval_total_ms'):
            eligible = memory_candidates(chunks, recent, config)
            rank_config = replace(pu.config_from_env(), pool=max(config.max_frames, 1))
            candidates = pu.rank(eligible, question, clip, stats, rank_config)[:config.max_frames]
        stop = 'no_candidates' if not candidates else 'cap_reached'
        for j, cand in enumerate(candidates, 1):
            frame = cand['frame'] if cand['frame'].size == recent.frames[0].size else cand['frame'].resize(recent.frames[0].size)
            trial = sorted(accepted + [(cand['end'], frame, cand)], key=lambda x: x[0])
            contexts[j] = [f for _t, f, _c in trial] + list(recent.frames)
            if not options:  # The hypothesis set may grow under the new context; re-score the current one on it.
                ev.answer(j, contexts[j])
            with ptimed(stats, 'context_uncertainty_ms'):
                se_trial, info = ev.se(j, contexts[j])
                if not options:
                    se_current, _ = ev.se(current, contexts[current])
            gain = se_current - se_trial
            ok = gain > config.lam
            trace.append(dict(depth=j, se=se_trial, gain=gain, accepted=ok, distance=cand['distance'], **info))
            if not ok:
                stop = 'gain_below_lambda'
                break
            accepted, current, se_current = trial, j, se_trial
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
                memory_distances=[c['distance'] for _t, _f, c in accepted], vlm_scoring_calls=ev.calls,
                recent_frame_hashes=None, baseline_control=False)
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
        with ptimed(stats, 'broad_history_decode_ms'):  # Only the memory store's reach is decoded.
            decode_start = max(0.0, float(end) - (config.reach + config.recent + 1.0), float(video_start or 0.0))
            chunks, backend = decode_video_to_chunks_qwen(video_path, 1.0, 1.0, None,
                                                          video_start=decode_start or None, video_end=video_end)
        with ptimed(stats, 'recent6_decode_ms'):
            recent = select_exact_current_recent_frames(qa, video_path, 1.0, 1.0, config.recent,
                                                        video_start=max(0.0, float(end) - config.recent), video_end=video_end)
    finally:
        if saved is not None:
            os.environ['QWEN_EXACT_RECENT_DECODE'] = saved
    stats['total_visual_preparation_ms'] = (time.perf_counter() - system_start) * 1000
    stats['video_io_ms'] = None
    before = base._reset_gpu_memory_peaks()
    sync()
    algorithm_start = time.perf_counter()
    clip.begin(os.path.realpath(video_path))
    import torch
    with torch.inference_mode():
        frames, final, meta = select(qa, chunks, recent, prompt, clip, stats, config)
    sync()
    stats['PRISM_algorithmic_latency_ms'] = (time.perf_counter() - algorithm_start) * 1000
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
    stats['memory_maintenance_total_ms'] = 0.0
    stats['final_frame_count'] = len(frames)
    stats['num_vision_tokens'] = qa._last_num_vision_tokens
    meta.update(timing=stats, latency_definition=DEFINITION, recent_chunk_ids=list(recent.final_chunk_ids))
    profile = base._build_profile(mode=MODE, decode_time=stats['total_visual_preparation_ms'] / 1000,
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
    return result, backend
