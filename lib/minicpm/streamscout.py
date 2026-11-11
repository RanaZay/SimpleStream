"""StreamScout zero-shot cascade (arXiv 2609.00291, training-free variant) reimplemented on our backbone.

Stream memory (Eqs. 1, 3): one caption per Delta = 16 s window from uniformly spaced frames, and a cache of at most
B = 64 raw frames (on overflow the frame in the most temporally dense region is evicted; one pass, never revisited).
Views (Eq. 4), nested: F1 = the n1 = 4 most recent cached frames (glance); F2 = F1 + n2 = 8 frames at stride 32 s
back from the present (look-back); F3 = F2 + the peak frames of the n3 = 12 most salient watershed segments of the
question-relevance curve (retrieval). Frames are woven into the caption timeline at their timestamps. At each view
the VLM either answers or replies <need_more_context/>; the last view must answer.

Not reproduced: the rolling QA summary (each benchmark question is answered independently here) and the trained
StreamScout-S / -R decision policies (no code or weights were released).
"""
from __future__ import annotations

import math
import time

WINDOW = 16.0
CACHE = 64
VIEWS = (4, 8, 12)
STRIDE = 32.0
CAPTION_FRAMES = 4
CAPTION_TOKENS = 48
ESCALATE = '<need_more_context/>'
CAPTION_PROMPT = ('Describe what happens in these frames in one or two sentences: the main people, objects, actions '
                  'and any visible text.')
DECIDE = ('\n\nYou are watching a live video stream. Above is a timeline of captions of the stream so far, with some '
          'frames shown at their times. If this is enough to answer, answer the question as instructed. If it is '
          f'not enough, reply with exactly {ESCALATE} and nothing else.')

_captions = {}  # (video, window start) -> caption; a stream captions each window once.


def evict_dense(times, budget=CACHE):
    """One-pass cache over arrival times: on overflow drop the frame whose neighbours are closest together."""
    cache = []
    for t in sorted(times):
        cache.append(t)
        if len(cache) > budget:
            # Interior frames only: the oldest and the newest bound the covered span.
            i = min(range(1, len(cache) - 1), key=lambda k: (cache[k + 1] - cache[k - 1], cache[k]))
            del cache[i]
    return cache


def look_back(cache_times, now, exclude, n=VIEWS[1], stride=STRIDE):
    """The cached frame nearest to each of now - stride, now - 2 stride, ... (n targets, distinct, not in exclude)."""
    out = []
    for k in range(1, n + 1):
        free = [t for t in cache_times if t not in exclude and t not in out]
        if not free:
            break
        target = now - k * stride
        out.append(min(free, key=lambda t: (abs(t - target), -t)))
    return out


def watershed_peaks(times, scores, n=VIEWS[2], exclude=()):
    """Peak frames of the n most salient segments of the relevance curve (segments split at local minima;
    salience = peak height above the higher bounding minimum), skipping excluded frames; top relevance fills up."""
    order = sorted(range(len(times)), key=lambda i: times[i])
    s = [scores[i] for i in order]
    cuts = [0] + [j for j in range(1, len(s) - 1) if s[j] <= s[j - 1] and s[j] <= s[j + 1]] + [len(s) - 1]
    segments = []
    for a, b in zip(cuts, cuts[1:]):
        inner = [j for j in range(a, b + 1) if times[order[j]] not in exclude]
        if not inner:
            continue
        peak = max(inner, key=lambda j: s[j])
        segments.append((s[peak] - max(s[a], s[b]), s[peak], times[order[peak]]))
    picked = []
    for _sal, _height, t in sorted(segments, key=lambda x: (-x[0], -x[1])):
        if t not in picked:
            picked.append(t)
        if len(picked) == n:
            return picked
    for i in sorted(range(len(times)), key=lambda i: -scores[i]):
        if times[i] not in exclude and times[i] not in picked:
            picked.append(times[i])
        if len(picked) == n:
            break
    return picked


def _stamp(t):
    return f'{int(t // 60):02d}:{t % 60:04.1f}'


def woven_content(captions, frames, prompt):
    """Captions and frames in time order, then the question."""
    items = [(start, 0, {'type': 'text', 'text': f'[{_stamp(start)}-{_stamp(start + WINDOW)}] {text}\n'})
             for start, text in captions]
    for t, image in frames:
        items.append((t, 1, {'type': 'text', 'text': f'[frame at {_stamp(t)}]'}))
        items.append((t, 2, {'type': 'image', 'image': image}))
    content = [item for _t, _k, item in sorted(items, key=lambda x: (x[0], x[1]))]
    content.append({'type': 'text', 'text': prompt})
    return content


def generate(qa, content):
    """qa.generate_from_frames with an interleaved message (same template, processor options and decoding)."""
    messages = [{'role': 'user', 'content': content}]
    template = dict(tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
    options = dict(downsample_mode=qa.downsample_mode, max_slice_nums=qa.max_slice_nums, use_image_id=False)
    try:
        inputs = qa.processor.apply_chat_template(messages, **template, processor_kwargs=options)
    except TypeError:
        inputs = qa.processor.apply_chat_template(messages, **template, **options)
    inputs = inputs.to(qa.model.device)
    qa._last_num_vision_frames = sum(c['type'] == 'image' for c in content)
    qa._last_num_vision_tokens = qa._estimate_vision_tokens(inputs)
    return qa._generate_from_model_inputs(prompt_length=int(inputs['input_ids'].shape[1]), **inputs)


def caption_windows(qa, video, frames_by_time, now, stats):
    """Captions of every completed window before `now`; new windows are captioned once and their time recorded."""
    out, new_ms = [], 0.0
    start = 0.0
    times = sorted(frames_by_time)
    while start + WINDOW <= now:
        key = (video, start)
        if key not in _captions:
            inside = [t for t in times if start <= t < start + WINDOW]
            if inside:
                step = len(inside) / CAPTION_FRAMES
                picks = sorted({inside[min(len(inside) - 1, int(step * (k + 0.5)))] for k in range(CAPTION_FRAMES)})
                begin = time.perf_counter()
                saved = qa.max_new_tokens
                qa.max_new_tokens = CAPTION_TOKENS
                try:
                    _captions[key] = qa.generate_from_frames([frames_by_time[t] for t in picks], CAPTION_PROMPT).strip()
                finally:
                    qa.max_new_tokens = saved
                new_ms += (time.perf_counter() - begin) * 1000
            else:
                _captions[key] = None
        if _captions[key]:
            out.append((start, _captions[key]))
        start += WINDOW
    stats['stream_upkeep_ms'] = stats.get('stream_upkeep_ms', 0.0) + new_ms
    stats['captions_new'] = stats.get('captions_new', 0) + int(new_ms > 0)
    return out


def select(qa, chunks, recent, prompt, clip, stats, config):
    from lib.minicpm import prism_unified as pu
    from lib.minicpm import prism_voi as pv
    from lib.minicpm.progressive_arbitration import extract_options, timed as ptimed
    question = pu.question_text(prompt)
    history = pv.memory_candidates(chunks() if callable(chunks) else chunks, recent, config)
    similarities, times = pv._pool_scores(history, recent, question, clip, stats)
    images = [c['frame'] for c in history] + list(recent.frames)
    size = recent.frames[0].size
    frames_by_time = {t: (im if im.size == size else im.resize(size)) for t, im in zip(times, images)}
    relevance = dict(zip(times, similarities))
    now = max(times)
    captions = caption_windows(qa, clip.video, frames_by_time, now, stats)
    with ptimed(stats, 'retrieval_total_ms'):
        cache = evict_dense(times)
        f1 = sorted(cache)[-VIEWS[0]:]
        f2 = f1 + look_back(cache, now, set(f1))
        f3 = f2 + watershed_peaks(cache, [relevance[t] for t in cache], exclude=set(f2))
    views = [sorted(f1), sorted(f2), sorted(f3)]
    trace, final = [], None
    for depth, view in enumerate(views):
        last = depth == len(views) - 1
        content = woven_content(captions, [(t, frames_by_time[t]) for t in view], prompt if last else prompt + DECIDE)
        start = time.perf_counter()
        text = generate(qa, content)
        ms = (time.perf_counter() - start) * 1000
        escalate = not last and ESCALATE.strip('</>') in text
        trace.append(dict(depth=depth, frames=len(view), escalate=escalate, ms=ms, answer=text))
        if not escalate:
            answer = text.replace('<answer>', '').replace('</answer>', '').strip()
            final = dict(text=answer, ms=ms, generate_ms=qa._last_model_generate_seconds * 1000,
                         ttft_ms=qa._last_ttft_seconds * 1000)
            break
    chosen = views[depth]
    meta = dict(mode='streamscout_zero_shot', config=pv.asdict(config),
                format='mcq' if extract_options(prompt) else 'open', question_text=question, trace=trace,
                stop_view=depth + 1, generations=len(trace), captions=len(captions), cache_size=len(cache),
                selected_times=chosen, selected_memory_depth=sum(t not in set(f1) for t in chosen),
                memory_triggered=depth > 0, memory_accepted=depth > 0, vlm_scoring_calls=0,
                recent_frame_hashes=None, baseline_control=False,
                _contexts={0: [frames_by_time[t] for t in chosen]})
    return [frames_by_time[t] for t in chosen], final, meta
