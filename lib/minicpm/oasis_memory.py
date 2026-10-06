"""Recent-6 + OASIS hierarchical event memory on frozen MiniCPM-V-4.6 (published-memory baseline).

OASIS: Liang et al., "On-Demand Hierarchical Event Memory for Streaming Video Reasoning",
CVPR 2026, arXiv:2604.17052. Ported from github.com/Solus-sano/OASIS at dbd342c.

Follows the paper (Sec. 3-4.2, Alg. 1, Figs. 7-10) where it and the release differ: medium buffer of
32 s at 1 fps, event forest (16 keyframes per node, root budget 4, merge score cos - 0.1 (d_j + d_k)),
QA history + QA summary, Qwen3-Embedding-0.6B retrieval of k_f = 2 events (ancestors and descendants
pruned) and k_q = 1 QA pair with the generated query I_i, the two-phase coarse -> tool call -> fine
policy with the final answer from the fine stage, and the paper's prompts. Changed for a matched
comparison only: the 2 fps NowWindow is the exact Recent-6 context used by PRISM and its control, and
MiniCPM-V-4.6 replaces Qwen3-VL in every call (greedy decoding; the release samples at T=0.1, top-p=0.001).

Timing scopes (ms per question):
  visual preparation  incremental 1 fps stream decode + exact Recent-6 decode
  memory maintenance  event summaries, root merges and QA-summary updates (asynchronous in the
                      paper; executed inline here and reported separately, never hidden)
  algorithmic         question arrival -> final answer: coarse call, retrieval, fine call
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
import re
import time
from typing import Any

from lib.minicpm.progressive_arbitration import extract_options, image_key, sync, timed

MODE = "recent6_oasis_hierarchical_event_memory"
DEFINITION = ("OASIS algorithmic latency runs from question arrival (memory already up to date) to the final answer: "
              "coarse MiniCPM call, embedding retrieval and the fine MiniCPM call when triggered. It excludes video "
              "decoding and memory maintenance, which are reported separately.")
SYSTEM_DEFINITION = ("Full-system latency includes incremental stream decoding, exact Recent-6 decoding, event "
                     "summarization/merging, the query itself and the post-answer QA-summary update.")


@dataclass(frozen=True)
class Config:
    recent: int = 6
    buffer_fps: float = 1.0
    buffer_frames: int = 32
    frames_per_node: int = 16
    root_limit: int = 4
    merge_lambda: float = 0.1
    event_k: int = 2
    qa_k: int = 1
    max_new_tokens: int = 1024
    decode_window_seconds: float = 300.0
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"


def config_from_env():
    return Config(max_new_tokens=int(os.environ.get('OASIS_MAX_NEW_TOKENS', '1024')),
                  buffer_frames=int(os.environ.get('OASIS_BUFFER_FRAMES', '32')),
                  embedding_model=os.environ.get('OASIS_EMBEDDING_MODEL', 'Qwen/Qwen3-Embedding-0.6B'))


QUERY_SYSTEM_PROMPT = """
You are an expert multimodal assistant on virtual reality headset for streaming video QA.
You are looking at a video of a real-world scene, and you are answering real-time questions for user

## Inputs (separate fields)
- now_window_frames: fine-grained frames from NowWindow, very recent video frames that the headset is looking at.
- short_term_frames: fine-grained frames from ShortTermWindow.
- long_term_events: textual summaries (and/or coarse frames) of longer segments.
- qa_history_summary: summary of prior Q/A.

## Decision Policy
- Read the user question + provided frames/summaries, briefly think step-by-step about the question
- After thinking, you can call `rag_retrieval` with a precise interest description, the tool will return the specific video
clips and question-and-answer history that are most relevant to your description.
- Once you confirm your final answer, place the final answer inside <answer> and </answer>.

## Forming the Retrieval Query
- Be specific and brief (≤ 20 words), prefer noun phrases.
- Include disambiguators when available: who/what, action, location/region, salient attributes (color/count).
- Avoid vague queries ("more info", "look again").

## Tool
You may fetch missing details by issuing a concise interest description (entity, action, time anchor, location, attributes).
<tools>
{
  "type": "function",
  "function": {
    "name_for_human": "rag_retrieval",
    "name": "rag_retrieval",
    "description": "Retrieve details based on a concise interest description.",
    "parameters": {
      "type": "object",
      "properties": {
        "text_input": {
          "type": "string",
          "description": "Short, specific description to retrieve."
        }
      },
      "required": ["text_input"]
    }
  }
}
</tools>

## Tool Call Format (example)
<tool_call>
{"name": "rag_retrieval", "arguments": {"text_input": "man opens car trunk, parking lot"}}
</tool_call>
"""

TOOL_CALL_PROMPT = """
Based on all the above information, answer the question inside <answer> and </answer>. DO NOT call function tool again.
"""

SUMMARY_PROMPT = """
You are an event summarizer for a Short-Term Memory (STM) video window.

Goal
- Produce ONE self-contained summary describing what happens inside this STM window only.

Inputs
- STM frames (authoritative evidence).

Hard Rules
1) Chronology: Narrate in temporal order within the STM window. No reordering across time.
2) No guessing: Do not infer intentions/causes not shown. If something is unclear, state "unidentified/unclear" rather than guessing.

Content Focus
- Who did what to whom/what, where, with what tool/object, and the immediate result.
- Include objects visible in STM

Style
- Active voice; present or simple past; concrete, observable facts.
- No titles, lists, timestamps, metadata, or markup.
- Length ≤ 300 words.

Output
- Output ONLY the summary text.
"""

MERGE_PROMPT = """
You are a summary merger.
I'm giving you two summaries and their timestamps, each describing the content of two adjacent clips from a video.
You need to merge them into one. Rules:

1. Do not add any new information that is not already present in either summary.
2. Maintain chronological order and keep the total word count under 300.

Here are the two summaries:
{summary_a}
{summary_b}

Now output your final summary directly, without any additional explanation or title, and you do not need to output the timestamps at the beginning:
"""

UPDATE_QA_SUMMARY_PROMPT = """
You are a QA aggregator. You receive the current QA history summary S and a new QA. Your task is to generate an updated S' for subsequent retrieval and low-cost reasoning.

Hard Rules:
1) Only use information from S and the new QA; no external knowledge or assumptions should be introduced.
2) Preserve the "who/what/key changes"; resolve pronouns and unify entity names.
3) De-duplicate and merge duplicate or synonymous statements; and remove redundant and irrelevant content.
4) Keep the total length to under 300 words.
5) Output only the updated summary text, without any explanations, titles, or additional notes.

Given the QA history summary S:
{QA_summary_all}

Given the new QA (including questions and answers):
{QA}

Now output the updated summary:
"""

QUESTION_TEMPLATE_MC = (
    "{question}\n\n"
    "Please briefly think step-by-step about this question. Keep your reasoning under 100 words.\n"
    "After thinking, You SHOULD CALL `rag_retrieval` to retrieve specific details from long-term "
    "memory to ensure the accuracy of your answers, unless the question explicitly asks about the "
    "content at the current moment(now, currently, etc.) or background(encyclopedic) knowledge "
    "unrelated to the video scene.\n"
    "Once you confirm your final answer, place the final answer inside <answer> and </answer>.\n"
    "Please provide only the single option letter (e.g., A, B, C, D, etc.) within the "
    "<answer> </answer> tags."
)

QUESTION_TEMPLATE_OPEN = (
    "Here is the question:\n{question}\n\n"
    "Please briefly think step-by-step about this question. Keep your reasoning under 100 words.\n"
    "After thinking, You SHOULD CALL `rag_retrieval` to retrieve specific details from long-term "
    "memory to ensure the accuracy of your answers, unless the question explicitly asks about the "
    "content at the current moment(now, currently, etc.) or the question is related to general knowledge.\n"
    "Once you confirm your final answer, place the final answer inside <answer> and </answer>."
)

# Benchmark wrappers ask for a bare letter; OASIS asks for reasoning, a tool call, then a tagged letter.
_PREAMBLE = re.compile(r'^You are an advanced video question-answering AI assistant\..*?(?=Question:)', re.S)
_MC_FORMAT = re.compile(r'\n\s*Only give the best option\'s letter[^\n]*\s*$', re.I)
_TOOL_CALL = re.compile(r'<tool_call>(.*?)</tool_call>', re.S)
_ANSWER = re.compile(r'<answer>\s*(.*?)\s*</answer>', re.S)


def build_query(prompt):
    """Return (OASIS question text, is_multiple_choice) from an evaluator prompt."""
    options = extract_options(prompt)
    if options:
        question = _MC_FORMAT.sub('', _PREAMBLE.sub('', prompt.strip())).strip()
        return QUESTION_TEMPLATE_MC.format(question=question), True
    # Open formats (OVO REC/SSR/CRR) keep their answer-format instruction: it defines the task.
    return QUESTION_TEMPLATE_OPEN.format(question=prompt.strip()), False


def parse_answer(response, multiple_choice):
    """The <answer> tag is the OASIS output contract; fallbacks only rescue format slips and are logged."""
    match = _ANSWER.search(response)
    if match:
        return match.group(1).strip(), 'answer_tag'
    if multiple_choice:
        found = re.findall(r'(?:answer|option)\s*(?:is|:)?\s*\(?([A-E])\b', response, flags=re.I)
        if found:
            return found[-1].upper(), 'answer_phrase'
        stripped = _TOOL_CALL.sub('', response).strip()
        if re.fullmatch(r'\(?[A-E]\)?\.?', stripped):
            return stripped.strip('().'), 'bare_letter'
        # Untagged reasoning must not be letter-scanned: "a" would read as option A.
        return '', 'unparsed'
    return _TOOL_CALL.sub('', response).strip(), 'untagged_text'


def parse_tool_query(block):
    """Retrieval query from a <tool_call> body; tolerates format slips a strict json.loads rejects."""
    try:
        call = json.loads(block)
        arguments = call.get('arguments', call) if isinstance(call, dict) else None
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if isinstance(arguments, dict) and isinstance(arguments.get('text_input'), str):
            return arguments['text_input'], 'json'
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    match = re.search(r'text_input[\'"]?\s*[:=]\s*[\'"]([^\'"]+)[\'"]', block)
    if match:
        return match.group(1), 'text_input_regex'
    plain = re.sub(r'[{}\[\]"]|\brag_retrieval\b|\bname\b|\barguments\b|:', ' ', block).strip(' ,\n')
    return (plain, 'plain_text') if plain else (None, 'unparsed')


def hhmmss(t):
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def uniform_keyframes(items, count):
    if len(items) <= count:
        return list(items)
    step = max(1, len(items) // count)
    return list(items[::step][:count])


@dataclass
class Packet:
    t: float
    frame: Any


@dataclass
class Node:
    id: str
    t_start: float
    t_end: float
    level: int
    summary: str
    embedding: Any
    frames: list
    parent: str | None = None
    children: list = field(default_factory=list)


@dataclass
class QA:
    t: float
    question: str
    answer: str
    embedding: Any


class Embedder:
    """Qwen3-Embedding with sentence-transformers' default pooling: last token, L2-normalized, no prompt."""
    def __init__(self, path, device):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left')
        self.model = AutoModel.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
        self.device = device

    def encode(self, texts):
        import torch
        with torch.inference_mode():
            batch = self.tokenizer(texts, padding=True, truncation=True, max_length=8192,
                                   return_tensors='pt').to(self.device)
            hidden = self.model(**batch).last_hidden_state[:, -1]
            return torch.nn.functional.normalize(hidden.float(), dim=-1).cpu()


class State:
    """One video's OASIS memory: ShortMemory buffer, event forest and QA memory."""
    def __init__(self, video, config):
        self.video = video
        self.config = config
        self.last_t = -math.inf
        self.buf, self.buf_for_event = [], []
        self.nodes, self.roots = {}, []
        self.qas, self.qa_summary = [], 'No QA history yet.'
        self.events_created = self.merges = 0

    def push(self, packet):
        """OASIS ShortMemory.push without the NowWindow (replaced by exact Recent-6)."""
        cuts = []
        # 1 ms slack: decoded 1 fps timestamps can land a hair under the 1 s spacing.
        if not self.buf or packet.t - self.buf[-1].t >= 1.0 / self.config.buffer_fps - 1e-3:
            self.buf.append(packet)
            self.buf_for_event.append(packet)
        if len(self.buf) > self.config.buffer_frames:  # Paper: 32 s at 1 fps (the release kept 31).
            self.buf.pop(0)
        if len(self.buf_for_event) >= self.config.buffer_frames:
            cuts.append(self.buf_for_event)
            self.buf_for_event = []
        return cuts

    def retrieve_events(self, query_vector):
        """Algorithm 1: greedy top-k_f with ancestors and descendants pruned."""
        ids = list(self.nodes)
        if not ids:
            return []
        import torch
        scores = (torch.stack([self.nodes[i].embedding for i in ids]) @ query_vector).tolist()
        masked, selected = set(), []
        def down(node_id):
            masked.add(node_id)
            for child in self.nodes[node_id].children:
                down(child)
        def up(node_id):
            masked.add(node_id)
            if self.nodes[node_id].parent is not None:
                up(self.nodes[node_id].parent)
        for index in sorted(range(len(ids)), key=lambda i: -scores[i]):
            if len(selected) >= self.config.event_k or len(masked) == len(ids):
                break
            if ids[index] in masked:
                continue
            selected.append(self.nodes[ids[index]])
            down(ids[index])
            up(ids[index])
        return selected

    def retrieve_qas(self, query_vector):
        if not self.qas:
            return []
        import torch
        scores = (torch.stack([q.embedding for q in self.qas]) @ query_vector).tolist()
        order = sorted(range(len(self.qas)), key=lambda i: -scores[i])
        return [self.qas[i] for i in order[:self.config.qa_k]]


def _render(qa, messages):
    """MiniCPM inputs for a chat; folds the system turn into the first user turn if the template drops it."""
    if getattr(qa, '_oasis_system_supported', True) is False and messages[0]['role'] == 'system':
        system = messages[0]['content'][0]['text']
        first = dict(messages[1], content=[{'type': 'text', 'text': system}, *messages[1]['content']])
        messages = [first, *messages[2:]]
    template = dict(tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
    options = dict(downsample_mode=qa.downsample_mode, max_slice_nums=qa.max_slice_nums, use_image_id=False)
    try:
        inputs = qa.processor.apply_chat_template(messages, **template, processor_kwargs=options)
    except TypeError:
        inputs = qa.processor.apply_chat_template(messages, **template, **options)
    return inputs.to(qa.model.device)


def generate(qa, messages, max_new_tokens):
    """One frozen MiniCPM call with the wrapper's greedy decoding, TTFT streamer and profiling."""
    sync()
    start = time.perf_counter()
    inputs = _render(qa, messages)
    sync()
    preprocess_ms = (time.perf_counter() - start) * 1000
    saved = qa.max_new_tokens
    qa.max_new_tokens = max_new_tokens
    try:
        text = qa._generate_from_model_inputs(prompt_length=int(inputs['input_ids'].shape[1]),
                                              downsample_mode=qa.downsample_mode, **inputs)
    finally:
        qa.max_new_tokens = saved
    qa._last_preprocess_seconds = preprocess_ms / 1000
    qa._last_num_vision_tokens = qa._estimate_vision_tokens(inputs)
    qa._last_num_vision_frames = sum(isinstance(c, dict) and c.get('type') == 'image'
                                     for m in messages for c in m['content'])
    return text, dict(preprocess_ms=preprocess_ms, generate_ms=qa._last_model_generate_seconds * 1000,
                      ttft_ms=qa._last_ttft_seconds * 1000, generated_tokens=qa._last_generated_tokens,
                      prompt_tokens=int(inputs['input_ids'].shape[1]), vision_tokens=qa._last_num_vision_tokens,
                      images=qa._last_num_vision_frames)


def text_call(qa, prompt, config):
    return generate(qa, [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}], config.max_new_tokens)


def _add(stats, key, value):
    stats[key] = (stats.get(key) or 0.0) + value


def summarize_event(qa, embedder, source, config, stats):
    with timed(stats, 'memory_event_summary_ms'):
        messages = [{'role': 'system', 'content': [{'type': 'text', 'text': SUMMARY_PROMPT}]},
                    {'role': 'user', 'content': [{'type': 'text', 'text': 'Here is some short-term memory frames: \n'},
                                                 *({'type': 'image', 'image': p.frame} for p in source),
                                                 {'type': 'text', 'text': 'Summarize the short-term memory video.'}]}]
        summary, call = generate(qa, messages, config.max_new_tokens)
    _add(stats, 'memory_generated_tokens', call['generated_tokens'])
    with timed(stats, 'memory_embedding_ms'):
        vector = embedder.encode([summary])[0]
    return summary, vector


def merge_pair(qa, embedder, state, a, b, stats):
    frames = uniform_keyframes(a.frames + b.frames, (len(a.frames) + len(b.frames)) // 2)
    prompt = MERGE_PROMPT.format(summary_a=f"time(s)[{a.t_start:.1f} - {a.t_end:.1f}]: {a.summary}",
                                 summary_b=f"time(s)[{b.t_start:.1f} - {b.t_end:.1f}]: {b.summary}")
    with timed(stats, 'memory_merge_ms'):
        summary, call = text_call(qa, prompt, state.config)
    _add(stats, 'memory_generated_tokens', call['generated_tokens'])
    with timed(stats, 'memory_embedding_ms'):
        vector = embedder.encode([summary])[0]
    node = Node(id=f"ev{len(state.nodes)}_{frames[0].t:.1f}_{frames[-1].t:.1f}",
                t_start=min(a.t_start, b.t_start), t_end=max(a.t_end, b.t_end), level=max(a.level, b.level) + 1,
                summary=summary, embedding=vector, frames=frames, children=[a.id, b.id])
    a.parent = b.parent = node.id  # Paper Algorithm 1 prunes ancestors; the release never set parents.
    state.merges += 1
    return node


def insert_root(qa, embedder, state, node, stats):
    import torch
    state.nodes[node.id] = node
    state.roots.append(node.id)
    while len(state.roots) > state.config.root_limit:
        roots = [state.nodes[r] for r in state.roots]
        scores = [float(torch.dot(roots[i].embedding, roots[i + 1].embedding))
                  - state.config.merge_lambda * (roots[i].level + roots[i + 1].level) for i in range(len(roots) - 1)]
        i = max(range(len(scores)), key=scores.__getitem__)
        merged = merge_pair(qa, embedder, state, roots[i], roots[i + 1], stats)
        state.nodes[merged.id] = merged
        state.roots = state.roots[:i] + [merged.id] + state.roots[i + 2:]


def video_duration(video_path):
    try:
        import decord
        reader = decord.VideoReader(video_path)
        return len(reader) / max(float(reader.get_avg_fps()), 1e-6)
    except Exception:
        import cv2
        capture = cv2.VideoCapture(video_path)
        try:
            return capture.get(cv2.CAP_PROP_FRAME_COUNT) / max(capture.get(cv2.CAP_PROP_FPS), 1e-6)
        finally:
            capture.release()


def stream_frames(video_path, start, end, config):
    """1 fps frames in (start, end), decoded in bounded windows so the decoder's frame cap never lowers the rate."""
    from lib.shared.recent_window import decode_video_to_chunks_qwen
    packets, low = [], max(0.0, start)
    while low < end - 1e-3:
        high = min(end, low + config.decode_window_seconds)
        chunks, _ = decode_video_to_chunks_qwen(video_path, 1.0, config.buffer_fps, None,
                                                video_start=low if low > 0 else None, video_end=high)
        packets += [Packet(t, f) for c in chunks for f, t in zip(c.frames, c.frame_timestamps)]
        low = high
    packets.sort(key=lambda p: p.t)
    out = []
    for p in packets:
        if p.t > start + 1e-6 and (not out or p.t > out[-1].t + 1e-6):
            out.append(p)
    return out


def coarse_messages(state, recent, recent_times, question, query_time):
    content = [{'type': 'text', 'text': (f"Here is the NowWindow, very recent video frames that the headset is looking at, "
                                         f"time[{hhmmss(recent_times[0])} - {hhmmss(recent_times[-1])}]: \n")},
               *({'type': 'image', 'image': f} for f in recent)]
    if state.buf:
        content.append({'type': 'text', 'text': (f"Here is short-term memory video window, "
                                                 f"time[{hhmmss(state.buf[0].t)} - {hhmmss(state.buf[-1].t)}]: \n")})
        content += [{'type': 'image', 'image': p.frame} for p in state.buf]
    content.append({'type': 'text', 'text': 'Here is some long-term events summary: \n'})
    content += [{'type': 'text', 'text': f"time[{hhmmss(n.t_start)} - {hhmmss(n.t_end)}]: {n.summary}\n"}
                for n in (state.nodes[r] for r in state.roots)]
    content += [{'type': 'text', 'text': f"Here is QA history summary: \n{state.qa_summary}"},
                {'type': 'text', 'text': f"Now process the question at time[{hhmmss(query_time)}]: {question}"}]
    return [{'role': 'system', 'content': [{'type': 'text', 'text': QUERY_SYSTEM_PROMPT}]},
            {'role': 'user', 'content': content}]


def warm_up(qa, embedder):
    """Untimed: compile kernels and detect whether the MiniCPM template keeps a system turn."""
    from PIL import Image
    marker = 'OASIS-SYSTEM-TURN-CHECK'
    try:
        rendered = qa.processor.apply_chat_template(
            [{'role': 'system', 'content': [{'type': 'text', 'text': marker}]},
             {'role': 'user', 'content': [{'type': 'text', 'text': 'hi'}]}], tokenize=False, add_generation_prompt=True)
        qa._oasis_system_supported = marker in (rendered if isinstance(rendered, str) else str(rendered))
    except Exception:
        qa._oasis_system_supported = False
    print(f'[OASIS] system turn kept by MiniCPM template: {qa._oasis_system_supported}', flush=True)
    frames = [Image.new('RGB', (224, 224), (i * 20, 40, 60)) for i in range(6)]
    generate(qa, [{'role': 'user', 'content': [*({'type': 'image', 'image': f} for f in frames),
                                                {'type': 'text', 'text': 'Describe.'}]}], 8)
    generate(qa, [{'role': 'user', 'content': [{'type': 'text', 'text': 'Say hi.'}]}], 8)
    embedder.encode(['warmup'])
    sync()


def query(qa, video_path, prompt, chunk_duration, fps, recent_frames_only, video_start=None, video_end=None,
          cdas_config=None):
    del recent_frames_only, video_start, cdas_config  # OASIS streams from t = 0; Recent-6 is fixed.
    from lib.minicpm import baseline as base
    from main_experiments.minicpm_v46.streamingbench.eval_prism_exact_recent_dist import select_exact_current_recent_frames
    if chunk_duration != 1.0 or fps != 1.0:
        raise ValueError('Matched Recent-6 control requires chunk_duration=1 and fps=1')
    config = getattr(qa, '_oasis_config', None) or config_from_env()
    qa._oasis_config = config
    embedder = getattr(qa, '_oasis_embedder', None)
    if embedder is None:
        embedder = qa._oasis_embedder = Embedder(config.embedding_model, qa.model.device)  # Outside all timers.
        warm_up(qa, embedder)
    sync()
    system_start = time.perf_counter()
    stats = {}
    video = os.path.realpath(video_path)
    state = getattr(qa, '_oasis_state', None)
    end = None if video_end is None else float(video_end)
    if state is None or state.video != video or end is None or end - 1e-4 < state.last_t:
        state = qa._oasis_state = State(video, config)  # New stream, or time went backwards.
    reused_until = state.last_t
    saved = os.environ.pop('QWEN_EXACT_RECENT_DECODE', None)
    try:
        with timed(stats, 'broad_history_decode_ms'):
            # OVO clips end at the question; OASIS answers after the whole clip has streamed in.
            stream_end = end if end is not None else video_duration(video_path)
            new = stream_frames(video_path, state.last_t if math.isfinite(state.last_t) else -1.0, stream_end, config)
        start = max(0., end - 6.) if end is not None else None
        with timed(stats, 'recent6_decode_ms'):
            # Identical call to PRISM's, so the Recent-6 hashes can be verified pairwise.
            recent = select_exact_current_recent_frames(qa, video_path, 1.0, 1.0, 6, video_start=start, video_end=end)
    finally:
        if saved is not None:
            os.environ['QWEN_EXACT_RECENT_DECODE'] = saved
    query_time = (end - 1e-4) if end is not None else max([stream_end, *(p.t + 1e-3 for p in new)])
    stats['total_visual_preparation_ms'] = (time.perf_counter() - system_start) * 1000
    stats['video_io_ms'] = None

    maintenance_start = time.perf_counter()
    for packet in new:
        if packet.t >= query_time:  # OASIS answers before pushing the frame at the question time.
            break
        state.last_t = packet.t
        for source in state.push(packet):
            summary, vector = summarize_event(qa, embedder, source, config, stats)
            node = Node(id=f"ev{len(state.nodes)}_{source[0].t:.1f}_{source[-1].t:.1f}", t_start=source[0].t, t_end=source[-1].t,
                        level=0, summary=summary, embedding=vector,
                        frames=uniform_keyframes(source, config.frames_per_node))
            state.events_created += 1
            insert_root(qa, embedder, state, node, stats)
    sync()
    stats['memory_update_before_query_ms'] = (time.perf_counter() - maintenance_start) * 1000

    question, multiple_choice = build_query(prompt)
    recent_times = recent.cdas_metadata['selected_timestamps']
    before = base._reset_gpu_memory_peaks()
    sync()
    algorithm_start = time.perf_counter()
    with timed(stats, 'context_assembly_ms'):
        messages = coarse_messages(state, recent.frames, recent_times, question, query_time)
    coarse, coarse_call = generate(qa, messages, config.max_new_tokens)
    final_call, response, tool_query, events, qas = coarse_call, coarse, None, [], []
    with timed(stats, 'tool_call_parse_ms'):
        calls = _TOOL_CALL.findall(coarse)
        # The release drops non-JSON calls; MiniCPM's format slips would otherwise silently disable retrieval.
        tool_query, tool_parse = parse_tool_query(calls[0]) if calls else (None, 'no_tool_call')
    stats['tool_call_emitted'] = bool(calls)
    if tool_query is not None:
        with timed(stats, 'retrieval_total_ms'):
            vector = embedder.encode([str(tool_query)])[0]
            events = state.retrieve_events(vector)
            qas = state.retrieve_qas(vector)  # Paper Sec. 3.2: both use E(I_i); the release used the question for QAs.
        with timed(stats, 'context_assembly_ms'):
            details = [{'type': 'text', 'text': 'The following are the details retrieved from the tool: \n'},
                       {'type': 'text', 'text': 'Here is some QA history retrieved from the tool: '},
                       *({'type': 'text', 'text': f"time[{hhmmss(q.t)}]: Question: {q.question}, Answer: {q.answer}\n"}
                         for q in qas),
                       {'type': 'text', 'text': 'Here is some event details retrieved from the tool: '}]
            for node in events:
                details.append({'type': 'text', 'text': f"time[{hhmmss(node.t_start)} - {hhmmss(node.t_end)}]: "})
                details += [{'type': 'image', 'image': p.frame} for p in node.frames]
            details.append({'type': 'text', 'text': TOOL_CALL_PROMPT})
            messages += [{'role': 'assistant', 'content': [{'type': 'text', 'text': coarse}]},
                         {'role': 'user', 'content': details}]
        response, final_call = generate(qa, messages, config.max_new_tokens)
    # Eq. 6: when retrieval fires, a_i is the fine-stage output (the release took the first tag of coarse + fine).
    answer, parse_source = parse_answer(response, multiple_choice)
    sync()
    stats['PRISM_algorithmic_latency_ms'] = (time.perf_counter() - algorithm_start) * 1000
    answer_profile = (qa._last_preprocess_seconds, qa._last_model_generate_seconds, qa._last_ttft_seconds,
                      qa._last_num_vision_tokens, qa._last_num_vision_frames, getattr(qa, '_last_generated_tokens', None),
                      qa._last_component_times)
    after_memory = base._capture_gpu_memory()

    with timed(stats, 'memory_qa_update_ms'):
        qa_vector = embedder.encode([f"Question: {question}, Answer: {response}"])[0]  # e_q = E(q_i ⊕ a_i)
        state.qas.append(QA(t=query_time, question=question, answer=response, embedding=qa_vector))
        state.qa_summary, update_call = text_call(qa, UPDATE_QA_SUMMARY_PROMPT.format(
            QA_summary_all=state.qa_summary,
            QA=f"timestep: {hhmmss(query_time)}, question: {question}, answer: {response}"), config)
    _add(stats, 'memory_generated_tokens', update_call['generated_tokens'])
    stats['full_system_latency_ms'] = (time.perf_counter() - system_start) * 1000
    (qa._last_preprocess_seconds, qa._last_model_generate_seconds, qa._last_ttft_seconds, qa._last_num_vision_tokens,
     qa._last_num_vision_frames, qa._last_generated_tokens, qa._last_component_times) = answer_profile

    fine_triggered = final_call is not coarse_call
    stats.update(
        oasis_coarse_generation_ms=coarse_call['generate_ms'], oasis_coarse_ttft_ms=coarse_call['ttft_ms'],
        oasis_coarse_generated_tokens=coarse_call['generated_tokens'], oasis_coarse_prompt_tokens=coarse_call['prompt_tokens'],
        oasis_coarse_images=coarse_call['images'],
        oasis_fine_generation_ms=final_call['generate_ms'] if fine_triggered else None,
        oasis_fine_ttft_ms=final_call['ttft_ms'] if fine_triggered else None,
        oasis_fine_generated_tokens=final_call['generated_tokens'] if fine_triggered else None,
        oasis_fine_prompt_tokens=final_call['prompt_tokens'] if fine_triggered else None,
        oasis_fine_images=final_call['images'] if fine_triggered else None,
        oasis_fine_triggered=fine_triggered, oasis_retrieved_events=len(events), oasis_retrieved_qas=len(qas),
        oasis_root_count=len(state.roots), oasis_node_count=len(state.nodes), oasis_qa_history=len(state.qas) - 1,
        oasis_events_created_total=state.events_created, oasis_merges_total=state.merges,
        oasis_buffer_frames=len(state.buf), oasis_reused_stream_until_s=reused_until if math.isfinite(reused_until) else None,
        final_generation_ms=final_call['generate_ms'], final_ttft_ms=final_call['ttft_ms'],
        final_preprocess_ms=final_call['preprocess_ms'], final_num_generated_tokens=final_call['generated_tokens'],
        num_vision_tokens=final_call['vision_tokens'], final_frame_count=final_call['images'],
        selected_memory_depth=len(events), max_depth_evaluated=len(events), num_option_scoring_forwards=0,
        memory_generated_tokens=stats.get('memory_generated_tokens') or 0)
    stats.setdefault('retrieval_total_ms', 0.0)
    stats['context_total_ms'] = stats['context_assembly_ms'] + coarse_call['preprocess_ms'] + (
        final_call['preprocess_ms'] if fine_triggered else 0.0)
    stats['vlm_selection_total_ms'] = coarse_call['generate_ms'] if fine_triggered else 0.0
    stats['decision_logic_total_ms'] = stats['tool_call_parse_ms']
    stats['PRISM_selection_latency_ms'] = (stats['context_total_ms'] + stats['vlm_selection_total_ms']
                                           + stats['retrieval_total_ms'] + stats['decision_logic_total_ms'])
    stats['PRISM_end_to_end_latency_ms'] = stats['PRISM_selection_latency_ms'] + stats['final_generation_ms']
    stats['latency_accounting_residual_ms'] = stats['PRISM_algorithmic_latency_ms'] - stats['PRISM_end_to_end_latency_ms']
    stats['request_to_answer_first_token_ms'] = stats['PRISM_selection_latency_ms'] + stats['final_ttft_ms']
    for key in ('memory_event_summary_ms', 'memory_merge_ms', 'memory_embedding_ms'):
        stats.setdefault(key, 0.0)
    stats['memory_maintenance_total_ms'] = stats['memory_update_before_query_ms'] + stats['memory_qa_update_ms']

    metadata = dict(
        mode=MODE, config=asdict(config), baseline_control=False, memory_triggered=fine_triggered,
        memory_accepted=bool(events), arbitration_supported=True, multiple_choice=multiple_choice,
        answer_parse_source=parse_source, tool_query=tool_query, tool_call_parse=tool_parse, coarse_response=coarse,
        fine_response=response if final_call is not coarse_call else None,
        retrieved_event_ids=[n.id for n in events], retrieved_event_spans=[[n.t_start, n.t_end] for n in events],
        retrieved_qa_times=[q.t for q in qas], root_ids=list(state.roots),
        system_turn_supported=getattr(qa, '_oasis_system_supported', None),
        recent_frame_hashes=[image_key(f) for f in recent.frames], recent_frame_count=len(recent.frames),
        recent_chunk_ids=list(recent.final_chunk_ids), baseline_recent_equivalence=dict(baseline_recent=recent.cdas_metadata),
        query_time=query_time, timing=stats, latency_definition=DEFINITION, full_system_definition=SYSTEM_DEFINITION,
        warmup_enabled=True)
    profile = base._build_profile(mode=MODE, decode_time=stats['total_visual_preparation_ms'] / 1000,
                                  selection_time=stats['PRISM_selection_latency_ms'] / 1000,
                                  generate_time=stats['final_generation_ms'] / 1000,
                                  before_memory=before, after_memory=after_memory, qa=qa)
    profile['adaptive'] = metadata
    stats['gpu_peak_allocated_mb'] = profile['gpu_peak_allocated_mb']
    stats['gpu_peak_reserved_mb'] = profile['gpu_peak_reserved_mb']
    profile['progressive_arbitration'] = stats
    profile['end_to_end_time_seconds'] = stats['full_system_latency_ms'] / 1000
    result = base.RecentWindowResult(answer=answer, final_chunk_ids=list(recent.final_chunk_ids),
                                     generate_time=stats['final_generation_ms'] / 1000,
                                     ttft_seconds=stats['final_ttft_ms'] / 1000, num_vision_tokens=final_call['vision_tokens'],
                                     num_vision_tokens_before=final_call['vision_tokens'],
                                     num_vision_tokens_after=final_call['vision_tokens'], num_frames=final_call['images'])
    result.profile_metadata = profile
    result.adaptive_metadata = metadata
    return result, 'oasis_stream'
