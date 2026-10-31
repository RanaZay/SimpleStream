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
