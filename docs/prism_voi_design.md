# PRISM-VoI: memory selection as value of information (design draft)

Status: design for review, not yet implemented. Replaces the hand-set thresholds of the format-agnostic
controller (`lib/minicpm/prism_unified.py`) with one decision rule.

## Idea

Add a past frame to the context only if it reduces the model's uncertainty about the answer by more than the
frame costs to process.

## Setup

- Question q at time t_q. The context always contains the exact recent window R (Recent-6).
- Memory store M: frames streamed before R, within a storage budget (e.g. last 30 s at 1 fps). Resource limit,
  not an accuracy knob.
- Frozen VLM gives p(a | C, q) for any context C.

## Format-agnostic uncertainty: semantic entropy

SE(C) = - sum_k p(c_k | C) log p(c_k | C), over meaning clusters c_k of the candidate answers Omega(q)
(Kuhn et al., ICLR 2023; Farquhar et al., Nature 2024). The question format only sets Omega(q):

| Format | Omega(q) | p(answer \| C) | Clusters |
|---|---|---|---|
| Multiple-choice | options | softmax over option-letter logits (one forward) | one per option (SE = option entropy) |
| Open-ended | greedy + M sampled answers under R, plus new answers under added frames | length-normalised teacher-forced likelihood (batched) | normalised match, then meaning-equivalence check |

## Decision rule

c = cost of one added frame (latency units). For candidate h:
DeltaSE(h | C) = SE(C) - SE(C u {h}); accept h iff DeltaSE(h | C) > lambda * c.

```
C <- R; compute SE(R)
if SE(R) <= lambda * c: answer from R          # no retrieval: even a perfect frame cannot pay for itself
order memory frames by question relevance (MobileCLIP)   # search order only
for h in candidates[:K]:
    if DeltaSE(h | C) > lambda * c: C <- C u {h}
    else: stop
answer from C
```

lambda -> infinity recovers Recent-6 / SimpleStream exactly; lambda = 0 adds frames up to the cap K.

## What it replaces

| Current (hand-set) | PRISM-VoI |
|---|---|
| tau_S | derived: retrieve iff SE(R) > lambda*c |
| alpha, beta, gamma | one measure (semantic entropy) |
| tau_M, delta_S, epsilon | DeltaSE > lambda*c |
| temporal-consistency guard | answer flips that raise uncertainty are rejected automatically |
| MMR lambda | redundant frames add ~0 information and are rejected |
| d_min / d_max band | memory store size (storage budget) |
| rollback | implicit: only accepted frames are kept |
| separate MCQ / open-ended paths | one rule; format only sets Omega(q) |

Remaining parameters: lambda (accuracy-compute trade-off, reported as a full curve), K (compute cap), memory
store size (storage budget), M (estimation budget). None is an accuracy threshold tuned on a benchmark.

## Justification

- Error is bounded by entropy (Hellman-Raviv: P_err <= H/2 in bits), so reducing SE at a stated price trades an
  upper bound on error against compute: a value-of-information decision.
- With diminishing returns (approximately submodular gain), greedy selection is within a constant factor of the
  best subset (Krause & Guestrin), which justifies stopping at the first rejected frame.

## Related work

- FLARE, DRAGIN: uncertainty decides when to retrieve; we also verify that each piece of evidence reduced
  uncertainty and price it against compute.
- Semantic entropy: used for hallucination detection; we use its reduction as the value of memory evidence.
- ShallowStream, OASIS: prompted/calibrated routers or tool calls; our gate comes from answer uncertainty with an
  explicit cost.
- SimpleStream: the lambda -> infinity special case.

## Cost per question

- Under R: 1 greedy answer + M samples (one batched generation) + scoring Omega (one batched forward).
- Per tried frame: scoring Omega under C u {h} (one forward) + 1 greedy answer (to catch new answers).
- Candidate ordering: MobileCLIP embeddings cached per frame on arrival.

## Experiments

1. Main tables: one PRISM-VoI variant vs Recent-6, OASIS, ShallowStream on OVO-Bench, StreamingBench,
   StreamBench, with paired significance tests.
2. Accuracy-latency curve over lambda.
3. OVO Backward focus.
4. Ablations: SE vs token entropy vs no verification; M; clustering; candidate ordering; K.
5. Calibration: does SE(R) predict Recent-6 errors?

## Risks

- Small-model calibration: measured first (step 1); change course if SE(R) does not track errors.
- Open-ended SE needs samples (latency); measure the smallest useful M.
- Gains may remain small; the lambda curve and significance tests report this either way.

## Build order

1. Calibration study on held-out data (cheap, decisive).
2. Implement PRISM-VoI mode (semantic entropy for both formats, MobileCLIP ordering).
3. Small lambda sweep on held-out split; then full single-variant runs.
4. Ablations and lambda curve.

Mobile deployment is handled by another contributor; we hand over the final model and configuration.
