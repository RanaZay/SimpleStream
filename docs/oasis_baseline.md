# Recent-6 + OASIS baseline (MiniCPM-V-4.6)

Published-memory baseline for the PRISM latency/accuracy comparison. Implementation:
`lib/minicpm/oasis_memory.py`, selected with `PRISM_OASIS=1` (mode `recent6_oasis_hierarchical_event_memory`).

## What is OASIS and what changed

OASIS (Liang et al., CVPR 2026, arXiv:2604.17052, code github.com/Solus-sano/OASIS@dbd342c) keeps a 1 fps,
32 s medium buffer, an event forest built from 32-frame segments (MLLM summary + Qwen3-Embedding-0.6B vector,
16 keyframes, at most 4 roots merged by cos - 0.1 (d_j + d_k)), and a QA history with an LLM-maintained summary.
Each question runs a coarse MiniCPM call on the short context, root summaries and QA summary; a
`<tool_call>` triggers retrieval of 2 event nodes (lineage-pruned) and 1 QA pair with the generated query,
then a fine call whose `<answer>` is final. Where the paper and release differ, the paper is followed.

Matched-comparison changes only: the NowWindow is PRISM's exact Recent-6 context (same selector call; hashes
are verified pairwise), and MiniCPM-V-4.6 (greedy) replaces Qwen3-VL for every call.

## Latency scopes

| Field | Meaning |
|---|---|
| `PRISM_end_to_end_latency_ms` | Question arrival to final answer (coarse + retrieval + fine); no decoding, no upkeep |
| `request_to_answer_first_token_ms` | Question arrival to first token of the final answer |
| `memory_maintenance_total_ms` | Event summaries, merges and QA-summary update (asynchronous in the paper) |
| `total_visual_preparation_ms` | Incremental stream decode + exact Recent-6 decode |
| `full_system_latency_ms` | All of the above |

Memory for a video is built incrementally across that video's questions on one worker, so upkeep is
charged to the question that first needs it. OVO clips are independent streams, rebuilt per question.

## Run

`OASIS_EMBEDDING_MODEL` must be a local Qwen3-Embedding-0.6B snapshot. OVO defaults to the paper's
Backward + Real-Time splits (`OVO_SPLITS`). Compare with existing PRISM runs:

```bash
python main_experiments/tools/compare_oasis_prism.py --benchmark streamingbench \
  --prism <prism_dir> --oasis <oasis_dir> [--recent6 <control_dir>] --output reports/oasis_vs_prism
```
