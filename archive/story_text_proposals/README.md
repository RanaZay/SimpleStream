# Story / Text Proposal Archive

This archive contains the story-memory and recent-frame-description experiments.
They were moved out of the active MiniCPM experiment folders because they were
conceptually useful but empirically too slow and generally degraded accuracy.

Archived families:

- `story_memory`: textual story memory with per-chunk evidence notes.
- `recursive_story_memory`: recursively rewritten single narrative memory.
- `recent_description`: descriptions for the recent visual frames.

Key observed results:

- Story memory V1 degraded OVO and StreamingBench accuracy and greatly increased
  end-to-end latency.
- Story memory V2 improved the prompt quality but still degraded accuracy and
  remained too slow.
- Recent-frame descriptions also degraded accuracy compared with the pure
  SimpleStream Recent-6 baseline.

These files are kept for reference only. Active novelty work should continue in
the semantic-memory, referential-memory, and routed-evidence/clip branches.
