#!/usr/bin/env python3
"""Per-question latency breakdown of a Recent-6 + OASIS run, in the row layout of the PRISM latency slides.

Means are over all questions (stages that did not run count as 0), in seconds per question.
Usage: python main_experiments/tools/oasis_latency_breakdown.py <OASIS result dir> [--output file.json]
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from main_experiments.tools.summarize_progressive_arbitration import distribution, profile, records

ALGORITHMIC = [
    ('Retrieval (event + QA embedding search)', 'retrieval_total_ms'),
    ('Coarse MiniCPM pass that requested memory', 'vlm_selection_total_ms'),
    ('Tool-call parsing / decision', 'decision_logic_total_ms'),
    ('Context assembly, normalization, preprocessing', 'context_total_ms'),
    ('Controller + preparation subtotal', 'PRISM_selection_latency_ms'),
    ('Final MiniCPM generation', 'final_generation_ms'),
    ('Unattributed timing residual', 'latency_accounting_residual_ms'),
    ('Measured total excluding video decoding', 'PRISM_end_to_end_latency_ms'),
]
EXTRA = [
    ('Question -> first answer token', 'request_to_answer_first_token_ms'),
    ('Memory upkeep total (OASIS, async in paper)', 'memory_maintenance_total_ms'),
    ('  event summaries (MiniCPM)', 'memory_event_summary_ms'),
    ('  event merges (MiniCPM)', 'memory_merge_ms'),
    ('  summary embeddings (Qwen3-Embedding)', 'memory_embedding_ms'),
    ('  QA-history summary (MiniCPM)', 'memory_qa_update_ms'),
    ('Video decoding / visual preparation', 'total_visual_preparation_ms'),
    ('Full system (everything)', 'full_system_latency_ms'),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('result_dir')
    p.add_argument('--output')
    args = p.parse_args()
    timings = [profile(r)[2] for r in records(Path(args.result_dir))]
    timings = [t for t in timings if t]
    n = len(timings)

    def mean_s(key):
        return distribution([t.get(key) or 0.0 for t in timings])['mean'] / 1000

    out = dict(questions=n, algorithmic={}, extra={})
    print(f'Recent-6 + OASIS latency, {n} questions (mean s/question)\n')
    print(f'{"Component":52s} {"OASIS":>10s}')
    for section, rows in (('algorithmic', ALGORITHMIC), ('extra', EXTRA)):
        print('-' * 63)
        for label, key in rows:
            out[section][label] = mean_s(key)
            print(f'{label:52s} {out[section][label]:10.6f}')
    e2e, upkeep = out['algorithmic']['Measured total excluding video decoding'], out['extra']['Memory upkeep total (OASIS, async in paper)']
    out['total_excl_decoding_plus_upkeep_s'] = e2e + upkeep
    print('-' * 63)
    print(f'{"Measured total excl. decoding + memory upkeep":52s} {e2e + upkeep:10.6f}')

    fine = [t for t in timings if t.get('oasis_fine_triggered')]
    out['fine_trigger_rate'] = len(fine) / n
    out['coarse_generation_mean_s'] = mean_s('oasis_coarse_generation_ms')
    out['fine_generation_mean_s_when_triggered'] = (
        distribution([t.get('oasis_fine_generation_ms') for t in fine])['mean'] or 0) / 1000
    out['mean_generated_tokens'] = {k: distribution([t.get(f'oasis_{k}_generated_tokens') for t in timings])['mean']
                                    for k in ('coarse', 'fine')}
    out['mean_final_frames'] = distribution([t.get('final_frame_count') for t in timings])['mean']
    print(f'\nFine stage (memory retrieval) triggered on {len(fine)}/{n} ({100 * out["fine_trigger_rate"]:.1f}%); '
          f'coarse pass {out["coarse_generation_mean_s"]:.3f} s mean, fine pass '
          f'{out["fine_generation_mean_s_when_triggered"]:.3f} s mean when triggered.')
    print(f'Mean generated tokens {out["mean_generated_tokens"]}; mean frames in final call {out["mean_final_frames"]:.1f}.')
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
