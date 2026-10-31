#!/usr/bin/env python3
"""PRISM-VoI calibration study (docs/prism_voi_design.md, step 1).

On a fixed random subset of questions, measures whether semantic entropy (SE) of the frozen VLM's answer
  (a) predicts when the Recent-6 answer is wrong (AUROC), compared with simpler confidence signals, and
  (b) for multiple-choice, whether adding retrieved frames that lower SE fixes more answers than it breaks,
      previewing the accept-if-DeltaSE > lambda*c rule at several lambda.
This is a diagnostic: nothing here is tuned for the final method.

Usage (one GPU): python main_experiments/tools/voi_calibration.py --benchmark streamingbench --n 300 --output DIR
                 python main_experiments/tools/voi_calibration.py --benchmark streambench --n 300 --output DIR \
                        --recent6-judged reports/streambench_judging/recent6_193879
"""
import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LAMBDAS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)


def env_defaults():
    for key, value in dict(MINICPM_DOWNSAMPLE_MODE='16x', MINICPM_MAX_SLICE_NUMS='1', MINICPM_SEED='42',
                           MINICPM_EXACT_RECENT_CANDIDATE_FPS='4.0', MINICPM_PSM_EXACT_RECENT_PRESERVE_SOURCE_IDS='1',
                           ATTN_IMPLEMENTATION='sdpa').items():
        os.environ.setdefault(key, value)
    os.environ.pop('QWEN_EXACT_RECENT_DECODE', None)


def load_streamingbench(n, seed):
    from main_experiments.minicpm_v46.streamingbench.eval_baseline_dist import _load_tasks
    from main_experiments.minicpm_v46.streamingbench.eval_baseline import build_prompt
    from lib.shared.recent_window import extract_mcq_answer, timestamp_to_seconds
    tasks, _ = _load_tasks(os.environ.get('STREAMINGBENCH_ANNO_PATH', 'data/streamingbench/questions_real.json'),
                           os.environ.get('STREAMINGBENCH_VIDEO_DIR', 'data/streamingbench/videos'))
    picked = random.Random(seed).sample(tasks, min(n, len(tasks)))
    out = []
    for t in picked:
        q = t['question']
        gt = extract_mcq_answer(str(q.get('answer', ''))) or str(q.get('answer', '')).strip().upper()
        out.append(dict(key=str(t['_index']), video=t['video_path'], end=float(timestamp_to_seconds(q['time_stamp'])) + 1e-4,
                        prompt=build_prompt(q), question=q.get('question', ''), gt=gt, task=q.get('task_type', '')))
    return out


def load_streambench(n, seed):
    from main_experiments.minicpm_v46.streambench_v03 import eval_streambench_v03_smoke as sb
    tasks = sb._load_tasks(Path(os.environ.get('STREAMBENCH_V03_ANNOTATIONS', 'data/streambench_v0_3/streaming_bench_v0.3.json')),
                           Path(os.environ.get('STREAMBENCH_V03_DATA_ROOT', 'data/streambench_v0_3')), 0, 0, '', 0)
    picked = random.Random(seed).sample(tasks, min(n, len(tasks)))
    return [dict(key=f"{t['video_index']}:{t['breakpoint_index']}", video=t['video_path'], end=float(t['time']) + 1e-4,
                 prompt=sb._build_prompt(t['question']), question=t['question'], gt=t['answer_gt'], task=t['subtask'])
            for t in picked]


def judged_correct(judged_dir):
    text = (Path(judged_dir) / 'judge' / 'judged.json').read_text().strip()
    rows = json.loads(text) if text.startswith('[') else [json.loads(l) for l in text.splitlines() if l.strip()]
    return {str(r['id']): (r.get('llama_pred') == 'yes', r.get('predict')) for r in rows}


def contexts(qa, item, clip, k):
    """Exact Recent-6 plus the top-k band candidates (question-conditioned CLIP-MMR), as frame lists K0..Kk."""
    from main_experiments.minicpm_v46.streamingbench.eval_prism_exact_recent_dist import select_exact_current_recent_frames
    from lib.shared.recent_window import decode_video_to_chunks_qwen
    from lib.minicpm.progressive_arbitration import initial_stats, prefilter
    from lib.minicpm import prism_unified as pu
    end = item['end']
    recent = select_exact_current_recent_frames(qa, item['video'], 1.0, 1.0, 6, video_start=max(0.0, end - 6.0), video_end=end)
    config = pu.config_from_env()
    chunks, _ = decode_video_to_chunks_qwen(item['video'], 1.0, 1.0, None,
                                            video_start=max(0.0, end - (config.d_max + config.recent + 1.0)) or None, video_end=end)
    stats = pu.initial_stats()
    clip.begin(os.path.realpath(item['video']))
    candidates = pu.rank(prefilter(chunks, recent, stats), pu.question_text(item['prompt']), clip, stats, config)[:k]
    out, accepted = [list(recent.frames)], []
    for c in candidates:
        frame = c['frame'] if c['frame'].size == recent.frames[0].size else c['frame'].resize(recent.frames[0].size)
        accepted = sorted(accepted + [(c['end'], frame)], key=lambda x: x[0])
        out.append([f for _t, f in accepted] + list(recent.frames))
    return out


def run_question(qa, clip, item, benchmark, samples, k, judged):
    from lib.minicpm import prism_voi as pv
    from lib.minicpm.progressive_arbitration import extract_options
    ctxs = contexts(qa, item, clip, k)
    row = dict(key=item['key'], task=item['task'], gt=item['gt'], candidates=len(ctxs) - 1, states=[])
    if benchmark == 'streamingbench':
        options = extract_options(item['prompt'])
        for frames in ctxs:
            probs, pred = pv.option_distribution(qa, frames, item['prompt'], options)
            row['states'].append(dict(se=pv.semantic_entropy_mcq(probs), max_prob=max(probs.values()), pred=pred,
                                      correct=pred == item['gt'], frames=len(frames)))
        return row
    greedy, sampled = pv.sample_answers(qa, ctxs[0], item['prompt'], samples)
    clusters = pv.cluster_answers([greedy] + sampled, qa, item['question'])
    from lib.minicpm.prism_unified import generate_state, Config
    token = generate_state(qa, ctxs[0], item['prompt'], Config())
    for j, frames in enumerate(ctxs):
        if j:  # New answers that appear under added evidence join the hypothesis set.
            new = qa.generate_from_frames(frames, item['prompt'])
            clusters = pv.cluster_answers([a for c in clusters for a in c] + [new], qa, item['question'])
        se, probs = pv.semantic_entropy_open(qa, frames, item['prompt'], clusters)
        row['states'].append(dict(se=se, max_prob=max(probs), frames=len(frames),
                                  answer=greedy if j == 0 else new))
    correct, judged_answer = judged.get(item['key'], (None, None))
    row.update(recent6_correct=correct, recent6_answer_matches=(judged_answer or '').strip() == greedy.strip(),
               token_margin=token['answer_margin'], token_entropy_conf=token['entropy_confidence'],
               clusters=[len(c) for c in clusters], samples=[greedy] + sampled)
    return row


def auroc(scores, labels):
    """P(score of an error > score of a correct answer); ties count half."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def summarize(rows, benchmark):
    lines = [f'# PRISM-VoI calibration: {benchmark} ({len(rows)} questions)', '']
    if benchmark == 'streamingbench':
        base = [r['states'][0] for r in rows]
        errors = [not s['correct'] for s in base]
        lines += [f"Recent-6 accuracy on the subset: {100 * (1 - statistics.mean(errors)):.2f}%", '',
                  '| Signal predicting a Recent-6 error | AUROC |', '|---|---:|',
                  f"| Semantic entropy SE(R) (= option entropy) | {auroc([s['se'] for s in base], errors):.3f} |",
                  f"| 1 - top option probability | {auroc([1 - s['max_prob'] for s in base], errors):.3f} |", '',
                  '| lambda (nats per frame) | Accuracy | Delta vs R6 | Fixed / broken | Frames added (mean) | Questions using memory |',
                  '|---:|---:|---:|---:|---:|---:|']
        base_acc = 100 * statistics.mean(s['correct'] for s in base)
        for lam in LAMBDAS:
            correct, added, used, fixed, broken = [], [], 0, 0, 0
            for r in rows:
                states, j = r['states'], 0
                while j + 1 < len(states) and states[j]['se'] - states[j + 1]['se'] > lam:
                    j += 1
                correct.append(states[j]['correct'])
                added.append(j)
                used += j > 0
                fixed += states[j]['correct'] and not states[0]['correct']
                broken += states[0]['correct'] and not states[j]['correct']
            acc = 100 * statistics.mean(correct)
            lines.append(f"| {lam:g} | {acc:.2f}% | {acc - base_acc:+.2f} | {fixed} / {broken} | {statistics.mean(added):.2f} | {used} |")
        oracle = 100 * statistics.mean(any(s['correct'] for s in r['states']) for r in rows)
        lines += ['', f"Oracle (best of K0..K3 per question): {oracle:.2f}%: the ceiling for any acceptance rule on these candidates."]
    else:
        judged = [r for r in rows if r.get('recent6_correct') is not None]
        errors = [not r['recent6_correct'] for r in judged]
        lines += [f"Questions with a Recent-6 judgment: {len(judged)}; greedy answer identical to the judged Recent-6 answer: "
                  f"{sum(r['recent6_answer_matches'] for r in judged)}", '',
                  '| Signal predicting a Recent-6 error | AUROC |', '|---|---:|',
                  f"| Semantic entropy SE(R) | {auroc([r['states'][0]['se'] for r in judged], errors):.3f} |",
                  f"| 1 - token margin (FLARE/DRAGIN-style) | {auroc([1 - r['token_margin'] for r in judged], errors):.3f} |",
                  f"| 1 - token entropy confidence | {auroc([1 - r['token_entropy_conf'] for r in judged], errors):.3f} |",
                  f"| 1 - top cluster probability | {auroc([1 - r['states'][0]['max_prob'] for r in judged], errors):.3f} |", '',
                  f"Mean meaning clusters per question: {statistics.mean(len(r['clusters']) for r in judged):.2f}; "
                  f"questions with a single cluster (SE = 0): {sum(len(r['clusters']) == 1 for r in judged)}"]
    lines += ['', 'AUROC: probability that a wrong Recent-6 answer gets a higher uncertainty than a right one '
              '(0.5 = no signal, 1.0 = perfect).']
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--benchmark', required=True, choices=['streamingbench', 'streambench'])
    p.add_argument('--n', type=int, default=300)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--samples', type=int, default=5, help='sampled answers for open-ended SE')
    p.add_argument('--candidates', type=int, default=3)
    p.add_argument('--recent6-judged', default='reports/streambench_judging/recent6_193879')
    p.add_argument('--output', required=True)
    p.add_argument('--summarize-only', action='store_true')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / f'{args.benchmark}_rows.jsonl'
    if not args.summarize_only:
        env_defaults()
        from lib.minicpm.baseline import RecentWindowQAModel
        from lib.minicpm.progressive_arbitration import ClipCache
        items = load_streamingbench(args.n, args.seed) if args.benchmark == 'streamingbench' else load_streambench(args.n, args.seed)
        judged = judged_correct(args.recent6_judged) if args.benchmark == 'streambench' else {}
        qa = RecentWindowQAModel(model_name='openbmb/MiniCPM-V-4.6', device='cuda:0',
                                 max_new_tokens=256 if args.benchmark == 'streambench' else 64,
                                 attn_implementation=os.environ['ATTN_IMPLEMENTATION'])
        clip = ClipCache(qa)
        done = {json.loads(l)['key'] for l in rows_path.read_text().splitlines() if l.strip()} if rows_path.exists() else set()
        with rows_path.open('a') as handle:
            for i, item in enumerate(items, 1):
                if item['key'] in done:
                    continue
                start = time.perf_counter()
                try:
                    row = run_question(qa, clip, item, args.benchmark, args.samples, args.candidates, judged)
                except Exception as exc:
                    row = dict(key=item['key'], error=repr(exc))
                row['seconds'] = time.perf_counter() - start
                handle.write(json.dumps(row) + '\n')
                handle.flush()
                print(f"[{i}/{len(items)}] {item['key']} {row.get('error') or ''}", flush=True)
    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get('error') and r.get('states')]
    text = summarize(rows, args.benchmark)
    (out / f'{args.benchmark}_summary.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
