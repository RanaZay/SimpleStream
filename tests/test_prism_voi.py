"""Deterministic CPU fixtures for the PRISM-VoI controller; no model accuracy or latency claims."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from lib.minicpm import prism_voi as pv
from lib.minicpm import prism_unified as pu

MCQ = 'Question: What is on the table?\nOptions:\nA. cup\nB. book\nC. phone\nD. pen\nOnly give the best option.'
OPEN = 'Answer the question based on the video frames. Give a concise direct answer.\n\nQuestion: What is on the table?'


def frame(i):
    return Image.new('RGB', (16, 16), (i, 40, 60))


RECENT = SimpleNamespace(frames=[frame(100 + i) for i in range(6)], selected_chunks=[], final_chunk_ids=list(range(6)))
CANDIDATES = [dict(frame=frame(i), end=float(10 + i), distance=10.0 + i) for i in range(3)]


class QA:
    """Generates the answer listed for the number of frames in the context."""
    def __init__(self, answers):
        self.answers = answers
        self.generated = 0
        self._last_model_generate_seconds = 0.01
        self._last_ttft_seconds = 0.005
        self._last_num_vision_tokens = 0

    def generate_from_frames(self, frames, prompt):
        self.generated += 1
        return self.answers[len(frames)]


def dist(p_top):
    rest = (1 - p_top) / 3
    return {'A': p_top, 'B': rest, 'C': rest, 'D': rest}


class Controller(unittest.TestCase):
    def run_select(self, qa, prompt, by_frames, config, chunks=(), **patches):
        def option_distribution(qa_, frames, prompt_, options):
            probs = by_frames[len(frames)]
            return probs, max(probs, key=probs.get)
        stats = {}
        with patch.object(pv, 'option_distribution', option_distribution), \
             patch.object(pu, 'rank', lambda eligible, q, clip, stats, cfg: list(CANDIDATES)), \
             patch.object(pv, 'memory_candidates', lambda chunks, recent, cfg: list(CANDIDATES)), \
             patch('lib.minicpm.progressive_arbitration.sync', lambda: None):
            for name, value in patches.items():
                patch.object(pv, name, value).start()
            try:
                return pv.select(qa, chunks, RECENT, prompt, None, stats, config)
            finally:
                patch.stopall()

    def test_keeps_frames_while_they_pay_and_stops_at_the_first_that_does_not(self):
        se = lambda p: pv.semantic_entropy_mcq(dist(p))
        by_frames = {6: dist(0.4), 7: dist(0.8), 8: dist(0.82), 9: dist(0.99)}
        self.assertGreater(se(0.4) - se(0.8), 0.1)
        self.assertLess(se(0.8) - se(0.82), 0.1)
        frames, final, meta = self.run_select(QA({7: 'A'}), MCQ, by_frames, pv.Config(lam=0.1))
        self.assertEqual(len(frames), 7)  # R + h1; h2 removed too little uncertainty, so the search stopped.
        self.assertEqual((meta['selected_memory_depth'], meta['stop_reason']), (1, 'gain_below_lambda'))
        self.assertEqual(final['text'], 'A')
        self.assertEqual(meta['format'], 'mcq')

    def test_confident_recent_answer_never_retrieves(self):
        ranked = []
        frames, final, meta = self.run_select(QA({6: 'A'}), MCQ, {6: dist(0.999)}, pv.Config(lam=0.1),
                                              memory_candidates=lambda *a: ranked.append(1) or [])
        self.assertEqual((len(frames), meta['stop_reason'], ranked), (6, 'confident', []))

    def test_history_is_decoded_only_when_retrieval_runs(self):
        decoded = []
        history = lambda: decoded.append(1) or []
        self.run_select(QA({6: 'A'}), MCQ, {6: dist(0.999)}, pv.Config(lam=0.1), chunks=history)
        self.assertEqual(decoded, [])
        self.run_select(QA({6: 'A', 7: 'A'}), MCQ, {6: dist(0.4), 7: dist(0.41)}, pv.Config(lam=0.1), chunks=history)
        self.assertEqual(decoded, [1])

    def test_lambda_infinity_is_recent6_and_zero_takes_every_improving_frame(self):
        by_frames = {6: dist(0.4), 7: dist(0.5), 8: dist(0.6), 9: dist(0.7)}
        frames, _f, _m = self.run_select(QA({6: 'A'}), MCQ, by_frames, pv.Config(lam=float('inf')))
        self.assertEqual(len(frames), 6)
        frames, _f, meta = self.run_select(QA({9: 'A'}), MCQ, by_frames, pv.Config(lam=0.0))
        self.assertEqual((len(frames), meta['stop_reason']), (9, 'cap_reached'))

    def test_open_ended_answers_from_the_chosen_context_without_regenerating(self):
        qa = QA({6: 'a cup', 7: 'a red cup', 8: 'a pen'})
        logp = {(6, 'a cup'): -0.7, (6, 'a mug'): -0.7, (6, 'a red cup'): -3.0, (6, 'a pen'): -3.0,
                (7, 'a cup'): -2.0, (7, 'a mug'): -2.0, (7, 'a red cup'): -0.05, (7, 'a pen'): -4.0,
                (8, 'a cup'): -2.0, (8, 'a mug'): -2.0, (8, 'a red cup'): -0.3, (8, 'a pen'): -0.4}
        frames, final, meta = self.run_select(
            qa, OPEN, {}, pv.Config(lam=0.1, samples=2),
            sample_answers=lambda qa_, frames, prompt, m: ('a cup', ['a mug', 'a cup']),
            cluster_answers=lambda answers, qa_=None, q=None: [[a] for a in dict.fromkeys(answers)],
            answer_logprobs=lambda qa_, frames, prompt, answers: {a: logp[(len(frames), a)] for a in answers})
        self.assertEqual(meta['format'], 'open')
        self.assertEqual((len(frames), final['text']), (7, 'a red cup'))
        generated = qa.generated
        self.assertEqual(generated, 3)  # R, R+h1, R+h1+h2: the chosen context's answer is reused, not regenerated.

    def test_trace_evaluates_every_candidate_but_answers_like_a_normal_run(self):
        by_frames = {6: dist(0.4), 7: dist(0.8), 8: dist(0.82), 9: dist(0.99)}
        normal, _f, meta_n = self.run_select(QA({7: 'A'}), MCQ, by_frames, pv.Config(lam=0.1))
        traced, _f, meta_t = self.run_select(QA({7: 'A'}), MCQ, by_frames, pv.Config(lam=0.1, trace=True))
        self.assertEqual(len(traced), len(normal))  # Same decision: greedy stops at the first step below lambda.
        self.assertEqual(len(meta_n['trace']), 3)   # Normal run stops after the rejected step.
        self.assertEqual(len(meta_t['trace']), 4)   # Trace keeps going so later lambdas can be replayed.
        self.assertTrue(all('prediction' in row and 'ms' in row for row in meta_t['trace']))

    def test_ranking_ablations_order_candidates_without_clip(self):
        eligible = [dict(frame=frame(i), end=float(10 + i), distance=10.0 - i) for i in range(5)]
        recency = pv.order_candidates(list(eligible), 'q', None, {}, pv.Config(ranking='recency'))
        self.assertEqual([c['end'] for c in recency], [14.0, 13.0, 12.0])
        first = pv.order_candidates(list(eligible), 'q', None, {}, pv.Config(ranking='random'))
        again = pv.order_candidates(list(eligible), 'q', None, {}, pv.Config(ranking='random'))
        self.assertEqual([c['end'] for c in first], [c['end'] for c in again])  # Same question, same frames.
        self.assertEqual(len(first), 3)
        with self.assertRaises(ValueError):
            pv.order_candidates(eligible, 'q', None, {}, pv.Config(ranking='nearest'))

    def test_scan_skips_a_frame_that_does_not_pay_and_tries_the_next(self):
        def by_content(frames):  # h1 (red 0) pays, h2 (red 1) does not, h3 (red 2) pays on top of h1.
            reds = {f.getpixel((0, 0))[0] for f in frames}
            return dist(0.4 + 0.3 * (0 in reds) + 0.25 * (2 in reds) + 0.01 * (1 in reds))
        frames, _f, meta = self.run_select(
            QA({7: 'A', 8: 'A'}), MCQ, {}, pv.Config(lam=0.1, stop='scan'),
            option_distribution=lambda qa_, frames, prompt, options: (by_content(frames), 'A'))
        reds = sorted(f.getpixel((0, 0))[0] for f in frames[:-6])
        self.assertEqual((reds, meta['selected_memory_depth']), ([0, 2], 2))
        frames, _f, meta = self.run_select(
            QA({7: 'A', 8: 'A'}), MCQ, {}, pv.Config(lam=0.1),
            option_distribution=lambda qa_, frames, prompt, options: (by_content(frames), 'A'))
        self.assertEqual((len(frames), meta['stop_reason']), (7, 'gain_below_lambda'))  # 'first' stops at h2.

    def test_wrws_entropy_follows_the_shape_of_the_relevance_distribution(self):
        times = [float(i) for i in range(10)]
        peaked, fused = pv.wrws_fusion([0.1] * 9 + [0.9], times)
        flat, _ = pv.wrws_fusion([0.2 + 0.001 * (i % 2) for i in range(10)], times)
        self.assertLess(peaked, flat)
        self.assertTrue(0.0 <= peaked <= 1.0 and 0.0 <= flat <= 1.0)
        self.assertEqual(max(range(10), key=lambda i: fused[i]), 9)
        self.assertEqual(pv.wrws_fusion([0.5], [0.0]), (0.0, [1.0]))
        uniform, scores = pv.wrws_fusion([0.3] * 4, [0.0, 1.0, 2.0, 3.0])
        self.assertAlmostEqual(uniform, 1.0)  # No relevance signal: WRWS falls back to recency.
        self.assertEqual(scores, sorted(scores))

    def test_wrws_answers_from_a_fixed_budget_chosen_by_the_fused_score(self):
        sims = [0.9, 0.1, 0.1] + [0.1] * 6  # One very relevant memory frame, the rest flat.
        times = [10.0, 11.0, 12.0] + [20.0 + i for i in range(6)]
        with patch.object(pv, 'memory_candidates', lambda chunks, recent, cfg: list(CANDIDATES)), \
             patch.object(pv, '_pool_scores', lambda *a: (sims, times)):
            frames, final, meta = pv.wrws_select(QA({6: 'A'}), [], RECENT, MCQ, None, {}, pv.Config(method='wrws'))
        self.assertEqual(len(frames), 6)  # Always the recent-window budget: WRWS never stays with R alone by rule.
        self.assertEqual(meta['memory_times'], [10.0])
        self.assertEqual(meta['selected_times'], sorted(meta['selected_times']))
        self.assertEqual((final['text'], meta['format']), ('A', 'mcq'))

    def test_env_overrides(self):
        import os
        with patch.dict(os.environ, {'PRISM_VOI_LAM': '0.3', 'PRISM_VOI_REACH': '120', 'PRISM_VOI_RANKING': 'recency'}):
            self.assertEqual((pv.config_from_env().lam, pv.config_from_env().reach), (0.3, 120.0))
            self.assertEqual(pv.config_from_env().ranking, 'recency')


if __name__ == '__main__':
    unittest.main()
