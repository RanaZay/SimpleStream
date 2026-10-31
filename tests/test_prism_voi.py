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
    def run_select(self, qa, prompt, by_frames, config, **patches):
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
                return pv.select(qa, [], RECENT, prompt, None, stats, config)
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
            answer_logprob=lambda qa_, frames, prompt, a: logp[(len(frames), a)])
        self.assertEqual(meta['format'], 'open')
        self.assertEqual((len(frames), final['text']), (7, 'a red cup'))
        generated = qa.generated
        self.assertEqual(generated, 3)  # R, R+h1, R+h1+h2: the chosen context's answer is reused, not regenerated.

    def test_env_overrides(self):
        import os
        with patch.dict(os.environ, {'PRISM_VOI_LAM': '0.3', 'PRISM_VOI_REACH': '120'}):
            self.assertEqual((pv.config_from_env().lam, pv.config_from_env().reach), (0.3, 120.0))


if __name__ == '__main__':
    unittest.main()
