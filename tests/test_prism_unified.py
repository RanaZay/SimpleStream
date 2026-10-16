"""Deterministic CPU fixtures for format-agnostic PRISM; no model accuracy or latency claims."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import torch

from lib.minicpm import prism_unified as pu


class Clip:
    """Every frame embeds to e0; a text embeds to e0 plus a word-hash direction, so answers differ by text."""
    def image_vectors(self, frames, stats, prefix='clip'):
        return torch.tensor([[1.] + [0.] * 7 for _ in frames])

    def text_vectors(self, texts):
        out = []
        for text in texts:
            v = torch.zeros(8)
            v[0] = 1.
            for word in text.lower().split():
                v[1 + hash(word) % 7] += .3
            out.append(torch.nn.functional.normalize(v, dim=0))
        return torch.stack(out)


def chunk(index, stamp):
    return SimpleNamespace(chunk_index=index, frame_timestamps=[stamp], start_time=stamp, end_time=stamp,
                           frames=[Image.new('RGB', (16, 16), (index % 255, 30, 70))])


def recent_window():
    chunks = [chunk(i, 45 + j) for j, i in enumerate(range(100, 106))]
    return SimpleNamespace(frames=[c.frames[0] for c in chunks], selected_chunks=chunks,
                           final_chunk_ids=list(range(100, 106)), cdas_metadata={})


HISTORY = [chunk(1, 20), chunk(2, 25), chunk(3, 30), chunk(4, 2), chunk(5, 44), chunk(6, 48)]
MCQ = 'Question: What is on the table?\nOptions:\nA. cup\nB. book\nOnly give the best option.'
OPEN = 'Answer the question based on the video frames. Give a concise direct answer.\n\nQuestion: What is on the table?'


class Text(unittest.TestCase):
    def test_question_text_is_format_agnostic(self):
        self.assertEqual(pu.question_text(MCQ), 'What is on the table?')
        self.assertEqual(pu.question_text(OPEN), 'What is on the table?')
        self.assertEqual(pu.question_text('Is the light on?'), 'Is the light on?')

    def test_answer_change(self):
        clip, config = Clip(), pu.Config()
        self.assertFalse(pu.answers_differ('B', 'b.', clip, config))
        self.assertTrue(pu.answers_differ('A', 'B', clip, config))
        self.assertFalse(pu.answers_differ('The answer is: a red cup', 'red cup', clip, config))
        self.assertTrue(pu.answers_differ('red cup', 'blue cup', clip, pu.Config(change_similarity=1.01)))
        self.assertFalse(pu.answers_differ('red cup', 'a red cup!', clip, pu.Config(change_similarity=1.01)))

    def test_token_confidence(self):
        sharp = torch.full((1, 50), -10.)
        sharp[0, 3] = 10.
        flat = torch.zeros((1, 50))
        margin, confidence, n = pu.token_confidence([sharp, flat, sharp], [3, 7, 0], {0}, 10)
        self.assertEqual(n, 2)  # The special token (id 0) is excluded.
        self.assertAlmostEqual(margin, 0.5, places=3)
        self.assertAlmostEqual(confidence, 0.5, places=3)
        self.assertEqual(pu.token_confidence([sharp], [0], {0}, 10), (0.0, 0.0, 0))

    def test_env_overrides(self):
        with patch.dict(os.environ, {'PRISM_U_TAU_S': '0.5', 'PRISM_U_ENTROPY_TOPK': '5'}):
            config = pu.config_from_env()
        self.assertEqual((config.tau_s, config.entropy_topk), (0.5, 5))


class Controller(unittest.TestCase):
    def run_select(self, prompt, accept_depth, answers=('cup', 'book', 'book', 'book')):
        def scorer(qa, frames, prompt, question, clip, stats, depth, config):
            stats[f'k{depth}_forward_calls'] = 1
            return dict(answer=answers[depth], answer_margin=.1 if depth > accept_depth else .7,
                        entropy_confidence=.5, sufficiency=.4 + depth * .05, support_raw=.5,
                        generate_ms=10., ttft_ms=2., preprocess_ms=1., generated_tokens=2, num_vision_tokens=396)
        with patch.object(pu, 'sync', lambda: None), patch('lib.minicpm.progressive_arbitration.sync', lambda: None):
            return pu.select(None, HISTORY, recent_window(), prompt, Clip(), pu.initial_stats(), pu.Config(),
                             scorer=scorer)

    def test_same_controller_for_mcq_and_open_ended(self):
        for prompt in (MCQ, OPEN):
            for expected in range(4):
                frames, memory, meta, final = self.run_select(prompt, expected)
                self.assertEqual(len(memory), expected)
                self.assertEqual(len(frames), 6 + expected)
                self.assertEqual(final['answer'], ('cup', 'book', 'book', 'book')[expected])
                self.assertTrue(meta['format_agnostic'])
                self.assertEqual(meta['question_text'], 'What is on the table?')

    def test_no_accepted_memory_returns_recent6_answer(self):
        frames, memory, meta, final = self.run_select(OPEN, 0)
        self.assertEqual(memory, [])
        self.assertEqual(final['answer'], 'cup')
        self.assertEqual([pu.image_key(f) for f in frames], [pu.image_key(f) for f in recent_window().frames])

    def test_close_candidate_must_support_new_answer(self):
        clip, config = Clip(), pu.Config()
        candidate = dict(distance=5., image_vector=torch.tensor([1.] + [0.] * 7))
        previous = dict(answer='cup', answer_margin=.7, sufficiency=.6)
        current = dict(answer='book', answer_margin=.8, sufficiency=.65)
        with patch.object(pu, 'support', side_effect=[.2, .3]):  # New answer supported less than the old one.
            self.assertFalse(pu.arbitrate(previous, current, candidate, 1, 'q', clip, config)['accepted'])
        with patch.object(pu, 'support', side_effect=[.3, .2]):
            self.assertTrue(pu.arbitrate(previous, current, candidate, 1, 'q', clip, config)['accepted'])
        far = dict(candidate, distance=20.)
        self.assertTrue(pu.arbitrate(previous, current, far, 1, 'q', clip, config)['temporal_consistency'])

    def test_timing_excludes_final_generation_from_selection(self):
        stats = pu.initial_stats()
        stats.update(k0_total_ms=30., k1_total_ms=40., final_generation_ms=25., PRISM_algorithmic_latency_ms=80.,
                     arbitration_k1_ms=1.)
        pu.aggregate_timing(stats)
        self.assertEqual(stats['vlm_selection_total_ms'], 45.)
        self.assertEqual(stats['PRISM_end_to_end_latency_ms'], stats['PRISM_selection_latency_ms'] + 25.)


if __name__ == '__main__':
    unittest.main()
