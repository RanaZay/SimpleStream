"""Deterministic CPU fixtures for the StreamScout zero-shot reimplementation; no model accuracy claims."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from lib.minicpm import prism_voi as pv
from lib.minicpm import streamscout as ss

MCQ = 'Question: What is on the table?\nOptions:\nA. cup\nB. book\nOnly give the best option.'


class Memory(unittest.TestCase):
    def test_cache_keeps_the_budget_and_evicts_where_frames_are_densest(self):
        times = [float(t) for t in range(100)]
        cache = ss.evict_dense(times, budget=10)
        self.assertEqual(len(cache), 10)
        self.assertEqual((cache[0], cache[-1]), (0.0, 99.0))  # The span stays covered end to end.
        gaps = [b - a for a, b in zip(cache, cache[1:])]
        self.assertLess(max(gaps), 30)  # Eviction spreads the cache instead of keeping only one end.

    def test_look_back_takes_the_frame_nearest_each_stride(self):
        cache = [float(t) for t in range(0, 200, 5)]
        out = ss.look_back(cache, 195.0, exclude={195.0, 190.0}, n=3, stride=32.0)
        self.assertEqual(out, [165.0, 130.0, 100.0])

    def test_watershed_takes_one_peak_per_salient_segment(self):
        times = [float(t) for t in range(9)]
        scores = [0.1, 0.9, 0.1, 0.2, 0.3, 0.1, 0.5, 0.45, 0.1]
        self.assertEqual(ss.watershed_peaks(times, scores, n=2), [1.0, 6.0])  # Not 6 and its neighbour 7.
        self.assertEqual(ss.watershed_peaks(times, scores, n=2, exclude={1.0}), [6.0, 4.0])

    def test_frames_are_woven_into_the_caption_timeline(self):
        img = Image.new('RGB', (4, 4))
        content = ss.woven_content([(0.0, 'a man enters'), (16.0, 'he sits')], [(20.0, img), (5.0, img)], 'Q?')
        kinds = [c['type'] if c['type'] == 'image' else c['text'][:6] for c in content]
        self.assertEqual(kinds, ['[00:00', '[frame', 'image', '[00:16', '[frame', 'image', 'Q?'])


class Cascade(unittest.TestCase):
    def run_cascade(self, replies):
        frames = [Image.new('RGB', (8, 8), (i, 0, 0)) for i in range(40)]
        history = [dict(frame=frames[i], end=float(i)) for i in range(34)]
        recent = SimpleNamespace(frames=frames[34:], selected_chunks=[])
        times = [float(i) for i in range(40)]
        replies = list(replies)
        views = []

        def generate(qa, content):
            views.append(sum(c['type'] == 'image' for c in content))
            return replies.pop(0)
        qa = SimpleNamespace(_last_model_generate_seconds=0.01, _last_ttft_seconds=0.005)
        clip = SimpleNamespace(video='v.mp4')
        with patch.object(pv, 'memory_candidates', lambda *a: list(history)), \
             patch.object(pv, '_pool_scores', lambda *a: ([0.1] * 40, times)), \
             patch.object(ss, 'caption_windows', lambda *a: [(0.0, 'start'), (16.0, 'middle')]), \
             patch.object(ss, 'generate', generate):
            out = ss.select(qa, [], recent, MCQ, clip, {}, pv.Config(method='streamscout'))
        return out, views

    def test_stops_at_the_first_view_that_answers(self):
        (frames, final, meta), views = self.run_cascade(['B'])
        self.assertEqual((views, final['text'], meta['stop_view'], len(frames)), ([4], 'B', 1, 4))

    def test_escalates_through_nested_views_and_the_last_must_answer(self):
        (frames, final, meta), views = self.run_cascade([ss.ESCALATE, ss.ESCALATE, 'A'])
        self.assertEqual(views, [4, 12, 24])
        self.assertEqual((final['text'], meta['generations'], meta['memory_triggered']), ('A', 3, True))


if __name__ == '__main__':
    unittest.main()
