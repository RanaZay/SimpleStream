"""Deterministic CPU fixtures for the ShallowStream MiniCPM port; no model accuracy or latency claims."""
import unittest

import torch

from lib.minicpm import shallowstream as ss


class Units(unittest.TestCase):
    def test_units_cover_frames_in_pairs(self):
        self.assertEqual(ss.units_of(5, 2), [[0, 1], [2, 3], [4]])

    def test_frame_token_runs(self):
        ids = torch.tensor([1, 9, 9, 2, 9, 9, 3])
        self.assertEqual(ss.frame_token_runs(ids, 9, 2), [[1, 2], [4, 5]])
        contiguous = torch.tensor([1, 9, 9, 9, 9, 3])  # No separator: split evenly.
        self.assertEqual(ss.frame_token_runs(contiguous, 9, 2), [[1, 2], [3, 4]])
        with self.assertRaises(RuntimeError):
            ss.frame_token_runs(torch.tensor([9, 9, 9]), 9, 2)


class Vectors(unittest.TestCase):
    def test_query_heads_are_averaged_per_kv_group(self):
        query = torch.zeros(3, 4, 2)  # [tokens, q_heads, dim], 2 KV heads -> groups (0,1), (2,3).
        query[:, 0, 0] = 1.
        query[:, 1, 0] = 3.
        query[:, 2, 1] = 2.
        query[:, 3, 1] = 2.
        vec = ss.query_vector(query, [0, 1, 2], kv_heads=2)
        self.assertTrue(torch.allclose(vec, torch.nn.functional.normalize(torch.tensor([2., 0., 0., 2.]), dim=0)))

    def test_key_vector_is_normalized_mean(self):
        key = torch.tensor([[[3., 0.]], [[1., 0.]]])
        self.assertTrue(torch.allclose(ss.key_vector(key, [0, 1]), torch.tensor([1., 0.])))


class Selection(unittest.TestCase):
    def setUp(self):
        self.units = [torch.nn.functional.normalize(torch.tensor([float(i == 3), 1.]), dim=0) for i in range(20)]
        self.query = torch.tensor([1., 0.])

    def test_realtime_route_uses_last_two_units(self):
        selected, seeds, _ = ss.select_units(self.units, None, False)
        self.assertEqual((selected, seeds), ([18, 19], []))

    def test_retrieval_route(self):
        selected, seeds, scored = ss.select_units(self.units, self.query, True)
        self.assertEqual(seeds[0], 3)  # Best-matching history unit ranks first.
        self.assertEqual(len(seeds), 8)
        self.assertTrue(all(s < 14 for s in seeds))  # The 6 recent units are never retrieval seeds.
        self.assertIn(1, selected)  # One earlier neighbour at stride 2.
        self.assertEqual(selected[-6:], list(range(14, 20)))
        self.assertEqual(selected, sorted(selected))

    def test_best_split(self):
        scored = [(-19.0, 'B'), (-18.7, 'B'), (-18.5, 'A'), (-17.8, 'A'), (-17.0, 'B'), (-16.7, 'A')]
        threshold, accuracy = ss.best_split(scored)
        self.assertAlmostEqual(threshold, -18.6)
        self.assertAlmostEqual(accuracy, 5 / 6)


class Incremental(unittest.TestCase):
    def test_frames_are_indexed_once_per_video(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        calls = []
        def index(qa, frames, stats, config=ss.CONFIG):
            calls.append(len(frames))
            return [(torch.ones(2), 1) for _ in frames], [torch.zeros(1) for _ in frames]
        qa = SimpleNamespace()
        with patch.object(ss, 'index_frames', index):
            ss.index_incremental(qa, 'v.mp4', list(range(10)), [float(t) for t in range(10)], {})
            _v, _f, reused = ss.index_incremental(qa, 'v.mp4', list(range(15)), [float(t) for t in range(15)], {})
            self.assertEqual((calls, reused), ([10, 5], 10))
            ss.index_incremental(qa, 'other.mp4', list(range(3)), [0., 1., 2.], {})
            self.assertEqual(calls[-1], 3)  # A new video resets the stream.
            ss.index_incremental(qa, 'other.mp4', list(range(2)), [0., 1.], {})
            self.assertEqual(calls[-1], 2)  # Time going backwards resets it too.


if __name__ == '__main__':
    unittest.main()
