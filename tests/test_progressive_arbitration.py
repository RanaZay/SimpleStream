"""Deterministic integration fixtures, not model accuracy or latency experiments."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import torch

from lib.minicpm import progressive_arbitration as pa


class Clip:
    def image_vectors(self, frames, stats):
        return torch.tensor([[1., i*.01] for i in range(len(frames))])

    def text_vectors(self, texts):
        return torch.tensor([[1.,0.], [0.,1.]])


def chunk(index,stamp):
    return SimpleNamespace(chunk_index=index,frame_timestamps=[stamp],start_time=stamp,end_time=stamp,
        frames=[Image.new('RGB',(16,16),(index%255,30,70))])


class IntegratedController(unittest.TestCase):
    def test_twenty_four_progressive_paths(self):
        for case in range(24):
            recent_chunks=[chunk(i,45+i) for i in range(100,106)]
            # Override times to an exact 45..50 recent window.
            for j,c in enumerate(recent_chunks):
                c.frame_timestamps=[45+j]
            recent=SimpleNamespace(frames=[c.frames[0] for c in recent_chunks],selected_chunks=recent_chunks,
                final_chunk_ids=list(range(100,106)),cdas_metadata={})
            history=[chunk(1,20),chunk(2,25),chunk(3,30),chunk(4,2),chunk(5,44),chunk(6,48)]
            expected=case%4
            def scorer(qa,frames,prompt,options,clip,stats,depth):
                stats[f'k{depth}_forward_calls']=1
                return dict(predicted_option='B' if depth==0 else 'A',answer_margin=.1 if depth>expected else .7,
                            sufficiency=.4+depth*.05,entropy_confidence=.5)
            with patch.object(pa,'sync',lambda:None):
                frames,memory,meta=pa.select(None,history,recent,'Question: test\nOptions:\nA. first\nB. second',
                    Clip(),pa.initial_stats(),scorer=scorer)
            self.assertEqual(len(memory),expected)
            self.assertEqual(len(frames),6+expected)
            self.assertEqual([pa.image_key(f) for f in frames[-6:]],[pa.image_key(f) for f in recent.frames])
            self.assertTrue(all(3<=c['distance']<=30 for c in meta['candidate_queue']))

    def test_consistency_and_no_flip_progress(self):
        previous=dict(predicted_option='A',answer_margin=.7,sufficiency=.6)
        current=dict(predicted_option='A',answer_margin=.8,sufficiency=.65)
        self.assertTrue(pa.arbitrate(previous,current,dict(best_supported_option='A',distance=5),2)['accepted'])
        self.assertFalse(pa.arbitrate(previous,current,dict(best_supported_option='B',distance=5),2)['accepted'])

    def test_unused_timings_are_null(self):
        stats=pa.initial_stats()
        self.assertIsNone(stats['k3_total_ms'])
        self.assertIsNone(stats['arbitration_k2_ms'])


if __name__=='__main__':
    unittest.main()
