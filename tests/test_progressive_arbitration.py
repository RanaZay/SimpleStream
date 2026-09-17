"""Deterministic integration fixtures, not model accuracy or latency experiments."""
import unittest
import os
from contextlib import ExitStack
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
    def test_short_context_preserved_for_both_modes(self):
        for count in range(1, 7):
            for control in (False, True):
                chunks = [chunk(i, i) for i in range(count)]
                recent = SimpleNamespace(frames=[c.frames[0] for c in chunks],
                    selected_chunks=chunks, final_chunk_ids=list(range(count)), cdas_metadata={})
                def scorer(*args):
                    return dict(predicted_option='A', answer_margin=.7, sufficiency=.8, entropy_confidence=.5)
                with patch.object(pa, 'sync', lambda: None):
                    frames, memory, metadata = pa.select(None, [], recent,
                        'Question: test\nOptions:\nA. one\nB. two', Clip(), pa.initial_stats(),
                        control=control, scorer=scorer)
                self.assertEqual(frames, recent.frames)
                self.assertEqual(memory, [])
                self.assertEqual(metadata['recent_frame_count'], count)
                self.assertEqual(metadata['short_recent_context'], count < 6)

    def test_timing_discrepancy_is_nonfatal_and_visible(self):
        stats = pa.initial_stats()
        stats.update(final_generation_ms=100., PRISM_algorithmic_latency_ms=250.)
        pa.aggregate_timing(stats)
        self.assertFalse(stats['latency_accounting_valid'])
        self.assertEqual(stats['latency_accounting_residual_ms'], 150.)
        self.assertEqual(stats['PRISM_algorithmic_latency_ms'], 250.)

    def test_option_labels_do_not_match_word_endings(self):
        texts = ['FIFA and La Liga.', 'UEFA and La Liga.', 'Bundesliga and La Liga.', 'Aeromexico and La Liga.']
        for separator in ('; ', '\n'):
            prompt = 'Question: logos?\nOptions: ' + separator.join(f'{label}. {text}' for label, text in zip('ABCD', texts))
            prompt += '\nAnswer with only the option letter.'
            self.assertEqual(pa.extract_options(prompt), [{'letter':label, 'text':text} for label,text in zip('ABCD',texts)])
        self.assertEqual(pa.extract_options('Describe the video.'), [])

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

    def test_control_never_decodes_history(self):
        recent = SimpleNamespace(frames=[Image.new('RGB', (16,16)) for _ in range(6)],
            final_chunk_ids=list(range(6)), downsample_mode='16x', selected_chunks=[], cdas_metadata={})
        qa = SimpleNamespace(_progressive_arbitration_clip=SimpleNamespace(begin=lambda video: None),
            _progressive_arbitration_warm=True, _last_model_generate_seconds=0.,
            _last_preprocess_seconds=0., _last_ttft_seconds=0., _last_num_vision_tokens=396,
            generate_from_frames=lambda *a, **k: 'A')
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, MINICPM_ADAPTIVE_MODE=pa.BASELINE_MODE))
            stack.enter_context(patch.object(pa, 'sync', lambda: None))
            decode = stack.enter_context(patch('lib.shared.recent_window.decode_video_to_chunks_qwen'))
            stack.enter_context(patch('main_experiments.minicpm_v46.streamingbench.eval_prism_exact_recent_dist.select_exact_current_recent_frames', return_value=recent))
            stack.enter_context(patch('lib.minicpm.baseline._reset_gpu_memory_peaks', return_value={}))
            stack.enter_context(patch('lib.minicpm.baseline._capture_gpu_memory', return_value={}))
            stack.enter_context(patch('lib.minicpm.baseline._build_profile', return_value={'gpu_peak_allocated_mb':0., 'gpu_peak_reserved_mb':0.}))
            result, _ = pa.query(qa, 'test.mp4', 'Question: test\nOptions:\nA. one\nB. two', 1., 1., 6, video_end=60.)
            decode.assert_not_called()
            self.assertEqual(result.num_frames, 6)
            self.assertEqual(result.profile_metadata['progressive_arbitration']['broad_history_decode_ms'], 0.)


if __name__=='__main__':
    unittest.main()
