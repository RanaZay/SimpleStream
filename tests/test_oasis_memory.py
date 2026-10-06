"""Deterministic CPU fixtures for the Recent-6 + OASIS baseline; no model accuracy or latency claims."""
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import torch

from lib.minicpm import oasis_memory as om

SB_PROMPT = ("You are an advanced video question-answering AI assistant. You have been provided with some frames "
             "from the video and a multiple-choice question. Your task is to analyze the video and provide the best "
             "answer.\n\nQuestion: What is on the table?\n\nOptions:\nA. cup\nB. book\nC. phone\nD. pen\n\n"
             "Only give the best option's letter (A, B, C, or D) directly.")


def image(i):
    return Image.new('RGB', (16, 16), (i % 255, 30, 70))


class Embedder:
    def encode(self, texts):
        out = []
        for text in texts:
            v = torch.zeros(8)
            for word in text.lower().split():
                v[hash(word) % 8] += 1
            out.append(torch.nn.functional.normalize(v + 1e-3, dim=0))
        return torch.stack(out)


class Prompts(unittest.TestCase):
    def test_streamingbench_prompt_becomes_oasis_mc(self):
        question, mc = om.build_query(SB_PROMPT)
        self.assertTrue(mc)
        self.assertTrue(question.startswith('Question: What is on the table?'))
        self.assertNotIn('advanced video question-answering', question)
        self.assertNotIn('directly', question)
        self.assertIn('A. cup', question)
        self.assertIn('<answer> </answer> tags', question)

    def test_egoschema_prompt(self):
        prompt = ("You are an advanced video question-answering AI assistant.\n"
                  "The video is processed causally up to the end of the clip. "
                  "Answer the multiple-choice question using the provided visual evidence.\n\n"
                  "Question: What did C do?\nOptions:\nA. cook\nB. clean\nC. read\nD. walk\nE. sleep\n\n"
                  "Answer with only the option letter.")
        question, mc = om.build_query(prompt)
        self.assertTrue(mc)
        self.assertTrue(question.startswith('Question: What did C do?'))
        self.assertIn('E. sleep', question)
        self.assertNotIn('Answer with only the option letter', question)

    def test_ovo_prompts(self):
        question, mc = om.build_query("Which tool?\nOptions: A. saw; B. drill; C. hammer; D. knife;\n"
                                      "Only give the best option's letter directly.")
        self.assertTrue(mc)
        self.assertNotIn('directly', question)
        question, mc = om.build_query('How many times did they jump?\nOnly give a number as answer.')
        self.assertFalse(mc)
        self.assertIn('Only give a number as answer.', question)

    def test_tool_call_variants(self):
        cases = {
            '\n{"name": "rag_retrieval", "arguments": {"text_input": "red cup"}}\n': ('red cup', 'json'),
            '{"name": "rag_retrieval", "arguments": "{\\"text_input\\": \\"red cup\\"}"}': ('red cup', 'json'),
            '{"text_input": "red cup"}': ('red cup', 'json'),
            "{'name': 'rag_retrieval', 'arguments': {'text_input': 'red cup'}}": ('red cup', 'text_input_regex'),
            '{"name": "rag_retrieval", "arguments": {"text_input": "red cup",}}': ('red cup', 'text_input_regex'),
            'red cup on table': ('red cup on table', 'plain_text'),
            '  ': (None, 'unparsed'),
        }
        for block, expected in cases.items():
            self.assertEqual(om.parse_tool_query(block), expected, block)

    def test_answer_parsing(self):
        self.assertEqual(om.parse_answer('think <answer>B</answer>', True), ('B', 'answer_tag'))
        self.assertEqual(om.parse_answer('So the answer is C.', True), ('C', 'answer_phrase'))
        self.assertEqual(om.parse_answer('B', True), ('B', 'bare_letter'))
        self.assertEqual(om.parse_answer('I see a cat on a mat.', True), ('', 'unparsed'))
        self.assertEqual(om.parse_answer('<tool_call>{}</tool_call> Yes', False), ('Yes', 'untagged_text'))


class Forest(unittest.TestCase):
    def setUp(self):
        self.config = om.Config()
        self.state = om.State('v', self.config)

    def test_buffer_holds_32_seconds_and_cuts_every_32_frames(self):
        cuts = []
        for t in range(70):
            cuts += self.state.push(om.Packet(t + 0.9995 * 0, image(t)))
        self.assertEqual(len(self.state.buf), 32)
        self.assertEqual([len(c) for c in cuts], [32, 32])
        self.assertEqual(self.state.buf[-1].t, 69)

    def test_subsecond_frames_are_skipped(self):
        for t in (0, 0.5, 0.9995, 1.4, 2.0):
            self.state.push(om.Packet(t, image(0)))
        self.assertEqual([p.t for p in self.state.buf], [0, 0.9995, 2.0])

    def test_root_budget_merges_and_lineage_pruning(self):
        emb = Embedder()
        stats = {}
        with patch.object(om, 'text_call', lambda qa, prompt, config: ('merged ' + prompt[-40:], dict(generated_tokens=3))), \
             patch.object(om, 'sync', lambda: None):
            for i in range(7):
                node = om.Node(id=f'leaf{i}', t_start=32 * i, t_end=32 * i + 31, level=0,
                               summary=f'event {i} kitchen' if i % 2 else f'event {i} street',
                               embedding=emb.encode([f'event {i}'])[0],
                               frames=[om.Packet(32 * i + j, image(j)) for j in range(16)])
                om.insert_root(None, emb, self.state, node, stats)
        self.assertEqual(len(self.state.roots), 4)
        self.assertEqual(self.state.merges, 3)
        self.assertGreater(stats['memory_merge_ms'], 0)
        roots = [self.state.nodes[r] for r in self.state.roots]
        self.assertEqual([r.t_start for r in roots], sorted(r.t_start for r in roots))
        merged = next(n for n in self.state.nodes.values() if n.children)
        self.assertTrue(all(self.state.nodes[c].parent == merged.id for c in merged.children))
        self.assertEqual(len(merged.frames), 16)
        picked = self.state.retrieve_events(merged.embedding)
        self.assertEqual(len(picked), 2)
        ids = {n.id for n in picked}
        for n in picked:
            self.assertFalse(set(n.children) & ids)
            self.assertNotIn(n.parent, ids)


class Query(unittest.TestCase):
    """Full query() control flow with MiniCPM, decoding and GPU profiling replaced by stubs."""

    def run_questions(self, coarse_text, times, video='v.mp4', fine_text='<answer>A</answer>'):
        calls = []

        def generate(qa, messages, max_new_tokens):
            images = sum(c.get('type') == 'image' for m in messages for c in m['content'])
            calls.append(dict(roles=[m['role'] for m in messages], images=images, text=messages[-1]['content'][-1]['text']))
            if messages[0]['role'] == 'system' and messages[0]['content'][0]['text'] is om.SUMMARY_PROMPT:
                text = 'person cooks pasta in kitchen'
            elif messages[-1]['role'] == 'user' and messages[-1]['content'][-1]['text'] == om.TOOL_CALL_PROMPT:
                text = fine_text
            elif messages[0]['role'] == 'system':
                text = coarse_text
            else:
                text = 'qa summary'
            return text, dict(preprocess_ms=1., generate_ms=10., ttft_ms=2., generated_tokens=5,
                              prompt_tokens=100, vision_tokens=64 * images, images=images)

        def frames(path, start, end, config):
            return [om.Packet(float(t), image(t)) for t in range(int(max(start, -1)) + 1, int(end) + 1) if t > start]

        def recent(qa, path, cd, fps, n, video_start=None, video_end=None):
            stamps = [video_end - 6 + i for i in range(6)]
            return SimpleNamespace(frames=[image(int(s)) for s in stamps], final_chunk_ids=list(range(6)),
                                   downsample_mode=None, cdas_metadata=dict(selected_timestamps=stamps))

        fake = types.ModuleType('fake_exact_recent')
        fake.select_exact_current_recent_frames = recent
        base = SimpleNamespace(_reset_gpu_memory_peaks=lambda: {}, _capture_gpu_memory=lambda: {},
                               _build_profile=lambda **kw: dict(gpu_peak_allocated_mb=0., gpu_peak_reserved_mb=0.),
                               RecentWindowResult=lambda **kw: SimpleNamespace(**kw))
        qa = SimpleNamespace(_oasis_embedder=Embedder(), _oasis_config=om.Config(), _last_preprocess_seconds=0.,
                             _last_model_generate_seconds=0., _last_ttft_seconds=0., _last_num_vision_tokens=0,
                             _last_num_vision_frames=0, _last_generated_tokens=0, _last_component_times={})
        results = []
        with patch.dict(sys.modules, {'main_experiments.minicpm_v46.streamingbench.eval_prism_exact_recent_dist': fake}), \
             patch('lib.minicpm.baseline', base, create=True), patch.object(om, 'generate', generate), \
             patch.object(om, 'stream_frames', frames), patch.object(om, 'sync', lambda: None), \
             patch.object(om, 'image_key', lambda f: str(f.getpixel((0, 0)))):
            import lib.minicpm as pkg
            with patch.object(pkg, 'baseline', base, create=True):
                for t in times:
                    results.append(om.query(qa, video, SB_PROMPT, 1.0, 1.0, 6, video_end=t + 1e-4)[0])
        return results, calls, qa

    def test_tool_call_triggers_fine_stage_and_answer_comes_from_it(self):
        coarse = 'Need memory. <answer>D</answer><tool_call>{"name": "rag_retrieval", "arguments": {"text_input": "pasta kitchen"}}</tool_call>'
        (result,), calls, qa = self.run_questions(coarse, [100])
        timing = result.profile_metadata['progressive_arbitration']
        meta = result.adaptive_metadata
        self.assertEqual(result.answer, 'A')
        self.assertTrue(timing['oasis_fine_triggered'])
        self.assertEqual(meta['tool_query'], 'pasta kitchen')
        self.assertEqual(timing['oasis_retrieved_events'], 2)
        self.assertEqual(qa._oasis_state.events_created, 3)  # 100 s at 1 fps -> 3 full 32-frame events.
        coarse_call = next(c for c in calls if c['roles'] == ['system', 'user'] and c['images'] == 6 + 32)
        fine_call = next(c for c in calls if c['roles'] == ['system', 'user', 'assistant', 'user'])
        self.assertEqual(fine_call['images'], 38 + 2 * 16)
        self.assertEqual(timing['final_generation_ms'], 10.)
        self.assertEqual(timing['vlm_selection_total_ms'], 10.)
        self.assertGreater(timing['memory_maintenance_total_ms'], 0)
        self.assertEqual(len(meta['recent_frame_hashes']), 6)
        self.assertIsNone(timing['video_io_ms'])

    def test_no_tool_call_answers_from_coarse_and_stream_is_reused(self):
        results, calls, qa = self.run_questions('It is a cup. <answer>A</answer>', [40, 90])
        first, second = (r.profile_metadata['progressive_arbitration'] for r in results)
        self.assertFalse(first['oasis_fine_triggered'])
        self.assertEqual(results[0].answer, 'A')
        self.assertEqual(first['vlm_selection_total_ms'], 0.)
        self.assertIsNone(first['oasis_reused_stream_until_s'])
        self.assertEqual(second['oasis_reused_stream_until_s'], 39.)
        self.assertEqual(second['oasis_qa_history'], 1)
        self.assertEqual(qa._oasis_state.events_created, 2)


if __name__ == '__main__':
    unittest.main()
