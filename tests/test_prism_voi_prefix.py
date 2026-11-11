"""Prefix-cached answer scoring must equal one teacher-forced forward per answer (tiny random LM on CPU)."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from lib.minicpm import prism_voi as pv


class Tokenizer:
    """One token per character; the 'template' wraps the prompt and an optional assistant reply."""
    def encode(self, text, add_special_tokens=False):
        return [3 + (ord(ch) % 60) for ch in text]


class Inputs(dict):
    def __getattr__(self, name):
        return self[name]

    def to(self, device):
        return self


def build(qa, frames, prompt, assistant_text=None):
    tok = qa.processor.tokenizer
    ids = [1] + [2] * len(frames) + tok.encode(prompt) + [1]  # BOS, an 'image' token per frame, prompt, turn marker
    if assistant_text is not None:
        ids += [2] + tok.encode(assistant_text) + [1]  # empty think block, the reply, end of turn
    ids = torch.tensor([ids])
    return Inputs(input_ids=ids, attention_mask=torch.ones_like(ids))


class PrefixScoring(unittest.TestCase):
    def setUp(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.manual_seed(0)
        config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
        self.qa = SimpleNamespace(model=LlamaForCausalLM(config).eval(), processor=SimpleNamespace(tokenizer=Tokenizer()))

    def test_cached_scores_match_one_forward_per_answer(self):
        answers = ['a cup', 'a red cup', 'two pens', 'a cup']
        with patch('lib.minicpm.progressive_sufficiency._build_minicpm_inputs', build), \
             patch.dict('os.environ', {'PRISM_VOI_PREFIX_CACHE': '1'}):
            cached = pv.answer_logprobs(self.qa, [None] * 3, 'What is on the table?', answers)
            single = {a: pv.answer_logprob(self.qa, [None] * 3, 'What is on the table?', a) for a in answers}
        self.assertEqual(set(cached), {'a cup', 'a red cup', 'two pens'})
        for a in cached:
            self.assertAlmostEqual(cached[a], single[a], places=4)


if __name__ == '__main__':
    unittest.main()
