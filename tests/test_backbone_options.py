"""MiniCPM-only image options reach the processor and generate() for MiniCPM, never for other HF backbones."""
import unittest
from types import SimpleNamespace

from lib.minicpm import baseline


class Processor:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        return 'inputs'


class BackboneOptions(unittest.TestCase):
    def test_minicpm_keeps_its_options(self):
        qa = SimpleNamespace(backbone='minicpm', downsample_mode='16x', max_slice_nums=1, processor=Processor())
        baseline.apply_template(qa, [], tokenize=True)
        self.assertEqual(qa.processor.calls[0]['processor_kwargs']['downsample_mode'], '16x')
        self.assertEqual(baseline.generation_options(qa), {'downsample_mode': '16x'})

    def test_qwen_gets_none_of_them(self):
        qa = SimpleNamespace(backbone='hf', downsample_mode=None, max_slice_nums=1, processor=Processor())
        baseline.apply_template(qa, [], tokenize=True)
        self.assertEqual(qa.processor.calls, [{'tokenize': True}])
        self.assertEqual(baseline.generation_options(qa), {})


if __name__ == '__main__':
    unittest.main()
