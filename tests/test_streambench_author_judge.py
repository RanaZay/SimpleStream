import unittest

from main_experiments.minicpm_v46.streambench_v03.run_author_judge import audit


class AuthorJudgeTests(unittest.TestCase):
    def setUp(self):
        self.inputs = [dict(id="0:0", question="What?", label="A cup", predict="A mug", **{"class": "OS"})]
        self.outputs = [dict(self.inputs[0], llama_pred="yes", score=4.8)]

    def test_semantic_accuracy_not_lexical_or_score(self):
        self.assertEqual(audit(self.inputs, self.outputs, 1)["accuracy_percent"], 100)

    def test_incomplete(self):
        with self.assertRaises(ValueError):
            audit(self.inputs, [], 1)

    def test_duplicate(self):
        with self.assertRaises(ValueError):
            audit(self.inputs, self.outputs * 2, 1)

    def test_invalid_judgment_not_silently_wrong(self):
        for changes in ({"llama_pred": "maybe"}, {"score": float("nan")}, {"score": True}):
            with self.assertRaises(ValueError):
                audit(self.inputs, [dict(self.outputs[0], **changes)], 1)

    def test_changed_prediction(self):
        with self.assertRaises(ValueError):
            audit(self.inputs, [dict(self.outputs[0], predict="changed")], 1)


if __name__ == "__main__":
    unittest.main()
