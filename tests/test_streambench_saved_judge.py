import copy
import unittest

from main_experiments.minicpm_v46.streambench_v03.evaluate_saved_answers import prepare, report, markdown


class SavedJudgeTests(unittest.TestCase):
    def setUp(self):
        self.payload = {'results': [dict(video_index=0, breakpoint_index=i, subtask='OS',
            question='What is shown?', answer_gt='A cup', methods=[dict(method='prism',
            prediction='A mug', adaptive={'mode': 'progressive_arbitration_exact_recent6_control'})])
            for i in range(2)]}

    def test_baseline_label_and_no_dropped_errors(self):
        self.payload['results'][1]['methods'][0] = dict(method='prism', error='decode failure')
        inputs, coverage = prepare(self.payload, 'prism', 2)
        summary = report(inputs, [dict(inputs[0], llama_pred='yes', score=5)], coverage)
        self.assertEqual(summary['method'], 'Recent-6')
        self.assertEqual(summary['accuracy_percent'], 50)
        self.assertEqual(summary['valid_predictions_accuracy_percent'], 100)
        self.assertFalse(summary['error_free_evaluation'])
        self.assertEqual(summary['subtasks']['OS']['total'], 2)
        self.assertIn('not an error-free', markdown(summary))

    def test_all_judged_and_correctness_not_score(self):
        inputs, coverage = prepare(self.payload, 'prism', 2)
        judged = [dict(row, llama_pred='no', score=3) for row in inputs]
        self.assertEqual(report(inputs, judged, coverage)['accuracy_percent'], 0)

    def test_duplicate_question_rejected(self):
        self.payload['results'][1] = copy.deepcopy(self.payload['results'][0])
        with self.assertRaises(ValueError):
            prepare(self.payload, 'prism', 2)

    def test_duplicate_method_rejected(self):
        self.payload['results'][0]['methods'] *= 2
        with self.assertRaises(ValueError):
            prepare(self.payload, 'prism', 2)

    def test_mixed_modes_rejected(self):
        self.payload['results'][1]['methods'][0]['adaptive']['mode'] = 'different'
        with self.assertRaises(ValueError):
            prepare(self.payload, 'prism', 2)

    def test_missing_output_rejected(self):
        inputs, coverage = prepare(self.payload, 'prism', 2)
        with self.assertRaises(ValueError):
            report(inputs, [dict(inputs[0], llama_pred='yes', score=5)], coverage)

    def test_empty_output_is_explicit_failure(self):
        self.payload['results'][1]['methods'][0]['prediction'] = ''
        inputs, coverage = prepare(self.payload, 'prism', 2)
        self.assertEqual(len(inputs), 1)
        self.assertEqual(coverage['failures'][0]['error'], 'empty_prediction')


if __name__ == '__main__':
    unittest.main()
