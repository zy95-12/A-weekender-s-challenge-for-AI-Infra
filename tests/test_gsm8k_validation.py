import unittest
import numpy as np
from scripts.gsm8k_validate import validate_alignment, logit_metrics


class LogitsTests(unittest.TestCase):
    def rows(self):
        return [dict(token_id=12, absolute_position=2, num_computed_tokens=0, query_len=3, phase='prefill'),
                dict(token_id=20, absolute_position=3, num_computed_tokens=3, query_len=1, phase='decode')]

    def test_same_position_and_forced_input_required(self):
        validate_alignment([10,11,12], [20,21], self.rows())
        rows = self.rows(); rows[1]['absolute_position'] = 4
        with self.assertRaises(ValueError): validate_alignment([10,11,12], [20,21], rows)
        rows = self.rows(); rows[1]['token_id'] = 99
        with self.assertRaises(ValueError): validate_alignment([10,11,12], [20,21], rows)

    def test_chunked_prefill_final_row_keeps_context(self):
        rows = self.rows(); rows[0].update(num_computed_tokens=2, query_len=1)
        validate_alignment([10,11,12], [20,21], rows)
        rows[0]['num_computed_tokens'] = 0
        with self.assertRaises(ValueError): validate_alignment([10,11,12], [20,21], rows)

    def test_gate_detects_logit_corruption_without_answer_scoring(self):
        values = np.array([1.,2.,3.])
        self.assertTrue(logit_metrics(values, values)['passed'])
        self.assertFalse(logit_metrics(values, values+1)['passed'])
        with self.assertRaises(ValueError): logit_metrics(values, np.array([1.,np.nan,3.]))
        # A top1 flip near a tie is observed, but does not override the numeric gate.
        result = logit_metrics(np.array([1.,1.00001]), np.array([1.00001,1.]))
        self.assertFalse(result['top1_agreement_observation'])
        self.assertTrue(result['passed'])


if __name__ == '__main__': unittest.main()
