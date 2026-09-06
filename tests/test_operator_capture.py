import tempfile
import unittest
from unittest.mock import patch

import torch
from split_poc.operator_capture import Capture, tensors


class OperatorCaptureTests(unittest.TestCase):
    def test_records_real_mixed_dtypes_without_changing_result(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = Capture('enterprise', 0, directory)
            capture.command = {'op': 'forward', 'phase': 'decode', 'batch_id': 'batch',
                               'items': [{'position': 4096, 'query_len': 1}]}
            a = torch.arange(6, dtype=torch.float16).reshape(2, 3)
            index = torch.tensor([1], dtype=torch.int64)
            expected = a.index_select(0, index)
            with patch('torch.cuda.nvtx.range_push'), patch('torch.cuda.nvtx.range_pop'), capture:
                actual = a.index_select(0, index)
            self.assertTrue(torch.equal(actual, expected))
            record = next(r for r in capture.records if 'index_select' in r['operator'])
            self.assertEqual([t['shape'] for t in record['inputs']], [[2, 3], [1]])
            self.assertEqual([t['dtype'] for t in record['inputs']], ['torch.float16', 'torch.int64'])
            self.assertEqual(record['items'][0]['position'], 4096)

    def test_nested_inputs_and_factory_no_tensor_inputs(self):
        tensor = torch.empty(2, 3).transpose(0, 1)
        result = tensors({'nested': [tensor]})
        self.assertEqual(result[0]['shape'], [3, 2])
        self.assertEqual(result[0]['stride'], [1, 3])
        self.assertEqual(tensors(((3, 2), torch.float16)), [])


if __name__ == '__main__':
    unittest.main()
