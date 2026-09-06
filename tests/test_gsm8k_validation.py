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


    def test_compare_records_single_gpu_reference_and_nonzero_error(self):
        import contextlib, hashlib, io, json, tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from scripts.gsm8k_validate import compare
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            prompts = [{'index': 1, 'prompt_ids': [10, 11, 12]}]
            (out/'prompts.json').write_text(json.dumps(prompts))
            (out/'manifest.json').write_text(json.dumps({'prompts_sha256':
                hashlib.sha256((out/'prompts.json').read_bytes()).hexdigest()}))
            values = np.tile(np.linspace(0,1,151936,dtype=np.float32),(2,1))
            for name, logits in [('native-baseline', values), ('split-baseline', values+.002)]:
                dest=out/name;dest.mkdir()
                np.savez(dest/'1.npz',logits=logits,prompt_ids=[10,11,12],tokens=[20,21],
                    forced_tokens=[20,21],rows_json=np.array(json.dumps(self.rows())))
            (out/'native-baseline/config.json').write_text(json.dumps({'tensor_parallel_size':1}))
            with contextlib.redirect_stdout(io.StringIO()):
                compare(SimpleNamespace(output=out,compare_variants=['baseline'],metric_mode='absolute'))
            result=json.loads((out/'summary.json').read_text())
            self.assertTrue(result['passed'])
            self.assertEqual(result['variants']['baseline']['native_config']['tensor_parallel_size'],1)
            self.assertGreater(result['variants']['baseline']['decode']['mae'],.0019)
            with contextlib.redirect_stdout(io.StringIO()):
                compare(SimpleNamespace(output=out,compare_variants=['baseline']))
            ranking=json.loads((out/'summary.json').read_text())
            self.assertEqual(ranking['metric_mode'],'ranking')
            self.assertNotIn('passed',ranking)
            self.assertNotIn('mae',ranking['variants']['baseline']['decode'])
            self.assertEqual(ranking['variants']['baseline']['decode']['top_k_overlap']['10']['mean'],1)

    def test_ranking_scale_invariance_and_overlap(self):
        from scripts.gsm8k_validate import ranking_metrics
        a=np.array([4.,3.,2.,1.])
        same=ranking_metrics(a,a*10,ks=(1,2,3))
        self.assertAlmostEqual(same['cosine_similarity'],1)
        self.assertTrue(same['top1_agreement'])
        self.assertEqual(same['top_k_overlap']['2'],1)
        swapped=ranking_metrics(a,np.array([4.,2.,3.,1.]),ks=(1,2,3))
        self.assertEqual(swapped['top_k_overlap']['2'],.5)
        self.assertEqual(swapped['top_k_overlap']['3'],1)
        self.assertNotIn('mae',swapped)

    def test_ranking_ties_and_zero_vectors(self):
        from scripts.gsm8k_validate import ranking_metrics
        tied=ranking_metrics(np.array([2.,2.,1.]),np.array([2.,2.,1.]),ks=(1,2))
        self.assertEqual(tied['native_top_tokens'],[0,1])
        with self.assertRaises(ValueError):ranking_metrics(np.zeros(3),np.ones(3),ks=(1,))


if __name__ == '__main__': unittest.main()
