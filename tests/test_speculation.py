import unittest
from split_poc.speculation import propose,confirm
from split_poc.runtime import KVPool

class SpecTests(unittest.TestCase):
    def test_lookup_and_no_match(self):
        self.assertEqual(propose([1,2,3,4,1,2],2),[3,4])
        self.assertEqual(propose([1,2,3,4],4),[])
        self.assertEqual(propose([1,2,1,2],0),[])
    def test_first_middle_and_full_acceptance(self):
        self.assertEqual(confirm([1,2,3],[9,2,3,4],9,set(),True),([9],0))
        self.assertEqual(confirm([1,2,3],[1,9,3,4],9,set(),True),([1,9],1))
        self.assertEqual(confirm([1,2,3],[1,2,3,4],9,set(),True),([1,2,3,4],3))
    def test_eos_and_output_limit(self):
        self.assertEqual(confirm([1,2,3],[1,2,3,4],9,{2},False),([1,2],2))
        self.assertEqual(confirm([1,2,3],[1,2,3,4],2,set(),True),([1,2],2))
        self.assertEqual(confirm([],[9],1,set(),True),([9],0))
    def test_rollback_frees_only_suffix_and_can_append(self):
        pool=KVPool(4)
        item={'request_id':'r','position':0,'query_len':35}
        pool.prepare([item]);pool.commit([item])
        old=pool.requests['r']['blocks'][:]
        pool.truncate({'r':17})
        self.assertEqual(pool.requests['r'],{'length':17,'blocks':old[:2]})
        self.assertEqual(len(pool.free),2)
        pool.prepare([{'request_id':'r','position':17,'query_len':3}])
        with self.assertRaises(ValueError):pool.truncate({'r':50})
        self.assertEqual(pool.requests['r']['length'],17)
        pool.release(['r'])
        self.assertEqual(len(pool.free),4)

class SchedulerSpecTests(unittest.TestCase):
    def test_rejection_rolls_back_both_sides_before_emitting_and_resumes(self):
        import tempfile,time
        from types import SimpleNamespace
        from unittest.mock import patch
        import numpy as np
        from split_poc.pipeline import PipelineScheduler
        from split_poc.server import Job
        class Executor:
            healthy=True
            def __init__(self):self.commands=[]
            def call(self,c,arrays=None):
                self.commands.append(c)
                if c['op'] in {'release','truncate'}:return {'kv_used_blocks':0}
                if c['op']=='front':
                    return {'arrays':[np.zeros((sum(i['query_len'] for i in c['items']),2))]*2,
                            'timings':{},'front_kv_used_blocks':1}
                tokens=[4,9,8] if c.get('verify') else [3 if c['items'][0]['position']==0 else 7]
                return {'tokens':tokens,'logits':None,'timings':{},'kv_used_blocks':1}
        def rpc(s,c,a):return a,{}
        with tempfile.TemporaryDirectory() as folder,patch('split_poc.server.httpx.Client') as http, \
                patch('split_poc.pipeline.http_client'),patch.object(PipelineScheduler,'rpc',rpc):
            args=SimpleNamespace(results=folder,pipeline_window=2,kv_blocks=16,max_active=4,
                cloud='http://unused',tcp_buffer_mib=0,wire_fast=True,prefill_chunk_size=0,
                scheduler_policy='decode-first',decode_quota=1,speculative_tokens=2)
            ex=Executor();s=PipelineScheduler(ex,args,set());job=Job([1,2,3,4,1,2],4,True)
            try:
                s.submit(job)
                events=[job.events.get(timeout=2) for _ in range(4)]
                self.assertEqual([e['token'] for e in events],[3,4,9,7])
                self.assertTrue(events[-1]['done'])
                fronts=[c for c in ex.commands if c['op']=='front']
                self.assertEqual([c['items'][0]['position'] for c in fronts],[0,6,8])
                self.assertEqual([c['items'][0]['query_len'] for c in fronts],[6,3,1])
                self.assertEqual([c['lengths'] for c in ex.commands if c['op']=='truncate'],[{job.id:8}])
                http.return_value.post.assert_any_call('/truncate',json={'lengths':{job.id:8}})
            finally:
                s.closed=True;s.thread.join(3);s.http.close();s.trace.close()
            self.assertFalse(s.thread.is_alive())

class VerificationShapeTests(unittest.TestCase):
    def test_grouped_transport_preserves_q1_calls_and_full_logits(self):
        import numpy as np
        from split_poc.runtime import Runner
        class RecordingRunner(Runner):
            def __init__(self):
                self.rank=0;self.args={'speculative_tokens':2};self.calls=[]
            def execute(self,command,arrays=None):
                if command['phase']=='verify':
                    return super().execute(command,arrays)
                self.calls.append((command,arrays))
                return {'tokens':[command['items'][0]['position']],
                        'logits':np.ones((1,3))*command['items'][0]['position'],
                        'timings':{'compute_ms':2},'kv_used_blocks':1}
        r=RecordingRunner()
        out=r.execute({'op':'back','phase':'verify','items':[{'request_id':'r','position':10,'query_len':3}],
                       'verify':True,'capture':True,'batch_id':'b'},[np.ones((3,2048))]*2)
        self.assertEqual([c['items'][0]['position'] for c,a in r.calls],[10,11,12])
        self.assertTrue(all(c['phase']=='decode' and c['items'][0]['query_len']==1 and a[0].shape==(1,2048)
                            for c,a in r.calls))
        self.assertEqual(out['tokens'],[10,11,12])
        self.assertEqual(out['logits'].shape,(3,3))
        self.assertEqual(out['timings']['compute_ms'],6)
