import concurrent.futures
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from split_poc.pd_scheduler import PDScheduler
from split_poc.pd_kv import KVTransfer
from split_poc.runtime import KVPool

class ReplicaTests(unittest.TestCase):
    def scheduler(self):
        s=PDScheduler.__new__(PDScheduler)
        s.args=SimpleNamespace(prefill_replicas=2,pd_prefill_window=3,pd_epoch='epoch',tcp_buffer_mib=0)
        s.window=2;s.routes={};s.active=[];s.reserving={};s.next_replica=0;s.inflight=[]
        s.prefill_clients=[Mock(),Mock()];s.prefill_controls=[Mock(),Mock()]
        s.prefill_urls=['http://p0','http://p1'];s.decode_http=Mock()
        return s

    def job(self,rid,n=4096):
        return SimpleNamespace(id=rid,ids=[1]*n,position=0,front_position=0,prefilled=False,
            outstanding=0,cancelled=False,finished=False)

    def test_work_balancing_includes_pending_reservations_and_sticky_routing(self):
        s=self.scheduler();a=self.job('a');b=self.job('b');c=self.job('c')
        self.assertEqual(s.assign_replica(a),0);s.reserving['a']=(a,None)
        self.assertEqual(s.assign_replica(b),1);s.active.append(b)
        a.position=3072
        self.assertEqual(s.assign_replica(c),0)
        command={'phase':'prefill','items':[{'request_id':'b'}]}
        self.assertIs(s.http_for(command),s.prefill_clients[1])
        command['phase']='decode';self.assertIs(s.http_for(command),s.decode_http)
        with self.assertRaisesRegex(ValueError,'spans'):
            s.control_post('/release',{'ids':['a','b']})
        s.control_post('/pd/wait',{'request_id':'b'})
        s.prefill_controls[1].post.assert_called_once()

    def test_full_peer_does_not_block_other_peer_or_decode(self):
        s=self.scheduler();a=self.job('a');b=self.job('b');d=self.job('d')
        d.front_position=4096;d.prefilled=True;d.pd_ready=concurrent.futures.Future()
        d.pd_ready.set_result({'state':'ready'});d.pd_logged=True
        s.routes={'a':0,'b':1,'d':0};s.active=[a,b,d]
        s.inflight=[{'command':{'phase':'prefill','items':[{'request_id':'a'}]}} for _ in range(3)]
        self.assertEqual(s.eligible(),[b,d]);self.assertTrue(s.can_submit())
        s.inflight.extend([{'command':{'phase':'decode'}}]*2)
        self.assertEqual(s.eligible(),[b])

    def test_shared_pool_ownership_and_independent_copy_queues(self):
        t=KVTransfer.__new__(KVTransfer);t.role='decode';t.args={'kv_blocks':8,'prefill_replicas':2}
        t.runner=SimpleNamespace(pool=KVPool(8));t.records={};t.released=set()
        t.reserve('a',17,33,0);t.reserve('b',17,33,1)
        self.assertTrue(set(t.runner.pool.requests['a']['blocks']).isdisjoint(t.runner.pool.requests['b']['blocks']))
        with self.assertRaisesRegex(ValueError,'another P'):
            t._execute({'op':'pd_release','ids':['b'],'prefill_replica':0})
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            t.reserve('b',17,33,0)
        f0=concurrent.futures.Future();f1=concurrent.futures.Future()
        t.threads={0:Mock(),1:Mock()};t.threads[0].submit.return_value=f0;t.threads[1].submit.return_value=f1
        t.start('a');t.start('b');f1.set_result({});t.commit('b')
        self.assertEqual(t.runner.pool.requests['b']['length'],17)
        self.assertEqual(t.runner.pool.requests['a']['length'],0)
        with self.assertRaisesRegex(ValueError,'in-flight'):t.release(['a'])
        t.release(['b']);self.assertIn('a',t.records)
        f0.set_result({});t.commit('a');t.release(['a'])
        self.assertEqual(len(t.runner.pool.free),8)

if __name__=='__main__':unittest.main()
