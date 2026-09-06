import concurrent.futures
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from split_poc.runtime import KVPool
from split_poc.pd_kv import KVTransfer,head_routes
from split_poc.pd_control import PDControl
from split_poc.pd_scheduler import PDScheduler


class PDTests(unittest.TestCase):
    def test_reserved_noncontiguous_pages_preserve_computed_length(self):
        pool=KVPool(8)
        pool.reserve('a',33);pool.reserve('b',17);pool.release(['a'])
        pool.reserve('c',65)
        blocks=pool.requests['c']['blocks'][:]
        self.assertEqual(pool.requests['c']['length'],0)
        self.assertNotEqual(blocks,list(range(min(blocks),min(blocks)+len(blocks))))
        items=[dict(request_id='c',position=0,query_len=17)]
        slots,_=pool.prepare(items);pool.commit(items)
        self.assertEqual(slots[16],blocks[1]*16)
        self.assertEqual(pool.requests['c']['length'],17)
        self.assertEqual(pool.requests['c']['blocks'],blocks)

    def test_insufficient_reservation_is_atomic(self):
        pool=KVPool(2);before=pool.free[:]
        with self.assertRaises(ValueError):pool.reserve('a',33)
        self.assertEqual(pool.free,before);self.assertEqual(pool.requests,{})

    def test_head_ownership_all_supported_tp_pairs(self):
        for source in [1,2]:
            for dest in [1,2]:
                routes=head_routes(source,dest)
                self.assertEqual([r[0] for r in routes],[0,1])
                for head,s,sl,d,dl in routes:
                    self.assertEqual(s*(2//source)+sl,head)
                    self.assertEqual(d*(2//dest)+dl,head)
        with self.assertRaises(ValueError):head_routes(4,1)

    def transfer(self):
        obj=KVTransfer.__new__(KVTransfer)
        obj.role='decode';obj.records={};obj.released=set();obj.args={'kv_blocks':8}
        obj.runner=SimpleNamespace(pool=KVPool(8))
        obj.reserve('a',17,33)
        return obj

    def test_no_commit_or_reuse_before_cuda_copy_completion(self):
        obj=self.transfer();future=concurrent.futures.Future()
        obj.records['a'].update(future=future,state='copying')
        with self.assertRaises(ValueError):obj.commit('a')
        with self.assertRaises(ValueError):obj.release(['a'])
        self.assertEqual(obj.runner.pool.requests['a']['length'],0)
        future.set_result({'checksums':{}})
        obj.commit('a');obj.commit('a')
        self.assertEqual(obj.runner.pool.requests['a']['length'],17)
        obj.release(['a']);obj.release(['a'])
        self.assertEqual(len(obj.runner.pool.free),8)
        with self.assertRaises(ValueError):obj.reserve('a',17,33)

    def test_failed_shard_cannot_become_ready(self):
        obj=self.transfer();future=concurrent.futures.Future()
        future.set_exception(RuntimeError('missing shard'))
        obj.records['a'].update(future=future,state='copying')
        with self.assertRaisesRegex(RuntimeError,'missing shard'):obj.commit('a')
        self.assertEqual(obj.runner.pool.requests['a']['length'],0)

    def test_cancel_drains_before_remote_release(self):
        obj=PDControl.__new__(PDControl);future=concurrent.futures.Future()
        obj.records={'a':{'future':future}};obj.condition=threading.Condition()
        obj.remote=Mock();obj.executor=Mock()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            release=pool.submit(obj.release,['a'])
            time.sleep(.01);obj.remote.assert_not_called()
            future.set_result(None);release.result(timeout=1)
        obj.remote.assert_called_once_with({'op':'pd_release','ids':['a']})
        self.assertEqual(obj.records,{})

    def test_pending_kv_and_full_prefill_window_do_not_block_ready_decode(self):
        obj=PDScheduler.__new__(PDScheduler);obj.window=2
        obj.inflight=[{'command':{'phase':'prefill'}} for _ in range(2)]
        def job(ready):
            f=concurrent.futures.Future()
            if ready:f.set_result({'state':'ready'})
            return SimpleNamespace(cancelled=False,finished=False,front_position=17,ids=[1]*17,
                                   prefilled=True,outstanding=0,pd_ready=f,pd_logged=True)
        waiting,ready=job(False),job(True);obj.active=[waiting,ready]
        self.assertEqual(obj.eligible(),[ready]);self.assertTrue(obj.can_submit())


if __name__=='__main__':unittest.main()
