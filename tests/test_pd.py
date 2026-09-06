import concurrent.futures
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from split_poc.runtime import KVPool,Executor
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
        obj.records['a'].update(future=future,state='copying',enqueued_until=17)
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
        obj.records['a'].update(future=future,state='copying',enqueued_until=17)
        with self.assertRaisesRegex(RuntimeError,'missing shard'):obj.commit('a')
        self.assertEqual(obj.runner.pool.requests['a']['length'],0)

    def test_pd_status_bypasses_busy_gpu_command_lock(self):
        obj=Executor.__new__(Executor);obj.lock=threading.Lock();obj.pd_lock=threading.Lock()
        obj.healthy=True;pipe=Mock();obj.pd_pipes=[pipe]
        pipe.poll.return_value=True;pipe.recv.return_value={'result':{'a':{'state':'copied'}}}
        pool=concurrent.futures.ThreadPoolExecutor()
        obj.lock.acquire()
        try:
            future=pool.submit(obj.call,{'op':'pd_status','ids':['a']})
            self.assertEqual(future.result(timeout=1),{'ranks':[{'a':{'state':'copied'}}]})
        finally:
            obj.lock.release();pool.shutdown(wait=True)

    def test_chunk_ranges_are_contiguous_idempotent_and_final_only_commit(self):
        obj=self.transfer();obj.thread=Mock()
        first=concurrent.futures.Future();last=concurrent.futures.Future()
        obj.thread.submit.side_effect=[first,last]
        with self.assertRaisesRegex(ValueError,'Partial'):obj.start('a',0,15)
        with self.assertRaisesRegex(ValueError,'Noncontiguous'):obj.start('a',16,17)
        obj.start('a',0,16);obj.start('a',0,16)
        self.assertEqual(obj.thread.submit.call_count,1)
        first.set_result({})
        with self.assertRaisesRegex(ValueError,'incomplete'):obj.commit('a')
        obj.start('a',16,17)
        with self.assertRaisesRegex(ValueError,'incomplete'):obj.commit('a')
        with self.assertRaisesRegex(ValueError,'in-flight'):obj.release(['a'])
        last.set_result({});obj.commit('a')
        self.assertEqual(obj.runner.pool.requests['a']['length'],17)
        obj.release(['a']);self.assertEqual(len(obj.runner.pool.free),8)

    def test_source_cannot_export_uncomputed_pages(self):
        obj=self.transfer();obj.role='prefill';obj.thread=Mock()
        with self.assertRaisesRegex(ValueError,'computed prefix'):obj.start('a',0,16)
        obj.records['a'].update(computed_until=16,event=object())
        obj.start('a',0,16)
        with self.assertRaisesRegex(ValueError,'computed prefix'):obj.start('a',16,17)

    def test_chunk_boundary_rounds_down_until_final(self):
        obj=PDControl.__new__(PDControl);obj.condition=threading.Condition()
        obj.args=SimpleNamespace(pd_chunk_transfer=True);obj.tasks=Mock()
        obj.records={'a':dict(signature=(33,49),enqueued_until=0,future=None,state='prefill')}
        def forward(pos,n):
            obj.after_forward(dict(phase='prefill',items=[dict(request_id='a',position=pos,query_len=n)]))
        forward(0,15);obj.tasks.submit.assert_not_called()
        forward(15,14);self.assertEqual(obj.tasks.submit.call_args.args[2:4],(0,16))
        previous=obj.records['a']['future']
        forward(29,4);self.assertEqual(obj.tasks.submit.call_args.args[2:],(16,33,previous))

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
        obj=PDScheduler.__new__(PDScheduler);obj.window=2;obj.args=SimpleNamespace(pd_prefill_window=0)
        obj.inflight=[{'command':{'phase':'prefill'}} for _ in range(2)]
        def job(ready):
            f=concurrent.futures.Future()
            if ready:f.set_result({'state':'ready'})
            return SimpleNamespace(cancelled=False,finished=False,front_position=17,ids=[1]*17,
                                   prefilled=True,outstanding=0,pd_ready=f,pd_logged=True)
        waiting,ready=job(False),job(True);obj.active=[waiting,ready]
        self.assertEqual(obj.eligible(),[ready]);self.assertTrue(obj.can_submit())

    def test_larger_prefill_window_preserves_decode_limit(self):
        obj=PDScheduler.__new__(PDScheduler);obj.window=2
        for size in [3,4]:
            obj.args=SimpleNamespace(pd_prefill_window=size)
            p=SimpleNamespace(cancelled=False,finished=False,front_position=0,ids=[1]*4096,prefilled=False,outstanding=0)
            f=concurrent.futures.Future();f.set_result({'state':'ready'})
            d=SimpleNamespace(cancelled=False,finished=False,front_position=4096,ids=[1]*4096,
                prefilled=True,outstanding=0,pd_ready=f,pd_logged=True)
            obj.active=[p,d]
            obj.inflight=[{'command':{'phase':'prefill'}} for _ in range(size-1)]+[{'command':{'phase':'decode'}}]*2
            self.assertEqual(obj.eligible(),[p]);self.assertTrue(obj.can_submit())
            obj.inflight.append({'command':{'phase':'prefill'}})
            self.assertEqual(obj.eligible(),[]);self.assertFalse(obj.can_submit())
            obj.inflight.pop(0)
            obj.inflight.pop()
            obj.inflight.pop()
            self.assertIn(d,obj.eligible())


if __name__=='__main__':unittest.main()
