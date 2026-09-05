import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from split_poc.pipeline import PipelineScheduler
from split_poc.server import Job


class FakeExecutor:
    healthy = True
    def __init__(self):
        self.calls=[]
    def call(self,command,arrays=None):
        self.calls.append(command)
        if command['op']=='release':
            return {'kv_used_blocks':0}
        n=sum(i['query_len'] for i in command['items'])
        if command['op']=='front':
            return {'arrays':[np.zeros((n,2))]*2,'timings':{},'front_kv_used_blocks':1}
        return {'tokens':[7]*len(command['items']),'logits':None,'timings':{},'kv_used_blocks':1}


class PipelineTests(unittest.TestCase):
    def make(self,folder):
        args=SimpleNamespace(results=folder,pipeline_window=2,kv_blocks=16,max_active=4,
            cloud='http://unused',tcp_buffer_mib=0,wire_fast=True,prefill_chunk_size=2,
            scheduler_policy='decode-first',decode_quota=1)
        return PipelineScheduler(FakeExecutor(),args,set())

    def test_second_front_runs_while_first_rpc_waits_and_cancel_drains(self):
        first=threading.Event(); second=threading.Event(); release=threading.Event()
        def rpc(scheduler,command,arrays):
            if command['items'][0]['position']==0:
                first.set()
                if not release.wait(3):
                    raise RuntimeError('test stalled')
            else:
                second.set()
            return arrays,{}
        with tempfile.TemporaryDirectory() as folder,patch('split_poc.server.httpx.Client'), \
                patch('split_poc.pipeline.http_client'),patch.object(PipelineScheduler,'rpc',rpc):
            scheduler=self.make(folder)
            job=Job([1]*5,2,True)
            try:
                scheduler.submit(job)
                self.assertTrue(first.wait(2))
                self.assertTrue(second.wait(2),'GPU front was blocked by network')
                self.assertEqual(job.outstanding,2)
                job.cancelled=True
                self.assertFalse(any(c['op']=='release' for c in scheduler.executor.calls))
                release.set()
                deadline=time.monotonic()+2
                while scheduler.active and time.monotonic()<deadline:
                    time.sleep(.01)
                self.assertFalse(scheduler.active)
                self.assertEqual(job.tokens,[])
                self.assertEqual(scheduler.admission.reservations,{})
                commands=scheduler.executor.calls
                self.assertEqual([c['items'][0]['position'] for c in commands if c['op']=='back'],[0,2])
                self.assertEqual(commands[-1]['op'],'release')
            finally:
                release.set();scheduler.closed=True;scheduler.thread.join(3)
                scheduler.http.close();scheduler.trace.close()
            self.assertFalse(scheduler.thread.is_alive())

    def test_rpc_failure_fails_closed_and_rejects_new_work(self):
        def rpc(*args):
            raise RuntimeError('injected WAN failure')
        with tempfile.TemporaryDirectory() as folder,patch('split_poc.server.httpx.Client'), \
                patch('split_poc.pipeline.http_client'),patch.object(PipelineScheduler,'rpc',rpc):
            scheduler=self.make(folder);job=Job([1]*5,2,True)
            try:
                scheduler.submit(job)
                self.assertIn('WAN failure',job.events.get(timeout=2)['error'])
                self.assertFalse(scheduler.executor.healthy)
                from fastapi import HTTPException
                with self.assertRaises(HTTPException):
                    scheduler.submit(Job([1],2,True))
                self.assertFalse(any(c['op']=='release' for c in scheduler.executor.calls))
            finally:
                scheduler.closed=True;scheduler.thread.join(3)
                scheduler.http.close();scheduler.trace.close()
