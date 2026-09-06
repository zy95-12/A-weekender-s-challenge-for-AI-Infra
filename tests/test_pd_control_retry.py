import concurrent.futures
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import httpx
from split_poc.transport import post_control,http_client
from split_poc.pd_kv import KVTransfer
from split_poc.runtime import KVPool

class ControlRetryTests(unittest.TestCase):
    def test_retry_after_applied_start_does_not_enqueue_a_second_copy(self):
        t=KVTransfer.__new__(KVTransfer);t.role='decode';t.args={'kv_blocks':8,'prefill_replicas':2}
        t.runner=SimpleNamespace(pool=KVPool(8));t.records={};t.released=set()
        t.reserve('a',17,33,1);t.threads={1:Mock()}
        t.threads[1].submit.return_value=concurrent.futures.Future()
        body={'op':'pd_start','request_id':'a','prefill_replica':1,'start':0,'end':17}
        client=Mock();fresh=Mock();manager=Mock();manager.__enter__=Mock(return_value=fresh);manager.__exit__=Mock(return_value=False)
        def dropped(*args,**kwargs):
            t._execute(body);raise httpx.RemoteProtocolError('response lost after operation')
        client.post.side_effect=dropped
        def replay(*args,**kwargs):
            result=t._execute(body);response=Mock();response.json.return_value=result;return response
        fresh.post.side_effect=replay
        with patch('split_poc.transport.http_client',return_value=manager):
            self.assertTrue(post_control(client,'http://d','/pd/local',body)['ok'])
        t.threads[1].submit.assert_called_once();fresh.post.assert_called_once()

    def test_never_retries_forward_or_http_error_and_bounds_transport_retry(self):
        client=Mock()
        with self.assertRaises(ValueError):post_control(client,'http://p','/forward',{})
        client.post.assert_not_called()
        response=Mock();response.raise_for_status.side_effect=ValueError('HTTP failure');client.post.return_value=response
        with patch('split_poc.transport.http_client') as factory:
            with self.assertRaises(ValueError):post_control(client,'http://p','/release',{'ids':['a']})
            factory.assert_not_called()
        client.post.side_effect=httpx.RemoteProtocolError('lost')
        with patch('split_poc.transport.http_client') as factory:
            factory.return_value.__enter__.return_value.post.side_effect=httpx.RemoteProtocolError('lost twice')
            with self.assertRaises(httpx.RemoteProtocolError):post_control(client,'http://p','/release',{'ids':['a']})
            factory.assert_called_once()

    def test_custom_tcp_transport_honors_short_idle_expiry(self):
        with http_client(16,limits=httpx.Limits(keepalive_expiry=1)) as client:
            self.assertEqual(client._transport._pool._keepalive_expiry,1)


class KeepaliveTests(unittest.TestCase):
    def test_server_idle_deadline_exceeds_client_idle_deadline(self):
        from split_poc.transport import serve
        with patch('uvicorn.run') as run:
            serve(Mock(),'127.0.0.1',8000)
            self.assertEqual(run.call_args.kwargs['timeout_keep_alive'],60)

    def test_decode_data_pool_expires_before_server(self):
        import tempfile
        from split_poc.pd_scheduler import PDScheduler
        with tempfile.TemporaryDirectory() as folder,patch('split_poc.pipeline.PipelineScheduler.__init__'):
            args=SimpleNamespace(max_active=96,cloud='http://p0',cloud_prefill_secondary='http://p1',
                cloud_decode='http://d',tcp_buffer_mib=0,prefill_replicas=2,results=folder)
            s=PDScheduler(None,args,set())
            try:
                for client in [s.decode_http]+s.prefill_clients+s.prefill_controls:
                    self.assertEqual(client._transport._pool._keepalive_expiry,1)
            finally:
                for client in [s.decode_http]+s.prefill_clients+s.prefill_controls:client.close()
                s.control.shutdown();s.pd_trace.close()

if __name__=='__main__':unittest.main()
