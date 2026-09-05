import socket
import unittest
from unittest.mock import patch
from split_poc.transport import PreconnectBackend, http_client, set_buffers


class FakeSocket:
    def __init__(self):
        self.events=[]
    def settimeout(self,value):
        self.events.append(('timeout',value))
    def setsockopt(self,*args):
        self.events.append(('option',args))
    def getsockopt(self,*args):
        return 32*1024**2
    def connect(self,address):
        self.events.append(('connect',address))
    def close(self):
        self.events.append(('close',))


class TransportTests(unittest.TestCase):
    def test_buffers_set_before_connect(self):
        fake=FakeSocket()
        with patch('split_poc.transport.socket.socket',return_value=fake):
            stream=PreconnectBackend(16).connect_tcp('127.0.0.1',8099,timeout=2)
        index=next(i for i,e in enumerate(fake.events) if e[0]=='connect')
        options=[e[1][1] for e in fake.events[:index] if e[0]=='option']
        self.assertIn(32,options)
        self.assertIn(33,options)
        stream.close()

    def test_disabled_uses_default_backend(self):
        with http_client(0) as client:
            self.assertNotIsInstance(client._transport._pool._network_backend,PreconnectBackend)

    def test_invalid_or_rejected_buffers_fail(self):
        with self.assertRaises(ValueError):
            set_buffers(FakeSocket(),0)
        fake=FakeSocket()
        fake.getsockopt=lambda *args:4096
        with self.assertRaises(RuntimeError):
            set_buffers(fake,16)


if __name__=='__main__':
    unittest.main()
