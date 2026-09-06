import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import httpx

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from mixed_arrival import pair


class Stream(httpx.AsyncByteStream):
    def __init__(self, body, truncate=False):
        self.body, self.truncate = body, truncate

    async def __aiter__(self):
        n = self.body["max_tokens"]
        for i in range(n):
            await asyncio.sleep(0)
            yield b'data: {"choices":[{"text":""}]}\n\n'
        yield ("data: "+json.dumps({"choices":[],"usage":{"prompt_tokens":len(self.body["prompt"]),
                                "completion_tokens":n}})+"\n\n").encode()
        if not self.truncate:
            yield b"data: [DONE]\n\n"


class MixedArrivalTests(unittest.IsolatedAsyncioTestCase):
    async def test_trigger_and_usage(self):
        async def handle(request):
            return httpx.Response(200,stream=Stream(json.loads(request.content)))
        args = SimpleNamespace(trigger_after=2,decode_osl=16,long_isl=32,long_osl=4)
        async with httpx.AsyncClient(base_url="http://test",transport=httpx.MockTransport(handle)) as client:
            rows = await pair(client,[1],args,"test-")
        self.assertEqual([len(x["event_monotonic_ns"]) for x in rows],[16,4])
        self.assertGreater(rows[0]["long_arrival_after_start_ms"],0)

    async def test_truncated_stream_fails_and_cancels_peer(self):
        async def handle(request):
            return httpx.Response(200,stream=Stream(json.loads(request.content),truncate=True))
        args = SimpleNamespace(trigger_after=2,decode_osl=16,long_isl=32,long_osl=4)
        async with httpx.AsyncClient(base_url="http://test",transport=httpx.MockTransport(handle)) as client:
            with self.assertRaises(AssertionError):
                await pair(client,[1],args,"test-")


if __name__ == "__main__":
    unittest.main()
