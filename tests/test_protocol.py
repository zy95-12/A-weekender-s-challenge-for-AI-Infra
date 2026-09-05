import json
import struct
import unittest
import numpy as np
from split_poc.wire import pack, unpack
from split_poc.runtime import KVPool


class ProtocolTests(unittest.TestCase):
    def test_lossless_fp16_wire(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(32, 2048)).astype(np.float16)
        b = rng.normal(size=(32, 2048)).astype(np.float16)
        meta, arrays = unpack(pack({"batch_id": "abc"}, [a, b]))
        self.assertEqual(meta["batch_id"], "abc")
        np.testing.assert_array_equal(arrays[0].view(np.uint16), a.view(np.uint16))
        np.testing.assert_array_equal(arrays[1].view(np.uint16), b.view(np.uint16))

    def test_truncated_and_trailing_payload_rejected(self):
        blob = pack({}, [np.zeros((1, 2048), dtype=np.float16)])
        for damaged in (blob[:-1], blob + b"garbage", b"\x00", struct.pack("!I", 2**31)):
            with self.assertRaises(ValueError):
                unpack(damaged)

    def test_no_object_deserialization(self):
        header = json.dumps({"meta": {}, "tensors": [{"shape": [1, 1], "dtype": "O", "bytes": 8}]}).encode()
        with self.assertRaises(ValueError):
            unpack(struct.pack("!I", len(header)) + header + b"12345678")


class KVTests(unittest.TestCase):
    def test_independent_requests_growth_and_release(self):
        pool = KVPool(8)
        a = {"request_id": "a", "position": 0, "query_len": 16}
        b = {"request_id": "b", "position": 0, "query_len": 17}
        slots, tables = pool.prepare([a, b])
        self.assertTrue(set(tables[0]).isdisjoint(tables[1]))
        self.assertEqual(len(set(slots)), 33)
        pool.commit([a, b])
        next_a = {"request_id": "a", "position": 16, "query_len": 1}
        pool.prepare([next_a])
        pool.commit([next_a])
        self.assertEqual(pool.requests["a"]["length"], 17)
        self.assertEqual(len(pool.requests["a"]["blocks"]), 2)
        pool.release(["a", "b", "a"])
        self.assertEqual(len(pool.free), 8)

    def test_replay_and_out_of_order_rejected(self):
        pool = KVPool(4)
        a = {"request_id": "a", "position": 0, "query_len": 1}
        pool.prepare([a])
        pool.commit([a])
        with self.assertRaises(ValueError):
            pool.prepare([a])
        with self.assertRaises(ValueError):
            pool.prepare([{**a, "position": 3}])

    def test_capacity_failure_is_atomic(self):
        pool = KVPool(1)
        with self.assertRaises(ValueError):
            pool.prepare([{"request_id": "a", "position": 0, "query_len": 17}])
        self.assertEqual(pool.requests, {})
        self.assertEqual(len(pool.free), 1)

    def test_duplicate_request_batch_rejected_atomically(self):
        pool = KVPool(4)
        item = {"request_id": "a", "position": 0, "query_len": 1}
        with self.assertRaises(ValueError):
            pool.prepare([item, item])
        self.assertEqual(pool.requests, {})


if __name__ == "__main__":
    unittest.main()
