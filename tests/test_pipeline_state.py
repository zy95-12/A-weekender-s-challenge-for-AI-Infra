import concurrent.futures
import threading
import unittest
from types import SimpleNamespace

from split_poc.pipeline_state import CausalGate, KVAdmission


def item(position, n=2, rid="a"):
    return [{"request_id":rid,"position":position,"query_len":n}]


class GateTests(unittest.TestCase):
    def test_out_of_order_arrival_does_not_hold_execution_lock(self):
        gate = CausalGate(timeout=2)
        order = []
        started = threading.Event()
        def later():
            started.set()
            return gate.run(item(2),lambda: order.append(2))
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            future = pool.submit(later)
            self.assertTrue(started.wait(1))
            gate.run(item(0),lambda: order.append(0))
            future.result(timeout=2)
        self.assertEqual(order,[0,2])
        self.assertEqual(gate.positions,{"a":4})

    def test_replay_and_missing_predecessor(self):
        gate = CausalGate(timeout=.01)
        gate.run(item(0),lambda: None)
        with self.assertRaises(ValueError):
            gate.run(item(0),lambda: self.fail("replay executed"))
        with self.assertRaises(TimeoutError):
            gate.run(item(4),lambda: self.fail("out of order executed"))
        self.assertIsNone(gate.failure)

    def test_partial_execution_failure_poisons_gate(self):
        gate = CausalGate()
        def failure():
            raise RuntimeError("GPU failed")
        with self.assertRaisesRegex(RuntimeError,"GPU failed"):
            gate.run(item(0),failure)
        with self.assertRaisesRegex(RuntimeError,"unhealthy"):
            gate.run(item(0,rid="other"),lambda: self.fail("unhealthy gate executed"))
        self.assertEqual(gate.positions,{})

    def test_release_runs_before_forgetting_position(self):
        gate = CausalGate()
        gate.run(item(0),lambda: None)
        gate.release(["a"],lambda: self.assertEqual(gate.positions["a"],2))
        self.assertEqual(gate.positions,{})


class AdmissionTests(unittest.TestCase):
    def test_capacity_and_release(self):
        budget = KVAdmission(3)
        a = SimpleNamespace(id="a",ids=[1]*17,limit=2)
        b = SimpleNamespace(id="b",ids=[1]*17,limit=2)
        self.assertTrue(budget.admit(a))
        self.assertFalse(budget.admit(b))
        budget.release(a)
        self.assertTrue(budget.admit(b))
        with self.assertRaises(ValueError):
            budget.admit(b)
        with self.assertRaises(ValueError):
            budget.admit(SimpleNamespace(id="big",ids=[1]*65,limit=2))


if __name__ == "__main__":
    unittest.main()
