import json
from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

from split_poc.scheduling import choose
from split_poc.server import Job, Scheduler


class PolicyTests(unittest.TestCase):
    def test_legacy_and_decode_quota(self):
        prefill, decode = Job([1, 2], 2, True), Job([1], 2, True)
        decode.prefilled = True
        active = [prefill, decode]
        self.assertEqual(choose(active, "legacy", 0, 2)[:2], ([prefill], "prefill"))
        rounds = 0
        phases = []
        for _ in range(6):
            batch, phase, rounds = choose(active, "decode-first", rounds, 2)
            phases.append(phase)
        self.assertEqual(phases, ["decode", "decode", "prefill"] * 2)

    def test_no_unnecessary_idle_and_fifo_prefill(self):
        a, b = Job([1], 2, True), Job([2], 2, True)
        self.assertEqual(choose([a, b], "decode-first", 0, 4)[:2], ([a], "prefill"))
        a.prefilled = True
        self.assertEqual(choose([a], "decode-first", 99, 1)[:2], ([a], "decode"))


class FakeExecutor:
    healthy = True

    def __init__(self):
        self.commands = []
        self.released = threading.Event()
        self.before_return = lambda command: None

    def call(self, command):
        self.commands.append(command)
        if command["op"] == "release":
            self.released.set()
            return {"kv_used_blocks": 0}
        self.before_return(command)
        n = len(command["items"])
        return {"tokens": [7] * n, "logits": [[.1, .2]] * n,
                "timings": {}, "kv_used_blocks": n}


class SchedulerTests(unittest.TestCase):
    def run_job(self, chunk, cancel=False):
        with tempfile.TemporaryDirectory() as folder, patch("split_poc.server.httpx.Client") as client:
            executor = FakeExecutor()
            job = Job([1, 2, 3, 4, 5], 2, True, forced=[9], capture=True)
            if cancel:
                executor.before_return = lambda command: setattr(job, "cancelled", True)
            args = SimpleNamespace(cloud="http://unused", results=folder, max_active=4,
                                   prefill_chunk_size=chunk, scheduler_policy="decode-first", decode_quota=1)
            scheduler = Scheduler(executor, args, set())
            try:
                scheduler.submit(job)
                if cancel:
                    self.assertTrue(executor.released.wait(5))
                    self.assertTrue(job.events.empty())
                else:
                    events = [job.events.get(timeout=5), job.events.get(timeout=5)]
                    self.assertEqual([event["done"] for event in events], [False, True])
                    self.assertEqual(job.tokens, [7, 7])
                    self.assertEqual(len(job.logits), 2)
                    self.assertEqual(scheduler.prompt_tokens, 5)
                    self.assertEqual(scheduler.generation_tokens, 2)
                    self.assertTrue(executor.released.wait(5))
            finally:
                scheduler.closed = True
                scheduler.thread.join(5)
                scheduler.trace.close()
                scheduler.http.close()
            self.assertFalse(scheduler.thread.is_alive())
            client.return_value.post.assert_called_with("/release", json={"ids": [job.id]})
            rows = [json.loads(line) for line in (Path(folder) / "split_trace.jsonl").read_text().splitlines()]
            return [cmd for cmd in executor.commands if cmd["op"] == "forward"], rows

    def test_chunks_preserve_positions_and_emit_only_final(self):
        commands, rows = self.run_job(2)
        self.assertEqual([cmd["items"][0]["query_len"] for cmd in commands], [2, 2, 1, 1])
        self.assertEqual([cmd["items"][0]["position"] for cmd in commands], [0, 2, 4, 5])
        self.assertEqual([cmd["emit"] for cmd in commands], [False, False, True, True])
        self.assertEqual(commands[-1]["token_ids"], [9])
        self.assertEqual([row["token_idx"] for row in rows], [-1, -1, 0, 1])
        self.assertTrue(all(row["queue_ms"] >= 0 for row in rows))

    def test_off_remains_one_prefill(self):
        commands, rows = self.run_job(0)
        self.assertEqual([cmd["items"][0]["query_len"] for cmd in commands], [5, 1])
        self.assertTrue(all(row["emits_token"] for row in rows))

    def test_cancel_after_intermediate_chunk_releases_both_sides(self):
        commands, rows = self.run_job(2, cancel=True)
        self.assertEqual(len(commands), 1)
        self.assertFalse(rows[0]["emits_token"])

    def test_midchunk_failure_fails_active_requests_and_rejects_new_work(self):
        with tempfile.TemporaryDirectory() as folder, patch("split_poc.server.httpx.Client"), \
                patch("traceback.print_exc"):
            executor = FakeExecutor()
            def fail_second(command):
                if len(executor.commands) == 2:
                    raise RuntimeError("simulated cloud failure")
            executor.before_return = fail_second
            args = SimpleNamespace(cloud="http://unused",results=folder,max_active=4,
                                   prefill_chunk_size=2,scheduler_policy="decode-first",decode_quota=1)
            # Pause thread startup so both requests are enqueued before scheduling.
            with patch("threading.Thread.start"):
                scheduler = Scheduler(executor,args,set())
            jobs = [Job([1]*5,2,True),Job([2]*5,2,True)]
            for job in jobs:
                scheduler.submit(job)
            scheduler.thread.start()
            try:
                for job in jobs:
                    self.assertIn("simulated cloud failure",job.events.get(timeout=5)["error"])
                    self.assertEqual(job.tokens,[])
                self.assertFalse(executor.healthy)
                from fastapi import HTTPException
                with self.assertRaises(HTTPException) as error:
                    scheduler.submit(Job([3],2,True))
                self.assertEqual(error.exception.status_code,503)
            finally:
                scheduler.closed = True
                scheduler.thread.join(5)
                scheduler.trace.close()
                scheduler.http.close()
            self.assertFalse(scheduler.thread.is_alive())


if __name__ == "__main__":
    unittest.main()
