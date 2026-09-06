"""Exercise dispatch under slow responses and failed requests without a GPU."""

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "open_loop_benchmark", SCRIPTS / "open_loop_benchmark.py"
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class Response:
    def raise_for_status(self):
        pass

    def json(self):
        return {
            "tokens": list(range(79)),
            "active": 0,
            "waiting": 0,
            "kv_used_blocks": 0,
        }


class Client:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def get(self, *args, **kwargs):
        return Response()

    async def post(self, *args, **kwargs):
        return Response()


class OpenLoopTests(unittest.TestCase):
    def test_nonfinite_rate_is_rejected_before_startup(self):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "open_loop_benchmark.py"),
                "--rate",
                "nan",
                "--label",
                "invalid",
                "--out",
                "/tmp/unused-open-loop-invalid",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Invalid load settings", result.stderr)

    def test_slow_responses_do_not_throttle_arrivals_and_errors_count(self):
        async def request(client, ids, rid, reference):
            start = time.perf_counter()
            await asyncio.sleep(0.25)
            rejected = rid.endswith("-0")
            return {
                "request_id": rid,
                "start": start,
                "end": time.perf_counter(),
                "ttft_ms": None if rejected else 100,
                "tpot_ms": None if rejected else 2,
                "error": "rejected" if rejected else None,
                "tokens": 0 if rejected else 79,
                "slo_pass": not rejected,
            }

        async def drained(client):
            return Response().json()

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "run").mkdir()
            launch = {"preset": "baseline", "wan": True}
            (root / "run/launch.json").write_text(json.dumps(launch))
            (root / "prompt.json").write_text(json.dumps({"prompt_ids": [1]}))
            (root / "reference.json").write_text(
                json.dumps({"greedy_ids": list(range(79)), "text": "ok"})
            )
            args = argparse.Namespace(
                out=root / "out",
                system="baseline",
                rate=30,
                seed=17,
                seconds=1,
                warmup=0,
                tail=1,
                prompt=root / "prompt.json",
                reference=root / "reference.json",
                label="test",
            )
            with patch.object(bench, "ROOT", root), patch.object(
                bench.subprocess, "check_output", return_value=json.dumps(launch)
            ), patch.object(bench, "check"), patch.object(
                bench, "verify"
            ), patch.object(
                bench, "snapshot", return_value={}
            ), patch.object(
                bench, "drained", drained
            ), patch.object(
                bench, "one", request
            ), patch.object(
                bench.httpx, "AsyncClient", return_value=Client()
            ):
                asyncio.run(bench.measure(args))
            report = json.loads((args.out / "summary.json").read_text())
            rows = json.loads((args.out / "requests.json").read_text())
            schedule = json.loads((args.out / "schedule.json").read_text())
            self.assertEqual(len(rows), len(schedule["offsets_s"]))
            self.assertGreater(report["peak_inflight"], 4)
            self.assertLess(report["arrival_cohort"]["p99_dispatch_lag_ms"], 50)
            self.assertEqual(report["arrival_cohort"]["errors"], 1)
            self.assertLess(report["arrival_cohort"]["joint_slo"], 1)
            self.assertFalse(report["all_outputs_match"])
            self.assertLess(
                report["successful_completed_qps"], report["completion_rate"]
            )


if __name__ == "__main__":
    unittest.main()
