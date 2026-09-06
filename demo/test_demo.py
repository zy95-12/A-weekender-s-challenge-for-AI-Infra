import importlib.util, json, tempfile, unittest, threading, urllib.request, urllib.error
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "demo_server", Path(__file__).with_name("server.py")
)
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)


class ContractTests(unittest.TestCase):
    def test_sim_rejects_silent_fallback(self):
        for p in (
            {"variant": "other", "concurrency": 1},
            {"variant": "optimized", "concurrency": True},
            {"variant": "baseline", "concurrency": 97},
        ):
            with self.assertRaises(ValueError):
                server.validate_sim(p)
        self.assertEqual(
            server.validate_sim({"variant": "optimized", "concurrency": 40}),
            {"variant": "optimized", "concurrency": 40},
        )

    def test_public_model_open_loop_contract(self):
        valid = {
            "variant": "optimized",
            "workload_mode": "open_loop",
            "arrival_rate_qps": 0.5,
            "model": "deepseek-v4-flash",
            "hardware": "ascend910b",
        }
        self.assertEqual(server.validate_sim(valid), valid)
        for change in [
            {"arrival_rate_qps": float("nan")},
            {"arrival_rate_qps": True},
            {"model": "deepseek-v3"},
            {"hardware": "unknown"},
            {"concurrency": 1},
        ]:
            with self.assertRaises(ValueError):
                server.validate_sim({**valid, **change})

    def test_attack_needs_successful_encode(self):
        with self.assertRaises(ValueError):
            server.launch("recover", {"encode_id": "../../x"})
        with self.assertRaises(ValueError):
            server.launch("encode", {"text": ""})

    def test_busy_does_not_launch(self):
        server.LOCK.acquire()
        try:
            with self.assertRaises(RuntimeError):
                server.launch("start", {})
        finally:
            server.LOCK.release()

    def test_health_without_owned_process_is_stopped(self):
        with tempfile.TemporaryDirectory() as d, patch.object(server, "ROOT", Path(d)):
            self.assertEqual(server.health(), {"status": "stopped"})

    def test_raw_breakdown_reconciles(self):
        d = json.loads(
            Path(__file__)
            .with_name("evidence")
            .joinpath("baseline-c16.json")
            .read_text()
        )
        for row in d["breakdown"].values():
            self.assertAlmostEqual(sum(row["parts_ms"].values()), row["total_ms"])

    def test_same_origin_and_bad_json(self):
        http = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        t = threading.Thread(target=http.serve_forever)
        t.start()
        try:
            url = f"http://127.0.0.1:{http.server_port}/api/service/start"
            for body, headers, code in [
                (
                    b"{}",
                    {"Origin": "http://elsewhere", "Content-Type": "application/json"},
                    403,
                ),
                (b"[]", {"Content-Type": "application/json"}, 400),
            ]:
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(
                        urllib.request.Request(url, data=body, headers=headers)
                    )
                self.assertEqual(caught.exception.code, code)
        finally:
            http.shutdown()
            t.join()
            http.server_close()


if __name__ == "__main__":
    unittest.main()
