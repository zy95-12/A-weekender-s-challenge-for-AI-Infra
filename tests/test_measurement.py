import argparse
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import network_state as network
import manage
import validate_poc
from benchmark_artifacts import export_trace


def observed(wan=True, delay=.005, rate=1_250_000_000):
    entries = [{"kind": "tbf", "root": True, "options": {"rate": rate}},
               {"kind": "netem", "parent": "1:1", "options": {"delay": {"delay": delay}}}] if wan else [{"kind": "noqueue"}]
    return {ns: {"exit_code": 0, "stdout": json.dumps(entries), "stderr": ""} for ns, _ in network.PAIRS}


class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.intent = {"wan": True, "delay_ms": 5, "bandwidth_gbps": 10}

    def test_matches_both_sides(self):
        self.assertTrue(network.verify(observed(), self.intent))
        self.assertTrue(network.verify(observed(False), {**self.intent, "wan": False}))

    def test_rejects_local_delay_rate_and_tc_failure(self):
        failed = observed()
        failed["split-cloud"]["exit_code"] = 1
        for state in (observed(False), observed(delay=.01), observed(rate=125_000_000), failed):
            with self.subTest(state=state), self.assertRaises(ValueError):
                network.verify(state, self.intent)

    def test_nondefault_parameters(self):
        self.assertTrue(network.verify(observed(delay=.01, rate=125_000_000),
                                       {"wan": True, "delay_ms": 10, "bandwidth_gbps": 1}))

    def test_local_rejects_shaping(self):
        with self.assertRaises(ValueError):
            network.verify(observed(), {**self.intent, "wan": False})

    def test_network_measurement_thresholds(self):
        ping = {"exit_code": 0, "stdout": "0% packet loss\nrtt min/avg/max/mdev = 10.01/10.02/10.03/0.01 ms"}
        iperf = {"exit_code": 0, "stdout": json.dumps({"end": {"sum_received": {"bits_per_second": 9.42e9}}})}
        self.assertEqual(network.validate_measurements(ping, iperf, self.intent)["throughput_gbps"], 9.42)
        for bad_ping, bad_iperf in (({**ping, "stdout": ping["stdout"].replace("10.02", "0.02")}, iperf),
                                  ({**ping, "stdout": ping["stdout"].replace("0%", "100%")}, iperf),
                                  (ping, {**iperf, "stdout": iperf["stdout"].replace("9420000000.0", "1000000000.0")})):
            with self.assertRaises(ValueError):
                network.validate_measurements(bad_ping, bad_iperf, self.intent)

    def test_configure_persists_only_verified_state(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(network, "RUN", Path(temp)), \
                patch.object(network.subprocess, "run"), patch.object(network, "snapshot", return_value=observed()):
            launch = Path(temp) / "launch.json"
            launch.write_text(json.dumps({"wan": False, "split": "1:3"}))
            network.configure(True)
            self.assertEqual(json.loads(launch.read_text())["bandwidth_gbps"], 10)
            with patch.object(network, "snapshot", return_value=observed(False)), self.assertRaises(ValueError):
                network.configure(True, 10)
            self.assertEqual(json.loads(launch.read_text())["delay"], 5)

    def test_lock_excludes_changes(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(network, "RUN", Path(temp)):
            with network.network_lock(), self.assertRaises(RuntimeError):
                network.configure(False)

    def test_reuse_reapplies_network_without_restart(self):
        args = argparse.Namespace(action="up", split="1:3", tp=2, enterprise_tp=None, cloud_tp=None,
                                  wan=True, delay=5, bandwidth_gbps=10, profile=False, phase_profile=False)
        with tempfile.TemporaryDirectory() as temp, patch.object(manage, "RUN", Path(temp)), \
                patch.object(manage.urllib.request, "urlopen", return_value=io.StringIO('{"split":"1:3","tp":2}')), \
                patch.object(manage, "configure") as configure, patch.object(manage, "stop") as stop:
            (Path(temp) / "enterprise.pid.json").write_text("{}")
            (Path(temp) / "launch.json").write_text(json.dumps({**vars(args), "wan": False}))
            manage.up(args)
            configure.assert_called_once_with(True, 5, 10)
            stop.assert_not_called()

    def test_failed_check_saves_failure(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(network, "snapshot", return_value=observed(False)):
            path = Path(temp) / "network_check.json"
            with self.assertRaises(ValueError):
                network.check(self.intent, path)
            self.assertEqual(json.loads(path.read_text())["result"], "FAIL")

    def test_validation_cannot_pass_without_network_check(self):
        commands = []
        def fake_run(command, **kwargs):
            commands.append(command)
            return argparse.Namespace(returncode=1 if "network-check" in command else 0)
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(sys, "argv", ["validate_poc.py", "--output", temp]), \
                patch.object(validate_poc.subprocess, "run", side_effect=fake_run):
            with self.assertRaisesRegex(RuntimeError, "network_check failed"):
                validate_poc.main()
            self.assertNotEqual(json.loads((Path(temp) / "validation.json").read_text())["result"], "PASS")
            self.assertFalse(any("scripts/benchmark.py" in command for command in commands))


class TraceTests(unittest.TestCase):
    def test_warmup_and_other_traffic_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder / "source.jsonl"
            source.write_text("".join(json.dumps({"client_request_id": rid}) + "\n"
                                      for rid in ("warmup-uuid", "exp-0", "exp-1", "exp-99", "another-client")))
            export_trace(source, 0, folder, "exp-", 2)
            measured = [json.loads(row) for row in (folder / "split_trace.jsonl").read_text().splitlines()]
            self.assertEqual(len(measured), 2)
            self.assertTrue(all(row["is_measured"] and row["is_warmup"] is False for row in measured))
            self.assertEqual(json.loads((folder / "trace_scope.json").read_text())["excluded_rows"], 3)


if __name__ == "__main__":
    unittest.main()
