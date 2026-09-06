"""POC-owned network configuration, observed state, and fail-closed checks."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "run"
PAIRS = (("split-enterprise", "split-e"), ("split-cloud", "split-c"))


@contextmanager
def network_lock():
    """Prevent managed network changes/iperf checks during a measurement."""
    RUN.mkdir(exist_ok=True)
    with (RUN / "network.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Network is in use by another benchmark/configuration/check") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def snapshot():
    return {ns: command(["ip", "netns", "exec", ns, "tc", "-j", "-s", "qdisc", "show", "dev", dev])
            for ns, dev in PAIRS}


def expected(deployment):
    if not deployment or "wan" not in deployment:
        raise ValueError("Missing network intent; run ./poc up first")
    return {"wan": deployment["wan"], "delay_ms": deployment.get("delay", 5),
            "bandwidth_gbps": deployment.get("bandwidth_gbps", 10)}


def verify(observed, intent):
    for ns, _ in PAIRS:
        saved = observed[ns]
        if saved["exit_code"]:
            raise ValueError(f"Cannot inspect {ns}: {saved['stderr']}")
        entries = json.loads(saved["stdout"])
        qdiscs = {q["kind"]: q for q in entries}
        if intent["wan"]:
            if set(qdiscs) != {"tbf", "netem"}:
                raise ValueError(f"{ns}: expected TBF + netem, got {list(qdiscs)}")
            tbf, netem = qdiscs["tbf"], qdiscs["netem"]
            if not tbf.get("root") or netem.get("parent") != "1:1":
                raise ValueError(f"{ns}: unexpected qdisc hierarchy")
            rate = tbf["options"]["rate"]
            delay = netem["options"].get("delay", {}).get("delay", 0)
            if not math.isclose(rate, intent["bandwidth_gbps"] * 1e9 / 8, rel_tol=1e-6):
                raise ValueError(f"{ns}: bandwidth mismatch: {rate} bytes/s")
            if not math.isclose(delay, intent["delay_ms"] / 1000, abs_tol=1e-6):
                raise ValueError(f"{ns}: delay mismatch: {delay} seconds")
        elif any(q["kind"] != "noqueue" for q in entries):
            raise ValueError(f"{ns}: local mode still has shaping: {list(qdiscs)}")
    return True


def configure(wan, delay=5, bandwidth_gbps=10):
    with network_lock():
        return _configure(wan, delay, bandwidth_gbps)


def _configure(wan, delay=5, bandwidth_gbps=10):
    if not math.isfinite(delay) or not 0 <= delay <= 100:
        raise ValueError("Delay must be 0..100 ms")
    if not math.isfinite(bandwidth_gbps) or not 0 < bandwidth_gbps <= 100:
        raise ValueError("Bandwidth must be >0 and <=100 Gbps")
    intent = {"wan": wan, "delay_ms": delay, "bandwidth_gbps": bandwidth_gbps}
    for ns, dev in PAIRS:
        base = ["ip", "netns", "exec", ns, "tc", "qdisc"]
        if wan:
            subprocess.run([*base, "replace", "dev", dev, "root", "handle", "1:", "tbf",
                            "rate", f"{bandwidth_gbps:g}gbit", "burst", "2mb", "latency", "100ms"], check=True)
            subprocess.run([*base, "replace", "dev", dev, "parent", "1:1", "handle", "10:",
                            "netem", "delay", f"{delay:g}ms", "limit", "100000"], check=True)
        else:
            # Deleting an absent root is harmless; the following inspection is mandatory.
            subprocess.run([*base, "del", "dev", dev, "root"], capture_output=True)
    observed = snapshot()
    verify(observed, intent)
    RUN.mkdir(exist_ok=True)
    state = {"configured_at_unix": time.time(), "intent": intent, "observed": observed}
    (RUN / "network.json").write_text(json.dumps(state, indent=2))
    launch = RUN / "launch.json"
    if launch.exists():
        data = json.loads(launch.read_text())
        data.update(wan=wan, delay=delay, bandwidth_gbps=bandwidth_gbps)
        launch.write_text(json.dumps(data, indent=2))
    return state


def validate_measurements(ping, iperf, intent, rtt_tolerance_ms=2, min_bandwidth_ratio=.8):
    if ping["exit_code"] or iperf["exit_code"]:
        raise ValueError("ping or iperf failed")
    match = re.search(r"= ([\d.]+)/([\d.]+)/([\d.]+)/", ping["stdout"])
    if not match or not re.search(r"\b0% packet loss", ping["stdout"]):
        raise ValueError("Missing RTT or nonzero packet loss")
    data = json.loads(iperf["stdout"])
    if data.get("error"):
        raise ValueError(data["error"])
    rtt = float(match[2])
    gbps = data["end"]["sum_received"]["bits_per_second"] / 1e9
    if intent["wan"]:
        if abs(rtt - 2 * intent["delay_ms"]) > rtt_tolerance_ms:
            raise ValueError(f"RTT {rtt} ms differs from expected {2 * intent['delay_ms']} ms")
        if not intent["bandwidth_gbps"] * min_bandwidth_ratio <= gbps <= intent["bandwidth_gbps"] * 1.1:
            raise ValueError(f"Throughput {gbps} Gbps outside configured tolerance")
    return {"rtt_ms": rtt, "throughput_gbps": gbps}


def check(intent, output, rtt_tolerance_ms=2, min_bandwidth_ratio=.8):
    report = {"result": "FAIL", "intent": intent, "observed_before": snapshot(),
              "rtt_tolerance_ms": rtt_tolerance_ms, "min_bandwidth_ratio": min_bandwidth_ratio,
              "max_bandwidth_ratio": 1.1, "started_at_unix": time.time()}
    try:
        verify(report["observed_before"], intent)
        report["ping"] = command(["ip", "netns", "exec", "split-enterprise", "ping", "-c", "5", "-W", "2", "10.205.0.2"])
        # Own a foreground one-shot server, so failures cannot leave a daemon behind.
        server = subprocess.Popen(["ip", "netns", "exec", "split-cloud", "iperf3", "-s", "-1", "-p", "5202"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            time.sleep(.3)
            if server.poll() is not None:
                raise ValueError("iperf server failed to start (port 5202 busy?)")
            report["iperf"] = command(["ip", "netns", "exec", "split-enterprise", "iperf3", "-c", "10.205.0.2",
                                       "-p", "5202", "-t", "5", "-P", "4", "-J"])
        finally:
            if server.poll() is None:
                server.terminate()
            server.wait(timeout=5)
        report["measured"] = validate_measurements(report["ping"], report["iperf"], intent,
                                                   rtt_tolerance_ms, min_bandwidth_ratio)
        report["observed_after"] = snapshot()
        verify(report["observed_after"], intent)
        report["result"] = "PASS"
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["wan", "local", "check"])
    parser.add_argument("legacy_delay", type=float, nargs="?")
    parser.add_argument("--delay-ms", type=float)
    parser.add_argument("--bandwidth-gbps", type=float, default=10)
    parser.add_argument("--output", default=str(RUN / "network_check.json"))
    parser.add_argument("--rtt-tolerance-ms", type=float, default=2)
    parser.add_argument("--min-bandwidth-ratio", type=float, default=.8)
    args = parser.parse_args()
    if not 0 < args.min_bandwidth_ratio <= 1 or not 0 <= args.rtt_tolerance_ms <= 100:
        parser.error("Invalid network-check tolerance")
    if args.action == "check":
        intent = expected(json.loads((RUN / "launch.json").read_text()))
        with network_lock():
            result = check(intent, args.output, args.rtt_tolerance_ms, args.min_bandwidth_ratio)
    else:
        delay = args.delay_ms if args.delay_ms is not None else args.legacy_delay if args.legacy_delay is not None else 5
        result = configure(args.action == "wan", delay, args.bandwidth_gbps)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
