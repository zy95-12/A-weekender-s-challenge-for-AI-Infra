"""Explicit destructive acceptance: stop only the owned Cloud process group.

Run while the demo is idle. The next `./poc up` restores both services.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import httpx

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-owned-cloud", action="store_true", required=True)
    parser.add_argument("--output", default="results/fault_check.json")
    args = parser.parse_args()
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=40, trust_env=False) as client:
        before = client.get("/health")
        before.raise_for_status()
        assert before.json()["active"] == 0 and before.json()["waiting"] == 0, "Demo must be idle"
        info = json.loads((ROOT / "run/cloud.pid.json").read_text())
        proc = Path(f"/proc/{info['pid']}/stat")
        assert proc.read_text().split()[21] == info["start_ticks"], "PID identity changed"
        assert os.getpgid(info["pid"]) == info["pid"], "Unexpected process group"
        assert "split-cloud" in info["command"] and "split_poc.server" in info["command"]
        os.killpg(info["pid"], signal.SIGTERM)
        time.sleep(1)
        start = time.perf_counter()
        response = client.post("/v1/chat/completions", json={"messages": [
            {"role": "user", "content": "Say hello."}], "max_tokens": 8, "temperature": 0})
        elapsed = time.perf_counter() - start
        after = client.get("/health")
        report = {"before": before.json(), "stopped_owned_cloud": info,
                  "request_status": response.status_code, "request_seconds": elapsed,
                  "response": response.text, "health_status": after.status_code,
                  "result": "PASS" if response.status_code == 503 and after.status_code == 503 else "FAIL",
                  "recovery": "Run ./poc up to restart the unhealthy deployment."}
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        assert report["result"] == "PASS"


if __name__ == "__main__":
    main()
