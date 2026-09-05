"""Low-frequency GPU and per-side metrics, separate from detailed profiling."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request

FIELDS = "index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=int, default=3600)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    dcgm_log = (root / "dcgm_metrics.txt").open("w")
    dcgm = subprocess.Popen(["dcgmi", "dmon", "-e", "1001,1002,1005,1009,1010,155,100,101,203,204,252",
                             "-d", "1000", "-c", str(args.seconds)], stdout=dcgm_log, stderr=subprocess.STDOUT)
    with (root / "gpu_metrics.csv").open("w", buffering=1) as gpu, (root / "runtime_metrics.jsonl").open("w", buffering=1) as runtime:
        gpu.write("time_ns," + FIELDS + "\n")
        for _ in range(args.seconds):
            start = time.monotonic()
            stamp = time.time_ns()
            result = subprocess.run(["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True)
            for line in result.stdout.splitlines():
                gpu.write(f"{stamp},{line}\n")
            for role, url in (("enterprise", "http://10.204.0.2:8000/metrics"),
                              ("cloud", "http://10.205.0.2:8001/metrics")):
                command = ["curl", "-fsS", "--max-time", "2", url]
                if role == "cloud":
                    command = ["ip", "netns", "exec", "split-enterprise", *command]
                response = subprocess.run(command, capture_output=True, text=True)
                runtime.write(json.dumps({"time_ns": stamp, "role": role,
                                          "metrics": response.stdout, "error": response.stderr}) + "\n")
            time.sleep(max(0, 1 - (time.monotonic() - start)))
    dcgm.wait(timeout=10)
    dcgm_log.close()


if __name__ == "__main__":
    main()
