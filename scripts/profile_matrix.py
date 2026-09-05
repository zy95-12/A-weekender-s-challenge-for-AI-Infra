"""Separate Nsight + detailed GPU stage capture for all baseline workloads."""
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import httpx
from transformers import AutoTokenizer
from manage import profiler_binary

ROOT = Path(__file__).resolve().parents[1]


def main():
    assert json.loads((ROOT / "run/launch.json").read_text())["profile"]
    folder = Path((ROOT / "run/current_results").read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "models/qwen"), local_files_only=True)
    unit = tokenizer.encode(" apple", add_special_tokens=False)
    assert len(unit) == 1
    rows = []
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=300, trust_env=False) as client:
        health = client.get("/health")
        health.raise_for_status()
        cloud = subprocess.run(["ip", "netns", "exec", "split-enterprise", "curl", "-fsS",
            "http://10.205.0.2:8001/health"], capture_output=True, text=True, check=True)
        environment = {"enterprise": health.json(), "cloud": json.loads(cloud.stdout)}
        for side in environment.values():
            for worker in side["worker_audits"]:
                if worker["tp"] > 1:
                    assert worker["vllm_nccl_version"] == ".".join(map(str, worker["torch_nccl_version"])), \
                        "Profiler selected a different NCCL build; do not mix these measurements"
        environment["profiler"] = {"binary": profiler_binary(), "version": subprocess.run(
            [profiler_binary(), "--version"], capture_output=True, text=True, check=True).stdout.strip()}
        (folder / "profile_environment.json").write_text(json.dumps(environment, indent=2))
        client.post("/start_profile").raise_for_status()
        telemetry = subprocess.Popen([sys.executable, str(ROOT / "scripts/telemetry.py"),
            "--output", str(folder), "--seconds", "3600"], start_new_session=True)
        try:
            for isl, osl in ((512, 128), (2048, 256), (8192, 256), (2048, 1024)):
                for concurrency in (1, 4):
                    def request(index):
                        rid = f"profile-isl-{isl}-osl-{osl}-c-{concurrency}-i-{index}"
                        start = time.perf_counter()
                        with httpx.Client(timeout=300, trust_env=False) as connection:
                            response = connection.post("http://127.0.0.1:8000/v1/completions",
                                headers={"X-Request-Id": rid}, json={"prompt": unit * isl,
                                "max_tokens": osl, "ignore_eos": True, "temperature": 0})
                            if response.is_error:
                                raise RuntimeError(f"{rid}: HTTP {response.status_code}: {response.text}")
                            output = response.json()
                        assert output["usage"]["prompt_tokens"] == isl and output["usage"]["completion_tokens"] == osl
                        return {"client_request_id": rid, "internal_request_id": output["id"],
                                "isl": isl, "osl": osl, "concurrency": concurrency,
                                "seconds": time.perf_counter() - start, "usage": output["usage"]}
                    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                        rows.extend(pool.map(request, range(concurrency)))
                    print(f"PROFILE {isl}/{osl} C={concurrency} complete", flush=True)
        finally:
            try:
                client.post("/stop_profile").raise_for_status()
            finally:
                if telemetry.poll() is None:
                    os.killpg(telemetry.pid, signal.SIGTERM)
                    telemetry.wait(timeout=10)
                (folder / "profile_cases.json").write_text(json.dumps({"classification": "profile_only",
                    "cases": rows, "not_for_baseline_throughput": True}, indent=2))


if __name__ == "__main__":
    main()
