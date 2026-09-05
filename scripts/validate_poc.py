"""Run real-model acceptance sequentially; leave the WAN demo running.

No synthetic-weight fixtures are invoked by this script. It stops at the first
failure and preserves logs/artifacts for diagnosis. Optional download wait only
observes the pinned checkpoint being downloaded by setup.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-model", action="store_true")
    parser.add_argument("--output", default="results/validation")
    parser.add_argument("--benchmark-requests", type=int, default=20)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    def run(name, command, env=None):
        print(f"START {name}", flush=True)
        start = time.time()
        with (output / f"{name}.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            print((output / f"{name}.log").read_text()[-7000:], flush=True)
            raise RuntimeError(f"{name} failed; see {output / (name + '.log')}")
        print(f"PASS {name} ({time.time() - start:.1f}s)", flush=True)

    if args.wait_model:
        deadline = time.monotonic() + 7200
        while not (ROOT / "models/qwen/revision.txt").exists():
            if time.monotonic() > deadline:
                raise RuntimeError("Timed out waiting for model download")
            total = sum(p.stat().st_blocks * 512 for p in (ROOT / "models/qwen").glob("*.safetensors*"))
            print(f"Waiting for verified real weights: {total / 1e9:.2f} / 6.17 GB on disk", flush=True)
            time.sleep(30)
    run("stop_previous", ["bash", "poc", "down"])
    run("model_integrity", [str(PYTHON), "scripts/model_views.py"])
    ref = output / "reference"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1", "VLLM_USE_V1": "0",
           "VLLM_ATTENTION_BACKEND": "FLASH_ATTN", "OMP_NUM_THREADS": "4",
           "NCCL_SOCKET_IFNAME": "lo", "GLOO_SOCKET_IFNAME": "lo"}
    run("native_reference", [str(PYTHON), "-m", "split_poc.reference", "--output", str(ref)], env)
    for split in ("1:3", "1:1", "3:1"):
        label = split.replace(":", "_")
        run("up_" + label, ["bash", "poc", "up", "--split", split])
        run("correctness_" + label, [str(PYTHON), "scripts/correctness.py", "--reference", str(ref),
                                    "--steps", "257", "--output", str(output / ("correctness_" + label))])
        run("acceptance_" + label, [str(PYTHON), "scripts/acceptance.py", "--output", str(output / ("acceptance_" + label + ".json"))])
        run("down_" + label, ["bash", "poc", "down"])
    run("up_wan", ["bash", "poc", "up", "--wan"])
    run("correctness_wan", [str(PYTHON), "scripts/correctness.py", "--reference", str(ref),
                            "--steps", "33", "--output", str(output / "correctness_wan")])
    run("benchmark_wan", [str(PYTHON), "scripts/benchmark.py", "--rates", "0.5,1,2",
                          "--requests", str(args.benchmark_requests), "--output", str(output / "benchmark_wan"),
                          "--correctness-report", str(output / "correctness_wan/summary.json")])
    (output / "validation.json").write_text(json.dumps({"result": "PASS", "real_weights": True,
        "splits": ["1:3", "1:1", "3:1"], "tp": [2, 2], "reference_decode_steps": 256,
        "benchmark_requests_per_point": args.benchmark_requests,
        "demo_url": "http://127.0.0.1:8000"}, indent=2))
    print("REAL POC VALIDATION PASS; WAN demo is running on http://127.0.0.1:8000", flush=True)


if __name__ == "__main__":
    main()
