"""Reproducible real-weight TP baseline, independent of SLO pass/fail.

Closed-loop client concurrency is explicit. Each repetition invokes the official
vLLM client including its untimed warmup. No profiler in measurement runs.
Use --resume to continue completed points; failed points are never marked done.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import httpx
from manage import profiler_binary

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / ".venv/bin/python")
WORKLOADS = [(512, 128), (2048, 256), (8192, 256), (2048, 1024)]


def run(name, command, output):
    print(f"START {name}", flush=True)
    start = time.monotonic()
    with (output / (name + ".log")).open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print((output / (name + ".log")).read_text()[-5000:], flush=True)
        raise RuntimeError(f"{name} failed ({result.returncode})")
    print(f"DONE {name}: {time.monotonic() - start:.1f}s", flush=True)


def audit(folder):
    with httpx.Client(timeout=10, trust_env=False) as client:
        enterprise = client.get("http://127.0.0.1:8000/health")
        enterprise.raise_for_status()
    cloud = subprocess.run(["ip", "netns", "exec", "split-enterprise", "curl", "-fsS",
                            "http://10.205.0.2:8001/health"], capture_output=True, text=True, check=True)
    sides = {"enterprise": enterprise.json(), "cloud": json.loads(cloud.stdout)}
    for role, health in sides.items():
        assert health["status"] == "ready" and len(health["worker_audits"]) == health["tp"]
        for worker in health["worker_audits"]:
            assert worker["has_embedding"] == (role == "enterprise")
            assert worker["has_final_norm"] == (role == "enterprise")
            assert worker["has_logits_processor"] == (role == "enterprise")
            assert worker["local_parameter_elements"] > 0
    front, middle, back = sides["enterprise"]["layer_split"]
    assert sides["enterprise"]["worker_audits"][0]["owned_layers"] == list(range(front)) + list(range(front + middle, 36))
    assert sides["cloud"]["worker_audits"][0]["owned_layers"] == list(range(front, front + middle))
    sides["checkpoint"] = json.loads((ROOT / "models/qwen/checksums.json").read_text())
    sides["gpu_processes"] = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
        "--format=csv"], capture_output=True, text=True, check=True).stdout
    (folder / "model_audit.json").write_text(json.dumps(sides, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/baseline_matrix")
    parser.add_argument("--split", default="1:3", choices=["1:3", "1:1", "3:1"])
    parser.add_argument("--pairs", default="1:1,1:2,2:1,2:2")
    parser.add_argument("--concurrency", default="1,4")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "workloads": [list(pair) for pair in WORKLOADS], "model": "Qwen/Qwen2.5-3B-Instruct",
              "dtype": "FP16", "network": {"gbps_each_direction": 10, "one_way_ms": 5},
              "arrival": "closed_loop", "requests_per_repeat": "max(8, 4 * concurrency)",
              "slo_is_acceptance_gate": False, "detailed_gpu_stage_timing_during_benchmark": False}
    manifest = output / "matrix_config.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        identity = ("split", "pairs", "concurrency", "repeats", "workloads", "dtype", "network")
        if not args.resume or any(previous.get(key) != config[key] for key in identity):
            raise RuntimeError("Existing matrix differs or --resume is missing; use a fresh output directory")
    manifest.write_text(json.dumps(config, indent=2))
    run("stop_previous", ["bash", "poc", "down"], output)
    for pair in args.pairs.split(","):
        enterprise_tp, cloud_tp = map(int, pair.split(":"))
        folder = output / f"tp_{enterprise_tp}_{cloud_tp}"
        folder.mkdir(exist_ok=True)
        expected_labels = [f"isl_{isl}_osl_{osl}_c_{c}_r_{r}" for isl, osl in WORKLOADS
                           for c in map(int, args.concurrency.split(",")) for r in range(args.repeats)]
        def complete_point(label):
            path = folder / label / "summary.json"
            if not path.exists():
                return False
            saved = json.loads(path.read_text())
            return saved["classification"] == "baseline" and bool(saved["points"]) and all(
                p["success_rate"] == 1 and p["fixed_output_length_valid"] for p in saved["points"])
        profile_cases = folder / "profiling/profile_cases.json"
        profile_complete = args.skip_profile or (profile_cases.exists() and
            len(json.loads(profile_cases.read_text())["cases"]) == 20 and all(
                (folder / "profiling" / f"{role}{suffix}").exists()
                for role in ("enterprise", "cloud") for suffix in
                (".nsys-rep", "_cuda_gpu_kern_sum.csv", "_nvtx_sum.csv")))
        if args.resume and profile_complete and all(complete_point(label) for label in expected_labels):
            print(f"RESUME completed topology {pair}", flush=True)
            continue
        common = ["bash", "poc", "up", "--wan", "--split", args.split,
                  "--enterprise-tp", str(enterprise_tp), "--cloud-tp", str(cloud_tp)]
        correctness = folder / "correctness"
        reference = "results/baseline_native_tp1" if enterprise_tp == cloud_tp == 1 else "results/validation_final/reference"
        mixed_tp = enterprise_tp != cloud_tp
        if not (args.resume and all(complete_point(label) for label in expected_labels)):
            run(f"up_{enterprise_tp}_{cloud_tp}", common, output)
            audit(folder)
            run("network_check", ["bash", "poc", "network-check"], folder)
            run("correctness", [PYTHON, "scripts/correctness.py", "--reference", reference,
                                "--steps", "33", "--output", str(correctness)] +
                                (["--cross-tp-observation"] if mixed_tp else []), folder)
        for isl, osl in WORKLOADS:
            for concurrency in map(int, args.concurrency.split(",")):
                for repeat in range(args.repeats):
                    label = f"isl_{isl}_osl_{osl}_c_{concurrency}_r_{repeat}"
                    point = folder / label
                    if args.resume and (point / "summary.json").exists():
                        saved = json.loads((point / "summary.json").read_text())
                        if saved["classification"] == "baseline" and all(p["success_rate"] == 1 and p["fixed_output_length_valid"] for p in saved["points"]):
                            print(f"RESUME {pair} {label}", flush=True)
                            continue
                    run(label, [PYTHON, "scripts/benchmark.py", "--baseline", "--rates", "inf",
                        "--max-concurrency", str(concurrency), "--requests", str(max(8, 4 * concurrency)),
                        "--input", str(isl), "--output-tokens", str(osl), "--seed", str(repeat),
                        "--output", str(point)] + ([] if mixed_tp else
                        ["--correctness-report", str(correctness / "summary.json")]), folder)
        run("down_measurement", ["bash", "poc", "down"], folder)
        if not args.skip_profile:
            run("up_profile", common + ["--profile"], folder)
            live = Path((ROOT / "run/current_results").read_text())
            (folder / "profile_source.txt").write_text(str(live))
            run("profile_workloads", [PYTHON, "scripts/profile_matrix.py"], folder)
            run("down_profile", ["bash", "poc", "down"], folder)
            profile = folder / "profiling"
            profile.mkdir(exist_ok=True)
            for source in live.iterdir():
                if source.is_file() and source.suffix != ".sqlite":
                    shutil.copyfile(source, profile / source.name)
            for role in ("enterprise", "cloud"):
                run("nsys_" + role, [profiler_binary(), "stats", "--report",
                    "cuda_gpu_kern_sum,nvtx_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum",
                    "--format", "csv", "--output", ".", str(profile / (role + ".nsys-rep"))], folder)
    run("restore_demo", ["bash", "poc", "up", "--wan"], output)
    (output / "completion.json").write_text(json.dumps({"result": "COMPLETED", "config": config,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "meaning": "Baseline measurement completed, not SLO acceptance"}, indent=2))
    print("BASELINE MATRIX COMPLETED", flush=True)


if __name__ == "__main__":
    main()
