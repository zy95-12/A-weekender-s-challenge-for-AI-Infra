"""Run the official vLLM client; report only measured SLO passing points."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal
import shutil
import httpx
from network_state import expected, snapshot, verify, check, network_lock
from benchmark_artifacts import export_trace

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rates", default="0.5,1,1.5,2")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--input", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--output", default="results/benchmark")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--correctness-report")
    parser.add_argument("--baseline", action="store_true", help="Measure a baseline without SLO acceptance gates")
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    launch_file = ROOT / "run/launch.json"
    deployment = json.loads(launch_file.read_text()) if launch_file.exists() else None
    intent = expected(deployment)
    if deployment and (deployment.get("profile") or deployment.get("phase_profile")):
        raise SystemExit("Restart the demo without --profile before measuring SLO")
    if args.correctness_report or (args.requests >= 1000 and not args.baseline):
        if not args.correctness_report or json.loads(Path(args.correctness_report).read_text()).get("result") != "PASS":
            raise SystemExit("Formal benchmark requires --correctness-report pointing to a PASS summary.json")
        validated = json.loads(Path(args.correctness_report).read_text())["server"]
        response = httpx.get(args.url + "/health", timeout=5, trust_env=False)
        response.raise_for_status()
        current = response.json()
        for key in ("split", "layer_split", "tp", "cloud_tp", "model_id", "revision", "protocol", "optimizations"):
            if current.get(key) != validated.get(key):
                raise SystemExit(f"Correctness configuration mismatch: {key}")
    dest = Path(args.output).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "summary.json").exists():
        raise RuntimeError(f"Refusing to overwrite existing summary: {dest}")
    for rate in args.rates.split(","):
        point_path = dest / f"qps_{rate}"
        if point_path.exists() and any(point_path.iterdir()):
            raise RuntimeError(f"Refusing to overwrite existing experiment: {point_path}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "models/qwen"), local_files_only=True)
    dataset = dest / "workload.jsonl"
    with dataset.open("w") as f:
        for index in range(args.requests):
            prefix = tokenizer.encode(f"Request {index}:", add_special_tokens=False)
            unit = tokenizer.encode(" apple", add_special_tokens=False)
            assert len(unit) == 1
            ids = prefix + unit * (args.input - len(prefix))
            text = tokenizer.decode(ids)
            assert len(tokenizer.encode(text, add_special_tokens=False)) == args.input
            f.write(json.dumps({"prompt": text}) + "\n")
    env = {**os.environ, "VLLM_USE_V1": "0", "VLLM_NO_USAGE_STATS": "1"}
    points = []
    for rate in args.rates.split(","):
        folder = dest / f"qps_{rate}"
        folder.mkdir(exist_ok=True)
        if any(folder.iterdir()):
            raise RuntimeError(f"Refusing to overwrite existing experiment: {folder}")
        prefix = f"qps-{rate}-{time.time_ns()}-"
        network_check = check(intent, folder / "network_check.json")
        network = snapshot()
        (folder / "network.json").write_text(json.dumps(network, indent=2))
        verify(network, intent)
        live = None
        pointer = ROOT / "run/current_results"
        if pointer.exists():
            live = Path(pointer.read_text().strip())
            for name in ("environment.json", "enterprise_config.json", "cloud_config.json"):
                if (live / name).exists():
                    shutil.copyfile(live / name, folder / ("launch_environment.json" if name == "environment.json" else name))
        subprocess.run([sys.executable, str(ROOT / "scripts/environment.py"), str(folder / "environment.json")],
                       cwd=ROOT, check=True)
        trace_path = live / "split_trace.jsonl" if live else None
        trace_offset = trace_path.stat().st_size if trace_path and trace_path.exists() else 0
        command = [str(ROOT / ".venv/bin/vllm"), "bench", "serve", "--backend", "openai",
            "--base-url", args.url, "--endpoint", "/v1/completions",
            "--model", str(ROOT / "models/qwen"), "--served-model-name", "Qwen/Qwen2.5-3B-Instruct",
            "--dataset-name", "custom", "--dataset-path", str(dataset),
            "--custom-output-len", str(args.output_tokens), "--custom-skip-chat-template",
            "--request-rate", rate, "--num-prompts", str(args.requests), "--ignore-eos",
            "--request-id-prefix", prefix,
            "--save-result", "--save-detailed", "--result-dir", str(folder),
            "--result-filename", "benchmark.json", "--percentile-metrics", "ttft,tpot,itl,e2el",
            "--metric-percentiles", "50,95,99", "--seed", str(args.seed)]
        if not args.baseline:
            command += ["--goodput", "ttft:3000", "tpot:100"]
        if args.max_concurrency:
            command += ["--max-concurrency", str(args.max_concurrency)]
        (folder / "config.json").write_text(json.dumps({"command": command, "deployment": deployment,
            "experiment_id": prefix, "network_intent": intent, "network_measured": network_check["measured"],
            "telemetry_scope": "client_process_including_warmup", **vars(args)}, indent=2))
        validation = {"result": "RUNNING", "experiment_id": prefix, "started_at_unix": time.time()}
        (folder / "measurement_validation.json").write_text(json.dumps(validation, indent=2))
        telemetry = None
        if live:
            telemetry = subprocess.Popen([sys.executable, str(ROOT / "scripts/telemetry.py"),
                "--output", str(folder), "--seconds", "36000"], cwd=ROOT, start_new_session=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            with (folder / "client.log").open("w") as logfile:
                print(f"Benchmark QPS={rate}, requests={args.requests}", flush=True)
                subprocess.run(command, cwd=ROOT, env=env, stdout=logfile, stderr=subprocess.STDOUT, check=True)
        except BaseException as error:
            validation.update(result="FAIL", error=str(error))
            (folder / "measurement_validation.json").write_text(json.dumps(validation, indent=2))
            raise
        finally:
            if telemetry and telemetry.poll() is None:
                os.killpg(telemetry.pid, signal.SIGTERM)
                telemetry.wait(timeout=10)
            if trace_path and trace_path.exists():
                export_trace(trace_path, trace_offset, folder, prefix, args.requests)
        after = snapshot()
        (folder / "network_after.json").write_text(json.dumps(after, indent=2))
        try:
            verify(after, intent)
            current_intent = expected(json.loads(launch_file.read_text()))
            if current_intent != intent:
                raise ValueError("Network intent changed during measurement")
        except Exception as error:
            validation.update(result="FAIL", error=str(error))
            (folder / "measurement_validation.json").write_text(json.dumps(validation, indent=2))
            raise
        result = json.loads((folder / "benchmark.json").read_text())
        success = result["completed"] / args.requests
        lengths_ok = all(length == args.output_tokens for length, error in
                         zip(result["output_lens"], result["errors"]) if not error)
        lengths_ok = lengths_ok and all(length == args.input for length in result["input_lens"])
        point = {"offered_qps": None if rate == "inf" else float(rate),
                 "arrival_mode": "closed_loop" if rate == "inf" else "poisson",
                 "max_concurrency": args.max_concurrency, "seed": args.seed,
                 "achieved_qps": result["request_throughput"],
                 "success_rate": success, "p99_ttft_ms": result["p99_ttft_ms"],
                 "p99_tpot_ms": result["p99_tpot_ms"], "fixed_output_length_valid": lengths_ok}
        for metric in ("mean_ttft_ms", "median_ttft_ms", "p50_ttft_ms", "p95_ttft_ms",
                       "mean_tpot_ms", "median_tpot_ms", "p50_tpot_ms", "p95_tpot_ms",
                       "mean_e2el_ms", "p50_e2el_ms", "p95_e2el_ms", "p99_e2el_ms",
                       "output_throughput", "duration", "completed"):
            if metric in result:
                point[metric] = result[metric]
        point["slo_pass"] = (success >= .99 and lengths_ok and point["p99_ttft_ms"] <= 3000
                             and point["p99_tpot_ms"] <= 100)
        points.append(point)
        if args.baseline and (success != 1 or not lengths_ok):
            validation.update(result="FAIL", error="Failed requests or invalid token lengths")
            (folder / "measurement_validation.json").write_text(json.dumps(validation, indent=2))
            raise RuntimeError("Baseline has failed requests or invalid token lengths; inspect raw client result")
        with (folder / "requests.jsonl").open("w") as f:
            for i, (ttft, itls, length, error) in enumerate(zip(result["ttfts"], result["itls"], result["output_lens"], result["errors"])):
                f.write(json.dumps({"client_request_id": prefix + str(i), "experiment_id": prefix,
                    "is_warmup": False, "is_measured": True, "ttft_ms": ttft * 1000,
                    "tpot_ms": sum(itls) * 1000 / (length - 1) if length > 1 else None,
                    "e2e_ms": (ttft + sum(itls)) * 1000, "output_tokens": length, "error": error}) + "\n")
        validation.update(result="PASS", finished_at_unix=time.time())
        (folder / "measurement_validation.json").write_text(json.dumps(validation, indent=2))
    passing = [p for p in points if p["slo_pass"] and p["offered_qps"] is not None]
    summary = {"points": points, "max_slo_offered_qps": (max((p["offered_qps"] for p in passing), default=None)
                                                        if args.requests >= 1000 and not args.baseline else None),
               "sample_count_per_point": args.requests,
               "classification": "baseline" if args.baseline else "formal" if args.requests >= 1000 else "smoke_only",
               "note": "Highest tested passing offered rate; achieved throughput is reported separately."}
    (dest / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    with network_lock():
        main()
