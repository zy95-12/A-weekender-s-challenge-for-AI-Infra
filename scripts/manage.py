"""Idempotent local startup; records and checks process ownership before stop."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import sysconfig
import time
import urllib.request
from network_state import configure

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "run"
PYTHON = ROOT / ".venv/bin/python"


def profiler_binary():
    isolated = Path.home() / ".local/opt/nsight-systems-2025.3.1/opt/nvidia/nsight-systems-cli/2025.3.1/target-linux-x64/nsys"
    return os.environ.get("SPLIT_NSYS_BIN", str(isolated) if isolated.is_file() else "nsys")


def spawn(name, command, env=None):
    logfile = (RUN / f"{name}.log").open("a")
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=logfile,
                               stderr=subprocess.STDOUT, start_new_session=True)
    logfile.close()
    info = {"pid": process.pid, "command": command,
            "start_ticks": Path(f"/proc/{process.pid}/stat").read_text().split()[21]}
    (RUN / f"{name}.pid.json").write_text(json.dumps(info))
    return process


def stop():
    for name in ("proxy", "enterprise", "cloud", "telemetry"):
        path = RUN / f"{name}.pid.json"
        if not path.exists():
            continue
        info = json.loads(path.read_text())
        proc = Path(f"/proc/{info['pid']}/stat")
        if proc.exists() and proc.read_text().split()[21] == info["start_ticks"]:
            os.killpg(info["pid"], signal.SIGTERM)
            for _ in range(450 if any(Path(arg).name == "nsys" for arg in info["command"]) else 100):
                if not proc.exists():
                    break
                time.sleep(0.1)
            if proc.exists():
                os.killpg(info["pid"], signal.SIGKILL)
        path.unlink()


def wait_ready(url, process, name, namespace=None):
    for i in range(600):
        if process.poll() is not None:
            raise RuntimeError(f"{name} exited; inspect run/{name}.log")
        try:
            if namespace:
                result = subprocess.run(["ip", "netns", "exec", namespace, "curl", "-fsS", "--max-time", "2", url],
                                        capture_output=True)
                if result.returncode == 0:
                    print(result.stdout.decode(), flush=True)
                    return
            else:
                with urllib.request.urlopen(url, timeout=2) as response:
                    print(response.read().decode(), flush=True)
                    return
        except Exception:
            pass
        if i % 30 == 0:
            print(f"Waiting for {name} GPU workers ({i}s)…", flush=True)
        time.sleep(1)
    raise RuntimeError(f"{name} startup timeout")


def up(args):
    RUN.mkdir(exist_ok=True)
    enterprise_tp = args.enterprise_tp or args.tp
    cloud_tp = args.cloud_tp or args.tp
    if (RUN / "enterprise.pid.json").exists():
        reusable = False
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3) as response:
                current = json.load(response)
            previous = json.loads((RUN / "launch.json").read_text())
            process_keys = set(vars(args)) - {"wan", "delay", "bandwidth_gbps"}
            reusable = (current["split"] == args.split and current["tp"] == enterprise_tp
                        and all(previous.get(key) == vars(args)[key] for key in process_keys))
        except Exception:
            pass
        if reusable:
            # Network errors must propagate, not trigger an unrelated GPU restart.
            configure(args.wan, args.delay, args.bandwidth_gbps)
            print("Already running; network reapplied and verified: http://127.0.0.1:8000")
            return
        stop()
    subprocess.run(["bash", str(ROOT / "scripts/network.sh"), "up"], check=True)
    subprocess.run([str(PYTHON), str(ROOT / "scripts/model_views.py")], cwd=ROOT, check=True)
    configure(args.wan, args.delay, args.bandwidth_gbps)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    results = ROOT / "results" / stamp
    results.mkdir(parents=True)
    subprocess.run([str(PYTHON), str(ROOT / "scripts/environment.py"), str(results / "environment.json")],
                   cwd=ROOT, check=True)
    (RUN / "current_results").write_text(str(results))
    env = {**os.environ, "PYTHONPATH": str(ROOT), "VLLM_USE_V1": "0", "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "NCCL_SOCKET_IFNAME": "lo", "GLOO_SOCKET_IFNAME": "lo",
           "OMP_NUM_THREADS": "4", "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
           "TOKENIZERS_PARALLELISM": "false"}
    nccl_library = Path(sysconfig.get_paths()["purelib"]) / "nvidia/nccl/lib/libnccl.so.2"
    if not nccl_library.is_file():
        raise RuntimeError("Pinned NCCL library is missing from the virtual environment")
    # Prevent Nsight LD_LIBRARY_PATH from selecting a different NCCL build.
    env["VLLM_NCCL_SO_PATH"] = str(nccl_library)
    def command_for(role):
        model = ROOT / "models" / ("qwen" if role == "enterprise" else "cloud")
        tp = enterprise_tp if role == "enterprise" else cloud_tp
        common = [str(PYTHON), "-m", "split_poc.server", "--model", str(model),
                  "--split", args.split, "--tp", str(tp), "--results", str(results), "--ipc-mode", args.ipc_mode]
        if args.wire_fast:
            common.append("--wire-fast")
        if args.profile or args.phase_profile:
            common.append("--phase-profile")
        if not args.profile:
            return common
        profiler = profiler_binary()
        help_text = subprocess.run([profiler, "profile", "--help"], capture_output=True, text=True, check=True).stdout
        event_options = ["--cuda-event-trace=false"] if "--cuda-event-trace" in help_text else []
        return [profiler, "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none", *event_options,
                "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--force-overwrite=true",
                "--output", str(results / role), *common]
    try:
        cloud = spawn("cloud", ["ip", "netns", "exec", "split-cloud", *command_for("cloud"), "--role", "cloud",
                       "--host", "10.205.0.2", "--port", "8001", "--dist-port", "29502"],
                      {**env, "CUDA_VISIBLE_DEVICES": "2,3" if cloud_tp == 2 else "2"})
        wait_ready("http://10.205.0.2:8001/health", cloud, "cloud", "split-enterprise")
        enterprise = spawn("enterprise", ["ip", "netns", "exec", "split-enterprise", *command_for("enterprise"),
                            "--role", "enterprise", "--host", "0.0.0.0", "--cloud", "http://10.205.0.2:8001",
                            "--diagnostics"],
                           {**env, "CUDA_VISIBLE_DEVICES": "0,1" if enterprise_tp == 2 else "0"})
        wait_ready("http://10.204.0.2:8000/health", enterprise, "enterprise")
        proxy = spawn("proxy", [str(PYTHON), str(ROOT / "scripts/proxy.py")])
        wait_ready("http://127.0.0.1:8000/health", proxy, "proxy")
        subprocess.run([str(PYTHON), str(ROOT / "scripts/smoke.py")], check=True)
        (RUN / "launch.json").write_text(json.dumps(vars(args), indent=2))
        print(f"\nDemo ready: http://127.0.0.1:8000\nResults: {results}\nSplit: {args.split}; TP: {enterprise_tp}+{cloud_tp}", flush=True)
    except BaseException:
        stop()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["up", "down", "status"])
    parser.add_argument("--split", choices=["1:3", "1:1", "3:1"], default="1:3")
    parser.add_argument("--tp", type=int, choices=[1, 2], default=2)
    parser.add_argument("--enterprise-tp", type=int, choices=[1, 2])
    parser.add_argument("--cloud-tp", type=int, choices=[1, 2])
    parser.add_argument("--wan", action="store_true")
    parser.add_argument("--delay", "--delay-ms", dest="delay", type=float, default=5)
    parser.add_argument("--bandwidth-gbps", type=float, default=10)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--ipc-mode", choices=["pipe", "shm"], default="pipe")
    parser.add_argument("--wire-fast", action="store_true")
    parser.add_argument("--phase-profile", action="store_true")
    args = parser.parse_args()
    if args.action == "up":
        up(args)
    elif args.action == "down":
        stop()
    else:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
            print(response.read().decode())
