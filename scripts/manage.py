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
import uuid
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
    for name in ("proxy", "enterprise", "cloud", "cloud_prefill", "cloud_decode", "telemetry"):
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
    if getattr(args,'pd',False):
        enterprise_tp=args.enterprise_tp or 1
        cloud_tp=args.prefill_tp
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
    if args.pd:args.pd_epoch=uuid.uuid4().hex
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
    if args.operator_profile:
        env["SPLIT_OPERATOR_CAPTURE"] = "1"
    def command_for(role):
        model = ROOT / "models" / ("qwen" if role == "enterprise" else "cloud")
        tp = enterprise_tp if role == "enterprise" else (args.decode_tp if role=='cloud_decode' else cloud_tp)
        common = [str(PYTHON), "-m", "split_poc.server", "--model", str(model),
                  "--split", args.split, "--tp", str(tp), "--results", str(results), "--ipc-mode", args.ipc_mode]
        if args.wire_fast:
            common.append("--wire-fast")
        common += ["--prefill-chunk-size", str(args.prefill_chunk_size),
                   "--scheduler-policy", args.scheduler_policy, "--decode-quota", str(args.decode_quota)]
        common += ["--tcp-buffer-mib", str(args.tcp_buffer_mib), "--pipeline-window", str(args.pipeline_window)]
        common += ["--max-active", str(args.max_active), "--kv-blocks", str(args.kv_blocks)]
        if args.pd:
            common += ['--pd-prefill-window',str(args.pd_prefill_window)]
            common += ['--pd','--pd-epoch',args.pd_epoch,'--prefill-tp',str(args.prefill_tp),
                       '--decode-tp',str(args.decode_tp),'--cloud-decode','http://10.205.0.2:8002']
            if role!='enterprise':common+=['--pd-role','decode' if role=='cloud_decode' else 'prefill']
            if args.pd_verify_kv:common+=['--pd-verify-kv']
            if args.pd_chunk_transfer:common+=['--pd-chunk-transfer']
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
        if args.pd:
            prefill_gpus='1,2' if args.prefill_tp==2 else '1'
            decode_gpus='3' if args.decode_tp==1 else '2,3'
            cloud=spawn('cloud_prefill',['ip','netns','exec','split-cloud',*command_for('cloud_prefill'),
                '--role','cloud','--host','10.205.0.2','--port','8001','--dist-port','29502'],
                {**env,'CUDA_VISIBLE_DEVICES':prefill_gpus})
            decode=spawn('cloud_decode',['ip','netns','exec','split-cloud',*command_for('cloud_decode'),
                '--role','cloud','--host','10.205.0.2','--port','8002','--dist-port','29503'],
                {**env,'CUDA_VISIBLE_DEVICES':decode_gpus})
            wait_ready('http://10.205.0.2:8002/health',decode,'cloud_decode','split-enterprise')
        else:
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
        topology=f'{enterprise_tp}+{cloud_tp}'+(f'+{args.decode_tp}' if args.pd else '')
        print(f"\nDemo ready: http://127.0.0.1:8000\nResults: {results}\nSplit: {args.split}; TP: {topology}", flush=True)
    except BaseException:
        stop()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["up", "down", "status"])
    parser.add_argument("--split", choices=["1:3", "1:1", "3:1", "4:30:2"], default="1:3")
    parser.add_argument("--tp", type=int, choices=[1, 2], default=2)
    parser.add_argument("--enterprise-tp", type=int, choices=[1, 2])
    parser.add_argument("--cloud-tp", type=int, choices=[1, 2])
    parser.add_argument("--wan", action="store_true")
    parser.add_argument("--delay", "--delay-ms", dest="delay", type=float, default=5)
    parser.add_argument("--bandwidth-gbps", type=float, default=10)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--operator-profile", action="store_true", help="Record operator tensor metadata and NVTX ranges; requires --profile")
    parser.add_argument("--kv-blocks", type=int, default=8192, help="KV pages per pool; size for the intended concurrency")
    parser.add_argument("--max-active", type=int, choices=range(1,129), default=8)
    parser.add_argument("--ipc-mode", choices=["pipe", "shm"], default="pipe")
    parser.add_argument("--wire-fast", action="store_true")
    parser.add_argument("--prefill-chunk-size", type=int, default=0)
    parser.add_argument("--scheduler-policy", choices=["legacy","decode-first"], default="legacy")
    parser.add_argument("--decode-quota", type=int, default=1)
    parser.add_argument("--tcp-buffer-mib", type=int, default=0)
    parser.add_argument("--phase-profile", action="store_true")
    parser.add_argument('--pd-prefill-window',type=int,default=0,help='PD prefill in-flight limit; 0 inherits pipeline-window; decode keeps pipeline-window')
    parser.add_argument("--pipeline-window", type=int, default=0)
    parser.add_argument('--pd',action='store_true',help='Separate cloud prefill and decode GPU groups')
    parser.add_argument('--prefill-tp',type=int,choices=[1,2],default=2)
    parser.add_argument('--decode-tp',type=int,choices=[1,2],default=1)
    parser.add_argument('--pd-verify-kv',action='store_true',help='Diagnostic exact KV copy hashes; excluded from benchmarks')
    parser.add_argument('--pd-chunk-transfer',action='store_true',help='Migrate completed KV pages after each prefill chunk')
    args = parser.parse_args()
    if args.pd and (args.prefill_tp+args.decode_tp!=3 or args.enterprise_tp not in (None,1) or not args.pipeline_window):
        parser.error('PD on this four-GPU host requires E1, P+D=3 and --pipeline-window')
    if args.operator_profile and not args.profile:
        parser.error("--operator-profile requires --profile")
    if args.kv_blocks < 1:parser.error("KV blocks must be positive")
    if not 0 <= args.prefill_chunk_size <= 16384 or args.decode_quota < 1:
        parser.error("Invalid prefill chunk or decode quota")
    if not 0 <= args.pd_prefill_window <= 8 or (args.pd_prefill_window and not args.pd):
        parser.error('PD prefill window requires PD and must be 0..8')
    if not 0 <= args.pipeline_window <= 8:
        parser.error("Pipeline window must be 0..8")
    if not 0 <= args.tcp_buffer_mib <= 64:
        parser.error("TCP buffer must be 0..64 MiB")
    if args.action == "up":
        up(args)
    elif args.action == "down":
        stop()
    else:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
            print(response.read().decode())
