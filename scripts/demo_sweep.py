"""Run the real optimized 4K/79 workload until the joint SLO fails."""

import argparse, json, subprocess, sys
from network_state import expected, check
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument(
    "--levels",
    nargs="+",
    type=int,
    default=[1, 8, 16, 24, 32, 40, 44, 48, 56, 64, 80, 96],
)
a = p.parse_args()
if len(set(a.levels)) != len(a.levels) or any(c < 1 or c > 96 for c in a.levels):
    p.error("Use distinct concurrency levels in 1–96")
launch = json.loads((ROOT / "run/launch.json").read_text())
required = {
    "pd": True,
    "prefill_replicas": 2,
    "prefill_tp": 1,
    "decode_tp": 1,
    "ipc_mode": "shm",
    "wire_fast": True,
    "prefill_chunk_size": 2048,
    "scheduler_policy": "decode-first",
    "decode_quota": 4,
    "pipeline_window": 2,
    "pd_prefill_window": 3,
    "pd_chunk_transfer": False,
    "pd_control_channel": True,
    "max_active": 96,
    "kv_blocks": 32768,
    "wan": True,
    "tcp_buffer_mib": 16,
}
if any(launch.get(k) != v for k, v in required.items()):
    raise RuntimeError("Start the exact optimized WAN preset before this sweep")
if launch.get("profile") or launch.get("operator_profile") or launch.get("phase_profile"):
    raise RuntimeError("Disable profiling before benchmarking")
a.output.mkdir(parents=True, exist_ok=False)
(a.output / "launch.json").write_text(json.dumps(launch, indent=2))
check(expected(launch), a.output / "network_check.json")
for c in a.levels:
    out = a.output / f"c{c}"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/pd_benchmark.py"),
            "--variant",
            f"live-demo-optimized-c{c}",
            "--out",
            str(out),
            "--prompt",
            str(ROOT / "Q1/data/current_prompt.json"),
            "--reference",
            str(ROOT / "Q1/data/current_reference.json"),
            "--concurrency",
            str(c),
            "--seconds",
            "60",
            "--cycles",
            "3",
        ],
        check=True,
        cwd=ROOT,
    )
    s = json.loads((out / "summary.json").read_text())
    if min(s["slo_attainment"], s["start_cohort_slo_attainment"]) < 0.99:
        print("Joint SLO boundary crossed; stopping coarse sweep.", flush=True)
        break
