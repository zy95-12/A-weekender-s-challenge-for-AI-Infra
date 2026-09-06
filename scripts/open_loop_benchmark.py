"""Independent Poisson arrivals; no client concurrency cap or completion feedback."""

import argparse
import asyncio
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from pd_benchmark import one, drained
from network_state import network_lock, expected, check, snapshot, verify

ROOT = Path(__file__).resolve().parents[1]


async def measure(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    launch = json.loads((ROOT / "run/launch.json").read_text())
    command = [
        sys.executable,
        str(ROOT / "scripts/manage.py"),
        "up",
        "--preset",
        args.system,
        "--wan",
        "--max-active",
        "96",
        "--kv-blocks",
        "32768",
        "--print-config",
    ]
    expected_launch = json.loads(subprocess.check_output(command, text=True))
    ignored = {"pd_epoch", "print_config"}
    if {k: v for k, v in launch.items() if k not in ignored} != {
        k: v for k, v in expected_launch.items() if k not in ignored
    }:
        raise RuntimeError(
            "Deployment differs from the selected preset with max_active=96 and kv_blocks=32768"
        )
    (out / "launch.json").write_text(json.dumps(launch, indent=2))
    intent = expected(launch)
    check(intent, out / "network_check.json")
    prompt = json.loads(args.prompt.read_text())["prompt_ids"]
    reference = json.loads(args.reference.read_text())
    rows, health, tasks = [], [], []
    rng = random.Random(args.seed)
    offsets = []
    t = 0
    total = args.warmup + args.seconds + args.tail
    while True:
        t += rng.expovariate(args.rate)
        if t >= total:
            break
        offsets.append(t)
    (out / "schedule.json").write_text(
        json.dumps(
            {
                "rate": args.rate,
                "seed": args.seed,
                "offsets_s": offsets,
                "warmup_s": args.warmup,
                "measurement_s": args.seconds,
                "tail_s": args.tail,
            },
            indent=2,
        )
    )
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8000",
        timeout=180,
        trust_env=False,
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=256),
    ) as client:
        (out / "initial_health.json").write_text(
            json.dumps(await drained(client), indent=2)
        )
        r = await client.post("/debug/greedy", json={"prompt_ids": prompt, "steps": 79})
        r.raise_for_status()
        assert r.json()["tokens"] == reference["greedy_ids"][:79]
        await drained(client)
        began = time.perf_counter() + 0.2
        start = began + args.warmup
        end = start + args.seconds

        async def request(index, scheduled):
            row = await one(
                client, prompt, f"open-{args.label}-{index}", reference["text"]
            )
            row["scheduled_start"] = scheduled
            row["dispatch_lag_ms"] = (row["start"] - scheduled) * 1000
            row["scheduled_ttft_ms"] = (
                None
                if row["ttft_ms"] is None
                else row["ttft_ms"] + row["dispatch_lag_ms"]
            )
            row["slo_pass"] = (
                row["slo_pass"]
                and row["tokens"] == 79
                and row["scheduled_ttft_ms"] <= 3000
            )
            rows.append(row)

        stopped = False

        async def monitor():
            while not stopped:
                now = time.perf_counter()
                try:
                    r = await client.get("/health")
                    r.raise_for_status()
                    health.append({"time": now, **r.json()})
                except Exception as exc:
                    health.append({"time": now, "error": repr(exc)})
                await asyncio.sleep(1)

        monitor_task = asyncio.create_task(monitor())
        try:
            for i, offset in enumerate(offsets):
                scheduled = began + offset
                await asyncio.sleep(max(0, scheduled - time.perf_counter()))
                tasks.append(asyncio.create_task(request(i, scheduled)))
            await asyncio.sleep(max(0, began + total - time.perf_counter()))
            await asyncio.gather(*tasks)
        finally:
            stopped = True
            await monitor_task
            (out / "requests.json").write_text(json.dumps(rows, indent=2))
            (out / "health.json").write_text(json.dumps(health, indent=2))
        final = await drained(client)
        (out / "final_health.json").write_text(json.dumps(final, indent=2))
    arrival = [r for r in rows if start <= r["scheduled_start"] < end]
    completion = [r for r in rows if start <= r["end"] < end]
    if not arrival or not completion:
        raise RuntimeError(
            "Empty measurement cohort; increase rate or measurement duration. Raw requests were saved."
        )

    def stats(rr):
        result = {
            "requests": len(rr),
            "errors": sum(r["error"] is not None or r["tokens"] != 79 for r in rr),
            "joint_slo": sum(r["slo_pass"] for r in rr) / len(rr),
        }
        for key in ["ttft_ms", "scheduled_ttft_ms", "tpot_ms", "dispatch_lag_ms"]:
            values = [r[key] for r in rr if r[key] is not None]
            result["mean_" + key] = float(np.mean(values)) if values else None
            result["p99_" + key] = float(np.percentile(values, 99)) if values else None
        result["ttft_failures"] = sum(
            r["scheduled_ttft_ms"] is None or r["scheduled_ttft_ms"] > 3000 for r in rr
        )
        result["tpot_failures"] = sum(
            r["tpot_ms"] is None or r["tpot_ms"] > 100 for r in rr
        )
        return result

    inflight = lambda t: sum(r["start"] <= t < r["end"] for r in rows)
    events = sorted(
        [
            (max(start, r["start"]), 1)
            for r in rows
            if r["start"] < end and r["end"] > start
        ]
        + [
            (min(end, r["end"]), -1)
            for r in rows
            if r["start"] < end and r["end"] > start
        ]
    )
    current = peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    report = {
        "label": args.label,
        "system": args.system,
        "target_arrival_rate": args.rate,
        "seed": args.seed,
        "measurement_start": start,
        "measurement_end": end,
        "duration_s": args.seconds,
        "actual_arrival_rate": len(arrival) / args.seconds,
        "completion_rate": len(completion) / args.seconds,
        "successful_completed_qps": sum(
            r["error"] is None and r["tokens"] == 79 for r in completion
        )
        / args.seconds,
        "goodput_qps": sum(r["slo_pass"] for r in completion) / args.seconds,
        "arrival_cohort": stats(arrival),
        "completion_cohort": stats(completion),
        "mean_inflight": sum(
            max(0, min(end, r["end"]) - max(start, r["start"])) for r in rows
        )
        / args.seconds,
        "peak_inflight": peak,
        "inflight_boundaries": [
            {"offset_s": i, "inflight": inflight(start + i)}
            for i in range(0, args.seconds + 1, 30)
        ],
        "all_outputs_match": all(
            r["error"] is None and r["tokens"] == 79 for r in rows
        ),
        "measurement_arrivals_completed_before_load_stop": all(
            r["end"] <= began + total for r in arrival
        ),
    }
    after = snapshot()
    verify(after, intent)
    (out / "network_after.json").write_text(json.dumps(after, indent=2))
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--system", choices=["baseline", "optimized"], default="optimized")
    p.add_argument("--rate", type=float, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--seconds", type=int, default=120)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--tail", type=int, default=30)
    p.add_argument("--prompt", type=Path, default=ROOT / "Q1/data/current_prompt.json")
    p.add_argument(
        "--reference", type=Path, default=ROOT / "Q1/data/current_reference.json"
    )
    a = p.parse_args()
    if (
        not math.isfinite(a.rate)
        or a.rate <= 0
        or a.seconds <= 0
        or a.warmup < 0
        or a.tail < 0
    ):
        p.error("Invalid load settings")
    with network_lock():
        asyncio.run(measure(a))
