"""Isolated adapter to the current mainline Q4 baseline / PD simulator."""

import argparse, json, sys
from pathlib import Path
from dataclasses import asdict, replace

ROOT = Path(__file__).resolve().parents[1]
Q4 = ROOT / "Q4"
sys.path.insert(0, str(Q4))
from split_serving_sim.config import load_config, validate_config
from split_serving_sim.catalog import build_config
from split_serving_sim.presets import configure_serving, ServingFeatures
from split_serving_sim.simulator import Simulator
from split_serving_sim.command_cost import CommandCostModel
from split_serving_sim.serving_runtime import VirtualServingSimulator
from split_serving_sim.serving_host import ServingHostWork


def run(payload):
    c = payload.get("concurrency")
    variant = payload["variant"]
    cfg, catalog = build_config(
        payload.get("model", "qwen2.5-3b"), payload.get("hardware", "a10"), variant
    )
    is_open = payload.get("workload_mode") == "open_loop"
    if is_open:
        workload = replace(
            cfg.workload,
            mode="open_loop",
            concurrency=0,
            warmup_requests=0,
            arrival_process="poisson",
            arrival_rate_qps=payload["arrival_rate_qps"],
            random_seed=17,
            warmup_duration_s=30,
            measurement_duration_s=120 if payload["arrival_rate_qps"] >= 4 else 60,
            arrival_tail_s=30,
        )
    else:
        workload = replace(
            cfg.workload,
            num_requests=1024,
            concurrency=c,
            warmup_requests=c,
            measurement_duration_s=60,
        )
        if variant == "baseline" and catalog["calibrated"]:
            cfg = replace(
                cfg,
                static_policy=replace(
                    cfg.static_policy, max_batch_size=8 if c == 1 else 16
                ),
                scheduler=replace(cfg.scheduler, max_num_seqs=8 if c == 1 else 16),
            )
    cfg = replace(
        cfg,
        workload=workload,
        simulation=replace(
            cfg.simulation,
            trace_enabled=False,
            max_trace_records=0,
            max_detailed_trace_records=0,
        ),
    )
    validate_config(cfg)
    if variant == "optimized":
        cost = (
            CommandCostModel.from_file(
                cfg,
                Q4 / "profiles/issue6_pd_tp1_c40_empirical_commands.json",
                sampling="empirical",
                seed=17,
            )
            if catalog["calibrated"]
            else None
        )
        host = (
            ServingHostWork.from_file(cfg, Q4 / "profiles/issue6_c40_serving_host.json")
            if catalog["calibrated"]
            else None
        )
        result = VirtualServingSimulator(cfg, cost, host).run()
    else:
        result = Simulator(cfg).run()
    s = result.summary
    s.pop("scheduling_decisions", None)
    return {
        "engine": "Q4",
        "source": "simulation",
        "variant": variant,
        "concurrency": c,
        "arrival_rate_qps": payload.get("arrival_rate_qps"),
        "input_tokens": 4096,
        "output_tokens": 79,
        "seed": 17,
        "catalog": catalog,
        "resolved_config": asdict(cfg),
        "workload_mode": "open_loop" if is_open else "closed_loop",
        "point": {
            "concurrency": c,
            "arrival_rate_qps": payload.get("arrival_rate_qps"),
            "qps": s["observed_request_throughput_qps"],
            "ttft_mean_ms": s["ttft_ms"]["mean"],
            "tpot_mean_ms": s["tpot_ms"]["mean"],
            "ttft_p99_ms": s["ttft_ms"]["p99"],
            "tpot_p99_ms": s["tpot_ms"]["p99"],
        },
        "summary": s,
        "calibration": catalog["assumptions"],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(json.loads(a.input.read_text())), indent=2))
