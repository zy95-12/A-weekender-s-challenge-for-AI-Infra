"""Isolated adapter to the current mainline Q4 baseline / PD simulator."""

import argparse, json, sys
from pathlib import Path
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
Q4 = ROOT / "Q4"
sys.path.insert(0, str(Q4))
from split_serving_sim.config import load_config
from split_serving_sim.presets import configure_serving, ServingFeatures
from split_serving_sim.simulator import Simulator
from split_serving_sim.command_cost import CommandCostModel
from split_serving_sim.serving_runtime import VirtualServingSimulator
from split_serving_sim.serving_host import ServingHostWork


def run(payload):
    c = payload["concurrency"]
    variant = payload["variant"]
    cfg = load_config(Q4 / "configs/issue6_baseline_host.json")
    cfg = configure_serving(
        cfg,
        ServingFeatures.optimized() if variant == "optimized" else ServingFeatures(),
    )
    cfg = replace(
        cfg,
        workload=replace(
            cfg.workload,
            num_requests=1024,
            concurrency=c,
            warmup_requests=c,
            measurement_duration_s=60,
        ),
        simulation=replace(
            cfg.simulation,
            trace_enabled=False,
            max_trace_records=0,
            max_detailed_trace_records=0,
        ),
    )
    if variant == "optimized":
        cost = CommandCostModel.from_file(
            cfg,
            Q4 / "profiles/issue6_pd_tp1_c40_empirical_commands.json",
            sampling="empirical",
            seed=17,
        )
        host = ServingHostWork.from_file(
            cfg, Q4 / "profiles/issue6_c40_serving_host.json"
        )
        result = VirtualServingSimulator(cfg, cost, host).run()
    else:
        cfg = replace(
            cfg,
            static_policy=replace(
                cfg.static_policy, max_batch_size=8 if c == 1 else 16
            ),
            scheduler=replace(cfg.scheduler, max_num_seqs=8 if c == 1 else 16),
        )
        result = Simulator(cfg).run()
    s = result.summary
    s.pop("scheduling_decisions", None)
    return {
        "engine": "Q4",
        "source": "simulation",
        "variant": variant,
        "concurrency": c,
        "input_tokens": 4096,
        "output_tokens": 79,
        "seed": 17,
        "point": {
            "concurrency": c,
            "qps": s["observed_request_throughput_qps"],
            "ttft_mean_ms": s["ttft_ms"]["mean"],
            "tpot_mean_ms": s["tpot_ms"]["mean"],
            "ttft_p99_ms": s["ttft_ms"]["p99"],
            "tpot_p99_ms": s["tpot_ms"]["p99"],
        },
        "summary": s,
        "calibration": "Optimized: Qwen2.5-3B/A10 C40 command+host empirical profile; baseline: operator+host model. Other concurrency points are predictions, not measured capacity.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(json.loads(a.input.read_text())), indent=2))
