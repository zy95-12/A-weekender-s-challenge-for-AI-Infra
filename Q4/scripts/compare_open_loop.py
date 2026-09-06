"""Replay archived measured arrivals without fitting costs to the replay results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

Q4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Q4))
from split_serving_sim.config import RequestSpec, load_config, validate_config
from split_serving_sim.presets import ServingFeatures, configure_serving
from split_serving_sim.simulator import Simulator
from split_serving_sim.command_cost import CommandCostModel
from split_serving_sim.serving_runtime import VirtualServingSimulator
from split_serving_sim.serving_host import ServingHostWork


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def run(root, out, only=None):
    out.mkdir(parents=True, exist_ok=True)
    profiles = [
        Q4 / "configs/issue6_baseline_host.json",
        *sorted((Q4 / "profiles").glob("*.json")),
    ]
    write(
        out / "provenance.json",
        {
            "measured_manifest": json.loads((root / "manifest.json").read_text()),
            "profile_sha256": {
                str(p.relative_to(Q4)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in profiles
            },
            "cost_sampling_seed": 17,
            "costs_refitted": False,
        },
    )
    comparisons = []
    for measured in json.loads((root / "summaries.json").read_text()):
        point = measured["path"]
        if only and point not in only:
            continue
        schedule_path = root / point / "schedule.json"
        schedule = json.loads(schedule_path.read_text())
        optimized = measured["system"] == "optimized"
        cfg = configure_serving(
            load_config(Q4 / "configs/issue6_baseline_host.json"),
            ServingFeatures.optimized() if optimized else ServingFeatures(),
        )
        cfg = replace(
            cfg,
            workload=replace(
                cfg.workload,
                mode="open_loop",
                concurrency=0,
                warmup_requests=0,
                warmup_duration_s=schedule["warmup_s"],
                measurement_duration_s=schedule["measurement_s"],
                arrival_tail_s=schedule["tail_s"],
                arrival_rate_qps=schedule["rate"],
                random_seed=schedule["seed"],
                arrival_process="poisson",
                requests=tuple(
                    RequestSpec(i, t * 1000, 4096, 79)
                    for i, t in enumerate(schedule["offsets_s"])
                ),
            ),
            static_policy=replace(cfg.static_policy, max_batch_size=96),
            scheduler=replace(
                cfg.scheduler,
                max_num_seqs=96,
                kv_cache=replace(cfg.scheduler.kv_cache, num_blocks=32768),
            ),
            simulation=replace(
                cfg.simulation,
                trace_enabled=False,
                max_trace_records=0,
                max_detailed_trace_records=0,
            ),
        )
        validate_config(cfg)
        if optimized:
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
            result = Simulator(cfg).run()
        s = result.summary
        assert len(result.requests) == len(schedule["offsets_s"]), "undrained requests"
        assert (
            s["arrival_cohort"]["num_requests"]
            == measured["arrival_cohort"]["requests"]
        )
        target = out / point
        target.mkdir(parents=True, exist_ok=True)
        s.pop("scheduling_decisions", None)
        write(target / "summary.json", s)
        write(target / "resolved_config.json", asdict(cfg))
        write(target / "requests.json", result.requests)
        row = dict(
            point=point,
            target_rate=schedule["rate"],
            measured_arrivals=s["arrival_cohort"]["num_requests"],
            schedule_sha256=hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
        )
        values = {
            "qps": (
                measured["successful_completed_qps"],
                s["observed_request_throughput_qps"],
            ),
            "ttft_mean_ms": (
                measured["arrival_cohort"]["mean_scheduled_ttft_ms"],
                s["ttft_ms"]["mean"],
            ),
            "ttft_p99_ms": (
                measured["arrival_cohort"]["p99_scheduled_ttft_ms"],
                s["ttft_ms"]["p99"],
            ),
            "tpot_mean_ms": (
                measured["arrival_cohort"]["mean_tpot_ms"],
                s["tpot_ms"]["mean"],
            ),
            "tpot_p99_ms": (
                measured["arrival_cohort"]["p99_tpot_ms"],
                s["tpot_ms"]["p99"],
            ),
            "mean_inflight": (measured["mean_inflight"], s["mean_inflight"]),
        }
        for name, (real, pred) in values.items():
            row.update(
                {
                    name + "_measured": real,
                    name + "_simulated": pred,
                    name + "_error_pct": (pred / real - 1) * 100 if real else None,
                }
            )
        for cohort in ("arrival_cohort", "completion_cohort"):
            real = measured[cohort]["joint_slo"]
            pred = s[cohort]["slo"]["attainment"]
            row.update(
                {
                    cohort + "_slo_measured": real,
                    cohort + "_slo_simulated": pred,
                    cohort + "_slo_error_pp": (pred - real) * 100,
                }
            )
        row["measured_pass"] = all(
            measured[c]["joint_slo"] >= 0.99
            for c in ("arrival_cohort", "completion_cohort")
        )
        row["simulated_pass"] = s["slo"]["pass"]
        comparisons.append(row)
        write(out / "comparison.json", comparisons)
        with (out / "comparison.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(comparisons)
        print(json.dumps(row), flush=True)
    return comparisons


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measured-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--point", action="append", help="e.g. optimized/refine-4.440; repeatable"
    )
    a = p.parse_args()
    run(a.measured_root, a.output_dir, a.point)
