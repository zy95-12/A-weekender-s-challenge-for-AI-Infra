#!/usr/bin/env python3
"""Compare Q4 analytical predictions with the measurements in PR #8.

The parent process never runs a full matrix point itself.  Every point runs in
an isolated child with an address-space limit and tracing disabled, so a bad
case cannot consume all RAM on the small demo host.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import statistics
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any


WORKLOADS = ((512, 128), (2048, 256), (8192, 256), (2048, 1024))
TOPOLOGIES = ((1, 1), (1, 2), (2, 1), (2, 2))


def git_text(repo: Path, ref: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{ref}:{path}"], cwd=repo, text=True
    )


def configure(base: Any, edge_tp: int, cloud_tp: int) -> Any:
    stages = tuple(
        replace(
            stage,
            tp_degree=cloud_tp if stage.name == "cloud_middle" else edge_tp,
        )
        for stage in base.stages
    )
    return replace(base, stages=stages)


def worker(args: argparse.Namespace) -> int:
    import resource

    limit = args.memory_limit_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    from split_serving_sim.config import load_config, load_performance_profile
    from split_serving_sim.simulator import Simulator

    base = load_config(args.config)
    if args.profile:
        base = replace(
            base, performance_profile=load_performance_profile(args.profile)
        )
    base = configure(base, args.edge_tp, args.cloud_tp)
    workload = replace(
        base.workload,
        num_requests=args.concurrency,
        arrival_interval_ms=0.0,
        input_tokens=args.isl,
        output_tokens=args.osl,
    )
    policy = replace(
        base.static_policy,
        max_batch_size=max(8, args.concurrency),
        max_batched_tokens=max(16_384, args.isl),
        prefill_token_budget=max(16_384, args.isl),
        prefill_chunk_size=max(4_096, args.isl),
    )
    simulation = replace(base.simulation, trace_enabled=False)
    result = Simulator(
        replace(base, workload=workload, static_policy=policy, simulation=simulation)
    ).run()
    payload = {
        "ttft_mean_ms": statistics.mean(row["ttft_ms"] for row in result.requests),
        "tpot_mean_ms": statistics.mean(
            row["mean_tpot_ms"] for row in result.requests
        ),
        "qps": result.summary["observed_request_throughput_qps"],
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    print(json.dumps(payload, separators=(",", ":")))
    return 0


def work_items(stage: Any, phase: Any, batch_size: int, tokens: int, context: int):
    from split_serving_sim.core import WorkItem

    return [
        WorkItem(
            id=index,
            request_id=index,
            phase=phase,
            stage=stage,
            token_start=0,
            token_count=tokens,
            context_tokens=context,
            produces_logits=stage.value == "edge_tail",
        )
        for index in range(batch_size)
    ]


def estimate_stage(model: Any, stage: Any, phase: Any, batch: int, isl: int, osl: int):
    tokens = isl if phase.value == "prefill" else 1
    context = isl if phase.value == "prefill" else int(isl + osl / 2)
    return model.estimate(
        stage.value, work_items(stage, phase, batch, tokens, context)
    )


def error_summary(values: list[tuple[float, float]]) -> dict[str, float]:
    signed = [100.0 * (predicted / actual - 1.0) for actual, predicted in values]
    return {
        "points": len(values),
        "mape_pct": statistics.mean(abs(value) for value in signed),
        "median_ape_pct": statistics.median(abs(value) for value in signed),
        "mean_signed_error_pct": statistics.mean(signed),
        "aggregate_error_pct": 100.0
        * (sum(predicted for _, predicted in values) / sum(actual for actual, _ in values) - 1.0),
    }


def pearson(values: list[tuple[float, float]]) -> float:
    actual = [value[0] for value in values]
    predicted = [value[1] for value in values]
    actual_mean, predicted_mean = statistics.mean(actual), statistics.mean(predicted)
    numerator = sum(
        (left - actual_mean) * (right - predicted_mean)
        for left, right in values
    )
    denominator = (
        sum((value - actual_mean) ** 2 for value in actual)
        * sum((value - predicted_mean) ** 2 for value in predicted)
    ) ** 0.5
    return numerator / denominator


def end_to_end_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in ("ttft_mean_ms", "tpot_mean_ms", "qps"):
        values = [
            (row[f"actual_{metric}"], row[f"sim_{metric}"]) for row in rows
        ]
        result[metric] = {**error_summary(values), "pearson": pearson(values)}
    return result


def classify_kernel(name: str) -> str | None:
    lowered = name.lower()
    if "nccldevkernel_allreduce" in lowered:
        return "collective"
    # Check split-KV first: its C++ template parameters also contain the text
    # ``Flash_fwd_kernel_traits`` used by the prefill kernel.
    if "flash_fwd_splitkv" in lowered:
        return "fa_decode"
    if "flash_fwd_kernel" in lowered:
        return "fa_prefill"
    if any(token in lowered for token in ("gemm", "gemv", "cutlass::kernel2", "splitkreduce")):
        return "mm"
    return None


def classify_modeled_operator(name: str) -> str | None:
    if name.endswith("_all_reduce"):
        return "collective"
    if name.endswith(".attention"):
        return "fa"
    if name == "lm_head" or name.rsplit(".", 1)[-1] in {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    }:
        return "mm"
    return None


def operator_accuracy(repo: Path, ref: str, base: Any) -> dict[str, Any]:
    from split_serving_sim.core import Phase, Stage
    from split_serving_sim.performance import RooflineModel

    results: dict[str, Any] = {}
    for edge_tp, cloud_tp in TOPOLOGIES:
        topology = f"tp_{edge_tp}_{cloud_tp}"
        config = configure(base, edge_tp, cloud_tp)
        model = RooflineModel(config)
        measured = {
            "fa_prefill": 0.0, "fa_decode": 0.0, "mm": 0.0,
            "collective": 0.0,
        }
        for side, tp in (("enterprise", edge_tp), ("cloud", cloud_tp)):
            text = git_text(
                repo,
                ref,
                f"results/baseline_20260905/{topology}/profiling/{side}_cuda_gpu_kern_sum.csv",
            )
            for row in csv.DictReader(io.StringIO(text)):
                category = classify_kernel(row["Name"])
                if category:
                    # Nsight summary adds both TP ranks; divide by TP to estimate
                    # the per-rank critical-path work represented by Roofline.
                    measured[category] += float(row["Total Time (ns)"]) / 1e6 / tp

        predicted = {
            "fa_prefill": 0.0, "fa_decode": 0.0, "mm": 0.0,
            "collective": 0.0,
        }
        for isl, osl in WORKLOADS:
            for stage in (Stage.EDGE_FRONT, Stage.CLOUD_MIDDLE, Stage.EDGE_TAIL):
                prefill = estimate_stage(model, stage, Phase.PREFILL, 1, isl, osl)
                for operation in prefill.sub_operations:
                    category = classify_modeled_operator(operation.name)
                    if category == "fa":
                        predicted["fa_prefill"] += operation.duration_s * 1e3 * 5
                    elif category == "mm":
                        predicted["mm"] += operation.duration_s * 1e3 * 5
                    elif category == "collective":
                        predicted["collective"] += operation.duration_s * 1e3 * 5
                for batch_size in (1, 4):
                    decode = estimate_stage(
                        model, stage, Phase.DECODE, batch_size, isl, osl
                    )
                    multiplier = osl - 1
                    for operation in decode.sub_operations:
                        category = classify_modeled_operator(operation.name)
                        if category == "fa":
                            predicted["fa_decode"] += (
                                operation.duration_s * 1e3 * multiplier
                            )
                        elif category == "mm":
                            predicted["mm"] += operation.duration_s * 1e3 * multiplier
                        elif category == "collective":
                            predicted["collective"] += (
                                operation.duration_s * 1e3 * multiplier
                            )
        results[topology] = {
            key: {
                "measured_ms": measured[key],
                "predicted_ms": predicted[key],
                "error_pct": (
                    100.0 * (predicted[key] / measured[key] - 1.0)
                    if measured[key] else None
                ),
            }
            for key in measured
        }
    return results


def stage_accuracy(profile_rows: list[dict[str, str]], base: Any) -> dict[str, Any]:
    from split_serving_sim.core import Phase, Stage
    from split_serving_sim.performance import NetworkModel, RooflineModel

    groups: dict[str, list[tuple[float, float]]] = {
        "front": [], "cloud": [], "back": [], "prefill_gpu": [], "decode_gpu": [],
        "upload": [], "download": [], "prefill_step": [], "decode_step": [],
        "prefill_front": [], "prefill_cloud": [], "prefill_back": [],
        "decode_front": [], "decode_cloud": [], "decode_back": [],
        "prefill_upload": [], "prefill_download": [],
        "decode_upload": [], "decode_download": [],
    }
    for row in profile_rows:
        edge_tp, cloud_tp = map(int, row["topology"].removeprefix("tp_").split("_"))
        config = configure(base, edge_tp, cloud_tp)
        roofline, network = RooflineModel(config), NetworkModel(config)
        phase = Phase(row["phase"])
        isl, osl, batch = int(row["isl"]), int(row["osl"]), int(row["actual_batch_size"])
        predicted_gpu: dict[str, float] = {}
        for name, stage, column in (
            ("front", Stage.EDGE_FRONT, f"enterprise_front_{phase.value}_ms_mean"),
            ("cloud", Stage.CLOUD_MIDDLE, f"cloud_middle_{phase.value}_ms_mean"),
            ("back", Stage.EDGE_TAIL, f"enterprise_back_{phase.value}_ms_mean"),
        ):
            actual_text = row[column]
            if not actual_text:
                continue
            predicted = estimate_stage(roofline, stage, phase, batch, isl, osl).total_time_s * 1e3
            actual = float(actual_text)
            groups[name].append((actual, predicted))
            groups[f"{phase.value}_{name}"].append((actual, predicted))
            groups[f"{phase.value}_gpu"].append((actual, predicted))
            predicted_gpu[name] = predicted
        tokens = isl if phase == Phase.PREFILL else 1
        context = isl if phase == Phase.PREFILL else int(isl + osl / 2)
        items = work_items(Stage.WAN_UP, phase, batch, tokens, context)
        upload = network.estimate(Stage.WAN_UP, items).total_time_s * 1e3
        download = network.estimate(Stage.WAN_DOWN, items).total_time_s * 1e3
        groups["upload"].append((float(row["upload_ms_mean"]), upload))
        groups["download"].append((float(row["download_ms_mean"]), download))
        groups[f"{phase.value}_upload"].append(
            (float(row["upload_ms_mean"]), upload)
        )
        groups[f"{phase.value}_download"].append(
            (float(row["download_ms_mean"]), download)
        )
        predicted_step = sum(predicted_gpu.values()) + upload + download
        groups[f"{phase.value}_step"].append((float(row["step_wall_ms_mean"]), predicted_step))
    return {name: error_summary(values) for name, values in groups.items()}


def run_parent(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    config = Path(args.config).resolve()
    pr8_commit = subprocess.check_output(
        ["git", "rev-parse", args.pr8_ref], cwd=repo, text=True
    ).strip()
    fingerprint = hashlib.sha256(
        config.read_bytes()
        + (Path(args.profile).read_bytes() if args.profile else b"")
        + pr8_commit.encode("ascii")
    ).hexdigest()
    baseline = list(
        csv.DictReader(
            io.StringIO(
                git_text(repo, args.pr8_ref, "results/baseline_20260905/baseline_summary.csv")
            )
        )
    )
    profile = list(
        csv.DictReader(
            io.StringIO(
                git_text(repo, args.pr8_ref, "results/baseline_20260905/profile_summary.csv")
            )
        )
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output.with_suffix(".matrix.jsonl")
    completed: dict[tuple[int, int, int, int, int], dict[str, Any]] = {}
    if checkpoint.exists() and not args.no_resume:
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item.get("fingerprint") != fingerprint:
                continue
            key = tuple(item[name] for name in (
                "enterprise_tp", "cloud_tp", "isl", "osl", "concurrency"
            ))
            completed[key] = item
    matrix: list[dict[str, Any]] = []
    for row in baseline:
        key = tuple(int(row[name]) for name in (
            "enterprise_tp", "cloud_tp", "isl", "osl", "client_concurrency"
        ))
        if key in completed:
            matrix.append(completed[key])
            continue
        command = [
            sys.executable, str(Path(__file__).resolve()), "--worker", "--config", str(config),
            "--edge-tp", row["enterprise_tp"], "--cloud-tp", row["cloud_tp"],
            "--isl", row["isl"], "--osl", row["osl"],
            "--concurrency", row["client_concurrency"],
            "--memory-limit-mb", str(args.memory_limit_mb),
        ]
        if args.profile:
            command.extend(("--profile", str(Path(args.profile).resolve())))
        predicted = json.loads(
            subprocess.check_output(command, text=True, timeout=args.point_timeout_s)
        )
        item: dict[str, Any] = {
            "fingerprint": fingerprint,
            "enterprise_tp": int(row["enterprise_tp"]),
            "cloud_tp": int(row["cloud_tp"]),
            "isl": int(row["isl"]), "osl": int(row["osl"]),
            "concurrency": int(row["client_concurrency"]),
            "peak_rss_mb": predicted.pop("peak_rss_kb") / 1024.0,
        }
        for metric, actual_column in (
            ("ttft_mean_ms", "ttft_mean_ms"),
            ("tpot_mean_ms", "tpot_mean_ms"),
            ("qps", "qps_repeat_mean"),
        ):
            actual = float(row[actual_column]); value = float(predicted[metric])
            item[f"actual_{metric}"] = actual
            item[f"sim_{metric}"] = value
            item[f"error_{metric}_pct"] = 100.0 * (value / actual - 1.0)
        matrix.append(item)
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, separators=(",", ":")) + "\n")

    from split_serving_sim.config import load_config, load_performance_profile

    base = load_config(config)
    if args.profile:
        base = replace(
            base, performance_profile=load_performance_profile(args.profile)
        )
    end_to_end = end_to_end_summary(matrix)
    by_topology = {
        f"tp_{edge_tp}_{cloud_tp}": end_to_end_summary(
            [
                row for row in matrix
                if (row["enterprise_tp"], row["cloud_tp"]) == (edge_tp, cloud_tp)
            ]
        )
        for edge_tp, cloud_tp in TOPOLOGIES
    }
    report = {
        "source": {
            "pr8_ref": args.pr8_ref,
            "pr8_commit": pr8_commit,
            "simulator_config": args.config,
            "performance_profile": args.profile,
            "calibrated_on_baseline": bool(args.profile),
        },
        "memory": {
            "child_limit_mb": args.memory_limit_mb,
            "max_observed_peak_rss_mb": max(row["peak_rss_mb"] for row in matrix),
        },
        "end_to_end": end_to_end,
        "end_to_end_by_topology": by_topology,
        "stage": stage_accuracy(profile, base),
        "operator": operator_accuracy(repo, args.pr8_ref, base),
        "matrix": matrix,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "matrix"}, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--config", default="configs/pr8_split_1_3.json")
    result.add_argument("--repo", default="..")
    result.add_argument("--pr8-ref", default="origin/pr8-review")
    result.add_argument("--output", default="outputs/validation/pr8_accuracy.json")
    result.add_argument("--memory-limit-mb", type=int, default=768)
    result.add_argument("--profile")
    result.add_argument("--point-timeout-s", type=int, default=120)
    result.add_argument("--no-resume", action="store_true")
    result.add_argument("--worker", action="store_true")
    result.add_argument("--edge-tp", type=int)
    result.add_argument("--cloud-tp", type=int)
    result.add_argument("--isl", type=int)
    result.add_argument("--osl", type=int)
    result.add_argument("--concurrency", type=int)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    raise SystemExit(worker(arguments) if arguments.worker else run_parent(arguments))
