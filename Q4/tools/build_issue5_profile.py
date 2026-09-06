#!/usr/bin/env python3
"""Convert issue #5 vLLM operator captures into Q4 calibration profiles.

The converter intentionally calibrates the physical vLLM DAG.  It takes the
critical rank (maximum correlated GPU-kernel duration) for an aligned operator
invocation; it never sums TP ranks or uses the CPU NVTX duration as GPU time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from split_serving_sim.config import (
    PerformanceProfileConfig,
    SimulationConfig,
    load_config,
)
from split_serving_sim.core import Phase, Stage, WorkItem
from split_serving_sim.performance import RooflineModel
from split_serving_sim.profiling import physical_signature


RAW_OPERATOR_MAP = {
    "aten.linear.default": ("mm", "aten.linear"),
    "aten.embedding.default": ("embedding", "aten.embedding"),
    "aten.zeros.default": ("tensor_allocation", "aten.zeros"),
    "_C.rms_norm.default": ("rms_norm", "vllm.rms_norm"),
    "_C.fused_add_rms_norm.default": (
        "fused_add_rms_norm",
        "vllm.fused_add_rms_norm",
    ),
    "_C.rotary_embedding.default": (
        "rotary_embedding",
        "vllm.rotary_embedding",
    ),
    "_C.silu_and_mul.default": ("silu_and_mul", "vllm.silu_and_mul"),
    "vllm.unified_attention_with_output.default": (
        "flash_attention",
        "vllm.unified_attention",
    ),
    "pynccl.all_reduce": ("collective", "pynccl.all_reduce"),
}

EVIDENCE_COMMIT = "670d3fc7d9d3ad042bef1933a37285eb47a5b3a1"
STAGE123_COMMIT = "b7d1b3c44d2cb817937a91cbe430853e9e9a5d17"


def _json(value: str) -> Any:
    return json.loads(value) if value else []


def _dtype(value: str) -> str:
    return value.removeprefix("torch.")


def normalized_items(row: dict[str, str]) -> list[dict[str, int]]:
    return [
        {
            "position": int(item["position"]),
            "query_len": int(item["query_len"]),
        }
        for item in _json(row["items"])
    ]


def raw_signature(
    row: dict[str, str], *, dtype: str, tp_degree: int
) -> tuple[str, str] | None:
    mapped = RAW_OPERATOR_MAP.get(row["operator"])
    if mapped is None:
        return None
    kind, physical_op = mapped
    inputs = [[int(value) for value in shape] for shape in _json(row["input_shapes"])]
    fields: dict[str, Any] = {
        "op": physical_op,
        "inputs": inputs,
        "dtype": dtype,
    }
    if kind == "embedding":
        dtypes = [_dtype(value) for value in _json(row["input_dtypes"])]
        fields["index_dtype"] = dtypes[1] if len(dtypes) > 1 else "int64"
    elif kind == "flash_attention":
        fields["items"] = normalized_items(row)
    elif kind == "collective":
        fields["tp_degree"] = tp_degree
    return kind, physical_signature(**fields)


def work_items(row: dict[str, str], phase: Phase, stage: Stage) -> list[WorkItem]:
    result = []
    for index, item in enumerate(normalized_items(row)):
        position, query_len = item["position"], item["query_len"]
        context = position + query_len if phase == Phase.PREFILL else position
        result.append(
            WorkItem(
                id=index,
                request_id=index,
                phase=phase,
                stage=stage,
                token_start=position,
                token_count=query_len,
                context_tokens=context,
                produces_logits=True,
            )
        )
    return result


def scenario_key(row: dict[str, str]) -> tuple[str, str]:
    return row["phase"], json.dumps(normalized_items(row), sort_keys=True)


def build_catalog(
    config: SimulationConfig,
    row: dict[str, str],
) -> dict[tuple[str, str], float]:
    """Return {(operator type, signature): analytical latency_ms}."""

    uncalibrated = replace(
        config,
        performance_profile=PerformanceProfileConfig(enabled=False),
    )
    model = RooflineModel(uncalibrated)
    phase = Phase(row["phase"])
    catalog: dict[tuple[str, str], float] = {}
    for stage in (Stage.EDGE_FRONT, Stage.CLOUD_MIDDLE, Stage.EDGE_TAIL):
        estimate = model.estimate(stage.value, work_items(row, phase, stage))
        for operation in estimate.sub_operations:
            if operation.profile_signature:
                catalog.setdefault(
                    (operation.profile_type, operation.profile_signature),
                    operation.duration_s * 1e3,
                )
    return catalog


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def file_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def load_collective_intervals(
    paths: Iterable[Path],
) -> dict[str, tuple[int, int]]:
    """Load correlated NCCL GPU intervals keyed by operator capture id."""

    intervals: dict[str, tuple[int, int]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["operator"] != "pynccl.all_reduce":
                    continue
                operator_id = row["operator_id"]
                start, end = int(row["gpu_start_ns"]), int(row["gpu_end_ns"])
                previous = intervals.get(operator_id)
                intervals[operator_id] = (
                    min(start, previous[0]) if previous else start,
                    max(end, previous[1]) if previous else end,
                )
    return intervals


def convert(
    config: SimulationConfig,
    paths: Iterable[Path],
    *,
    label: str,
    kernel_paths: Iterable[Path] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = tuple(paths)
    kernel_paths = tuple(kernel_paths)
    if config.operator_backend.name != "vllm":
        raise ValueError("profile conversion requires operator_backend.name=vllm")
    tp_degrees = {stage.tp_degree for stage in config.stages}
    if len(tp_degrees) != 1:
        raise ValueError("issue #5 conversion currently requires one TP degree")
    tp_degree = next(iter(tp_degrees))

    scenario_catalogs: dict[tuple[str, str], dict[tuple[str, str], float]] = {}
    occurrence_by_rank: Counter[tuple[Any, ...]] = Counter()
    collective_intervals = load_collective_intervals(kernel_paths)
    invocations: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    roofline_by_invocation: dict[tuple[Any, ...], float] = {}
    raw_calls: Counter[str] = Counter()
    mapped_calls: Counter[str] = Counter()
    unmatched_calls: Counter[str] = Counter()
    gpu_ns_total = 0
    gpu_ns_mapped = 0

    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                kernel_ns = int(row.get("gpu_kernel_ns") or 0)
                if kernel_ns <= 0:
                    continue
                gpu_ns_total += kernel_ns
                raw_calls[row["operator"]] += 1
                mapped = raw_signature(
                    row,
                    dtype=config.model.dtype,
                    tp_degree=tp_degree,
                )
                if mapped is None:
                    unmatched_calls[row["operator"]] += 1
                    continue
                key = scenario_key(row)
                catalog = scenario_catalogs.get(key)
                if catalog is None:
                    catalog = build_catalog(config, row)
                    scenario_catalogs[key] = catalog
                if mapped not in catalog:
                    unmatched_calls[f"{row['operator']} (signature)"] += 1
                    continue

                kind, signature = mapped
                rank_key = (
                    str(path), row["phase"], row["batch_id"], row["role"],
                    row["rank"], kind, signature,
                )
                ordinal = occurrence_by_rank[rank_key]
                occurrence_by_rank[rank_key] += 1
                invocation_key = (
                    str(path), row["phase"], row["batch_id"], row["role"],
                    kind, signature, ordinal,
                )
                interval = collective_intervals.get(row["id"])
                invocations[invocation_key].append(
                    {
                        "kernel_sum_ms": kernel_ns / 1e6,
                        "gpu_start_ns": interval[0] if interval else None,
                        "gpu_end_ns": interval[1] if interval else None,
                    }
                )
                roofline_by_invocation[invocation_key] = catalog[mapped]
                mapped_calls[row["operator"]] += 1
                gpu_ns_mapped += kernel_ns

    # Each invocation has one value per TP rank.  The stage cannot finish until
    # the slowest rank completes, so max is the correct non-summing reduction.
    grouped: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
    rank_widths: Counter[int] = Counter()
    collective_reductions = Counter()
    collective_before_ms = 0.0
    collective_after_ms = 0.0
    for key, rank_values in invocations.items():
        phase, kind, signature = key[1], key[4], key[5]
        kernel_sum_critical = max(value["kernel_sum_ms"] for value in rank_values)
        measured_ms = kernel_sum_critical
        if kind == "collective" and all(
            value["gpu_start_ns"] is not None for value in rank_values
        ):
            # The last rank arrival is when intrinsic collective progress can
            # begin. Completion is the last rank end. This removes time an
            # early rank spent waiting inside its NCCL kernel.
            ready_ns = max(value["gpu_start_ns"] for value in rank_values)
            finish_ns = max(value["gpu_end_ns"] for value in rank_values)
            intrinsic_ms = (finish_ns - ready_ns) / 1e6
            if intrinsic_ms >= 0:
                measured_ms = intrinsic_ms
                collective_reductions["raw_timestamp_intrinsic"] += 1
            else:
                collective_reductions["invalid_interval_fallback"] += 1
        elif kind == "collective":
            collective_reductions["kernel_sum_fallback"] += 1
        if kind == "collective":
            collective_before_ms += kernel_sum_critical
            collective_after_ms += measured_ms
        grouped[(phase, kind, signature)].append(
            (measured_ms, roofline_by_invocation[key])
        )
        rank_widths[len(rank_values)] += 1

    samples: list[dict[str, Any]] = []
    operator_stats: dict[str, dict[str, Any]] = {}
    operator_errors: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (phase, kind, signature), values in sorted(grouped.items()):
        measured = [value[0] for value in values]
        analytical = [value[1] for value in values]
        profiled = statistics.median(measured)
        samples.append(
            {
                "operator_type": kind,
                "signature": signature,
                "match": {
                    "tp_degree": tp_degree,
                    "dtype": config.model.dtype,
                    "operator_backend": "vllm",
                },
                "latency_ms": profiled,
                "roofline_ms": statistics.median(analytical),
                "source": f"issue5 {label} vLLM GPU kernels, TP critical rank",
                "captured_phase": phase,
                "sample_count": len(values),
                "p10_ms": percentile(measured, 0.10),
                "p90_ms": percentile(measured, 0.90),
            }
        )
        stats = operator_stats.setdefault(
            kind,
            {"signatures": 0, "invocations": 0, "measured_ms": 0.0, "roofline_ms": 0.0},
        )
        stats["signatures"] += 1
        stats["invocations"] += len(values)
        stats["measured_ms"] += sum(measured)
        stats["roofline_ms"] += sum(analytical)
        for actual, roofline in zip(measured, analytical):
            if actual > 0:
                operator_errors[kind]["roofline_ape"].append(
                    abs(roofline - actual) / actual
                )
                operator_errors[kind]["profile_ape"].append(
                    abs(profiled - actual) / actual
                )
                operator_errors[kind]["profile_signed"].append(
                    (profiled - actual) / actual
                )

    for kind, stats in operator_stats.items():
        roofline_ms = stats["roofline_ms"]
        stats["aggregate_correction_factor"] = (
            stats["measured_ms"] / roofline_ms if roofline_ms else 1.0
        )
        errors = operator_errors[kind]
        stats["roofline_mape_percent"] = (
            100.0 * statistics.mean(errors["roofline_ape"])
        )
        stats["exact_profile_mape_percent"] = (
            100.0 * statistics.mean(errors["profile_ape"])
        )
        stats["exact_profile_systematic_bias_percent"] = (
            100.0 * statistics.mean(errors["profile_signed"])
        )

    profile = {
        "version": 1,
        "enabled": True,
        "metadata": {
            "source_issue": 5,
            "source_evidence_commit": EVIDENCE_COMMIT,
            "stage123_implementation_commit": STAGE123_COMMIT,
            "dataset": label,
            "operator_backend": "vllm",
            "model": config.model.name,
            "dtype": config.model.dtype,
            "tp_degree": tp_degree,
            "reduction": "median across invocations after max across TP ranks",
            "collective_reduction": (
                "max(gpu_end_rank) - max(gpu_start_rank) after last-rank arrival"
                if collective_intervals
                else "critical rank correlated kernel sum (provisional)"
            ),
        },
        "samples": samples,
    }
    report = {
        "dataset": label,
        "input_files": [file_metadata(path) for path in paths],
        "kernel_files": [file_metadata(path) for path in kernel_paths],
        "raw_gpu_operator_calls": sum(raw_calls.values()),
        "mapped_gpu_operator_calls": sum(mapped_calls.values()),
        "call_coverage": (
            sum(mapped_calls.values()) / sum(raw_calls.values()) if raw_calls else 0.0
        ),
        "gpu_duration_coverage": gpu_ns_mapped / gpu_ns_total if gpu_ns_total else 0.0,
        "critical_invocations": len(invocations),
        "rank_width_histogram": dict(sorted(rank_widths.items())),
        "collective_reductions": dict(collective_reductions),
        "collective_kernel_sum_critical_ms": collective_before_ms,
        "collective_intrinsic_ms": collective_after_ms,
        "profile_samples": len(samples),
        "mapped_raw_operators": dict(mapped_calls.most_common()),
        "unmatched_raw_operators": dict(unmatched_calls.most_common()),
        "operator_stats": operator_stats,
        "limitations": [
            "CPU NVTX durations are not used as GPU latency.",
            "TP ranks are reduced by max, never summed.",
            (
                "NCCL arrival skew is removed using raw GPU timestamps."
                if collective_intervals
                else "The compact CSV lacks raw kernel timestamps, so NCCL arrival skew cannot yet be removed."
            ),
            "Framework-only broadcast/gather and request plumbing remain outside the layer cost model.",
        ],
    }
    return profile, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/issue6_stage123.json")
    parser.add_argument("--baseline-prefill", type=Path, required=True)
    parser.add_argument("--baseline-decode", type=Path, required=True)
    parser.add_argument("--stage123-prefill", type=Path, required=True)
    parser.add_argument("--stage123-decode", type=Path, required=True)
    parser.add_argument("--baseline-kernels", type=Path)
    parser.add_argument("--stage123-kernels", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("profiles"))
    args = parser.parse_args()

    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"version": 1, "datasets": {}}
    datasets = {
        "baseline": (
            (args.baseline_prefill, args.baseline_decode),
            (args.baseline_kernels,) if args.baseline_kernels else (),
        ),
        "stage123": (
            (args.stage123_prefill, args.stage123_decode),
            (args.stage123_kernels,) if args.stage123_kernels else (),
        ),
    }
    for label, (paths, kernel_paths) in datasets.items():
        profile, dataset_report = convert(
            config, paths, label=label, kernel_paths=kernel_paths
        )
        profile_path = args.output_dir / f"issue5_a10_tp2_{label}.json"
        profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        report["datasets"][label] = dataset_report
        print(f"wrote {len(profile['samples'])} samples to {profile_path}")

    report_path = args.output_dir / "issue5_mapping_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote mapping report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
