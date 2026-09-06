#!/usr/bin/env python3
"""Sweep closed-loop concurrency and compare simulated Issue #6 curves."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import resource
from dataclasses import replace
from pathlib import Path
from typing import Any

from split_serving_sim.config import load_config
from split_serving_sim.simulator import Simulator


METRICS = {
    "completed_qps": ("observed_request_throughput_qps", None),
    "mean_ttft_ms": ("ttft_ms", "mean"),
    "p99_ttft_ms": ("ttft_ms", "p99"),
    "mean_tpot_ms": ("tpot_ms", "mean"),
    "p99_tpot_ms": ("tpot_ms", "p99"),
}


def simulated_value(summary: dict[str, Any], field: str) -> float:
    outer, inner = METRICS[field]
    value = summary[outer] if inner is None else summary[outer][inner]
    return float(value)


def write_svg(points: list[dict[str, Any]], path: Path) -> None:
    variants = list(dict.fromkeys(point["variant"] for point in points))
    panel_specs = [
        (variant, label, mean_key, p99_key)
        for variant in variants
        for label, mean_key, p99_key in (
            ("TTFT", "mean_ttft_ms", "p99_ttft_ms"),
            ("TPOT", "mean_tpot_ms", "p99_tpot_ms"),
        )
    ]
    panels = [
        (*spec, 40 + index % 2 * 580, 55 + index // 2 * 350)
        for index, spec in enumerate(panel_specs)
    ]
    width, height = 1200, 60 + math.ceil(len(panels) / 2) * 350
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:system-ui,sans-serif;fill:#222}.axis{stroke:#444}.grid{stroke:#ddd}.actual{stroke:#2672d4}.sim{stroke:#d1495b}.mean{stroke-width:3;fill:none}.p99{stroke-width:2;fill:none;stroke-dasharray:7 5}.dot{stroke:none}</style>',
    ]
    for variant, label, mean_key, p99_key, left, top in panels:
        selected = [point for point in points if point["variant"] == variant]
        plot_w, plot_h = 520, 260
        all_x = [point[prefix + "completed_qps"] for point in selected for prefix in ("actual_", "sim_")]
        all_y = [point[prefix + key] for point in selected for prefix in ("actual_", "sim_") for key in (mean_key, p99_key)]
        xmax, ymax = max(all_x) * 1.08, max(all_y) * 1.08
        x = lambda value: left + 55 + value / xmax * (plot_w - 70)
        y = lambda value: top + plot_h - 35 - value / ymax * (plot_h - 60)
        parts.append(f'<text x="{left + 8}" y="{top + 18}" font-size="16" font-weight="600">{variant} {label} vs completed QPS</text>')
        parts.append(f'<line class="axis" x1="{left+55}" y1="{top+25}" x2="{left+55}" y2="{top+plot_h-35}"/>')
        parts.append(f'<line class="axis" x1="{left+55}" y1="{top+plot_h-35}" x2="{left+plot_w-15}" y2="{top+plot_h-35}"/>')
        for fraction in (0.25, 0.5, 0.75, 1.0):
            gy = y(ymax * fraction)
            parts.append(f'<line class="grid" x1="{left+55}" y1="{gy:.1f}" x2="{left+plot_w-15}" y2="{gy:.1f}"/>')
            parts.append(f'<text x="{left+50}" y="{gy+4:.1f}" text-anchor="end" font-size="10">{ymax*fraction:.0f}</text>')
            gx = x(xmax * fraction)
            parts.append(f'<text x="{gx:.1f}" y="{top+plot_h-18}" text-anchor="middle" font-size="10">{xmax*fraction:.2f}</text>')
        for prefix, css in (("actual_", "actual"), ("sim_", "sim")):
            for key, style in ((mean_key, "mean"), (p99_key, "p99")):
                coords = " ".join(f'{x(point[prefix+"completed_qps"]):.1f},{y(point[prefix+key]):.1f}' for point in selected)
                parts.append(f'<polyline class="{css} {style}" points="{coords}"/>')
        parts.append(f'<text x="{left+plot_w-220}" y="{top+18}" font-size="11" fill="#2672d4">blue=actual</text>')
        parts.append(f'<text x="{left+plot_w-145}" y="{top+18}" font-size="11" fill="#d1495b">red=sim</text>')
        parts.append(f'<text x="{left+plot_w-75}" y="{top+18}" font-size="11">solid=mean dash=P99</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/issue6_qps_sweep.json",
        help="curve experiment JSON; all relative paths are resolved from it",
    )
    args = parser.parse_args()
    experiment_path = Path(args.config).resolve()
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    base_dir = experiment_path.parent

    def resolved(value: str) -> Path:
        return (base_dir / value).resolve()

    memory_limit_mib = int(experiment.get("memory_limit_mib", 768))
    duration_s = float(experiment.get("measurement_duration_s", 60.0))
    request_cap = int(experiment.get("request_cap", 512))
    concurrency_levels = {int(value) for value in experiment["concurrency"]}
    max_active_by_concurrency = {
        int(key): int(value)
        for key, value in experiment["server_max_active"].items()
    }
    if memory_limit_mib:
        limit = memory_limit_mib * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    evidence_path = resolved(experiment["evidence"])
    evidence = [
        row
        for row in csv.DictReader(evidence_path.open(encoding="utf-8"))
        if int(row["concurrency"]) in concurrency_levels
    ]
    configs = {
        variant: load_config(resolved(path))
        for variant, path in experiment["variants"].items()
    }
    evidence = [row for row in evidence if row["variant"] in configs]
    if not evidence:
        raise ValueError("sweep config selected no evidence rows")
    points: list[dict[str, Any]] = []
    output_dir = resolved(experiment["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint.jsonl"
    checkpoint.write_text("", encoding="utf-8")
    for row in evidence:
        variant = row["variant"]
        concurrency = int(row["concurrency"])
        max_active = max_active_by_concurrency[concurrency]
        base = configs[variant]
        config = replace(
            base,
            workload=replace(
                base.workload,
                num_requests=request_cap,
                concurrency=concurrency,
                warmup_requests=concurrency,
                measurement_duration_s=duration_s,
            ),
            static_policy=replace(base.static_policy, max_batch_size=max_active),
            scheduler=replace(base.scheduler, max_num_seqs=max_active),
            simulation=replace(
                base.simulation,
                trace_enabled=False,
                max_trace_records=0,
                max_detailed_trace_records=0,
            ),
        )
        print(f"simulate {variant} C={concurrency}", flush=True)
        result = Simulator(config).run()
        point: dict[str, Any] = {
            "variant": variant,
            "concurrency": concurrency,
            "server_max_active": max_active,
            "sim_requests": result.summary["num_requests"],
            "measurement_duration_s": duration_s,
        }
        for field in METRICS:
            actual = float(row[field])
            simulated = simulated_value(result.summary, field)
            point[f"actual_{field}"] = actual
            point[f"sim_{field}"] = simulated
            point[f"error_pct_{field}"] = (simulated - actual) / actual * 100.0
        points.append(point)
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(point) + "\n")
        del result
        gc.collect()

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for variant in configs:
        selected = [point for point in points if point["variant"] == variant]
        aggregate[variant] = {}
        for field in METRICS:
            errors = [point[f"error_pct_{field}"] for point in selected]
            aggregate[variant][field] = {
                "mape_pct": sum(abs(value) for value in errors) / len(errors),
                "signed_bias_pct": sum(errors) / len(errors),
            }
    report = {
        "kind": "issue6_closed_loop_qps_curve_validation",
        "measurement": (
            f"{duration_s:g}s steady-state completion cohort; "
            "warmup and drain excluded"
        ),
        "experiment_config": str(Path(args.config)),
        "memory_limit_mib": memory_limit_mib,
        "request_cap_per_point": request_cap,
        "operator_calibration": "Issue #5 vLLM fused operator profiles",
        "wan_calibration": "Issue #5 piecewise directional component interpolation",
        "evidence_source": "https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/blob/f1c25cfd17c99131043e74afa51e634496562861/docs/evidence/issue-6/concurrency-4k-tp22/curves.csv",
        "points": points,
        "aggregate": aggregate,
        "limitations": [
            "WAN calibration is C1 and does not directly identify concurrent CPU/HTTP contention.",
            "Operator captures are C1; unmatched batched shapes use per-operator average correction.",
        ],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    columns = list(points[0])
    with (output_dir / "points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(points)
    write_svg(points, output_dir / "curves.svg")
    checkpoint.unlink()
    print(json.dumps(aggregate, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
