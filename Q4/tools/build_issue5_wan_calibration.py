#!/usr/bin/env python3
"""Convert Issue #5 WAN probe curves into simulator interpolation knots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SOURCE_COMMIT = "366b71a4c516095d19993397f1410847c550cafb"
SOURCE_PATH = "docs/evidence/issue-5/wan-calibration/curves.csv"

UPLOAD_COLUMNS = (
    ("device_to_host", "enterprise_d2h_ms_mean"),
    ("host_pack", "enterprise_pack_ms_mean"),
    ("wan_path", "upload_path_ms_mean"),
    ("cloud_unpack", "cloud_unpack_ms_mean"),
    ("cloud_dispatch", "cloud_dispatch_ms_mean"),
    ("host_to_device", "cloud_h2d_tp_ms_mean"),
)
DOWNLOAD_COLUMNS = (
    ("device_to_host", "cloud_d2h_ms_mean"),
    ("cloud_ipc", "cloud_ipc_residual_ms_mean"),
    ("host_pack", "cloud_pack_ms_mean"),
    ("wan_path", "download_path_ms_mean"),
    ("host_unpack", "enterprise_unpack_ms_mean"),
    ("host_to_device", "enterprise_h2d_tp_ms_mean"),
)


def components(row: dict[str, str], columns: tuple[tuple[str, str], ...]) -> dict[str, float]:
    return {name: float(row[column]) for name, column in columns}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Issue #5 wan-calibration curves.csv")
    parser.add_argument("--output", type=Path, default=Path("profiles/issue5_wan_calibration.json"))
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    variants: dict[str, dict[str, object]] = {}
    for variant in sorted({row["variant"] for row in rows}):
        selected = sorted(
            (row for row in rows if row["variant"] == variant),
            key=lambda row: float(row["activation_mib_each_direction"]),
        )
        knots = []
        for row in selected:
            upload = components(row, UPLOAD_COLUMNS)
            download = components(row, DOWNLOAD_COLUMNS)
            reconstructed = sum(upload.values()) + sum(download.values())
            observed = float(row["gpu_ready_rtt_ms_mean"])
            if abs(reconstructed - observed) > 0.05:
                raise ValueError(
                    f"{variant} {row['activation_mib_each_direction']} MiB: "
                    f"components={reconstructed}ms but RTT={observed}ms"
                )
            knots.append(
                {
                    "activation_mib": float(row["activation_mib_each_direction"]),
                    "upload_components_ms": upload,
                    "download_components_ms": download,
                    "observed_gpu_ready_rtt_ms": observed,
                    "reconstructed_gpu_ready_rtt_ms": reconstructed,
                }
            )
        variants[variant] = {
            "one_way_latency_ms": 5.0,
            "knots": knots,
        }
    output = {
        "schema_version": 1,
        "source": (
            "https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/"
            f"blob/{SOURCE_COMMIT}/{SOURCE_PATH}"
        ),
        "scope": "C1 persistent HTTP, TP2+2, 10Gbps, 5ms one-way; no model compute",
        "interpolation": "piecewise linear in activation MiB; analytical fallback outside range",
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
