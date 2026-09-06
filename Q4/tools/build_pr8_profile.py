#!/usr/bin/env python3
"""Build a Q4 performance profile from the compact artifacts in PR #8."""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from split_serving_sim.config import load_config
from split_serving_sim.core import Phase, Stage, WorkItem
from split_serving_sim.performance import NetworkModel


def git_text(repo: Path, ref: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{ref}:{path}"], cwd=repo, text=True
    )


def network_item(phase: Phase, stage: Stage, batch: int, isl: int) -> list[WorkItem]:
    tokens = isl if phase == Phase.PREFILL else 1
    return [
        WorkItem(index, index, phase, stage, 0, tokens, isl)
        for index in range(batch)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="..")
    parser.add_argument("--pr8-ref", default="origin/pr8-review")
    parser.add_argument("--config", default="configs/pr8_split_1_3.json")
    parser.add_argument("--accuracy", default="outputs/validation/pr8_accuracy.json")
    parser.add_argument("--output", default="profiles/pr8_a10.json")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    accuracy = json.loads(Path(args.accuracy).read_text(encoding="utf-8"))
    samples: list[dict] = []
    for topology, tp in (("tp_1_1", 1), ("tp_2_2", 2)):
        measured = accuracy["operator"][topology]
        for source_name, kind, phase in (
            ("fa_prefill", "flash_attention", "prefill"),
            ("fa_decode", "flash_attention", "decode"),
            ("mm", "mm", None),
            ("collective", "collective", None),
        ):
            value = measured[source_name]
            if not value["measured_ms"]:
                continue
            match = {"tp_degree": tp}
            if phase:
                match["phase"] = phase
            samples.append(
                {
                    "operator_type": kind,
                    "match": match,
                    "latency_ms": value["measured_ms"],
                    "roofline_ms": value["predicted_ms"],
                    "source": f"PR8 {topology} aggregate CUDA kernels",
                }
            )

    config = load_config(args.config)
    network = NetworkModel(config)
    rows = csv.DictReader(
        io.StringIO(
            git_text(
                repo,
                args.pr8_ref,
                "results/baseline_20260905/profile_summary.csv",
            )
        )
    )
    grouped: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        phase = Phase(row["phase"])
        batch, isl = int(row["actual_batch_size"]), int(row["isl"])
        for direction, stage, column in (
            ("up", Stage.WAN_UP, "upload_ms_mean"),
            ("down", Stage.WAN_DOWN, "download_ms_mean"),
        ):
            estimate = network.estimate(
                stage, network_item(phase, stage, batch, isl)
            )
            operation = estimate.sub_operations[0]
            grouped[(direction, phase.value, operation.profile_signature)].append(
                (float(row[column]), estimate.total_time_s * 1e3)
            )
    for (direction, phase, signature), values in sorted(grouped.items()):
        samples.append(
            {
                "operator_type": "network_transfer",
                "signature": signature,
                "match": {"direction": direction, "phase": phase},
                "latency_ms": sum(value[0] for value in values) / len(values),
                "roofline_ms": sum(value[1] for value in values) / len(values),
                "source": "PR8 profile_summary exact payload shape mean",
            }
        )

    commit = subprocess.check_output(
        ["git", "rev-parse", args.pr8_ref], cwd=repo, text=True
    ).strip()
    output = {
        "version": 1,
        "enabled": True,
        "metadata": {
            "source_pr": 8,
            "source_commit": commit,
            "note": "Operator entries are aggregate correction samples; network entries are exact payload-shape samples.",
        },
        "samples": samples,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(samples)} samples to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
