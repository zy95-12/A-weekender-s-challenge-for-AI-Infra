from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from split_serving_sim.config import parse_config
from tools.build_issue5_profile import convert
from tests.helpers import copied_toy_config


class Issue5ProfileConverterTest(unittest.TestCase):
    def test_uses_slowest_tp_rank_and_emits_exact_physical_signature(self) -> None:
        data = copied_toy_config()
        data["operator_backend"] = {"name": "vllm", "version": "test"}
        data["attention_backend"] = {"mode": "unified"}
        # Give every split the same TP degree, as in the issue #5 experiment.
        data["hardware"]["edge"]["count"] = 2
        for stage in data["topology"]["stages"]:
            stage["tp_degree"] = 2
        config = parse_config(data)

        fieldnames = [
            "role", "rank", "pid", "phase", "batch_id", "id", "operator",
            "items", "input_shapes", "input_dtypes", "inputs",
            "cpu_range_ns", "gpu_kernel_ns", "gpu_kernel_count",
        ]
        rows = []
        for rank, duration in ((0, 10_000), (1, 14_000)):
            rows.append(
                {
                    "role": "cloud",
                    "rank": str(rank),
                    "pid": str(rank + 1),
                    "phase": "prefill",
                    "batch_id": "batch",
                    "id": f"op:{rank}",
                    "operator": "aten.linear.default",
                    "items": '[{"request_id":"r","position":0,"query_len":16}]',
                    # TP2: local Q=32 and local KV=8+8 => fused output 48.
                    "input_shapes": "[[16,64],[48,64]]",
                    "input_dtypes": '["torch.bfloat16","torch.bfloat16"]',
                    "inputs": "[]",
                    "cpu_range_ns": "999999",
                    "gpu_kernel_ns": str(duration),
                    "gpu_kernel_count": "1",
                }
            )
            rows.append(
                {
                    "role": "cloud",
                    "rank": str(rank),
                    "pid": str(rank + 1),
                    "phase": "prefill",
                    "batch_id": "batch",
                    "id": f"nccl:{rank}",
                    "operator": "pynccl.all_reduce",
                    "items": '[{"request_id":"r","position":0,"query_len":16}]',
                    "input_shapes": "[[16,64]]",
                    "input_dtypes": '["torch.bfloat16"]',
                    "inputs": "[]",
                    "cpu_range_ns": "999999",
                    "gpu_kernel_ns": "100000" if rank == 0 else "50000",
                    "gpu_kernel_count": "1",
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "operators.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            kernel_path = Path(directory) / "kernel_instances.csv"
            kernel_fields = [
                "role", "pid", "kernel", "gpu_start_ns", "gpu_end_ns",
                "gpu_duration_ns", "device_id", "stream_id", "operator_id",
                "operator", "phase", "rank",
            ]
            with kernel_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=kernel_fields)
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "role": "cloud", "pid": "1", "kernel": "nccl",
                            "gpu_start_ns": "100000", "gpu_end_ns": "200000",
                            "gpu_duration_ns": "100000", "device_id": "0",
                            "stream_id": "1", "operator_id": "nccl:0",
                            "operator": "pynccl.all_reduce", "phase": "prefill",
                            "rank": "0",
                        },
                        {
                            "role": "cloud", "pid": "2", "kernel": "nccl",
                            "gpu_start_ns": "160000", "gpu_end_ns": "210000",
                            "gpu_duration_ns": "50000", "device_id": "1",
                            "stream_id": "1", "operator_id": "nccl:1",
                            "operator": "pynccl.all_reduce", "phase": "prefill",
                            "rank": "1",
                        },
                    ]
                )
            profile, report = convert(
                config, [path], label="unit", kernel_paths=[kernel_path]
            )

        self.assertEqual(report["rank_width_histogram"], {2: 2})
        self.assertEqual(report["call_coverage"], 1.0)
        self.assertEqual(len(profile["samples"]), 2)
        sample = next(
            value for value in profile["samples"] if value["operator_type"] == "mm"
        )
        self.assertEqual(sample["operator_type"], "mm")
        self.assertAlmostEqual(sample["latency_ms"], 0.014)
        self.assertIn('"op":"aten.linear"', sample["signature"])
        self.assertNotIn("phase", sample["match"])
        self.assertEqual(
            report["collective_reductions"], {"raw_timestamp_intrinsic": 1}
        )
        collective = next(
            value
            for value in profile["samples"]
            if value["operator_type"] == "collective"
        )
        self.assertAlmostEqual(collective["latency_ms"], 0.05)


if __name__ == "__main__":
    unittest.main()
