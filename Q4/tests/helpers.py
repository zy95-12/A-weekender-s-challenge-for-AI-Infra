from __future__ import annotations

from copy import deepcopy
from typing import Any


def toy_config() -> dict[str, Any]:
    return {
        "model": {
            "name": "toy",
            "num_layers": 4,
            "hidden_size": 64,
            "intermediate_size": 256,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 128,
            "dtype": "bfloat16",
            "dtype_bytes": 2,
        },
        "topology": {
            "stages": [
                {
                    "name": "edge_front",
                    "layer_start": 0,
                    "layer_end": 1,
                    "resource": "edge",
                    "tp_degree": 1,
                },
                {
                    "name": "cloud_middle",
                    "layer_start": 1,
                    "layer_end": 3,
                    "resource": "cloud",
                    "tp_degree": 2,
                },
                {
                    "name": "edge_tail",
                    "layer_start": 3,
                    "layer_end": 4,
                    "resource": "edge",
                    "tp_degree": 1,
                },
            ]
        },
        "hardware": {
            "edge": {
                "count": 1,
                "peak_flops_tflops": 1,
                "hbm_bandwidth_gb_s": 100,
                "memory_gb": 8,
            },
            "cloud": {
                "count": 2,
                "peak_flops_tflops": 1,
                "hbm_bandwidth_gb_s": 100,
                "memory_gb": 8,
            },
        },
        "network": {
            "uplink_gbps": 10,
            "downlink_gbps": 10,
            "rtt_ms": 10,
        },
        "static_policy": {
            "max_batch_size": 4,
            "max_batched_tokens": 128,
            "prefill_token_budget": 32,
            "prefill_chunk_size": 16,
            "pipeline_depth": 2,
            "max_outstanding_batches": 8,
        },
        "workload": {
            "mode": "synthetic",
            "num_requests": 2,
            "arrival_interval_ms": 0,
            "input_tokens": 32,
            "output_tokens": 4,
        },
        "slo": {"ttft_ms": 1000, "tpot_ms": 100},
    }


def copied_toy_config() -> dict[str, Any]:
    return deepcopy(toy_config())
