from __future__ import annotations

import unittest
from pathlib import Path

from split_serving_sim.config import ConfigError, load_config, parse_config

from tests.helpers import copied_toy_config


class ConfigTest(unittest.TestCase):
    def test_parses_issue6_stage123_controls(self) -> None:
        data = copied_toy_config()
        data["data_path"] = {
            "enabled": True,
            "ipc_mode": "shm",
            "wire_fast": True,
            "tcp_buffer_mib": 16,
        }
        data["execution"] = {
            "max_inflight_transactions": 2,
            "buffer_pool_mib": 1,
        }
        data["scheduler"] = {
            "decode_first": True,
            "max_consecutive_decode_batches": 3,
            "max_decode_tokens_per_batch": 2,
            "max_prefill_wait_ms": 5,
        }
        data["workload"].update(
            {
                "mode": "closed_loop", "num_requests": 3,
                "concurrency": 2, "warmup_requests": 1,
            }
        )
        config = parse_config(data)
        self.assertEqual(config.data_path.ipc_mode, "shm")
        self.assertTrue(config.data_path.wire_fast)
        self.assertEqual(config.execution.max_inflight_transactions, 2)
        self.assertTrue(config.scheduler.decode_first)
        self.assertEqual(config.workload.concurrency, 2)

    def test_parses_attention_scheduler_and_nested_continuous_batching(self) -> None:
        data = copied_toy_config()
        data["static_policy"]["continuous_batching"] = {"enabled": False}
        data["attention_backend"] = {"mode": "separate"}
        data["scheduler"] = {"policy": "priority", "max_num_seqs": 2}
        config = parse_config(data)
        self.assertFalse(config.static_policy.continuous_batching)
        self.assertEqual(config.attention_backend.mode, "separate")
        self.assertEqual(config.scheduler.policy, "priority")
    def test_valid_config(self) -> None:
        config = parse_config(copied_toy_config())
        self.assertEqual(config.model.name, "toy")
        self.assertEqual([stage.num_layers for stage in config.stages], [1, 2, 1])

    def test_parses_qwen2_synchronous_rpc_and_dual_tensor_network(self) -> None:
        data = copied_toy_config()
        data["model"].update(
            {"model_type": "qwen2", "attention_bias": True}
        )
        data["execution"] = {"mode": "synchronous_rpc"}
        data["network"].update(
            {
                "activation_tensor_count": 2,
                "protocol_overhead_bytes": 256,
                "sender_overhead_ms": 0.2,
                "receiver_overhead_ms": 0.3,
            }
        )
        data["scheduler"] = {
            "policy": "split_poc_naive",
            "enable_chunked_prefill": False,
        }

        config = parse_config(data)
        self.assertEqual(config.model.architecture, "qwen2")
        self.assertTrue(config.model.attention_bias)
        self.assertEqual(config.execution.mode, "synchronous_rpc")
        self.assertTrue(config.execution.preserve_batch_across_stages)
        self.assertEqual(config.network.activation_tensor_count, 2)

    def test_rejects_non_contiguous_split(self) -> None:
        data = copied_toy_config()
        data["topology"]["stages"][1]["layer_start"] = 2
        with self.assertRaisesRegex(ConfigError, "contiguous"):
            parse_config(data)

    def test_rejects_chunk_larger_than_token_budget(self) -> None:
        data = copied_toy_config()
        data["static_policy"]["prefill_chunk_size"] = 256
        with self.assertRaisesRegex(ConfigError, "prefill_chunk_size"):
            parse_config(data)

    def test_accepts_batch_size_input_name(self) -> None:
        data = copied_toy_config()
        data["static_policy"]["batch_size"] = data["static_policy"].pop(
            "max_batch_size"
        )
        config = parse_config(data)
        self.assertEqual(config.static_policy.max_batch_size, 4)

    def test_accepts_continuous_batch_alias(self) -> None:
        data = copied_toy_config()
        data["static_policy"]["continuous_batch"] = False
        config = parse_config(data)
        self.assertFalse(config.static_policy.continuous_batching)

    def test_loads_checked_in_hugging_face_profile(self) -> None:
        config = load_config(Path(__file__).parents[1] / "configs" / "example.json")
        self.assertEqual(config.model.architecture, "qwen3")
        self.assertEqual(config.model.num_layers, 64)
        self.assertEqual(config.model.hidden_size, 5120)
        self.assertEqual(config.model.intermediate_size, 25600)
        self.assertEqual(config.model.num_attention_heads, 64)
        self.assertEqual(config.model.num_key_value_heads, 8)
        self.assertEqual(config.model.dtype, "bfloat16")
        cloud = config.stage("cloud_middle")
        self.assertEqual(cloud.pp_degree, 2)
        self.assertEqual(cloud.pipeline_layer_range(0), (6, 34))
        self.assertEqual(cloud.pipeline_layer_range(1), (34, 62))
        total_parameters = (
            config.model.parameters_per_layer * config.model.num_layers
            + 2 * config.model.vocab_size * config.model.hidden_size
        )
        self.assertEqual(total_parameters, 32_762_118_144)

    def test_rejects_parallel_plan_larger_than_device_pool(self) -> None:
        data = copied_toy_config()
        data["topology"]["stages"][1]["pp_degree"] = 2
        with self.assertRaisesRegex(ConfigError, "TP times PP times replicas"):
            parse_config(data)

    def test_shared_resource_requires_one_parallel_layout(self) -> None:
        data = copied_toy_config()
        data["hardware"]["edge"]["count"] = 2
        data["topology"]["stages"][2]["replicas"] = 2
        with self.assertRaisesRegex(ConfigError, "same TP/PP/replica layout"):
            parse_config(data)


if __name__ == "__main__":
    unittest.main()
