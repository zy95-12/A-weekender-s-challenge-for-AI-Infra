from __future__ import annotations

import unittest
from pathlib import Path

from split_serving_sim.config import load_config, parse_config
from split_serving_sim.core import Phase, Stage, WorkItem
from split_serving_sim.performance import NetworkModel, RooflineModel

from tests.helpers import copied_toy_config


def item(item_id: int, tokens: int = 16) -> WorkItem:
    return WorkItem(
        id=item_id,
        request_id=item_id,
        phase=Phase.PREFILL,
        stage=Stage.EDGE_FRONT,
        token_start=0,
        token_count=tokens,
        context_tokens=tokens,
    )


class PerformanceTest(unittest.TestCase):
    def test_wan_calibration_interpolates_components_without_double_counting_rtt(self) -> None:
        data = copied_toy_config()
        # One token carries 128 bytes in the toy model. These knots bracket
        # 8192 tokens == 1 MiB of activation payload.
        data["data_path"] = {
            "wan_calibration": {
                "enabled": True,
                "variant": "test",
                "one_way_latency_ms": 5,
                "knots": [
                    {
                        "activation_mib": 0.5,
                        "upload_components_ms": {"host_pack": 1, "wan_path": 7},
                        "download_components_ms": {"wan_path": 8, "host_unpack": 2},
                    },
                    {
                        "activation_mib": 1.5,
                        "upload_components_ms": {"host_pack": 3, "wan_path": 11},
                        "download_components_ms": {"wan_path": 12, "host_unpack": 4},
                    },
                ],
            }
        }
        estimate = NetworkModel(parse_config(data)).estimate(
            Stage.WAN_UP, [item(0, 8192)]
        )
        self.assertAlmostEqual(estimate.total_time_s * 1000, 11.0)
        self.assertEqual(
            [operation.name for operation in estimate.sub_operations],
            ["host_pack", "wan_serialization", "wan_propagation"],
        )
        self.assertAlmostEqual(estimate.sub_operations[-1].duration_s * 1000, 5.0)

    def test_vllm_backend_matches_captured_fusion_counts(self) -> None:
        config = load_config(
            Path(__file__).parents[1] / "configs" / "issue6_stage123.json"
        )
        model = RooflineModel(config)
        base = WorkItem(
            id=0,
            request_id=0,
            phase=Phase.PREFILL,
            stage=Stage.EDGE_FRONT,
            token_start=0,
            token_count=4096,
            context_tokens=4096,
            produces_logits=True,
        )
        estimates = {
            stage: model.estimate(
                stage,
                [WorkItem(**{**base.__dict__, "stage": Stage(stage)})],
            )
            for stage in ("edge_front", "cloud_middle", "edge_tail")
        }

        def count(stage: str, kind: str) -> int:
            return sum(
                operation.profile_type == kind
                for operation in estimates[stage].sub_operations
            )

        self.assertEqual(
            [count(stage, "mm") for stage in estimates], [16, 108, 21]
        )
        self.assertEqual(
            [count(stage, "collective") for stage in estimates], [9, 54, 10]
        )
        self.assertEqual(
            [count(stage, "fused_add_rms_norm") for stage in estimates],
            [7, 54, 11],
        )
        all_names = {
            operation.name
            for estimate in estimates.values()
            for operation in estimate.sub_operations
        }
        self.assertIn("layer_00.qkv_proj", all_names)
        self.assertNotIn("layer_00.q_proj", all_names)
        self.assertIn("layer_00.gate_up_proj", all_names)
        self.assertNotIn("layer_00.gate_proj", all_names)

    def test_stage1_data_path_switches_change_mechanical_costs(self) -> None:
        baseline_data = copied_toy_config()
        baseline_data["data_path"] = {
            "enabled": True,
            "ipc_mode": "copy",
            "wire_fast": False,
            "tcp_buffer_mib": 16,
        }
        optimized_data = copied_toy_config()
        optimized_data["data_path"] = {
            "enabled": True,
            "ipc_mode": "shm",
            "wire_fast": True,
            "tcp_buffer_mib": 16,
        }
        baseline = NetworkModel(parse_config(baseline_data)).estimate(
            Stage.WAN_UP, [item(0)]
        )
        optimized = NetworkModel(parse_config(optimized_data)).estimate(
            Stage.WAN_UP, [item(0)]
        )
        self.assertLess(optimized.total_time_s, baseline.total_time_s)
        names = [operation.name for operation in optimized.sub_operations]
        self.assertEqual(
            names,
            [
                "sender_staging", "device_to_host", "host_pack",
                "wan_serialization", "wan_propagation", "cloud_ipc",
                "host_unpack", "host_to_device", "receiver_staging",
            ],
        )

        small_window_data = copied_toy_config()
        small_window_data["data_path"] = {
            "enabled": True,
            "tcp_buffer_mib": 0.001,
        }
        small_window = NetworkModel(parse_config(small_window_data)).estimate(
            Stage.WAN_UP, [item(0)]
        )
        self.assertGreater(small_window.total_time_s, baseline.total_time_s)

    def test_qwen2_omits_qk_norm_and_network_can_send_hidden_and_residual(self) -> None:
        data = copied_toy_config()
        data["model"]["model_type"] = "qwen2"
        data["network"].update(
            {
                "activation_tensor_count": 2,
                "protocol_overhead_bytes": 128,
                "sender_overhead_ms": 0.1,
                "receiver_overhead_ms": 0.2,
            }
        )
        config = parse_config(data)
        work_item = item(0, 16)
        estimate = RooflineModel(config).estimate("cloud_middle", [work_item])
        names = {operation.name for operation in estimate.sub_operations}
        self.assertFalse(any(name.endswith((".q_norm", ".k_norm")) for name in names))

        network = NetworkModel(config).estimate(Stage.WAN_UP, [work_item])
        tensor_bytes = 16 * config.model.hidden_size * config.model.dtype_bytes * 2
        self.assertEqual(network.communication_bytes, tensor_bytes + 128)
        self.assertEqual(
            [operation.name for operation in network.sub_operations],
            ["sender_staging", "wan_serialization", "wan_propagation", "receiver_staging"],
        )

    def test_mixed_attention_backend_can_be_unified_or_separate(self) -> None:
        mixed = [item(0), WorkItem(id=99, request_id=99, phase=Phase.DECODE, stage=Stage.EDGE_FRONT, token_start=32, token_count=1, context_tokens=32)]
        unified = RooflineModel(parse_config(copied_toy_config())).estimate("edge_front", mixed)
        self.assertTrue(any(op.name.endswith(".mix attention") for op in unified.sub_operations))
        data = copied_toy_config()
        data["attention_backend"] = {"mode": "separate"}
        separate = RooflineModel(parse_config(data)).estimate("edge_front", mixed)
        names = [op.name for op in separate.sub_operations]
        self.assertTrue(any(name.endswith(".prefill attention") for name in names))
        self.assertTrue(any(name.endswith(".decode attention") for name in names))
        self.assertGreater(separate.total_time_s, unified.total_time_s)
    def test_roofline_is_batch_aware_and_reuses_weights(self) -> None:
        model = RooflineModel(parse_config(copied_toy_config()))
        single = model.estimate("edge_front", [item(0)])
        batch = model.estimate("edge_front", [item(0), item(1)])
        self.assertGreater(batch.total_time_s, single.total_time_s)
        self.assertLess(batch.memory_bytes, 2 * single.memory_bytes)

    def test_network_batches_payload_but_pays_one_propagation_delay(self) -> None:
        model = NetworkModel(parse_config(copied_toy_config()))
        single = model.estimate(Stage.WAN_UP, [item(0)])
        batch = model.estimate(Stage.WAN_UP, [item(0), item(1)])
        self.assertEqual(batch.communication_bytes, 2 * single.communication_bytes)
        self.assertLess(batch.total_time_s, 2 * single.total_time_s)

    def test_gpu_trace_has_operator_and_tp_communication_subflows(self) -> None:
        model = RooflineModel(parse_config(copied_toy_config()))
        estimate = model.estimate("cloud_middle", [item(0), item(1)])
        names = [operation.name for operation in estimate.sub_operations]
        self.assertIn("layer_01.q_proj", names)
        self.assertIn("layer_02.attention", names)
        self.assertIn("layer_01.attention_all_reduce", names)
        self.assertIn("layer_02.mlp_all_reduce", names)
        self.assertEqual(
            sum(name.endswith("all_reduce") for name in names),
            2 * 2,
        )
        self.assertEqual(estimate.input_shape, "bfloat16 [32, 64]")
        self.assertNotIn("context", estimate.input_shape)

    def test_qwen3_gqa_uses_local_query_and_kv_heads(self) -> None:
        model = RooflineModel(parse_config(copied_toy_config()))
        estimate = model.estimate("cloud_middle", [item(0)])
        attention = next(
            operation
            for operation in estimate.sub_operations
            if operation.name == "layer_01.attention"
        )
        self.assertIn("Q[16, 4, 8]", attention.input_shape)
        self.assertIn("KV[B, 1, L_i, 8]", attention.input_shape)
        self.assertEqual(
            attention.dependencies,
            ("layer_01.rope", "layer_01.kv_cache_update"),
        )

    def test_lm_head_only_runs_for_logit_producing_items(self) -> None:
        model = RooflineModel(parse_config(copied_toy_config()))
        ordinary = item(0)
        logit_item = WorkItem(**{**ordinary.__dict__, "produces_logits": True})
        no_logits = model.estimate("edge_tail", [ordinary])
        with_logits = model.estimate("edge_tail", [logit_item])
        self.assertNotIn("lm_head", {op.name for op in no_logits.sub_operations})
        self.assertIn("lm_head", {op.name for op in with_logits.sub_operations})

    def test_pipeline_rank_only_costs_its_layers_and_sends_activation(self) -> None:
        data = copied_toy_config()
        data["hardware"]["cloud"]["count"] = 4
        data["topology"]["stages"][1]["pp_degree"] = 2
        model = RooflineModel(parse_config(data))
        first_item = WorkItem(**{**item(0).__dict__, "pipeline_rank": 0})
        second_item = WorkItem(**{**item(0).__dict__, "pipeline_rank": 1})
        first = model.estimate("cloud_middle", [first_item])
        second = model.estimate("cloud_middle", [second_item])
        first_names = {op.name for op in first.sub_operations}
        second_names = {op.name for op in second.sub_operations}
        self.assertIn("layer_01.q_proj", first_names)
        self.assertNotIn("layer_02.q_proj", first_names)
        self.assertIn("pp_0_send", first_names)
        self.assertIn("layer_02.q_proj", second_names)
        self.assertNotIn("pp_0_send", second_names)


if __name__ == "__main__":
    unittest.main()
