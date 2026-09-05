from __future__ import annotations

import unittest

from split_serving_sim.config import parse_config
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
