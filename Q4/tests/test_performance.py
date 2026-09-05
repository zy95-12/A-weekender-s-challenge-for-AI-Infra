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


if __name__ == "__main__":
    unittest.main()
