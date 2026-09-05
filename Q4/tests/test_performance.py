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


if __name__ == "__main__":
    unittest.main()
