from __future__ import annotations

import unittest

from split_serving_sim.config import parse_config
from split_serving_sim.simulator import Simulator

from tests.helpers import copied_toy_config


class SimulatorTest(unittest.TestCase):
    def test_simulation_produces_request_metrics_and_batches(self) -> None:
        result = Simulator(parse_config(copied_toy_config())).run()
        self.assertEqual(len(result.requests), 2)
        self.assertTrue(all(len(row["token_timestamps_ms"]) == 4 for row in result.requests))
        self.assertGreater(result.summary["ttft_ms"]["p99"], 0)
        self.assertGreater(result.summary["tpot_ms"]["p99"], 0)
        self.assertTrue(any(row["batch_size"] == 2 for row in result.trace))

    def test_trace_respects_every_dependency(self) -> None:
        result = Simulator(parse_config(copied_toy_config())).run()
        finish_by_item = {
            item_id: row["end_time_ms"]
            for row in result.trace
            for item_id in row["work_item_ids"]
        }
        for row in result.trace:
            for dependencies in row["dependency_ids"]:
                for dependency in dependencies:
                    self.assertLessEqual(finish_by_item[dependency], row["start_time_ms"])

    def test_edge_front_and_tail_share_one_resource(self) -> None:
        result = Simulator(parse_config(copied_toy_config())).run()
        edge = sorted(
            (row for row in result.trace if row["resource"] == "edge"),
            key=lambda row: row["start_time_ms"],
        )
        for previous, current in zip(edge, edge[1:]):
            self.assertLessEqual(previous["end_time_ms"], current["start_time_ms"])

    def test_single_output_token_has_no_tpot_but_can_pass_slo(self) -> None:
        data = copied_toy_config()
        data["workload"]["output_tokens"] = 1
        result = Simulator(parse_config(data)).run()
        self.assertIsNone(result.summary["tpot_ms"]["p99"])
        self.assertTrue(result.summary["slo"]["tpot_pass"])

    def test_stage_replicas_route_requests_to_sticky_resources(self) -> None:
        data = copied_toy_config()
        data["hardware"]["cloud"]["count"] = 4
        data["topology"]["stages"][1]["replicas"] = 2
        result = Simulator(parse_config(data)).run()
        cloud_rows = [
            row for row in result.trace if row["stage"] == "cloud_middle"
        ]
        self.assertEqual(
            {row["resource"] for row in cloud_rows},
            {"cloud/replica_0", "cloud/replica_1"},
        )
        for row in cloud_rows:
            for request_id in row["request_ids"]:
                self.assertTrue(row["resource"].endswith(str(request_id % 2)))


if __name__ == "__main__":
    unittest.main()
