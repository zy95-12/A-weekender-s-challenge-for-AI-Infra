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
        self.assertEqual(result.summary["static_policy"]["scheduler"], "fcfs")
        self.assertTrue(result.summary["static_policy"]["continuous_batching"])
        self.assertEqual(result.summary["parallel_plan"][1]["tp_degree"], 2)

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

    def test_pipeline_parallel_ranks_are_independent_resources(self) -> None:
        data = copied_toy_config()
        data["hardware"]["cloud"]["count"] = 4
        data["topology"]["stages"][1]["pp_degree"] = 2
        result = Simulator(parse_config(data)).run()
        cloud_rows = [row for row in result.trace if row["stage"] == "cloud_middle"]
        self.assertEqual(
            {row["resource"] for row in cloud_rows},
            {"cloud/pp_0", "cloud/pp_1"},
        )
        self.assertEqual(
            {tuple(row["layer_range"]) for row in cloud_rows},
            {(1, 2), (2, 3)},
        )

    def test_edge_pipeline_parallelism_respects_prefill_backpressure(self) -> None:
        data = copied_toy_config()
        data["model"]["num_layers"] = 5
        data["topology"]["stages"][0].update(
            {"layer_start": 0, "layer_end": 2, "pp_degree": 2}
        )
        data["topology"]["stages"][1].update(
            {"layer_start": 2, "layer_end": 3}
        )
        data["topology"]["stages"][2].update(
            {"layer_start": 3, "layer_end": 5, "pp_degree": 2}
        )
        data["hardware"]["edge"]["count"] = 2
        data["static_policy"]["pipeline_depth"] = 1
        result = Simulator(parse_config(data)).run()
        self.assertTrue(all(len(row["token_timestamps_ms"]) == 4 for row in result.requests))
        self.assertIn("edge/pp_1", result.summary["resource_utilization"])

    def test_continuous_batching_switch_controls_slot_refill(self) -> None:
        base = copied_toy_config()
        base["static_policy"]["max_batch_size"] = 2
        base["workload"] = {
            "mode": "trace",
            "requests": [
                {
                    "request_id": request_id,
                    "arrival_time_ms": 0,
                    "input_tokens": 16,
                    "output_tokens": [1, 3, 2][request_id],
                }
                for request_id in range(3)
            ],
        }

        continuous = Simulator(parse_config(base)).run()
        static_data = copied_toy_config()
        static_data.update({key: value for key, value in base.items() if key != "static_policy"})
        static_data["static_policy"] = dict(base["static_policy"])
        static_data["static_policy"]["continuous_batching"] = False
        static = Simulator(parse_config(static_data)).run()

        continuous_request_2_start = min(
            row["start_time_ms"] for row in continuous.trace if 2 in row["request_ids"]
        )
        continuous_first_cohort_finish = max(
            request["e2e_ms"] for request in continuous.requests if request["request_id"] < 2
        )
        continuous_first_slot_free = min(
            request["e2e_ms"] for request in continuous.requests if request["request_id"] < 2
        )
        static_request_2_start = min(
            row["start_time_ms"] for row in static.trace if 2 in row["request_ids"]
        )
        static_first_cohort_finish = max(
            request["e2e_ms"] for request in static.requests if request["request_id"] < 2
        )
        self.assertGreaterEqual(continuous_request_2_start, continuous_first_slot_free)
        self.assertLess(continuous_request_2_start, continuous_first_cohort_finish)
        self.assertGreaterEqual(static_request_2_start, static_first_cohort_finish)
        self.assertEqual(
            {row["cohort_ids"][0] for row in static.trace if 2 in row["request_ids"]},
            {1},
        )


if __name__ == "__main__":
    unittest.main()
