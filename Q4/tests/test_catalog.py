import unittest
from dataclasses import replace
from split_serving_sim.catalog import build_config, stage_storage
from split_serving_sim.performance import RooflineModel, NetworkModel
from split_serving_sim.core import WorkItem, Phase, Stage
from split_serving_sim.config import RequestSpec, validate_config, ConfigError
from split_serving_sim.serving_runtime import VirtualServingSimulator
from split_serving_sim.simulator import Simulator


class CatalogTests(unittest.TestCase):
    def test_public_combinations_capacity_and_calibration_isolation(self):
        for model in ("qwen2.5-3b", "qwen3-32b", "deepseek-v4-flash"):
            for device in ("a10", "l20", "h20", "ascend910b"):
                for variant in ("baseline", "optimized"):
                    with self.subTest(model=model, device=device, variant=variant):
                        c, meta = build_config(model, device, variant)
                        validate_config(c)
                        self.assertEqual(sum(meta["split"]), c.model.num_layers)
                        if not meta["calibrated"]:
                            self.assertFalse(c.performance_profile.enabled)
                            self.assertFalse(c.execution.host_submission.enabled)
                            self.assertFalse(c.data_path.wan_calibration.enabled)
                            for r in meta["capacity_plan"]:
                                self.assertLessEqual(
                                    r["weight_gb_per_rank"] + r["kv_gb_per_rank"],
                                    0.8 * meta["hardware"]["memory_gb"],
                                )

    def test_roofline_scales_with_compute_and_bandwidth(self):
        c, _ = build_config("deepseek-v4-flash", "h20")
        item = WorkItem(0, 0, Phase.DECODE, Stage.CLOUD_MIDDLE, 4096, 1, 4097)
        original = RooflineModel(c).estimate("cloud_middle", [item])
        faster = replace(
            c,
            hardware={
                k: replace(
                    v,
                    peak_flops_tflops=v.peak_flops_tflops * 2,
                    hbm_bandwidth_gb_s=v.hbm_bandwidth_gb_s * 2,
                )
                for k, v in c.hardware.items()
            },
        )
        self.assertLess(
            RooflineModel(faster).estimate("cloud_middle", [item]).total_time_s,
            original.total_time_s,
        )
        names = [o.name for o in original.sub_operations]
        for family in [
            "compressor",
            "indexer_scores_topk",
            "routed_shared_experts",
            "mhc",
            "tp_all_reduce_moe",
        ]:
            self.assertTrue(any(family in n for n in names))
        self.assertTrue(
            all(o.profile_source == "roofline" for o in original.sub_operations)
        )

    def test_moe_batch_shares_touched_weights_and_keeps_all_experts_resident(self):
        c, _ = build_config("deepseek-v4-flash", "h20")
        cost = RooflineModel(c)

        def items(n):
            return [
                WorkItem(i, i, Phase.DECODE, Stage.CLOUD_MIDDLE, 4096, 1, 4097)
                for i in range(n)
            ]

        one = cost.estimate("cloud_middle", items(1))
        batch = cost.estimate("cloud_middle", items(16))
        self.assertGreater(batch.memory_bytes, one.memory_bytes)
        self.assertLess(batch.memory_bytes, 16 * one.memory_bytes)
        w, _ = stage_storage(c.model, [c.stages[1]], 8)
        self.assertGreater(w, one.memory_bytes)
        # Four residual streams cross WAN; one shared K=V cache, not dense 2*heads KV.
        self.assertEqual(c.model.activation_width, 4 * c.model.hidden_size)
        payload = (
            NetworkModel(c).payload_bytes(items(1)) - c.network.protocol_overhead_bytes
        )
        self.assertEqual(payload, 4 * c.model.hidden_size * 2)

    def test_new_models_run_both_schedulers(self):
        for model in ("qwen3-32b", "deepseek-v4-flash"):
            for variant in ("baseline", "optimized"):
                c, _ = build_config(model, "h20", variant)
                c = replace(
                    c,
                    workload=replace(
                        c.workload,
                        mode="open_loop",
                        concurrency=0,
                        warmup_requests=0,
                        warmup_duration_s=0,
                        measurement_duration_s=30,
                        arrival_tail_s=0,
                        requests=(RequestSpec(0, 0, 4096, 3),),
                    ),
                    simulation=replace(c.simulation, trace_enabled=False),
                )
                result = (
                    VirtualServingSimulator(c)
                    if variant == "optimized"
                    else Simulator(c)
                ).run()
                self.assertEqual(len(result.requests), 1)
                self.assertGreater(result.requests[0]["ttft_ms"], 0)
                self.assertGreater(result.requests[0]["mean_tpot_ms"], 0)
