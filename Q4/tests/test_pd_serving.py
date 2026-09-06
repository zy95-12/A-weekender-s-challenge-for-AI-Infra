from dataclasses import replace
from pathlib import Path
import unittest

from split_serving_sim.config import load_config, parse_config, validate_config, ConfigError
from split_serving_sim.core import Stage, Phase
from split_serving_sim.presets import ServingFeatures, configure_serving
from split_serving_sim.simulator import Simulator
from split_serving_sim.scheduler import PDScheduler
from tests.helpers import copied_toy_config


def toy_pd(**features):
    base = parse_config(copied_toy_config())
    opts = replace(ServingFeatures.optimized(), chunk_size=16, **features)
    return configure_serving(base, opts)


class PDServingTest(unittest.TestCase):
    def test_all_off_preserves_baseline_configuration_and_results(self):
        base = load_config(Path(__file__).parents[1] / 'configs/issue6_baseline_host.json')
        base = replace(base, workload=replace(base.workload, concurrency=2, num_requests=4,
                       warmup_requests=0, measurement_duration_s=0, output_tokens=3))
        restored = configure_serving(base, ServingFeatures())
        self.assertEqual(base, restored)
        self.assertEqual(Simulator(base).run(), Simulator(restored).run())

    def test_phase_topology_and_resource_counts(self):
        for p_tp, d_tp, replicas in [(1,1,2),(2,1,1),(1,2,1)]:
            cfg = toy_pd(prefill_tp=p_tp, decode_tp=d_tp, prefill_replicas=replicas)
            sim = Simulator(cfg)
            cloud = cfg.stage('cloud_middle')
            self.assertEqual(cloud.for_phase('prefill').tp_degree, p_tp)
            self.assertEqual(cloud.for_phase('decode').tp_degree, d_tp)
            self.assertEqual(len([r for r in sim.resources if r.startswith('cloud_prefill')]), replicas)
            self.assertEqual(len([r for r in sim.resources if r.startswith('cloud_decode')]), 1)
            self.assertIsInstance(sim.scheduler, PDScheduler)
            sim.run()

    def test_ttft_independent_of_kv_and_decode_gated(self):
        cfg = toy_pd()
        cfg = replace(cfg, workload=replace(cfg.workload, num_requests=1))
        slow = replace(cfg, scheduler=replace(cfg.scheduler, pd_disaggregation=replace(
            cfg.scheduler.pd_disaggregation, kv_transfer_latency_ms=1000)))
        fast_result = Simulator(cfg).run()
        sim = Simulator(slow); result = sim.run()
        self.assertEqual(result.requests[0]['ttft_ms'], fast_result.requests[0]['ttft_ms'])
        kv = next(row for row in result.trace if row['stage']=='pd_kv_transfer')
        first_decode = next(row for row in result.trace if row['stage']=='edge_front' and row['phases']==['decode'])
        self.assertGreaterEqual(first_decode['start_time_ms'], kv['end_time_ms'])
        tail = next(row for row in result.trace if row['stage']=='edge_tail' and row['chunk_indices']==[1])
        self.assertLess(kv['start_time_ms'], tail['end_time_ms'])
        self.assertEqual(kv['communication_bytes'], 32 * 2 * 2 * 16 * 2)

    def test_chunk_transfer_no_duplicate_bytes_and_ordered_ranges(self):
        for chunked in (False, True):
            sim = Simulator(toy_pd(chunk_transfer=chunked)); result = sim.run()
            for rid in sim.requests:
                items = [i for i in sim.dag.items.values() if i.stage==Stage.PD_KV_TRANSFER and i.request_id==rid]
                self.assertEqual(len(items), 2 if chunked else 1)
                self.assertEqual(sum(i.token_count for i in items), 32)
                if chunked:
                    self.assertEqual([i.token_start for i in items], [0,16])
                    start = sim.dag.items[items[1].dependencies[0]]
                    self.assertTrue(any(sim.dag.items[dep].stage == Stage.PD_KV_COMMIT
                                        for dep in start.dependencies))

    def test_one_token_drains_handoff_before_freeing_credit(self):
        cfg = toy_pd()
        cfg = replace(cfg, workload=replace(cfg.workload, output_tokens=1), scheduler=replace(
            cfg.scheduler, pd_disaggregation=replace(cfg.scheduler.pd_disaggregation, kv_transfer_latency_ms=1000)))
        sim = Simulator(cfg); sim.run()
        self.assertEqual(sim.pd_ready, set(sim.requests))
        self.assertFalse(sim.active_continuous_requests)
        self.assertFalse(sim.inflight_transactions)
        self.assertTrue(all(r.finish_time < sim.clock for r in sim.requests.values()))

    def test_replica_affinity_and_window_credits(self):
        cfg = toy_pd(prefill_window=1, decode_window=1)
        cfg = replace(cfg, workload=replace(cfg.workload, num_requests=8))
        sim = Simulator(cfg); original = sim._reserve_front_batch
        observed = []
        def reserve(items):
            original(items)
            if items[0].stage != Stage.EDGE_FRONT:
                return
            if items[0].phase == Phase.PREFILL:
                self.assertEqual(len({sim.pd_routes[i.request_id] for i in items}), 1)
            buckets = {}
            for members in sim.inflight_transaction_members.values():
                rid, phase, _, _ = next(iter(members))
                key = (phase, sim.pd_routes[rid] if phase=='prefill' else 0)
                buckets[key] = buckets.get(key,0)+1
            self.assertTrue(all(count<=1 for count in buckets.values()))
            observed.append(buckets)
        sim._reserve_front_batch = reserve
        result = sim.run()
        self.assertTrue(observed)
        self.assertEqual(set(sim.pd_routes.values()), {0,1})
        for rid in sim.requests:
            routes = {row['resource'] for row in result.trace if row['stage']=='cloud_middle'
                      and row['phases']==['prefill'] and rid in row['request_ids']}
            self.assertEqual(len(routes),1)

    def test_control_channel_changes_command_lane_without_blocking_bulk_transfer(self):
        for independent in (False, True):
            cfg = toy_pd(control_channel=independent, control_latency_ms=1)
            result = Simulator(cfg).run()
            for row in result.trace:
                if row['stage'] == 'pd_kv_start':
                    self.assertTrue(row['resource'].startswith('pd_control_prefill' if independent else 'cloud_prefill'))
                if row['stage'] == 'pd_kv_commit':
                    self.assertEqual(row['resource'], 'pd_control_decode' if independent else 'cloud_decode')
                if row['stage'] == 'pd_kv_transfer':
                    self.assertEqual(row['resource'], 'pd_kv_transfer')

    def test_invalid_features_and_analytical_backend(self):
        with self.assertRaises(ValueError):
            toy_pd(pipeline=False)
        cfg = toy_pd()
        invalid = replace(cfg, scheduler=replace(cfg.scheduler, allow_mixed_batch=True))
        with self.assertRaises(ConfigError):
            validate_config(invalid)
        baseline = parse_config(copied_toy_config())
        cfg = configure_serving(baseline, ServingFeatures(), 'roofline')
        self.assertFalse(cfg.performance_profile.enabled)

if __name__ == '__main__':
    unittest.main()
