import unittest
from pathlib import Path
from dataclasses import replace
from split_serving_sim.config import HostSubmissionConfig, parse_config, load_config, ConfigError
from split_serving_sim.core import SubOperation, Stage, Phase, WorkItem
from split_serving_sim.host_submission import schedule_submissions
from split_serving_sim.simulator import Simulator
from split_serving_sim.performance import RooflineModel
from tests.helpers import copied_toy_config


def op(name, us, dependencies=(), kind="mm"):
    return SubOperation(name, "compute", us * 1e-6, "", dependencies, profile_type=kind)


class HostSubmissionTest(unittest.TestCase):
    def test_calibrated_gpu_costs_are_unchanged(self):
        cfg = load_config(Path(__file__).parents[1] / 'configs/issue6_baseline_host.json')
        off = replace(cfg, execution=replace(cfg.execution,
                      host_submission=replace(cfg.execution.host_submission, enabled=False)))
        for batch_size in [1, 16]:
            items = [WorkItem(i, i, Phase.DECODE, Stage.CLOUD_MIDDLE, 4096, 1, 4096)
                     for i in range(batch_size)]
            before = RooflineModel(off).estimate('cloud_middle', items)
            after = RooflineModel(cfg).estimate('cloud_middle', items)
            self.assertEqual(before.total_time_s, after.total_time_s)
            self.assertEqual(before.sub_operations, after.sub_operations)

    def test_submission_is_hidden_by_long_kernel(self):
        s = schedule_submissions((op('a', 20), op('b', 20, ('a',))), 0, 0, HostSubmissionConfig(True, 6))
        self.assertAlmostEqual(s.end_time * 1e6, 46)
        self.assertAlmostEqual(s.cpu_busy_s * 1e6, 12)
        self.assertAlmostEqual(s.exposed_delay_s * 1e6, 6)
        self.assertLess(s.cpu_intervals[1][1], s.gpu_intervals[0][1])

    def test_short_kernels_expose_submission_bubbles(self):
        s = schedule_submissions((op('a', 2), op('b', 2, ('a',))), 0, 0, HostSubmissionConfig(True, 6))
        self.assertAlmostEqual(s.end_time * 1e6, 14)
        self.assertAlmostEqual(s.exposed_delay_s * 1e6, 10)

    def test_cpu_lane_backlog_and_type_override(self):
        cfg = HostSubmissionConfig(True, 6, (('flash_attention', 18),))
        s = schedule_submissions((op('a', 2, kind='flash_attention'),), 10e-6, 20e-6, cfg)
        self.assertAlmostEqual(s.cpu_intervals[0][0] * 1e6, 20)
        self.assertAlmostEqual(s.end_time * 1e6, 40)

    def test_invalid_costs(self):
        for value in [-1, float('nan'), float('inf')]:
            data = copied_toy_config()
            data['execution'] = {'host_submission': {'submit_us': value}}
            with self.assertRaises(ConfigError):
                parse_config(data)

    def test_raw_roofline_does_not_charge_serial_launch_twice(self):
        data = copied_toy_config()
        cfg = parse_config(data)
        items = [WorkItem(0, 0, Phase.DECODE, Stage.CLOUD_MIDDLE, 32, 1, 32)]
        before = RooflineModel(cfg).estimate('cloud_middle', items)
        cfg = replace(cfg, execution=replace(cfg.execution, host_submission=HostSubmissionConfig(True, 6)))
        after = RooflineModel(cfg).estimate('cloud_middle', items)
        self.assertLess(after.total_time_s, before.total_time_s)
        self.assertEqual(after.overhead_time_s, 0)

    def test_end_to_end_zero_cost_matches_and_trace_contains_cpu_intervals(self):
        data = copied_toy_config()
        # Zero legacy overhead isolates scheduling from Roofline cost changes.
        for hw in data['hardware'].values():
            hw['kernel_overhead_us'] = 0
        reference = Simulator(parse_config(data)).run()
        data['execution'] = {'host_submission': {'enabled': True, 'submit_us': 0}}
        zero = Simulator(parse_config(data)).run()
        for expected, actual in zip(reference.requests, zero.requests):
            for key, value in expected.items():
                if isinstance(value, float):
                    self.assertAlmostEqual(value, actual[key], places=9)
                elif key == 'token_timestamps_ms':
                    self.assertEqual(len(value), len(actual[key]))
                    for a, b in zip(value, actual[key]):
                        self.assertAlmostEqual(a, b, places=9)
                else:
                    self.assertEqual(value, actual[key])
        data['execution']['host_submission']['submit_us'] = 6
        result = Simulator(parse_config(data)).run()
        self.assertEqual(result.summary['num_requests'], reference.summary['num_requests'])
        self.assertTrue(result.summary['execution']['host_submission']['resource_totals'])
        ops = [o for row in result.trace for o in row['sub_operations'] if o['host_submit_end_ms'] is not None]
        self.assertTrue(ops)
        self.assertTrue(all(o['start_time_ms'] >= o['host_submit_end_ms'] for o in ops))
