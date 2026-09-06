from dataclasses import replace
import unittest

from split_serving_sim.config import RequestSpec, validate_config, ConfigError
from split_serving_sim.open_loop import request_specs, summary
from split_serving_sim.serving_runtime import VirtualServingSimulator
from split_serving_sim.simulator import Simulator
from tests.test_serving_runtime import config


def open_config():
    cfg = config()
    return replace(
        cfg,
        workload=replace(
            cfg.workload,
            mode="open_loop",
            concurrency=0,
            warmup_requests=0,
            warmup_duration_s=0.05,
            measurement_duration_s=0.1,
            arrival_tail_s=0.05,
            requests=tuple(
                RequestSpec(i, t, 32, 3) for i, t in enumerate([1, 2, 60, 61, 62, 180])
            ),
        ),
    )


class OpenLoopTests(unittest.TestCase):
    def test_both_backends_keep_arrivals_independent_and_do_not_close_during_idle_gap(
        self,
    ):
        cfg = open_config()
        cfg = replace(cfg, scheduler=replace(cfg.scheduler, max_num_seqs=1))
        validate_config(cfg)
        for engine in [Simulator, VirtualServingSimulator]:
            result = engine(cfg).run()
            self.assertEqual(
                [r["arrival_time_ms"] for r in result.requests], [1, 2, 60, 61, 62, 180]
            )
            self.assertEqual(len(result.requests), 6)
            self.assertEqual(result.summary["arrival_cohort"]["num_requests"], 3)
            self.assertGreater(result.summary["peak_inflight"], 1)
            self.assertEqual(result.summary["measurement_window"]["start_ms"], 50)
            self.assertTrue(
                all(r["finish_time_ms"] > r["arrival_time_ms"] for r in result.requests)
            )

    def test_empty_measurement_cohort_has_zero_throughput(self):
        cfg = open_config()
        cfg = replace(
            cfg, workload=replace(cfg.workload, requests=(RequestSpec(0, 180, 32, 3),))
        )
        for engine in (Simulator, VirtualServingSimulator):
            s = engine(cfg).run().summary
            self.assertEqual(s["arrival_cohort"]["num_requests"], 0)
            self.assertEqual(s["observed_request_throughput_qps"], 0)
            self.assertFalse(s["slo"]["pass"])
            self.assertEqual(s["measurement_window"]["mode"], "fixed_open_loop")

    def test_poisson_matches_real_benchmark_schedule(self):
        cfg = open_config()
        w = replace(
            cfg.workload,
            requests=(),
            warmup_duration_s=30,
            measurement_duration_s=120,
            arrival_tail_s=30,
            arrival_process="poisson",
            arrival_rate_qps=4.135823836327486,
            random_seed=17,
        )
        generated = request_specs(w)
        # Archived benchmark uses exponential gaps including the first arrival.
        import random

        rng = random.Random(17)
        t = 0
        expected = []
        while True:
            t += rng.expovariate(w.arrival_rate_qps)
            if t >= 180:
                break
            expected.append(t * 1000)
        self.assertEqual([r.arrival_time_ms for r in generated], expected)
        self.assertEqual(
            sum(30000 <= r.arrival_time_ms < 150000 for r in generated), 532
        )

    def test_half_open_cohorts_keep_slow_requests_and_exclude_drain_from_qps(self):
        cfg = open_config()
        w = replace(
            cfg.workload,
            warmup_duration_s=1,
            measurement_duration_s=1,
            arrival_tail_s=1,
        )

        def row(i, a, f, ttft=5, tpot=10):
            return dict(
                request_id=i,
                arrival_time_ms=a,
                finish_time_ms=f,
                ttft_ms=ttft,
                e2e_ms=f - a,
                mean_tpot_ms=tpot,
                output_tokens=2,
                token_timestamps_ms=[a + ttft, f],
            )

        rows = [
            row(0, 999, 1200),
            row(1, 1000, 2000, tpot=200),
            row(2, 1500, 2500),
            row(3, 2000, 2600),
        ]
        s = summary(rows, [], 1, 2, cfg.slo, w)
        self.assertEqual(s["arrival_cohort"]["num_requests"], 2)
        self.assertEqual(s["completion_cohort"]["num_requests"], 1)
        self.assertEqual(s["observed_request_throughput_qps"], 1)
        self.assertEqual(s["arrival_cohort"]["tpot_ms"]["mean"], 105)
        self.assertEqual([r["measured"] for r in rows], [False, True, True, False])
        self.assertEqual(s["mean_inflight"], 1.7)

    def test_reject_invalid_open_loop_configuration(self):
        cfg = open_config()
        for changes in [
            dict(concurrency=1),
            dict(warmup_requests=1),
            dict(measurement_duration_s=0),
            dict(warmup_duration_s=float("nan")),
            dict(requests=(), arrival_rate_qps=float("inf")),
            dict(requests=(RequestSpec(0, 201, 32, 3),)),
        ]:
            with self.assertRaises(ConfigError):
                validate_config(replace(cfg, workload=replace(cfg.workload, **changes)))
