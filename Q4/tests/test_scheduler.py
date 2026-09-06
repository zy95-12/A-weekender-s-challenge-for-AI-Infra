from __future__ import annotations

import unittest

from split_serving_sim.config import parse_config
from split_serving_sim.core import Phase, Stage, WorkItem
from split_serving_sim.scheduler import FCFSScheduler, SchedulerSnapshot

from tests.helpers import copied_toy_config


class SchedulerTest(unittest.TestCase):
    def test_mixed_batch_switch_and_decode_fairness(self) -> None:
        prefill = WorkItem(id=1, request_id=1, phase=Phase.PREFILL,
                          stage=Stage.EDGE_FRONT, token_start=0,
                          token_count=16, context_tokens=16)
        decode = WorkItem(id=0, request_id=0, phase=Phase.DECODE,
                         stage=Stage.EDGE_FRONT, token_start=32,
                         token_count=1, context_tokens=32)
        for mixed in (False, True):
            with self.subTest(mixed=mixed):
                data = copied_toy_config()
                data['scheduler'] = dict(allow_mixed_batch=mixed,
                                         decode_first=True,
                                         max_consecutive_decode_batches=1)
                config = parse_config(data)
                scheduler = FCFSScheduler(config.static_policy, config.scheduler)
                snapshot = SchedulerSnapshot(0.0, {}, 0)
                self.assertEqual(scheduler.form_batch([prefill, decode], snapshot),
                                 [decode, prefill] if mixed else [decode])
                if not mixed:
                    self.assertEqual(scheduler.form_batch([prefill, decode], snapshot), [prefill])

    def test_unmixed_can_skip_blocked_prefill(self) -> None:
        data = copied_toy_config()
        data['scheduler'] = dict(allow_mixed_batch=False)
        config = parse_config(data)
        scheduler = FCFSScheduler(config.static_policy, config.scheduler)
        prefill = WorkItem(id=0, request_id=0, phase=Phase.PREFILL,
                          stage=Stage.EDGE_FRONT, token_start=0,
                          token_count=16, context_tokens=16)
        decode = WorkItem(id=1, request_id=1, phase=Phase.DECODE,
                         stage=Stage.EDGE_FRONT, token_start=32,
                         token_count=1, context_tokens=32)
        snapshot = SchedulerSnapshot(0.0, {0: config.static_policy.pipeline_depth}, 0)
        self.assertEqual(scheduler.form_batch([prefill, decode], snapshot), [decode])

    def test_bounded_decode_first_forces_prefill_progress(self) -> None:
        data = copied_toy_config()
        data["static_policy"]["max_batch_size"] = 1
        data["scheduler"] = {
            "decode_first": True,
            "max_consecutive_decode_batches": 1,
        }
        config = parse_config(data)
        scheduler = FCFSScheduler(config.static_policy, config.scheduler)
        decode = WorkItem(
            id=0, request_id=0, phase=Phase.DECODE,
            stage=Stage.EDGE_FRONT, token_start=32, token_count=1,
            context_tokens=32,
        )
        prefill = WorkItem(
            id=1, request_id=1, phase=Phase.PREFILL,
            stage=Stage.EDGE_FRONT, token_start=0, token_count=16,
            context_tokens=16,
        )
        snapshot = SchedulerSnapshot(0.0, {}, 0)
        self.assertEqual(scheduler.form_batch([prefill, decode], snapshot), [decode])
        self.assertEqual(scheduler.form_batch([prefill, decode], snapshot), [prefill])

    def test_fcfs_uses_ready_time_and_prefill_respects_budget(self) -> None:
        config = parse_config(copied_toy_config()).static_policy
        scheduler = FCFSScheduler(config)
        candidates = [
            WorkItem(
                id=0,
                request_id=0,
                phase=Phase.PREFILL,
                stage=Stage.CLOUD_MIDDLE,
                token_start=0,
                token_count=16,
                context_tokens=16,
            ),
            WorkItem(
                id=1,
                request_id=1,
                phase=Phase.PREFILL,
                stage=Stage.CLOUD_MIDDLE,
                token_start=0,
                token_count=16,
                context_tokens=16,
            ),
            WorkItem(
                id=2,
                request_id=2,
                phase=Phase.PREFILL,
                stage=Stage.CLOUD_MIDDLE,
                token_start=0,
                token_count=16,
                context_tokens=16,
            ),
            WorkItem(
                id=3,
                request_id=3,
                phase=Phase.DECODE,
                stage=Stage.CLOUD_MIDDLE,
                token_start=32,
                token_count=1,
                context_tokens=32,
                ready_time=1.0,
            ),
        ]
        selected = scheduler.form_batch(
            candidates,
            SchedulerSnapshot(0.0, {}, 0),
        )
        self.assertEqual([item.id for item in selected], [0, 1, 3])
        self.assertEqual(
            sum(item.token_count for item in selected if item.phase == Phase.PREFILL),
            config.prefill_token_budget,
        )
        self.assertEqual(len(selected), 3)


if __name__ == "__main__":
    unittest.main()
