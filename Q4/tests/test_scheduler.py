from __future__ import annotations

import unittest

from split_serving_sim.config import parse_config
from split_serving_sim.core import Phase, Stage, WorkItem
from split_serving_sim.scheduler import NaiveScheduler, SchedulerSnapshot

from tests.helpers import copied_toy_config


class SchedulerTest(unittest.TestCase):
    def test_decode_is_selected_first_and_prefill_respects_budget(self) -> None:
        config = parse_config(copied_toy_config()).static_policy
        scheduler = NaiveScheduler(config)
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
            ),
        ]
        selected = scheduler.form_batch(
            candidates,
            SchedulerSnapshot(0.0, {}, 0),
        )
        self.assertEqual(selected[0].phase, Phase.DECODE)
        self.assertEqual(
            sum(item.token_count for item in selected if item.phase == Phase.PREFILL),
            config.prefill_token_budget,
        )
        self.assertEqual(len(selected), 3)


if __name__ == "__main__":
    unittest.main()
