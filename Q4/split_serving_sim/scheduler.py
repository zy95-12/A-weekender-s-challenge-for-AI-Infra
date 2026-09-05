from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import StaticPolicyConfig
from .core import Phase, Stage, WorkItem


@dataclass(frozen=True)
class SchedulerSnapshot:
    current_time: float
    outstanding_prefill_chunks: dict[int, int]
    total_outstanding_prefill_chunks: int


class SchedulerPolicy(Protocol):
    def form_batch(
        self,
        candidates: list[WorkItem],
        snapshot: SchedulerSnapshot,
    ) -> list[WorkItem]: ...


class FCFSScheduler:
    """Deterministic FCFS batching subject to configured capacity limits."""

    def __init__(self, config: StaticPolicyConfig):
        self.config = config

    @staticmethod
    def _priority(item: WorkItem) -> tuple[float, int, int]:
        return (item.ready_time, item.request_id, item.id)

    def form_batch(
        self,
        candidates: list[WorkItem],
        snapshot: SchedulerSnapshot,
    ) -> list[WorkItem]:
        if not candidates:
            return []
        ordered = sorted(candidates, key=self._priority)
        # A shared Edge resource can have both front and tail work. If one stage
        # is backpressured, try the next stage rather than leaving the GPU idle.
        attempted_stages: set[Stage] = set()
        for anchor in ordered:
            if anchor.stage in attempted_stages:
                continue
            attempted_stages.add(anchor.stage)
            compatible = [item for item in ordered if item.stage == anchor.stage]
            selected: list[WorkItem] = []
            selected_requests: set[int] = set()
            token_count = 0
            prefill_token_count = 0
            new_prefill_chunks = 0
            for item in compatible:
                if item.request_id in selected_requests:
                    continue
                if len(selected) >= self.config.max_batch_size:
                    break
                if token_count + item.token_count > self.config.max_batched_tokens:
                    continue
                if (
                    item.phase == Phase.PREFILL
                    and prefill_token_count + item.token_count
                    > self.config.prefill_token_budget
                ):
                    continue
                if (
                    item.stage == Stage.EDGE_FRONT
                    and item.phase == Phase.PREFILL
                    and item.pipeline_rank == 0
                ):
                    request_outstanding = snapshot.outstanding_prefill_chunks.get(
                        item.request_id, 0
                    )
                    if request_outstanding >= self.config.pipeline_depth:
                        continue
                    if (
                        snapshot.total_outstanding_prefill_chunks + new_prefill_chunks
                        >= self.config.max_outstanding_prefill_chunks
                    ):
                        continue
                    new_prefill_chunks += 1
                selected.append(item)
                selected_requests.add(item.request_id)
                token_count += item.token_count
                if item.phase == Phase.PREFILL:
                    prefill_token_count += item.token_count
            if selected:
                return selected
        return []


# Backward-compatible import name for existing users of the first MVP.
NaiveScheduler = FCFSScheduler
