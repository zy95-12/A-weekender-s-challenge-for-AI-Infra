from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import SchedulerConfig, StaticPolicyConfig
from .core import Phase, Stage, WorkItem


@dataclass(frozen=True)
class SchedulerSnapshot:
    current_time: float
    outstanding_prefill_chunks: dict[int, int]
    total_outstanding_prefill_chunks: int
    running_request_ids: frozenset[int] = frozenset()
    request_priorities: dict[int, int] | None = None
    request_arrival_times: dict[int, float] | None = None
    request_input_tokens: dict[int, int] | None = None


class SchedulerPolicy(Protocol):
    def form_batch(self, candidates: list[WorkItem], snapshot: SchedulerSnapshot) -> list[WorkItem]: ...


class VLLMScheduler:
    """V1-like running-first batching with configurable waiting-queue order."""

    def __init__(self, static: StaticPolicyConfig, scheduler: SchedulerConfig | None = None):
        self.static = static
        self.scheduler = scheduler or SchedulerConfig(max_num_seqs=static.max_batch_size)

    def _request_key(self, item: WorkItem, snapshot: SchedulerSnapshot) -> tuple:
        arrival = (snapshot.request_arrival_times or {}).get(item.request_id, item.ready_time)
        if self.scheduler.policy == "priority":
            return ((snapshot.request_priorities or {}).get(item.request_id, 0), arrival, item.request_id, item.id)
        if self.scheduler.policy == "shortest_prefill":
            return ((snapshot.request_input_tokens or {}).get(item.request_id, item.token_count), arrival, item.request_id, item.id)
        return (arrival, item.request_id, item.id)

    def form_batch(self, candidates: list[WorkItem], snapshot: SchedulerSnapshot) -> list[WorkItem]:
        ordered = sorted(candidates, key=lambda item: (
            0 if item.request_id in snapshot.running_request_ids else 1,
            self._request_key(item, snapshot),
        ))
        attempted: set[tuple[Stage, int]] = set()
        for anchor in ordered:
            group = (anchor.stage, anchor.pipeline_rank)
            if group in attempted:
                continue
            attempted.add(group)
            compatible = [item for item in ordered if (item.stage, item.pipeline_rank) == group]
            selected: list[WorkItem] = []
            request_ids: set[int] = set()
            new_running: set[int] = set()
            tokens = prefill_tokens = chunks = 0
            for item in compatible:
                if item.request_id in request_ids:
                    continue
                running = item.request_id in snapshot.running_request_ids
                if not running and len(snapshot.running_request_ids | new_running) >= self.scheduler.max_num_seqs:
                    continue
                if len(selected) >= self.static.max_batch_size:
                    break
                if tokens + item.token_count > self.static.max_batched_tokens:
                    continue
                if item.phase == Phase.PREFILL and prefill_tokens + item.token_count > self.static.prefill_token_budget:
                    continue
                if item.stage == Stage.EDGE_FRONT and item.phase == Phase.PREFILL and item.pipeline_rank == 0:
                    if snapshot.outstanding_prefill_chunks.get(item.request_id, 0) >= self.static.pipeline_depth:
                        continue
                    if snapshot.total_outstanding_prefill_chunks + chunks >= self.static.max_outstanding_prefill_chunks:
                        continue
                    chunks += 1
                selected.append(item); request_ids.add(item.request_id)
                if not running: new_running.add(item.request_id)
                tokens += item.token_count
                if item.phase == Phase.PREFILL: prefill_tokens += item.token_count
            if selected:
                return selected
        return []


FCFSScheduler = VLLMScheduler
NaiveScheduler = VLLMScheduler
