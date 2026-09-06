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
        self.consecutive_decode_batches = 0

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
            if (
                self.scheduler.policy == "split_poc_naive"
                and anchor.stage == Stage.EDGE_FRONT
                and anchor.pipeline_rank == 0
            ):
                prefills = [item for item in compatible if item.phase == Phase.PREFILL]
                # PR #8 baseline: one whole/chunked prefill monopolizes the
                # next end-to-end call; otherwise decode every active request.
                compatible = prefills[:1] if prefills else [
                    item for item in compatible if item.phase == Phase.DECODE
                ]
            elif self.scheduler.decode_first:
                prefills = [item for item in compatible if item.phase == Phase.PREFILL]
                decodes = [item for item in compatible if item.phase == Phase.DECODE]
                waited_too_long = bool(
                    prefills
                    and self.scheduler.max_prefill_wait_ms > 0
                    and (snapshot.current_time - min(item.ready_time for item in prefills)) * 1000.0
                    >= self.scheduler.max_prefill_wait_ms
                )
                force_prefill = bool(
                    prefills
                    and decodes
                    and (
                        waited_too_long
                        or self.consecutive_decode_batches
                        >= self.scheduler.max_consecutive_decode_batches
                    )
                )
                compatible = (
                    prefills + decodes if force_prefill else decodes + prefills
                )
            selected: list[WorkItem] = []
            request_ids: set[int] = set()
            new_running: set[int] = set()
            tokens = prefill_tokens = decode_tokens = chunks = 0
            for item in compatible:
                if (
                    not self.scheduler.allow_mixed_batch
                    and selected
                    and item.phase != selected[0].phase
                ):
                    continue
                if item.request_id in request_ids:
                    continue
                running = item.request_id in snapshot.running_request_ids
                if not running and len(snapshot.running_request_ids | new_running) >= self.scheduler.max_num_seqs:
                    continue
                if len(selected) >= self.static.max_batch_size:
                    break
                if tokens + item.token_count > self.static.max_batched_tokens:
                    continue
                if (
                    item.phase == Phase.DECODE
                    and self.scheduler.max_decode_tokens_per_batch > 0
                    and decode_tokens + item.token_count
                    > self.scheduler.max_decode_tokens_per_batch
                ):
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
                if item.phase == Phase.PREFILL:
                    prefill_tokens += item.token_count
                else:
                    decode_tokens += item.token_count
            if selected:
                if anchor.stage == Stage.EDGE_FRONT and anchor.pipeline_rank == 0:
                    if any(item.phase == Phase.PREFILL for item in selected):
                        self.consecutive_decode_batches = 0
                    elif all(item.phase == Phase.DECODE for item in selected):
                        self.consecutive_decode_batches += 1
                return selected
        return []


FCFSScheduler = VLLMScheduler
NaiveScheduler = VLLMScheduler


class PDScheduler(VLLMScheduler):
    """PD front fairness plus bounded decode priority for returned RPCs.

    Replica affinity and per-role credits are enforced by the simulator because
    they span multiple resources and survive individual GPU batches.
    """
    def __init__(self, static, scheduler):
        super().__init__(static, scheduler)
        self.consecutive_decode_backs = 0

    def form_batch(self, candidates, snapshot):
        selected = super().form_batch(candidates, snapshot)
        if selected and selected[0].stage == Stage.EDGE_FRONT and selected[0].phase == Phase.PREFILL:
            return selected[:1]
        return selected

    def ready_backs(self, candidates):
        backs = [item for item in candidates if item.stage == Stage.EDGE_TAIL]
        if not backs:
            return candidates
        decodes = [item for item in backs if item.phase == Phase.DECODE]
        prefills = [item for item in backs if item.phase == Phase.PREFILL]
        if self.scheduler.decode_first and decodes and (
            not prefills or self.consecutive_decode_backs < self.scheduler.max_consecutive_decode_batches
        ):
            self.consecutive_decode_backs += 1
            return decodes
        self.consecutive_decode_backs = 0
        return prefills or backs
