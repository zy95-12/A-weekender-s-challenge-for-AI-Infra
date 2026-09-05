from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .config import RequestSpec, SimulationConfig
from .core import GPU_STAGES, NETWORK_STAGES, PerformanceEstimate, Phase, Stage, WorkItem
from .dag import ExecutionDAG
from .metrics import RequestRuntime, build_summary
from .performance import NetworkModel, RooflineModel
from .scheduler import NaiveScheduler, SchedulerSnapshot


class EventType(str, Enum):
    REQUEST_ARRIVAL = "request_arrival"
    BATCH_FINISH = "batch_finish"


@dataclass(order=True)
class Event:
    timestamp: float
    sequence_no: int
    event_type: EventType = field(compare=False)
    payload: Any = field(compare=False)


@dataclass
class Resource:
    resource_id: str
    running_batch_id: int | None = None
    busy_time: float = 0.0

    @property
    def idle(self) -> bool:
        return self.running_batch_id is None


@dataclass
class BatchExecution:
    batch_id: int
    resource_id: str
    stage: Stage
    items: list[WorkItem]
    estimate: PerformanceEstimate
    start_time: float
    end_time: float


@dataclass(frozen=True)
class SimulationResult:
    summary: dict[str, Any]
    requests: list[dict[str, Any]]
    trace: list[dict[str, Any]]


class Simulator:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.dag = ExecutionDAG()
        self.scheduler = NaiveScheduler(config.static_policy)
        self.roofline = RooflineModel(config)
        self.network = NetworkModel(config)
        resource_names = {
            self._replica_resource_id(stage.resource, replica, stage.replicas)
            for stage in config.stages
            for replica in range(stage.replicas)
        }
        resource_names.update({"wan_up", "wan_down"})
        self.resources = {name: Resource(name) for name in sorted(resource_names)}
        self.queues: dict[str, list[WorkItem]] = {name: [] for name in self.resources}
        self.requests: dict[int, RequestRuntime] = {}
        self.events: list[Event] = []
        self.trace: list[dict[str, Any]] = []
        self.clock = 0.0
        self._event_sequence = 0
        self._batch_sequence = 0
        self.outstanding_prefill_chunks: dict[int, int] = {}
        self.total_outstanding_prefill_chunks = 0

    def run(self) -> SimulationResult:
        request_specs = self._request_specs()
        if not request_specs:
            raise ValueError("workload generated no requests")
        for spec in request_specs:
            self.requests[spec.request_id] = RequestRuntime(
                request_id=spec.request_id,
                arrival_time=spec.arrival_time_ms / 1000.0,
                input_tokens=spec.input_tokens,
                output_tokens=spec.output_tokens,
            )
            self._push_event(
                spec.arrival_time_ms / 1000.0,
                EventType.REQUEST_ARRIVAL,
                spec.request_id,
            )

        start_time = min(request.arrival_time for request in self.requests.values())
        while self.events:
            first = heapq.heappop(self.events)
            if first.timestamp > self.config.simulation.max_time_s:
                raise RuntimeError(
                    f"simulation exceeded max_time_s={self.config.simulation.max_time_s}"
                )
            self.clock = first.timestamp
            simultaneous = [first]
            while self.events and self.events[0].timestamp == self.clock:
                simultaneous.append(heapq.heappop(self.events))
            for event in simultaneous:
                self._process_event(event)
            self._schedule_idle_resources()

        unfinished = [
            request.request_id
            for request in self.requests.values()
            if request.finish_time is None
        ]
        if unfinished:
            queued = sum(len(queue) for queue in self.queues.values())
            raise RuntimeError(f"simulation deadlocked; unfinished={unfinished}, queued={queued}")

        request_metrics = [
            self.requests[request_id].to_metrics()
            for request_id in sorted(self.requests)
        ]
        end_time = max(request.finish_time or 0.0 for request in self.requests.values())
        summary = build_summary(
            request_metrics,
            self.trace,
            {name: resource.busy_time for name, resource in self.resources.items()},
            start_time,
            end_time,
            self.config.slo,
        )
        summary["static_policy"] = {
            "batch_size": self.config.static_policy.max_batch_size,
            "max_batched_tokens": self.config.static_policy.max_batched_tokens,
            "prefill_token_budget": self.config.static_policy.prefill_token_budget,
            "prefill_chunk_size": self.config.static_policy.prefill_chunk_size,
        }
        return SimulationResult(summary=summary, requests=request_metrics, trace=self.trace)

    def _request_specs(self) -> list[RequestSpec]:
        workload = self.config.workload
        if workload.mode == "trace":
            return sorted(workload.requests, key=lambda item: (item.arrival_time_ms, item.request_id))
        random_source = random.Random(workload.random_seed)
        arrival_time_ms = 0.0
        specs: list[RequestSpec] = []
        for request_id in range(workload.num_requests):
            specs.append(
                RequestSpec(
                    request_id=request_id,
                    arrival_time_ms=arrival_time_ms,
                    input_tokens=workload.input_tokens,
                    output_tokens=workload.output_tokens,
                )
            )
            if workload.arrival_process == "poisson":
                assert workload.arrival_rate_qps is not None
                arrival_time_ms += (
                    random_source.expovariate(workload.arrival_rate_qps) * 1000.0
                )
            else:
                arrival_time_ms += workload.arrival_interval_ms
        return specs

    def _push_event(self, timestamp: float, event_type: EventType, payload: Any) -> None:
        event = Event(timestamp, self._event_sequence, event_type, payload)
        self._event_sequence += 1
        heapq.heappush(self.events, event)

    def _process_event(self, event: Event) -> None:
        if event.event_type == EventType.REQUEST_ARRIVAL:
            request = self.requests[event.payload]
            ready = self.dag.add_prefill(
                request.request_id,
                request.input_tokens,
                self.config.static_policy.prefill_chunk_size,
                self.clock,
            )
            self._enqueue(ready)
            return
        if event.event_type == EventType.BATCH_FINISH:
            self._finish_batch(event.payload)
            return
        raise RuntimeError(f"unknown event: {event.event_type}")

    def _finish_batch(self, batch: BatchExecution) -> None:
        resource = self.resources[batch.resource_id]
        if resource.running_batch_id != batch.batch_id:
            raise RuntimeError(f"resource/batch mismatch: {batch.resource_id}")
        resource.running_batch_id = None
        resource.busy_time += batch.end_time - batch.start_time

        for item in batch.items:
            ready = self.dag.mark_complete(item.id, self.clock)
            self._enqueue(ready)
            if item.phase == Phase.PREFILL and item.stage == Stage.EDGE_TAIL:
                self.outstanding_prefill_chunks[item.request_id] -= 1
                self.total_outstanding_prefill_chunks -= 1
            if item.stage == Stage.EDGE_TAIL:
                if item.phase == Phase.PREFILL and self.dag.is_final_prefill(item):
                    self._token_ready(item.request_id)
                elif item.phase == Phase.DECODE:
                    self._token_ready(item.request_id)

    def _token_ready(self, request_id: int) -> None:
        request = self.requests[request_id]
        request.token_times.append(self.clock)
        if request.first_token_time is None:
            request.first_token_time = self.clock
        if len(request.token_times) >= request.output_tokens:
            request.finish_time = self.clock
            return
        context = request.input_tokens + len(request.token_times)
        ready = self.dag.add_decode(
            request_id=request_id,
            iteration=len(request.token_times),
            context_tokens=context,
            ready_time=self.clock,
        )
        self._enqueue(ready)

    def _enqueue(self, items: list[WorkItem]) -> None:
        for item in items:
            self.queues[self._resource_for_item(item)].append(item)

    @staticmethod
    def _replica_resource_id(resource: str, replica: int, replicas: int) -> str:
        return resource if replicas == 1 else f"{resource}/replica_{replica}"

    def _resource_for_item(self, item: WorkItem) -> str:
        if item.stage == Stage.WAN_UP:
            return "wan_up"
        if item.stage == Stage.WAN_DOWN:
            return "wan_down"
        stage = self.config.stage(item.stage.value)
        # Naive but KV-safe routing: a request remains sticky to one replica.
        replica = item.request_id % stage.replicas
        return self._replica_resource_id(stage.resource, replica, stage.replicas)

    def _schedule_idle_resources(self) -> None:
        for resource_id in sorted(self.resources):
            resource = self.resources[resource_id]
            if not resource.idle or not self.queues[resource_id]:
                continue
            snapshot = SchedulerSnapshot(
                current_time=self.clock,
                outstanding_prefill_chunks=dict(self.outstanding_prefill_chunks),
                total_outstanding_prefill_chunks=self.total_outstanding_prefill_chunks,
            )
            items = self.scheduler.form_batch(self.queues[resource_id], snapshot)
            if not items:
                continue
            selected_ids = {item.id for item in items}
            self.queues[resource_id] = [
                item for item in self.queues[resource_id] if item.id not in selected_ids
            ]
            self._start_batch(resource, items)

    def _start_batch(self, resource: Resource, items: list[WorkItem]) -> None:
        stage = items[0].stage
        if any(item.stage != stage for item in items):
            raise RuntimeError("a batch cannot mix execution stages")
        if stage in GPU_STAGES:
            estimate = self.roofline.estimate(stage.value, items)
        elif stage in NETWORK_STAGES:
            estimate = self.network.estimate(stage, items)
        else:
            raise RuntimeError(f"unsupported stage: {stage}")

        batch_id = self._batch_sequence
        self._batch_sequence += 1
        end_time = self.clock + estimate.total_time_s
        batch = BatchExecution(
            batch_id=batch_id,
            resource_id=resource.resource_id,
            stage=stage,
            items=items,
            estimate=estimate,
            start_time=self.clock,
            end_time=end_time,
        )
        resource.running_batch_id = batch_id
        for item in items:
            queue_time = self.clock - item.ready_time
            request = self.requests[item.request_id]
            if item.phase == Phase.PREFILL:
                request.prefill_queue_time += queue_time
            else:
                request.decode_queue_time += queue_time
            if item.stage == Stage.EDGE_FRONT and item.phase == Phase.PREFILL:
                self.outstanding_prefill_chunks[item.request_id] = (
                    self.outstanding_prefill_chunks.get(item.request_id, 0) + 1
                )
                self.total_outstanding_prefill_chunks += 1
        if self.config.simulation.trace_enabled:
            self.trace.append(self._trace_record(batch))
        self._push_event(end_time, EventType.BATCH_FINISH, batch)

    def _trace_record(self, batch: BatchExecution) -> dict[str, Any]:
        cursor = batch.start_time
        sub_operations: list[dict[str, Any]] = []
        for operation in batch.estimate.sub_operations:
            operation_end = cursor + operation.duration_s
            sub_operations.append(
                {
                    "name": operation.name,
                    "category": operation.category,
                    "start_time_ms": cursor * 1000.0,
                    "end_time_ms": operation_end * 1000.0,
                    "duration_ms": operation.duration_s * 1000.0,
                    "input_shape": operation.input_shape,
                    "dependencies": list(operation.dependencies),
                }
            )
            cursor = operation_end
        return {
            "batch_id": batch.batch_id,
            "resource": batch.resource_id,
            "stage": batch.stage.value,
            "phases": sorted({item.phase.value for item in batch.items}),
            "request_ids": [item.request_id for item in batch.items],
            "work_item_ids": [item.id for item in batch.items],
            "chunk_indices": [item.chunk_index for item in batch.items],
            "decode_iterations": [item.iteration for item in batch.items],
            "dependency_ids": [list(item.dependencies) for item in batch.items],
            "ready_times_ms": [item.ready_time * 1000.0 for item in batch.items],
            "queue_delays_ms": [
                (batch.start_time - item.ready_time) * 1000.0 for item in batch.items
            ],
            "start_time_ms": batch.start_time * 1000.0,
            "end_time_ms": batch.end_time * 1000.0,
            "duration_ms": (batch.end_time - batch.start_time) * 1000.0,
            "batch_size": len(batch.items),
            "total_tokens": sum(item.token_count for item in batch.items),
            "input_shape": batch.estimate.input_shape,
            "sub_operations": sub_operations,
            "flops": batch.estimate.flops,
            "memory_bytes": batch.estimate.memory_bytes,
            "communication_bytes": batch.estimate.communication_bytes,
        }
