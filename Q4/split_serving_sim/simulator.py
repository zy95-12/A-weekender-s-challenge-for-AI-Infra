from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from .config import RequestSpec, SimulationConfig
from .core import GPU_STAGES, NETWORK_STAGES, PerformanceEstimate, Phase, Stage, WorkItem
from .dag import ExecutionDAG
from .metrics import RequestRuntime, build_summary
from .kv_cache import PagedKVCache
from .performance import NetworkModel, RooflineModel
from .scheduler import SchedulerSnapshot, VLLMScheduler


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
    transaction_id: int | None = None


@dataclass(frozen=True)
class SimulationResult:
    summary: dict[str, Any]
    requests: list[dict[str, Any]]
    trace: list[dict[str, Any]]


class Simulator:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.dag = ExecutionDAG(
            {Stage(stage.name): stage.pp_degree for stage in config.stages}
        )
        self.scheduler = VLLMScheduler(config.static_policy, config.scheduler)
        self.roofline = RooflineModel(config)
        self.network = NetworkModel(config)
        resource_names = {
            self._parallel_resource_id(
                stage.resource_for_phase(phase),
                replica,
                stage.replicas,
                pipeline_rank,
                stage.pp_degree,
            )
            for stage in config.stages
            for phase in ("prefill", "decode")
            for replica in range(stage.replicas)
            for pipeline_rank in range(stage.pp_degree)
        }
        resource_names.update({"wan_up", "wan_down"})
        if config.scheduler.pd_disaggregation.enabled:
            resource_names.add("pd_kv_transfer")
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
        self.waiting_request_ids: list[int] = []
        self.active_continuous_requests: set[int] = set()
        self.active_static_cohort: set[int] = set()
        self.request_cohorts: dict[int, int] = {}
        self._cohort_sequence = 0
        self.request_specs: dict[int, RequestSpec] = {}
        self.request_priorities: dict[int, int] = {}
        self.request_arrival_times: dict[int, float] = {}
        self.request_cached_prefix_tokens: dict[int, int] = {}
        self.kv_cache = PagedKVCache(config.scheduler.kv_cache) if config.scheduler.kv_cache.enabled else None
        self.kv_events: list[dict[str, Any]] = []
        self.running_batches: dict[int, BatchExecution] = {}
        self.preempted_request_ids: set[int] = set()
        self.recompute_pending_stages: dict[int, set[Stage]] = {}
        self.sync_signature: dict[int, tuple[Phase, int | None, int | None]] | None = None
        self.sync_expected: tuple[Stage, int] | None = None
        self.sync_transaction_id: int | None = None
        self._sync_transaction_sequence = 0
        self.trace_records_dropped = 0
        self.trace_detail_records_dropped = 0
        self.inflight_transactions: dict[tuple[int, str, int | None, int | None], float] = {}
        self.inflight_bytes = 0.0
        self.max_inflight_transactions_observed = 0
        self.max_inflight_bytes_observed = 0.0
        self._closed_loop_specs: list[RequestSpec] = []
        self._next_closed_loop_index = 0
        self._closed_loop_warmup_completed = 0
        self.measured_request_ids: set[int] = set()

    def run(self) -> SimulationResult:
        request_specs = self._request_specs()
        if not request_specs:
            raise ValueError("workload generated no requests")
        for index, spec in enumerate(request_specs):
            self.request_specs[spec.request_id] = spec
            self.request_priorities[spec.request_id] = spec.priority
            self.request_arrival_times[spec.request_id] = spec.arrival_time_ms / 1000.0
            self.requests[spec.request_id] = RequestRuntime(
                request_id=spec.request_id,
                arrival_time=spec.arrival_time_ms / 1000.0,
                input_tokens=spec.input_tokens,
                output_tokens=spec.output_tokens,
            )
            if index >= self.config.workload.warmup_requests:
                self.measured_request_ids.add(spec.request_id)
        if self.config.workload.mode == "closed_loop":
            self._closed_loop_specs = request_specs
            initial_count = (
                min(self.config.workload.concurrency, self.config.workload.warmup_requests)
                if self.config.workload.warmup_requests
                else self.config.workload.concurrency
            )
            scheduled_specs = request_specs[:initial_count]
            self._next_closed_loop_index = len(scheduled_specs)
        else:
            scheduled_specs = request_specs
        for spec in scheduled_specs:
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
            self._admit_waiting_requests()
            self._schedule_idle_resources()

        unfinished = [
            request.request_id
            for request in self.requests.values()
            if request.finish_time is None
        ]
        if unfinished:
            queued = sum(len(queue) for queue in self.queues.values())
            raise RuntimeError(f"simulation deadlocked; unfinished={unfinished}, queued={queued}")

        request_metrics = []
        for request_id in sorted(self.requests):
            metrics = self.requests[request_id].to_metrics()
            metrics["measured"] = request_id in self.measured_request_ids
            request_metrics.append(metrics)
        measured_metrics = [row for row in request_metrics if row["measured"]]
        end_time = max(request.finish_time or 0.0 for request in self.requests.values())
        measured_start = min(row["arrival_time_ms"] for row in measured_metrics) / 1000.0
        measured_end = max(row["finish_time_ms"] for row in measured_metrics) / 1000.0
        measured_trace = [
            row for row in self.trace
            if row["end_time_ms"] > measured_start * 1000.0
            and row["start_time_ms"] < measured_end * 1000.0
        ]
        summary = build_summary(
            measured_metrics,
            measured_trace,
            {name: resource.busy_time for name, resource in self.resources.items()},
            measured_start,
            measured_end,
            self.config.slo,
        )
        full_duration = max(end_time - start_time, 0.0)
        summary["resource_utilization"] = {
            name: resource.busy_time / full_duration if full_duration else 0.0
            for name, resource in self.resources.items()
        }
        summary["resource_utilization_scope"] = "full_run_including_warmup"
        summary["total_requests_including_warmup"] = len(request_metrics)
        summary["warmup_requests"] = len(request_metrics) - len(measured_metrics)
        summary["static_policy"] = {
            "batch_size": self.config.static_policy.max_batch_size,
            "max_batched_tokens": self.config.static_policy.max_batched_tokens,
            "prefill_token_budget": self.config.static_policy.prefill_token_budget,
            "prefill_chunk_size": self.config.static_policy.prefill_chunk_size,
            "scheduler": self.config.static_policy.scheduler,
            "continuous_batching": self.config.static_policy.continuous_batching,
            "attention_backend": self.config.attention_backend.mode,
            "scheduler_policy": self.config.scheduler.policy,
            "max_num_seqs": self.config.scheduler.max_num_seqs,
            "decode_first": self.config.scheduler.decode_first,
            "max_consecutive_decode_batches": (
                self.config.scheduler.max_consecutive_decode_batches
            ),
            "max_decode_tokens_per_batch": (
                self.config.scheduler.max_decode_tokens_per_batch
            ),
            "max_prefill_wait_ms": self.config.scheduler.max_prefill_wait_ms,
        }
        summary["model"] = {
            "name": self.config.model.name,
            "architecture": self.config.model.architecture,
            "num_layers": self.config.model.num_layers,
            "dtype": self.config.model.dtype,
        }
        summary["execution"] = {
            "mode": self.config.execution.mode,
            "preserve_batch_across_stages": (
                self.config.execution.preserve_batch_across_stages
            ),
            "max_inflight_transactions": self.config.execution.max_inflight_transactions,
            "buffer_pool_mib": self.config.execution.buffer_pool_mib,
            "max_inflight_transactions_observed": self.max_inflight_transactions_observed,
            "max_inflight_bytes_observed": self.max_inflight_bytes_observed,
        }
        summary["data_path"] = {
            "enabled": self.config.data_path.enabled,
            "ipc_mode": self.config.data_path.ipc_mode,
            "wire_fast": self.config.data_path.wire_fast,
            "tcp_buffer_mib": self.config.data_path.tcp_buffer_mib,
        }
        summary["workload"] = {
            "mode": self.config.workload.mode,
            "concurrency": self.config.workload.concurrency,
        }
        summary["performance_profile"] = {
            "enabled": self.config.performance_profile.enabled,
            "samples": len(self.config.performance_profile.samples),
        }
        summary["network"] = {
            "uplink_gbps": self.config.network.uplink_gbps,
            "downlink_gbps": self.config.network.downlink_gbps,
            "rtt_ms": self.config.network.rtt_ms,
            "activation_tensor_count": self.config.network.activation_tensor_count,
            "protocol_overhead_bytes": self.config.network.protocol_overhead_bytes,
        }
        summary["trace"] = {
            "enabled": self.config.simulation.trace_enabled,
            "records": len(self.trace),
            "records_dropped": self.trace_records_dropped,
            "detailed_records": sum(
                bool(row["sub_operations"]) for row in self.trace
            ),
            "detail_records_dropped": self.trace_detail_records_dropped,
            "max_trace_records": self.config.simulation.max_trace_records,
            "max_detailed_trace_records": (
                self.config.simulation.max_detailed_trace_records
            ),
        }
        if self.kv_cache is not None:
            summary["kv_cache"] = {
                "capacity_blocks": self.kv_cache.config.num_blocks,
                "allocation_mode": self.kv_cache.config.allocation_mode,
                "used_blocks_at_end": self.kv_cache.used_blocks,
                "events": self.kv_events,
            }
        summary["parallel_plan"] = [
            {
                "stage": stage.name,
                "layer_range": [stage.layer_start, stage.layer_end],
                "tp_degree": stage.tp_degree,
                "pp_degree": stage.pp_degree,
                "pp_layer_ranges": [
                    list(stage.pipeline_layer_range(rank))
                    for rank in range(stage.pp_degree)
                ],
                "replicas": stage.replicas,
                "devices": stage.tp_degree * stage.pp_degree * stage.replicas,
            }
            for stage in self.config.stages
        ]
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
                    arrival_time_ms=(0.0 if workload.mode == "closed_loop" else arrival_time_ms),
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

    def _launch_closed_loop_spec(self, spec: RequestSpec) -> None:
        self.requests[spec.request_id].arrival_time = self.clock
        self.request_arrival_times[spec.request_id] = self.clock
        self._push_event(self.clock, EventType.REQUEST_ARRIVAL, spec.request_id)

    def _process_event(self, event: Event) -> None:
        if event.event_type == EventType.REQUEST_ARRIVAL:
            self.waiting_request_ids.append(event.payload)
            return
        if event.event_type == EventType.BATCH_FINISH:
            self._finish_batch(event.payload)
            return
        raise RuntimeError(f"unknown event: {event.event_type}")

    def _start_request(self, request_id: int) -> None:
        request = self.requests[request_id]
        spec = self.request_specs[request_id]
        cached = self.kv_cache.cached_tokens(spec.prefix_id, spec.prefix_tokens) if self.kv_cache else 0
        self.request_cached_prefix_tokens[request_id] = cached
        ready = self.dag.add_prefill(
            request.request_id,
            request.input_tokens,
            self.config.static_policy.prefill_chunk_size if self.config.scheduler.enable_chunked_prefill else request.input_tokens,
            self.clock,
            token_offset=cached,
        )
        self._enqueue(ready)

    def _admit_waiting_requests(self) -> None:
        if self.config.execution.mode == "synchronous_rpc" and self.sync_signature:
            return
        if self.config.static_policy.continuous_batching:
            available_slots = (
                self.config.scheduler.max_num_seqs
                - len(self.active_continuous_requests)
            )
            ordered = self._ordered_waiting_requests()
            # A full sequence table must not prevent a higher-priority request
            # from reaching the safe-point preemption path.  This mirrors the
            # running-first behavior: work already executing finishes, then a
            # lower-priority sequence can yield its slot and KV allocation.
            while (
                ordered
                and available_slots <= 0
                and self._preempt_for(ordered[0])
            ):
                available_slots = (
                    self.config.scheduler.max_num_seqs
                    - len(self.active_continuous_requests)
                )
                ordered = self._ordered_waiting_requests()
            admitted = []
            for request_id in ordered:
                admission_limit = (
                    min(1, available_slots)
                    if self.config.scheduler.policy == "split_poc_naive"
                    else available_slots
                )
                if len(admitted) >= admission_limit:
                    break
                if self._reserve_kv(request_id):
                    admitted.append(request_id)
            self.waiting_request_ids = [request_id for request_id in self.waiting_request_ids if request_id not in set(admitted)]
            for request_id in admitted:
                self.active_continuous_requests.add(request_id)
                if request_id in self.preempted_request_ids:
                    self.preempted_request_ids.remove(request_id)
                else:
                    self._start_request(request_id)
            return
        if self.active_static_cohort or not self.waiting_request_ids:
            return
        cohort = []
        for request_id in self._ordered_waiting_requests():
            if len(cohort) >= self.config.static_policy.max_batch_size:
                break
            if self._reserve_kv(request_id):
                cohort.append(request_id)
        self.waiting_request_ids = [request_id for request_id in self.waiting_request_ids if request_id not in set(cohort)]
        cohort_id = self._cohort_sequence
        self._cohort_sequence += 1
        self.active_static_cohort = set(cohort)
        for request_id in cohort:
            self.request_cohorts[request_id] = cohort_id
            self._start_request(request_id)

    def _ordered_waiting_requests(self) -> list[int]:
        if self.config.scheduler.policy == "priority":
            return sorted(self.waiting_request_ids, key=lambda request_id: (self.request_priorities[request_id], self.request_arrival_times[request_id], request_id))
        if self.config.scheduler.policy == "shortest_prefill":
            return sorted(self.waiting_request_ids, key=lambda request_id: (self.requests[request_id].input_tokens, self.request_arrival_times[request_id], request_id))
        return list(self.waiting_request_ids)

    def _reserve_kv(self, request_id: int) -> bool:
        if self.kv_cache is None or request_id in self.kv_cache.allocations:
            return True
        spec = self.request_specs[request_id]
        reserved_tokens = (
            0
            if self.config.scheduler.kv_cache.allocation_mode == "on_demand"
            else spec.input_tokens + spec.output_tokens
        )
        admission = self.kv_cache.reserve(request_id, reserved_tokens, prefix_id=spec.prefix_id, prefix_tokens=spec.prefix_tokens)
        while admission is None and self._preempt_for(request_id):
            admission = self.kv_cache.reserve(request_id, reserved_tokens, prefix_id=spec.prefix_id, prefix_tokens=spec.prefix_tokens)
        if admission is None:
            return False
        self.kv_events.append({"time_ms": self.clock * 1000.0, "event": "allocate", "request_id": request_id, "blocks": admission.allocated_blocks, "prefix_hit_blocks": admission.prefix_hit_blocks, "evicted_prefix_blocks": admission.evicted_prefix_blocks})
        return True

    def _preempt_for(self, incoming_id: int) -> bool:
        if self.kv_cache is None or not self.config.scheduler.kv_cache.enable_preemption or self.config.scheduler.policy != "priority":
            return False
        busy = {item.request_id for batch in self.running_batches.values() for item in batch.items}
        incoming_priority = self.request_priorities[incoming_id]
        victims = []
        for request_id in self.active_continuous_requests:
            queued = [item for queue in self.queues.values() for item in queue if item.request_id == request_id]
            if request_id not in busy and queued and all(item.stage == Stage.EDGE_FRONT for item in queued) and self.outstanding_prefill_chunks.get(request_id, 0) == 0 and self.request_priorities[request_id] > incoming_priority:
                victims.append(request_id)
        if not victims:
            return False
        victim = max(victims, key=lambda request_id: (self.request_priorities[request_id], self.request_arrival_times[request_id]))
        freed = self.kv_cache.free_request(victim)
        self.active_continuous_requests.remove(victim)
        if victim not in self.waiting_request_ids:
            self.waiting_request_ids.append(victim)
        self.preempted_request_ids.add(victim)
        self.recompute_pending_stages[victim] = set(GPU_STAGES)
        self.kv_events.append({"time_ms": self.clock * 1000.0, "event": "preempt_recompute", "request_id": victim, "for_request_id": incoming_id, "freed_blocks": freed})
        return freed > 0

    def _finish_batch(self, batch: BatchExecution) -> None:
        resource = self.resources[batch.resource_id]
        if resource.running_batch_id != batch.batch_id:
            raise RuntimeError(f"resource/batch mismatch: {batch.resource_id}")
        resource.running_batch_id = None
        self.running_batches.pop(batch.batch_id, None)
        resource.busy_time += batch.end_time - batch.start_time

        for item in batch.items:
            ready = self.dag.mark_complete(item.id, self.clock)
            self._enqueue(ready)
            if (
                item.phase == Phase.PREFILL
                and item.stage == Stage.EDGE_TAIL
                and item.pipeline_rank
                == self.config.stage(Stage.EDGE_TAIL.value).pp_degree - 1
            ):
                self.outstanding_prefill_chunks[item.request_id] -= 1
                self.total_outstanding_prefill_chunks -= 1
            if (
                item.stage == Stage.EDGE_TAIL
                and item.pipeline_rank
                == self.config.stage(Stage.EDGE_TAIL.value).pp_degree - 1
            ):
                self._release_transaction(item)
            if (
                item.stage == Stage.EDGE_TAIL
                and item.pipeline_rank
                == self.config.stage(Stage.EDGE_TAIL.value).pp_degree - 1
            ):
                if item.phase == Phase.PREFILL and self.dag.is_final_prefill(item):
                    if self.kv_cache is not None:
                        spec = self.request_specs[item.request_id]
                        blocks = self.kv_cache.publish_prefix(item.request_id, spec.prefix_id, spec.prefix_tokens)
                        if blocks:
                            self.kv_events.append({"time_ms": self.clock * 1000.0, "event": "publish_prefix", "request_id": item.request_id, "prefix_id": spec.prefix_id, "blocks": blocks})
                    if self.config.scheduler.pd_disaggregation.enabled:
                        self._enqueue(self.dag.add_pd_kv_transfer(item.request_id, self.requests[item.request_id].input_tokens, self.clock, item.id))
                    else:
                        self._token_ready(item.request_id)
                elif item.phase == Phase.DECODE:
                    self._token_ready(item.request_id)
            elif item.stage == Stage.PD_KV_TRANSFER:
                self._token_ready(item.request_id)
        self._advance_sync_transaction(batch)

    def _token_ready(self, request_id: int) -> None:
        request = self.requests[request_id]
        request.token_times.append(self.clock)
        if request.first_token_time is None:
            request.first_token_time = self.clock
        if len(request.token_times) >= request.output_tokens:
            request.finish_time = self.clock
            if self.kv_cache is not None:
                freed = self.kv_cache.free_request(request_id)
                self.kv_events.append({"time_ms": self.clock * 1000.0, "event": "free", "request_id": request_id, "blocks": freed})
            if self.config.static_policy.continuous_batching:
                self.active_continuous_requests.remove(request_id)
            if (
                not self.config.static_policy.continuous_batching
                and self.active_static_cohort
                and all(
                    self.requests[active_id].finish_time is not None
                    for active_id in self.active_static_cohort
                )
            ):
                self.active_static_cohort.clear()
            if (
                self.config.workload.mode == "closed_loop"
            ):
                warmup = self.config.workload.warmup_requests
                if request_id not in self.measured_request_ids:
                    self._closed_loop_warmup_completed += 1
                    if self._next_closed_loop_index < warmup:
                        spec = self._closed_loop_specs[self._next_closed_loop_index]
                        self._next_closed_loop_index += 1
                        self._launch_closed_loop_spec(spec)
                    elif self._closed_loop_warmup_completed == warmup:
                        stop = min(
                            len(self._closed_loop_specs),
                            self._next_closed_loop_index
                            + self.config.workload.concurrency,
                        )
                        while self._next_closed_loop_index < stop:
                            spec = self._closed_loop_specs[self._next_closed_loop_index]
                            self._next_closed_loop_index += 1
                            self._launch_closed_loop_spec(spec)
                elif self._next_closed_loop_index < len(self._closed_loop_specs):
                    spec = self._closed_loop_specs[self._next_closed_loop_index]
                    self._next_closed_loop_index += 1
                    self._launch_closed_loop_spec(spec)
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
    def _parallel_resource_id(
        resource: str,
        replica: int,
        replicas: int,
        pipeline_rank: int,
        pp_degree: int,
    ) -> str:
        parts = [resource]
        if replicas > 1:
            parts.append(f"replica_{replica}")
        if pp_degree > 1:
            parts.append(f"pp_{pipeline_rank}")
        return "/".join(parts)

    def _resource_for_item(self, item: WorkItem) -> str:
        if item.stage == Stage.WAN_UP:
            return "wan_up"
        if item.stage == Stage.WAN_DOWN:
            return "wan_down"
        if item.stage == Stage.PD_KV_TRANSFER:
            return "pd_kv_transfer"
        stage = self.config.stage(item.stage.value)
        # Naive but KV-safe routing: a request remains sticky to one replica.
        replica = item.request_id % stage.replicas
        return self._parallel_resource_id(
            stage.resource_for_phase(item.phase.value),
            replica,
            stage.replicas,
            item.pipeline_rank,
            stage.pp_degree,
        )

    @staticmethod
    def _transaction_key(item: WorkItem) -> tuple[int, str, int | None, int | None]:
        return (item.request_id, item.phase.value, item.chunk_index, item.iteration)

    def _transaction_bytes(self, item: WorkItem) -> float:
        one_way = (
            item.token_count
            * self.config.model.hidden_size
            * self.config.model.dtype_bytes
            * self.config.network.activation_tensor_count
            + self.config.network.protocol_overhead_bytes
        )
        return float(2 * one_way)

    def _can_admit_front_batch(self, items: list[WorkItem]) -> bool:
        if items[0].stage != Stage.EDGE_FRONT or items[0].pipeline_rank != 0:
            return True
        new_items = [
            item for item in items
            if self._transaction_key(item) not in self.inflight_transactions
        ]
        if not new_items:
            return True
        limit = self.config.execution.max_inflight_transactions
        if limit and len(self.inflight_transactions) + len(new_items) > limit:
            return False
        additional_bytes = sum(self._transaction_bytes(item) for item in new_items)
        capacity = self.config.execution.buffer_pool_mib * 1024.0 * 1024.0
        if capacity and additional_bytes > capacity:
            raise RuntimeError(
                "one batch requires more bytes than execution.buffer_pool_mib"
            )
        return not capacity or self.inflight_bytes + additional_bytes <= capacity

    def _admissible_front_candidates(self, items: list[WorkItem]) -> list[WorkItem]:
        selected: list[WorkItem] = []
        added_keys: set[tuple[int, str, int | None, int | None]] = set()
        added_bytes = 0.0
        limit = self.config.execution.max_inflight_transactions
        capacity = self.config.execution.buffer_pool_mib * 1024.0 * 1024.0
        for item in items:
            if item.stage != Stage.EDGE_FRONT or item.pipeline_rank != 0:
                selected.append(item)
                continue
            key = self._transaction_key(item)
            if key in self.inflight_transactions or key in added_keys:
                selected.append(item)
                continue
            size = self._transaction_bytes(item)
            if capacity and size > capacity:
                raise RuntimeError(
                    "one transaction requires more bytes than execution.buffer_pool_mib"
                )
            if limit and len(self.inflight_transactions) + len(added_keys) >= limit:
                continue
            if capacity and self.inflight_bytes + added_bytes + size > capacity:
                continue
            selected.append(item)
            added_keys.add(key)
            added_bytes += size
        return selected

    def _reserve_front_batch(self, items: list[WorkItem]) -> None:
        if items[0].stage != Stage.EDGE_FRONT or items[0].pipeline_rank != 0:
            return
        for item in items:
            key = self._transaction_key(item)
            if key in self.inflight_transactions:
                continue
            size = self._transaction_bytes(item)
            self.inflight_transactions[key] = size
            self.inflight_bytes += size
        self.max_inflight_transactions_observed = max(
            self.max_inflight_transactions_observed, len(self.inflight_transactions)
        )
        self.max_inflight_bytes_observed = max(
            self.max_inflight_bytes_observed, self.inflight_bytes
        )

    def _release_transaction(self, item: WorkItem) -> None:
        key = self._transaction_key(item)
        size = self.inflight_transactions.pop(key, 0.0)
        self.inflight_bytes -= size

    def _schedule_idle_resources(self) -> None:
        if self.config.execution.mode == "synchronous_rpc" and self.running_batches:
            return
        for resource_id in sorted(self.resources):
            resource = self.resources[resource_id]
            candidates = self.queues[resource_id]
            candidates = self._admissible_front_candidates(candidates)
            if self.sync_signature is not None:
                expected = self.sync_expected
                candidates = [
                    item
                    for item in candidates
                    if expected == (item.stage, item.pipeline_rank)
                    and self.sync_signature.get(item.request_id)
                    == (item.phase, item.chunk_index, item.iteration)
                ]
            if not resource.idle or not candidates:
                continue
            snapshot = SchedulerSnapshot(
                current_time=self.clock,
                outstanding_prefill_chunks=dict(self.outstanding_prefill_chunks),
                total_outstanding_prefill_chunks=self.total_outstanding_prefill_chunks,
                running_request_ids=frozenset(self.active_continuous_requests | self.active_static_cohort),
                request_priorities=self.request_priorities,
                request_arrival_times=self.request_arrival_times,
                request_input_tokens={request_id: spec.input_tokens for request_id, spec in self.request_specs.items()},
            )
            items = self.scheduler.form_batch(candidates, snapshot)
            if not items:
                continue
            if not self._can_admit_front_batch(items):
                continue
            if not self._grow_kv_for_batch(items):
                continue
            selected_ids = {item.id for item in items}
            self.queues[resource_id] = [
                item for item in self.queues[resource_id] if item.id not in selected_ids
            ]
            self._start_batch(resource, items)
            if self.config.execution.mode == "synchronous_rpc":
                break

    def _grow_kv_for_batch(self, items: list[WorkItem]) -> bool:
        if (
            self.kv_cache is None
            or self.config.scheduler.kv_cache.allocation_mode != "on_demand"
            or items[0].stage != Stage.EDGE_FRONT
            or items[0].pipeline_rank != 0
        ):
            return True
        targets: dict[int, int] = {}
        for item in items:
            targets[item.request_id] = max(
                targets.get(item.request_id, 0), item.context_tokens
            )
        deltas = self.kv_cache.grow_batch(targets)
        if deltas is None:
            return False
        for request_id, blocks in deltas.items():
            if blocks:
                self.kv_events.append(
                    {
                        "time_ms": self.clock * 1000.0,
                        "event": "grow",
                        "request_id": request_id,
                        "blocks": blocks,
                        "target_tokens": targets[request_id],
                    }
                )
        return True

    def _start_batch(self, resource: Resource, items: list[WorkItem]) -> None:
        adjusted = []
        for item in items:
            pending = self.recompute_pending_stages.get(item.request_id)
            if pending and item.stage in pending and item.stage in GPU_STAGES:
                item = replace(item, recompute_tokens=item.context_tokens)
                pending.remove(item.stage)
                if not pending:
                    self.recompute_pending_stages.pop(item.request_id, None)
            adjusted.append(item)
        items = adjusted
        stage = items[0].stage
        if any(item.stage != stage for item in items):
            raise RuntimeError("a batch cannot mix execution stages")
        if any(item.pipeline_rank != items[0].pipeline_rank for item in items):
            raise RuntimeError("a batch cannot mix pipeline ranks")
        if self.config.execution.mode == "synchronous_rpc" and self.sync_signature is None:
            if stage != Stage.EDGE_FRONT or items[0].pipeline_rank != 0:
                raise RuntimeError("synchronous RPC transaction must start at edge_front")
            self.sync_signature = {
                item.request_id: (item.phase, item.chunk_index, item.iteration)
                for item in items
            }
            self.sync_expected = (stage, items[0].pipeline_rank)
            self.sync_transaction_id = self._sync_transaction_sequence
            self._sync_transaction_sequence += 1
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
            transaction_id=self.sync_transaction_id,
        )
        resource.running_batch_id = batch_id
        self.running_batches[batch_id] = batch
        self._reserve_front_batch(items)
        for item in items:
            queue_time = self.clock - item.ready_time
            request = self.requests[item.request_id]
            if item.phase == Phase.PREFILL:
                request.prefill_queue_time += queue_time
            else:
                request.decode_queue_time += queue_time
            if (
                item.stage == Stage.EDGE_FRONT
                and item.phase == Phase.PREFILL
                and item.pipeline_rank == 0
            ):
                self.outstanding_prefill_chunks[item.request_id] = (
                    self.outstanding_prefill_chunks.get(item.request_id, 0) + 1
                )
                self.total_outstanding_prefill_chunks += 1
        if self.config.simulation.trace_enabled:
            if len(self.trace) < self.config.simulation.max_trace_records:
                include_details = (
                    len(self.trace)
                    < self.config.simulation.max_detailed_trace_records
                )
                self.trace.append(
                    self._trace_record(batch, include_sub_operations=include_details)
                )
                if not include_details and batch.estimate.sub_operations:
                    self.trace_detail_records_dropped += 1
            else:
                self.trace_records_dropped += 1
                if batch.estimate.sub_operations:
                    self.trace_detail_records_dropped += 1
        self._push_event(end_time, EventType.BATCH_FINISH, batch)

    def _advance_sync_transaction(self, batch: BatchExecution) -> None:
        if self.config.execution.mode != "synchronous_rpc":
            return
        rank = batch.items[0].pipeline_rank
        if batch.stage in GPU_STAGES:
            degree = self.config.stage(batch.stage.value).pp_degree
            if rank + 1 < degree:
                self.sync_expected = (batch.stage, rank + 1)
                return
        next_stage = {
            Stage.EDGE_FRONT: Stage.WAN_UP,
            Stage.WAN_UP: Stage.CLOUD_MIDDLE,
            Stage.CLOUD_MIDDLE: Stage.WAN_DOWN,
            Stage.WAN_DOWN: Stage.EDGE_TAIL,
        }.get(batch.stage)
        if next_stage is not None:
            self.sync_expected = (next_stage, 0)
            return
        if batch.stage == Stage.EDGE_TAIL:
            pd_requests = {
                item.request_id
                for item in batch.items
                if item.phase == Phase.PREFILL and self.dag.is_final_prefill(item)
            }
            if self.config.scheduler.pd_disaggregation.enabled and pd_requests:
                self.sync_signature = {
                    request_id: (Phase.PREFILL, None, None)
                    for request_id in pd_requests
                }
                self.sync_expected = (Stage.PD_KV_TRANSFER, 0)
                return
            self.sync_signature = None
            self.sync_expected = None
            self.sync_transaction_id = None
            return
        if batch.stage == Stage.PD_KV_TRANSFER:
            self.sync_signature = None
            self.sync_expected = None
            self.sync_transaction_id = None

    def _trace_record(
        self, batch: BatchExecution, *, include_sub_operations: bool = True
    ) -> dict[str, Any]:
        cursor = batch.start_time
        sub_operations: list[dict[str, Any]] = []
        operations = (
            batch.estimate.sub_operations if include_sub_operations else ()
        )
        for operation in operations:
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
                    "profile_type": operation.profile_type,
                    "profile_signature": operation.profile_signature,
                    "profile_source": operation.profile_source,
                    "profile_correction_factor": (
                        operation.profile_correction_factor
                    ),
                }
            )
            cursor = operation_end
        return {
            "batch_id": batch.batch_id,
            "transaction_id": batch.transaction_id,
            "resource": batch.resource_id,
            "stage": batch.stage.value,
            "pipeline_rank": batch.items[0].pipeline_rank,
            "layer_range": list(
                self.config.stage(batch.stage.value).pipeline_layer_range(
                    batch.items[0].pipeline_rank
                )
            ) if batch.stage in GPU_STAGES else None,
            "phases": sorted({item.phase.value for item in batch.items}),
            "request_ids": [item.request_id for item in batch.items],
            "prefill_request_ids": sorted({item.request_id for item in batch.items if item.phase == Phase.PREFILL}),
            "decode_request_ids": sorted({item.request_id for item in batch.items if item.phase == Phase.DECODE}),
            "cohort_ids": [
                self.request_cohorts.get(item.request_id) for item in batch.items
            ],
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
            "recompute_tokens": sum(item.recompute_tokens for item in batch.items),
            "input_shape": batch.estimate.input_shape,
            "sub_operations": sub_operations,
            "sub_operations_omitted": bool(
                batch.estimate.sub_operations and not include_sub_operations
            ),
            "flops": batch.estimate.flops,
            "memory_bytes": batch.estimate.memory_bytes,
            "communication_bytes": batch.estimate.communication_bytes,
            "attention_backend": self.config.attention_backend.mode,
            "scheduler_policy": self.config.scheduler.policy,
            "execution_mode": self.config.execution.mode,
            "kv_blocks_used": self.kv_cache.used_blocks if self.kv_cache else None,
            "kv_blocks_free": self.kv_cache.free_blocks if self.kv_cache else None,
            "kv_allocation_mode": (
                self.config.scheduler.kv_cache.allocation_mode
                if self.kv_cache else None
            ),
            "inflight_transactions": len(self.inflight_transactions),
            "inflight_bytes": self.inflight_bytes,
        }
