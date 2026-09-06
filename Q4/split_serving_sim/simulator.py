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
from . import open_loop
from .kv_cache import PagedKVCache
from .performance import NetworkModel, RooflineModel
from .scheduler import SchedulerSnapshot, VLLMScheduler, PDScheduler
from .host_submission import schedule_submissions


class EventType(str, Enum):
    REQUEST_ARRIVAL = "request_arrival"
    BATCH_FINISH = "batch_finish"
    PD_LIFECYCLE = "pd_lifecycle"


@dataclass(order=True)
class Event:
    timestamp: float
    sequence_no: int
    event_type: EventType = field(compare=False)
    payload: Any = field(compare=False)


@dataclass
class Resource:
    resource_id: str
    capacity: int = 1
    running_batch_ids: set[int] = field(default_factory=set)
    busy_time: float = 0.0

    @property
    def idle(self) -> bool:
        return len(self.running_batch_ids) < self.capacity


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
    resource_busy_time: float | None = None
    sub_operation_intervals: tuple[tuple[float, float], ...] = ()
    host_submission_intervals: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class SimulationResult:
    summary: dict[str, Any]
    requests: list[dict[str, Any]]
    trace: list[dict[str, Any]]


class Simulator:
    lifecycle_event_type = EventType.PD_LIFECYCLE

    def __init__(self, config: SimulationConfig, gpu_cost_model=None):
        self.config = config
        self.dag = ExecutionDAG(
            {Stage(stage.name): stage.pp_degree for stage in config.stages}
        )
        self.scheduler = (PDScheduler if config.scheduler.policy == "split_poc_pd" else VLLMScheduler)(config.static_policy, config.scheduler)
        self.roofline = gpu_cost_model or RooflineModel(config)
        self.command_costs = bool(getattr(self.roofline, "includes_host_staging", False))
        if self.command_costs:
            from .command_cost import CommandNetworkModel
            if config.data_path.wan_calibration.enabled:
                raise ValueError("command costs require uncalibrated WAN to avoid double-counting staging")
            self.network = CommandNetworkModel(config)
        else:
            self.network = NetworkModel(config)
        resource_names = {
            self._parallel_resource_id(
                stage.resource_for_phase(phase),
                replica,
                stage.replicas,
                pipeline_rank,
                stage.pp_degree,
            )
            for original in config.stages
            for phase in ("prefill", "decode")
            for stage in (original.for_phase(phase),)
            for replica in range(stage.replicas)
            for pipeline_rank in range(stage.pp_degree)
        }
        resource_names.update({"wan_up", "wan_down"})
        if config.scheduler.pd_disaggregation.enabled:
            resource_names.add("pd_kv_transfer")
            if config.scheduler.pd_disaggregation.control_channel:
                replicas = config.stage("cloud_middle").for_phase("prefill").replicas
                resource_names.add("pd_control_decode")
                resource_names.update(self._parallel_resource_id("pd_control_prefill", r, replicas, 0, 1)
                                      for r in range(replicas))
        self.resources = {
            name: Resource(
                name,
                capacity=(
                    config.static_policy.pipeline_depth
                    if name in {"wan_up", "wan_down"}
                    else 1
                ),
            )
            for name in sorted(resource_names)
        }
        self.pd_routes: dict[int, int] = {}
        self.pd_remaining: dict[int, int] = {}
        self.pd_next_replica = 0
        self.pd_last_steps: dict[int, float] = {}
        self.pd_ready: set[int] = set()
        self.pd_waiting_decode: set[int] = set()
        self.network_link_available = {"wan_up": 0.0, "wan_down": 0.0}
        self.host_available: dict[str, float] = {}
        self.host_totals: dict[str, dict[str, float]] = {}
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
        self._dispatch_group_sequence = 0
        self.work_item_dispatch_groups: dict[int, int] = {}
        self.trace_records_dropped = 0
        self.trace_detail_records_dropped = 0
        self.inflight_transactions: dict[int, float] = {}
        self.inflight_transaction_members: dict[
            int, set[tuple[int, str, int | None, int | None]]
        ] = {}
        self.inflight_bytes = 0.0
        self.max_inflight_transactions_observed = 0
        self.max_inflight_bytes_observed = 0.0
        self._closed_loop_specs: list[RequestSpec] = []
        self._next_closed_loop_index = 0
        self._closed_loop_warmup_completed = 0
        self.measured_request_ids: set[int] = set()
        self.launched_request_ids: set[int] = set()
        self._measurement_start_time: float | None = None
        self._measurement_end_time: float | None = None
        from .pd_admission import PDAdmission
        self.pd_admission = PDAdmission(self) if config.scheduler.pd_disaggregation.admission_enabled else None

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
            if not self.config.workload.warmup_requests:
                self._measurement_start_time = 0.0
                if self.config.workload.measurement_duration_s:
                    self._measurement_end_time = (
                        self.config.workload.measurement_duration_s
                    )
        else:
            scheduled_specs = request_specs
        for spec in scheduled_specs:
            self.launched_request_ids.add(spec.request_id)
            self._push_event(
                spec.arrival_time_ms / 1000.0,
                EventType.REQUEST_ARRIVAL,
                spec.request_id,
            )

        start_time = min(
            self.requests[request_id].arrival_time
            for request_id in self.launched_request_ids
        )
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

        if self.pd_admission and (
            self.pd_admission.reserving or self.pd_admission.releasing
            or self.pd_admission.gateway_busy or any(self.pd_admission.queues.values())):
            raise RuntimeError("simulation ended before PD lifecycle drained")
        unfinished = [
            request.request_id
            for request_id, request in self.requests.items()
            if request_id in self.launched_request_ids
            if request.finish_time is None
        ]
        if unfinished:
            queued = sum(len(queue) for queue in self.queues.values())
            raise RuntimeError(f"simulation deadlocked; unfinished={unfinished}, queued={queued}")

        request_metrics = []
        for request_id in sorted(self.launched_request_ids):
            metrics = self.requests[request_id].to_metrics()
            request_metrics.append(metrics)
        end_time = max(
            self.requests[request_id].finish_time or 0.0
            for request_id in self.launched_request_ids
        )
        if self.config.workload.mode == "open_loop":
            measured_start = self.config.workload.warmup_duration_s
            measured_end = measured_start + self.config.workload.measurement_duration_s
            selected_ids = {r["request_id"] for r in request_metrics if measured_start*1000 <= r["arrival_time_ms"] < measured_end*1000}
        elif self.config.workload.measurement_duration_s:
            if self._measurement_start_time is None or self._measurement_end_time is None:
                raise RuntimeError("closed-loop measurement window never started")
            measured_start = self._measurement_start_time
            measured_end = self._measurement_end_time
            if end_time < measured_end:
                raise RuntimeError(
                    "closed-loop request cap exhausted before measurement window ended; "
                    "increase workload.num_requests"
                )
            selected_ids = {
                row["request_id"]
                for row in request_metrics
                if row["request_id"] in self.measured_request_ids
                and measured_start <= row["finish_time_ms"] / 1000.0 <= measured_end
            }
        else:
            selected_ids = self.measured_request_ids & self.launched_request_ids
            selected_rows = [row for row in request_metrics if row["request_id"] in selected_ids]
            measured_start = min(row["arrival_time_ms"] for row in selected_rows) / 1000.0
            measured_end = max(row["finish_time_ms"] for row in selected_rows) / 1000.0
        for row in request_metrics:
            row["measured"] = row["request_id"] in selected_ids
        measured_metrics = [row for row in request_metrics if row["measured"]]
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
        if self.config.workload.mode == "open_loop":
            summary = open_loop.summary(request_metrics, measured_trace, measured_start, measured_end, self.config.slo, self.config.workload)
        full_duration = max(end_time - start_time, 0.0)
        summary["resource_utilization"] = {
            name: resource.busy_time / full_duration if full_duration else 0.0
            for name, resource in self.resources.items()
        }
        summary["resource_utilization_scope"] = "full_run_including_warmup"
        summary["total_requests_including_warmup_and_drain"] = len(request_metrics)
        launched_warmup = (sum(r["arrival_time_ms"] < measured_start*1000 for r in request_metrics)
                           if self.config.workload.mode == "open_loop" else
                           min(self.config.workload.warmup_requests, len(self.launched_request_ids)))
        summary["warmup_requests"] = launched_warmup
        summary["drain_requests_excluded"] = max(
            len(request_metrics) - len(measured_metrics) - launched_warmup, 0
        )
        summary["measurement_window"] = {
            "mode": (
                "fixed_open_loop" if self.config.workload.mode == "open_loop" else
                "fixed_duration_excluding_warmup_and_drain"
                if self.config.workload.measurement_duration_s
                else "finite_requests_including_final_drain"
            ),
            "start_ms": measured_start * 1000.0,
            "end_ms": measured_end * 1000.0,
            "duration_ms": (measured_end - measured_start) * 1000.0,
        }
        summary["static_policy"] = {
            "batch_size": self.config.static_policy.max_batch_size,
            "max_batched_tokens": self.config.static_policy.max_batched_tokens,
            "prefill_token_budget": self.config.static_policy.prefill_token_budget,
            "prefill_chunk_size": self.config.static_policy.prefill_chunk_size,
            "scheduler": self.config.static_policy.scheduler,
            "continuous_batching": self.config.static_policy.continuous_batching,
            "attention_backend": self.config.attention_backend.mode,
            "operator_backend": self.config.operator_backend.name,
            "scheduler_policy": self.config.scheduler.policy,
            "max_num_seqs": self.config.scheduler.max_num_seqs,
            "decode_first": self.config.scheduler.decode_first,
            "allow_mixed_batch": self.config.scheduler.allow_mixed_batch,
            "prefer_ready_back": self.config.scheduler.prefer_ready_back,
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
            "operator_backend": self.config.operator_backend.name,
            "operator_backend_version": self.config.operator_backend.version,
        }
        summary["execution"] = {
            "host_submission": {
                "enabled": self.config.execution.host_submission.enabled and not self.command_costs,
                "configured_enabled": self.config.execution.host_submission.enabled,
                "submit_us": self.config.execution.host_submission.submit_us,
                "submit_us_by_type": dict(self.config.execution.host_submission.submit_us_by_type),
                "resource_totals": self.host_totals,
                "scope": "full_run_including_warmup_and_drain",
                "gpu_model": "one ordered stream and one host lane per TP resource group",
            },
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
            "wan_calibration": {
                "enabled": self.config.data_path.wan_calibration.enabled,
                "variant": self.config.data_path.wan_calibration.variant,
                "source": self.config.data_path.wan_calibration.source,
                "interpolation": "piecewise_linear_with_analytical_fallback",
            },
        }
        summary["workload"] = {
            "mode": self.config.workload.mode,
            "concurrency": self.config.workload.concurrency,
        }
        summary["pd_admission"] = {
            "enabled": self.pd_admission is not None,
            "events": self.pd_admission.events if self.pd_admission else [],
            "command_lane": "shared_with_forward",
        }
        summary["performance_profile"] = {
            "enabled": self.config.performance_profile.enabled,
            "samples": len(self.config.performance_profile.samples),
            "operator_estimate_counts": self.roofline.coverage,
            "coverage_scope": "command batches" if self.command_costs else "all simulated GPU operator estimates, including warmup",
            "backend": "measured_command" if self.command_costs else "operator_profile_or_roofline",
            "busy_time_scope": "serialized command wall time, including CPU and copies" if self.command_costs else "modeled GPU execution",

        }
        summary["network"] = {
            "uplink_gbps": self.config.network.uplink_gbps,
            "downlink_gbps": self.config.network.downlink_gbps,
            "rtt_ms": self.config.network.rtt_ms,
            "activation_tensor_count": self.config.network.activation_tensor_count,
            "protocol_overhead_bytes": self.config.network.protocol_overhead_bytes,
            "rpc_window": self.config.static_policy.pipeline_depth,
            "link_scheduling": "serialized_bytes_overlapped_propagation",
            "utilization_scope": "wan serialization only",
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
        if workload.mode == "open_loop":
            return open_loop.request_specs(workload)
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
        self.launched_request_ids.add(spec.request_id)
        self.requests[spec.request_id].arrival_time = self.clock
        self.request_arrival_times[spec.request_id] = self.clock
        self._push_event(self.clock, EventType.REQUEST_ARRIVAL, spec.request_id)

    def _process_event(self, event: Event) -> None:
        if event.event_type == EventType.PD_LIFECYCLE:
            event.payload()
            return
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
        if self.config.scheduler.policy == "split_poc_pd":
            replicas = self.config.stage("cloud_middle").for_phase("prefill").replicas
            loads = [sum(self.pd_remaining.get(rid, 0) for rid, route in self.pd_routes.items()
                         if route == replica) for replica in range(replicas)]
            route = min(range(replicas), key=lambda r: (loads[r], (r - self.pd_next_replica) % replicas))
            self.pd_routes[request_id] = route
            self.pd_remaining[request_id] = request.input_tokens
            self.pd_next_replica = (route + 1) % replicas
        if self.pd_admission is not None:
            self.pd_admission.reserve(request_id)
            return
        self._start_prefill(request_id)

    def _start_prefill(self, request_id: int) -> None:
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
        if self.pd_admission:
            enterprise = self.config.stage('edge_front').resource
            if not self.resources[enterprise].idle:
                return
            self.pd_admission.poll()
        if self.config.execution.mode == "synchronous_rpc" and self.sync_signature:
            return
        if self.config.static_policy.continuous_batching:
            available_slots = (
                self.config.scheduler.max_num_seqs
                - len(self.active_continuous_requests)
                - (len(self.pd_admission.reserving) if self.pd_admission else 0)
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
                if self.pd_admission is None:
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
        if batch.batch_id not in resource.running_batch_ids:
            raise RuntimeError(f"resource/batch mismatch: {batch.resource_id}")
        resource.running_batch_ids.remove(batch.batch_id)
        self.running_batches.pop(batch.batch_id, None)
        resource.busy_time += (
            batch.resource_busy_time
            if batch.resource_busy_time is not None
            else batch.end_time - batch.start_time
        )

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
                self.pd_last_steps[item.request_id] = self.clock
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
                    self._token_ready(item.request_id)
                elif item.phase == Phase.DECODE:
                    self._token_ready(item.request_id)
            elif item.stage == Stage.PD_KV_COMMIT:
                if item.context_tokens == self.requests[item.request_id].input_tokens:
                    self.pd_ready.add(item.request_id)
                    if self.pd_admission:
                        self.pd_admission.source_release(item.request_id)
                    if item.request_id in self.pd_waiting_decode:
                        self.pd_waiting_decode.remove(item.request_id)
                        self._enqueue_decode(item.request_id)
                    if self.requests[item.request_id].finish_time is not None:
                        self._release_request(item.request_id)
            if (self.config.scheduler.pd_disaggregation.enabled
                and item.stage == Stage.CLOUD_MIDDLE and item.phase == Phase.PREFILL
                and item.pipeline_rank == self.config.stage("cloud_middle").pp_degree - 1):
                final = item.context_tokens == self.requests[item.request_id].input_tokens
                chunked = self.config.scheduler.pd_disaggregation.chunk_transfer
                if chunked or final:
                    self._enqueue(self.dag.add_pd_kv_transfer(
                        item.request_id, item.token_count if chunked else item.context_tokens,
                        self.clock, item.id, item.token_start if chunked else 0))
            if item.stage == Stage.EDGE_TAIL and item.phase == Phase.PREFILL and item.pipeline_rank == self.config.stage("edge_tail").pp_degree - 1:
                if item.request_id in self.pd_remaining:
                    self.pd_remaining[item.request_id] -= item.token_count
        self._advance_sync_transaction(batch)

    def _token_ready(self, request_id: int) -> None:
        request = self.requests[request_id]
        request.token_times.append(self.clock)
        if request.first_token_time is None:
            request.first_token_time = self.clock
        if len(request.token_times) >= request.output_tokens:
            request.finish_time = self.clock
            if not self.config.scheduler.pd_disaggregation.enabled or request_id in self.pd_ready:
                self._release_request(request_id)
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
                    if self._closed_loop_warmup_completed == warmup:
                        self._measurement_start_time = self.clock
                        if self.config.workload.measurement_duration_s:
                            self._measurement_end_time = (
                                self.clock
                                + self.config.workload.measurement_duration_s
                            )
                    if self._next_closed_loop_index < len(self._closed_loop_specs):
                        spec = self._closed_loop_specs[self._next_closed_loop_index]
                        self._next_closed_loop_index += 1
                        self._launch_closed_loop_spec(spec)
                    if self._closed_loop_warmup_completed == warmup:
                        active = sum(
                            self.requests[active_id].finish_time is None
                            for active_id in self.launched_request_ids
                        )
                        while (
                            active < self.config.workload.concurrency
                            and self._next_closed_loop_index < len(self._closed_loop_specs)
                        ):
                            spec = self._closed_loop_specs[self._next_closed_loop_index]
                            self._next_closed_loop_index += 1
                            self._launch_closed_loop_spec(spec)
                            active += 1
                elif (
                    (
                        not self.config.workload.measurement_duration_s
                        or self._measurement_end_time is None
                        or self.clock < self._measurement_end_time
                    )
                    and self._next_closed_loop_index < len(self._closed_loop_specs)
                ):
                    spec = self._closed_loop_specs[self._next_closed_loop_index]
                    self._next_closed_loop_index += 1
                    self._launch_closed_loop_spec(spec)
            return
        if self.config.scheduler.pd_disaggregation.enabled and request_id not in self.pd_ready:
            self.pd_waiting_decode.add(request_id)
            return
        self._enqueue_decode(request_id)

    def _release_request(self, request_id: int) -> None:
        if self.pd_admission:
            self.pd_admission.release(request_id)
            return
        self._free_local_kv(request_id)
        self.active_continuous_requests.discard(request_id)

    def _free_local_kv(self, request_id: int) -> None:
        if self.kv_cache is not None:
            freed = self.kv_cache.free_request(request_id)
            self.kv_events.append({"time_ms": self.clock * 1000.0, "event": "free", "request_id": request_id, "blocks": freed})

    def _enqueue_decode(self, request_id: int) -> None:
        request = self.requests[request_id]
        context = request.input_tokens + len(request.token_times)
        ready = self.dag.add_decode(
            request_id=request_id,
            iteration=len(request.token_times),
            context_tokens=context,
            ready_time=self.clock,
            token_start=context - 1 if self.config.scheduler.pd_disaggregation.enabled else None,
        )
        self._enqueue(ready)

    def _enqueue(self, items: list[WorkItem]) -> None:
        for item in items:
            if not (
                item.stage == Stage.EDGE_FRONT and item.pipeline_rank == 0
            ):
                parent_group = next(
                    (
                        self.work_item_dispatch_groups[dependency]
                        for dependency in item.dependencies
                        if dependency in self.work_item_dispatch_groups
                    ),
                    None,
                )
                if parent_group is None:
                    raise RuntimeError(
                        f"cannot determine dispatch group for work item {item.id}"
                    )
                # The DAG lists the same transaction's immediate path edge
                # before cross-chunk pipeline dependencies. The latter can
                # legitimately belong to an older dispatch group.
                self.work_item_dispatch_groups[item.id] = parent_group
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
        if item.stage in {Stage.PD_KV_START, Stage.PD_KV_COMMIT}:
            phase = "prefill" if item.stage == Stage.PD_KV_START else "decode"
            cloud = self.config.stage("cloud_middle").for_phase(phase)
            if self.config.scheduler.pd_disaggregation.control_channel:
                replica = self.pd_routes.get(item.request_id, item.request_id % cloud.replicas) if phase == "prefill" else 0
                return self._parallel_resource_id(f"pd_control_{phase}", replica, cloud.replicas, 0, 1)
            replica = self.pd_routes.get(item.request_id, item.request_id % cloud.replicas) if phase == "prefill" else 0
            return self._parallel_resource_id(cloud.resource, replica, cloud.replicas, 0, cloud.pp_degree)
        stage = self.config.stage(item.stage.value).for_phase(item.phase.value)
        # Naive but KV-safe routing: a request remains sticky to one replica.
        replica = (self.pd_routes[item.request_id] if item.stage == Stage.CLOUD_MIDDLE
                   and item.phase == Phase.PREFILL and item.request_id in self.pd_routes
                   else item.request_id % stage.replicas)
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
        if not all(self._pd_front_credit(item) for item in items):
            return False
        limit = self.config.execution.max_inflight_transactions
        if limit and len(self.inflight_transactions) >= limit:
            return False
        additional_bytes = sum(self._transaction_bytes(item) for item in items)
        capacity = self.config.execution.buffer_pool_mib * 1024.0 * 1024.0
        if capacity and additional_bytes > capacity:
            raise RuntimeError(
                "one batch requires more bytes than execution.buffer_pool_mib"
            )
        return not capacity or self.inflight_bytes + additional_bytes <= capacity

    def _pd_front_credit(self, item: WorkItem) -> bool:
        if self.config.scheduler.policy != "split_poc_pd":
            return True
        pd = self.config.scheduler.pd_disaggregation
        route = self.pd_routes[item.request_id]
        count = sum(any(phase == item.phase.value and
                        (item.phase == Phase.DECODE or self.pd_routes[rid] == route)
                        for rid, phase, _, _ in members)
                    for members in self.inflight_transaction_members.values())
        return count < (pd.prefill_window if item.phase == Phase.PREFILL else pd.decode_window)

    def _admissible_front_candidates(self, items: list[WorkItem]) -> list[WorkItem]:
        if not items:
            return items
        limit = self.config.execution.max_inflight_transactions
        selected: list[WorkItem] = []
        added_bytes = 0.0
        capacity = self.config.execution.buffer_pool_mib * 1024.0 * 1024.0
        for item in items:
            is_front = item.stage == Stage.EDGE_FRONT and item.pipeline_rank == 0
            if not is_front:
                selected.append(item)
                continue
            if not self._pd_front_credit(item):
                continue
            if limit and len(self.inflight_transactions) >= limit:
                continue
            size = self._transaction_bytes(item)
            if capacity and size > capacity:
                raise RuntimeError(
                    "one transaction requires more bytes than execution.buffer_pool_mib"
                )
            if capacity and self.inflight_bytes + added_bytes + size > capacity:
                continue
            selected.append(item)
            added_bytes += size
        return selected

    def _reserve_front_batch(self, items: list[WorkItem]) -> None:
        if items[0].stage != Stage.EDGE_FRONT or items[0].pipeline_rank != 0:
            return
        group_ids = {self.work_item_dispatch_groups[item.id] for item in items}
        if len(group_ids) != 1:
            raise RuntimeError("front batch must have one dispatch group")
        group_id = group_ids.pop()
        size = sum(self._transaction_bytes(item) for item in items)
        self.inflight_transactions[group_id] = size
        self.inflight_transaction_members[group_id] = {
            self._transaction_key(item) for item in items
        }
        self.inflight_bytes += size
        self.max_inflight_transactions_observed = max(
            self.max_inflight_transactions_observed, len(self.inflight_transactions)
        )
        self.max_inflight_bytes_observed = max(
            self.max_inflight_bytes_observed, self.inflight_bytes
        )

    def _release_transaction(self, item: WorkItem) -> None:
        group_id = self.work_item_dispatch_groups[item.id]
        members = self.inflight_transaction_members[group_id]
        members.discard(self._transaction_key(item))
        if members:
            return
        self.inflight_transaction_members.pop(group_id)
        size = self.inflight_transactions.pop(group_id)
        self.inflight_bytes -= size

    def _schedule_idle_resources(self) -> None:
        if self.config.execution.mode == "synchronous_rpc" and self.running_batches:
            return
        for resource_id in sorted(self.resources):
            resource = self.resources[resource_id]
            while resource.idle:
                candidates = self.queues[resource_id]
                candidates = self._admissible_front_candidates(candidates)
                if self.pd_admission and self.pd_admission.try_start(resource, candidates):
                    continue
                if self.sync_signature is not None:
                    expected = self.sync_expected
                    candidates = [
                        item
                        for item in candidates
                        if expected == (item.stage, item.pipeline_rank)
                        and self.sync_signature.get(item.request_id)
                        == (item.phase, item.chunk_index, item.iteration)
                    ]
                if not candidates:
                    break
                # The split pipeline completes returned RPCs before submitting
                # another front on the shared enterprise GPU resource.
                if isinstance(self.scheduler, PDScheduler):
                    candidates = self.scheduler.ready_backs(candidates)
                elif self.config.scheduler.prefer_ready_back:
                    backs = [item for item in candidates if item.stage == Stage.EDGE_TAIL]
                    if backs:
                        candidates = backs
                snapshot = SchedulerSnapshot(
                    current_time=self.clock,
                    outstanding_prefill_chunks=dict(self.outstanding_prefill_chunks),
                    total_outstanding_prefill_chunks=self.total_outstanding_prefill_chunks,
                    running_request_ids=frozenset(self.active_continuous_requests | self.active_static_cohort),
                    request_priorities=self.request_priorities,
                    request_arrival_times=self.request_arrival_times,
                    request_input_tokens={request_id: spec.input_tokens for request_id, spec in self.request_specs.items()},
                )
                first = min(
                    candidates, key=lambda item: (item.ready_time, item.id)
                )
                if isinstance(self.scheduler, PDScheduler) and first.stage == Stage.EDGE_TAIL:
                    first = min(candidates, key=lambda item: (
                        self.pd_last_steps.get(item.request_id, 0) if item.phase == Phase.DECODE
                        else self.work_item_dispatch_groups[item.id], item.id))
                preserve_group = (
                    self.config.execution.preserve_batch_across_stages
                    and first.stage != Stage.EDGE_FRONT
                )
                if first.stage in {Stage.PD_KV_TRANSFER, Stage.PD_KV_START, Stage.PD_KV_COMMIT}:
                    items = [first]
                elif preserve_group:
                    group_id = self.work_item_dispatch_groups[first.id]
                    items = [
                        item
                        for item in candidates
                        if self.work_item_dispatch_groups.get(item.id) == group_id
                    ]
                else:
                    items = self.scheduler.form_batch(candidates, snapshot)
                if (items and self.config.scheduler.policy == "split_poc_pd"
                    and items[0].stage == Stage.EDGE_FRONT and items[0].phase == Phase.PREFILL):
                    route = self.pd_routes[items[0].request_id]
                    items = [item for item in items if self.pd_routes[item.request_id] == route]
                if not items or not self._can_admit_front_batch(items):
                    break
                if not self._grow_kv_for_batch(items):
                    break
                selected_ids = {item.id for item in items}
                self.queues[resource_id] = [
                    item
                    for item in self.queues[resource_id]
                    if item.id not in selected_ids
                ]
                self._start_batch(resource, items)
                if self.config.execution.mode == "synchronous_rpc":
                    return

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
        if stage == Stage.EDGE_FRONT and items[0].pipeline_rank == 0:
            transaction_id = self._dispatch_group_sequence
            self._dispatch_group_sequence += 1
            for item in items:
                self.work_item_dispatch_groups[item.id] = transaction_id
        else:
            group_ids = {self.work_item_dispatch_groups[item.id] for item in items}
            if (
                len(group_ids) != 1
                and self.config.execution.preserve_batch_across_stages
            ):
                raise RuntimeError("a downstream batch cannot mix dispatch groups")
            transaction_id = group_ids.pop() if len(group_ids) == 1 else None
        if self.config.execution.mode == "synchronous_rpc" and self.sync_signature is None:
            if stage != Stage.EDGE_FRONT or items[0].pipeline_rank != 0:
                raise RuntimeError("synchronous RPC transaction must start at edge_front")
            self.sync_signature = {
                item.request_id: (item.phase, item.chunk_index, item.iteration)
                for item in items
            }
            self.sync_expected = (stage, items[0].pipeline_rank)
            self.sync_transaction_id = transaction_id
        if stage in GPU_STAGES:
            estimate = self.roofline.estimate(stage.value, items)
        elif stage in NETWORK_STAGES:
            estimate = self.network.estimate(stage, items)
        else:
            raise RuntimeError(f"unsupported stage: {stage}")

        batch_id = self._batch_sequence
        self._batch_sequence += 1
        intervals: tuple[tuple[float, float], ...] = ()
        host_intervals: tuple[tuple[float, float], ...] = ()
        resource_busy_time: float | None = None
        if stage in {Stage.WAN_UP, Stage.WAN_DOWN}:
            intervals, end_time, resource_busy_time = self._schedule_network_path(
                resource.resource_id, estimate
            )
        elif (stage in GPU_STAGES and not self.command_costs and self.config.execution.host_submission.enabled
              and estimate.sub_operations):
            schedule = schedule_submissions(
                estimate.sub_operations, self.clock,
                self.host_available.get(resource.resource_id, self.clock),
                self.config.execution.host_submission,
            )
            intervals, host_intervals = schedule.gpu_intervals, schedule.cpu_intervals
            end_time = schedule.end_time
            self.host_available[resource.resource_id] = schedule.cpu_end
            totals = self.host_totals.setdefault(resource.resource_id, {
                "cpu_busy_ms": 0.0, "gpu_busy_ms": 0.0, "exposed_delay_ms": 0.0,
            })
            for key, value in [("cpu_busy_ms", schedule.cpu_busy_s),
                               ("gpu_busy_ms", schedule.gpu_busy_s),
                               ("exposed_delay_ms", schedule.exposed_delay_s)]:
                totals[key] += value * 1000.0
        else:
            end_time = self.clock + estimate.total_time_s
        batch = BatchExecution(
            batch_id=batch_id,
            resource_id=resource.resource_id,
            stage=stage,
            items=items,
            estimate=estimate,
            start_time=self.clock,
            end_time=end_time,
            transaction_id=transaction_id,
            resource_busy_time=resource_busy_time,
            sub_operation_intervals=intervals,
            host_submission_intervals=host_intervals,
        )
        if self.pd_admission and stage == Stage.CLOUD_MIDDLE:
            self.pd_admission.forward_gate_available[resource.resource_id] = end_time
        resource.running_batch_ids.add(batch_id)
        self.running_batches[batch_id] = batch
        self._reserve_front_batch(items)
        if self.pd_admission and stage == Stage.EDGE_FRONT:
            for item in items:
                if item.phase == Phase.PREFILL and item.chunk_index == 0:
                    self.pd_admission.record(item.request_id, 'first_front')
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

    def _schedule_network_path(
        self,
        resource_id: str,
        estimate: PerformanceEstimate,
    ) -> tuple[tuple[tuple[float, float], ...], float, float]:
        """Place one RPC on a finite worker window and a serialized WAN link."""

        cursor = self.clock
        intervals: list[tuple[float, float]] = []
        serialization_busy = 0.0
        for operation in estimate.sub_operations:
            if operation.name == "wan_serialization":
                cursor = max(cursor, self.network_link_available[resource_id])
            start = cursor
            cursor += operation.duration_s
            intervals.append((start, cursor))
            if operation.name == "wan_serialization":
                self.network_link_available[resource_id] = cursor
                serialization_busy += operation.duration_s
        return tuple(intervals), cursor, serialization_busy

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
        for index, operation in enumerate(operations):
            if batch.sub_operation_intervals:
                operation_start, operation_end = batch.sub_operation_intervals[index]
            else:
                operation_start = cursor
                operation_end = cursor + operation.duration_s
            sub_operations.append(
                {
                    "name": operation.name,
                    "category": operation.category,
                    "start_time_ms": operation_start * 1000.0,
                    "end_time_ms": operation_end * 1000.0,
                    "duration_ms": operation.duration_s * 1000.0,
                    "input_shape": operation.input_shape,
                    "dependencies": list(operation.dependencies),
                    "profile_type": operation.profile_type,
                    "host_submit_start_ms": (batch.host_submission_intervals[index][0] * 1000.0
                                             if batch.host_submission_intervals else None),
                    "host_submit_end_ms": (batch.host_submission_intervals[index][1] * 1000.0
                                           if batch.host_submission_intervals else None),
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
