from __future__ import annotations

from collections.abc import Sequence

from .config import SimulationConfig, StageConfig
from .core import OperatorWorkload, PerformanceEstimate, Phase, Stage, WorkItem


class DenseWorkloadModel:
    """Simple analytical workload model; intended for behavioral simulation."""

    def __init__(self, config: SimulationConfig):
        self.model = config.model

    def build(self, stage: StageConfig, items: Sequence[WorkItem]) -> OperatorWorkload:
        if not items:
            raise ValueError("cannot build an empty workload")
        hidden = self.model.hidden_size
        dtype_bytes = self.model.dtype_bytes
        layers = stage.num_layers
        total_tokens = sum(item.token_count for item in items)

        # Dense attention projections + MLP. Each parameter contributes roughly
        # one multiply-add (two FLOPs) per processed token.
        parameters_per_layer = self.model.params_per_layer_factor * hidden * hidden
        linear_flops = 2.0 * parameters_per_layer * total_tokens * layers

        # Attention score/value work grows with the visible prefix. This is an
        # intentionally simple approximation that preserves the right trend.
        attention_flops = 0.0
        kv_bytes = 0.0
        for item in items:
            visible_context = max(item.context_tokens, item.token_count)
            attention_flops += (
                4.0 * hidden * item.token_count * visible_context * layers
            )
            kv_bytes += (
                2.0 * hidden * visible_context * dtype_bytes * layers
            )
            if item.phase == Phase.PREFILL:
                kv_bytes += 2.0 * hidden * item.token_count * dtype_bytes * layers

        # Weights are read once per batch, while activation/KV traffic scales
        # with the members of the batch.
        weight_bytes = parameters_per_layer * dtype_bytes * layers
        activation_bytes = 4.0 * total_tokens * hidden * dtype_bytes * layers
        communication_bytes = 2.0 * total_tokens * hidden * dtype_bytes * layers
        return OperatorWorkload(
            flops=linear_flops + attention_flops,
            memory_bytes=weight_bytes + activation_bytes + kv_bytes,
            communication_bytes=communication_bytes,
        )


class RooflineModel:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.workloads = DenseWorkloadModel(config)

    def estimate(self, stage_name: str, items: Sequence[WorkItem]) -> PerformanceEstimate:
        stage = self.config.stage(stage_name)
        hardware = self.config.hardware[stage.resource]
        workload = self.workloads.build(stage, items)
        tp = stage.tp_degree

        peak_flops = (
            hardware.peak_flops_tflops
            * 1e12
            * tp
            * hardware.compute_efficiency
        )
        memory_bandwidth = (
            hardware.hbm_bandwidth_gb_s
            * 1e9
            * tp
            * hardware.memory_efficiency
        )
        compute_time = workload.flops / peak_flops
        memory_time = workload.memory_bytes / memory_bandwidth

        collective_time = 0.0
        if tp > 1:
            ring_factor = 2.0 * (tp - 1) / tp
            bytes_on_link = workload.communication_bytes * ring_factor / tp
            num_collectives = 2 * stage.num_layers
            collective_time = (
                num_collectives * hardware.interconnect_latency_us * 1e-6
                + bytes_on_link / (hardware.interconnect_bandwidth_gb_s * 1e9)
            )

        overhead = hardware.kernel_overhead_us * 1e-6 * stage.num_layers
        total = max(compute_time, memory_time) + collective_time + overhead
        return PerformanceEstimate(
            flops=workload.flops,
            memory_bytes=workload.memory_bytes,
            communication_bytes=workload.communication_bytes,
            compute_time_s=compute_time,
            memory_time_s=memory_time,
            collective_time_s=collective_time,
            overhead_time_s=overhead,
            total_time_s=total,
        )


class NetworkModel:
    def __init__(self, config: SimulationConfig):
        self.config = config

    def payload_bytes(self, items: Sequence[WorkItem]) -> float:
        return float(
            sum(item.token_count for item in items)
            * self.config.model.hidden_size
            * self.config.model.dtype_bytes
        )

    def estimate(self, stage: Stage, items: Sequence[WorkItem]) -> PerformanceEstimate:
        if stage not in {Stage.WAN_UP, Stage.WAN_DOWN}:
            raise ValueError(f"not a network stage: {stage}")
        payload = self.payload_bytes(items)
        bandwidth_gbps = (
            self.config.network.uplink_gbps
            if stage == Stage.WAN_UP
            else self.config.network.downlink_gbps
        )
        bytes_per_second = bandwidth_gbps * 1e9 / 8.0 * self.config.network.efficiency
        serialization = payload / bytes_per_second
        propagation = self.config.network.rtt_ms / 2000.0
        total = serialization + propagation
        return PerformanceEstimate(
            flops=0.0,
            memory_bytes=0.0,
            communication_bytes=payload,
            compute_time_s=0.0,
            memory_time_s=serialization,
            collective_time_s=0.0,
            overhead_time_s=propagation,
            total_time_s=total,
        )
