from __future__ import annotations

from collections.abc import Sequence

from .config import SimulationConfig, StageConfig
from .core import (
    OperatorWorkload,
    PerformanceEstimate,
    Phase,
    Stage,
    SubOperation,
    WorkItem,
)


class DenseWorkloadModel:
    """Simple analytical workload model; intended for behavioral simulation."""

    def __init__(self, config: SimulationConfig):
        self.model = config.model

    def build_operators(
        self, stage: StageConfig, items: Sequence[WorkItem]
    ) -> list[tuple[str, OperatorWorkload]]:
        if not items:
            raise ValueError("cannot build an empty workload")
        hidden = self.model.hidden_size
        dtype_bytes = self.model.dtype_bytes
        layers = stage.num_layers
        total_tokens = sum(item.token_count for item in items)

        attention_parameters = 4.0 * hidden * hidden
        mlp_parameters = max(
            self.model.params_per_layer_factor - 4.0, 0.0
        ) * hidden * hidden
        projection_flops = 2.0 * attention_parameters * total_tokens * layers
        mlp_flops = 2.0 * mlp_parameters * total_tokens * layers
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

        projection_memory = (
            attention_parameters * dtype_bytes * layers
            + 2.0 * total_tokens * hidden * dtype_bytes * layers
        )
        attention_memory = kv_bytes + total_tokens * hidden * dtype_bytes * layers
        mlp_memory = (
            mlp_parameters * dtype_bytes * layers
            + 2.0 * total_tokens * hidden * dtype_bytes * layers
        )
        return [
            (
                "attention_projection",
                OperatorWorkload(projection_flops, projection_memory),
            ),
            ("attention", OperatorWorkload(attention_flops, attention_memory)),
            ("mlp", OperatorWorkload(mlp_flops, mlp_memory)),
        ]

    def build(self, stage: StageConfig, items: Sequence[WorkItem]) -> OperatorWorkload:
        operators = self.build_operators(stage, items)
        communication_bytes = (
            2.0
            * sum(item.token_count for item in items)
            * self.model.hidden_size
            * self.model.dtype_bytes
            * stage.num_layers
        )
        return OperatorWorkload(
            flops=sum(workload.flops for _, workload in operators),
            memory_bytes=sum(workload.memory_bytes for _, workload in operators),
            communication_bytes=communication_bytes,
        )

    def input_shape(self, items: Sequence[WorkItem]) -> str:
        tokens = [item.token_count for item in items]
        contexts = [item.context_tokens for item in items]
        return (
            f"B={len(items)}, tokens={tokens}, contexts={contexts}, "
            f"hidden={self.model.hidden_size}"
        )


class RooflineModel:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.workloads = DenseWorkloadModel(config)

    def estimate(self, stage_name: str, items: Sequence[WorkItem]) -> PerformanceEstimate:
        stage = self.config.stage(stage_name)
        hardware = self.config.hardware[stage.resource]
        workload = self.workloads.build(stage, items)
        operator_workloads = self.workloads.build_operators(stage, items)
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

        shape = self.workloads.input_shape(items)
        sub_operations: list[SubOperation] = []
        for name, operator in operator_workloads:
            duration = max(
                operator.flops / peak_flops,
                operator.memory_bytes / memory_bandwidth,
            )
            sub_operations.append(
                SubOperation(
                    name=name,
                    category="compute",
                    duration_s=duration,
                    input_shape=shape,
                )
            )

        collective_time = 0.0
        if tp > 1:
            ring_factor = 2.0 * (tp - 1) / tp
            bytes_on_link = workload.communication_bytes * ring_factor / tp
            num_collectives = 2 * stage.num_layers
            collective_time = (
                num_collectives * hardware.interconnect_latency_us * 1e-6
                + bytes_on_link / (hardware.interconnect_bandwidth_gb_s * 1e9)
            )
            sub_operations.append(
                SubOperation(
                    name="tp_collective",
                    category="node_communication",
                    duration_s=collective_time,
                    input_shape=(
                        f"[{sum(item.token_count for item in items)}, "
                        f"{self.config.model.hidden_size}], tp={tp}"
                    ),
                )
            )

        overhead = hardware.kernel_overhead_us * 1e-6 * stage.num_layers
        if overhead:
            sub_operations.append(
                SubOperation(
                    name="kernel_overhead",
                    category="overhead",
                    duration_s=overhead,
                    input_shape=f"layers={stage.num_layers}",
                )
            )
        total = sum(operation.duration_s for operation in sub_operations)
        return PerformanceEstimate(
            flops=workload.flops,
            memory_bytes=workload.memory_bytes,
            communication_bytes=workload.communication_bytes,
            compute_time_s=compute_time,
            memory_time_s=memory_time,
            collective_time_s=collective_time,
            overhead_time_s=overhead,
            total_time_s=total,
            input_shape=shape,
            sub_operations=tuple(sub_operations),
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
        shape = (
            f"hidden_state=[{sum(item.token_count for item in items)}, "
            f"{self.config.model.hidden_size}], dtype_bytes={self.config.model.dtype_bytes}"
        )
        return PerformanceEstimate(
            flops=0.0,
            memory_bytes=0.0,
            communication_bytes=payload,
            compute_time_s=0.0,
            memory_time_s=serialization,
            collective_time_s=0.0,
            overhead_time_s=propagation,
            total_time_s=total,
            input_shape=shape,
            sub_operations=(
                SubOperation(
                    name="wan_serialization",
                    category="communication",
                    duration_s=serialization,
                    input_shape=f"{shape}, bytes={int(payload)}",
                ),
                SubOperation(
                    name="wan_propagation",
                    category="communication",
                    duration_s=propagation,
                    input_shape=f"one_way_rtt={self.config.network.rtt_ms / 2:.3f} ms",
                ),
            ),
        )
