from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .config import SimulationConfig, StageConfig
from .core import OperatorWorkload, PerformanceEstimate, Phase, Stage, SubOperation, WorkItem


@dataclass(frozen=True)
class ModeledOperator:
    name: str
    category: str
    workload: OperatorWorkload
    input_shape: str
    dependencies: tuple[str, ...] = ()


class Qwen3WorkloadModel:
    """Build a Qwen3 operator sequence from Hugging Face model dimensions."""

    def __init__(self, config: SimulationConfig):
        self.model = config.model

    def _shape(self, *dimensions: int | str) -> str:
        return f"{self.model.dtype} [{', '.join(str(value) for value in dimensions)}]"

    def _linear(
        self,
        name: str,
        layer: int | None,
        tokens: int,
        input_width: int,
        output_width: int,
        *,
        local_output_width: int | None = None,
        local_input_width: int | None = None,
        dependencies: tuple[str, ...] = (),
    ) -> ModeledOperator:
        local_in = local_input_width or input_width
        local_out = local_output_width or output_width
        elements = self.model.dtype_bytes
        workload = OperatorWorkload(
            flops=2.0 * tokens * local_in * local_out,
            memory_bytes=(
                local_in * local_out + tokens * (local_in + local_out)
            )
            * elements,
        )
        prefix = f"layer_{layer:02d}." if layer is not None else ""
        return ModeledOperator(
            name=f"{prefix}{name}",
            category="compute",
            workload=workload,
            input_shape=self._shape(tokens, local_in),
            dependencies=dependencies,
        )

    def _elementwise(
        self,
        name: str,
        layer: int | None,
        tokens: int,
        width: int,
        flop_factor: float,
        memory_factor: float = 2.0,
        dependencies: tuple[str, ...] = (),
    ) -> ModeledOperator:
        prefix = f"layer_{layer:02d}." if layer is not None else ""
        return ModeledOperator(
            name=f"{prefix}{name}",
            category="compute",
            workload=OperatorWorkload(
                flops=flop_factor * tokens * width,
                memory_bytes=memory_factor * tokens * width * self.model.dtype_bytes,
            ),
            input_shape=self._shape(tokens, width),
            dependencies=dependencies,
        )

    def _collective(
        self,
        name: str,
        layer: int,
        tokens: int,
        dependencies: tuple[str, ...],
    ) -> ModeledOperator:
        payload = tokens * self.model.hidden_size * self.model.dtype_bytes
        return ModeledOperator(
            name=f"layer_{layer:02d}.{name}",
            category="communication",
            workload=OperatorWorkload(0.0, 0.0, float(payload)),
            input_shape=self._shape(tokens, self.model.hidden_size),
            dependencies=dependencies,
        )

    def build_operators(
        self, stage: StageConfig, items: Sequence[WorkItem]
    ) -> list[ModeledOperator]:
        if not items:
            raise ValueError("cannot build an empty workload")
        model = self.model
        tokens = sum(item.token_count for item in items)
        tp = stage.tp_degree
        local_q_heads = model.num_attention_heads // tp
        local_q_width = local_q_heads * model.head_dim
        if model.num_key_value_heads >= tp:
            local_kv_heads = model.num_key_value_heads // tp
        else:
            # When TP is wider than the number of KV heads, replicate KV heads.
            local_kv_heads = model.num_key_value_heads
        local_kv_width = local_kv_heads * model.head_dim
        local_intermediate = model.intermediate_size // tp
        operators: list[ModeledOperator] = []

        if stage.layer_start == 0:
            operators.append(
                ModeledOperator(
                    name="embed_tokens",
                    category="compute",
                    workload=OperatorWorkload(
                        flops=0.0,
                        memory_bytes=2.0 * tokens * model.hidden_size * model.dtype_bytes,
                    ),
                    input_shape="int64 [B, T]",
                )
            )

        previous_output = "embed_tokens" if stage.layer_start == 0 else ""

        context_pairs = sum(
            (
                item.token_count * item.token_start
                + item.token_count * (item.token_count + 1) / 2
            )
            if item.phase == Phase.PREFILL
            else item.token_count * max(item.context_tokens, item.token_count)
            for item in items
        )
        cache_tokens = sum(max(item.context_tokens, item.token_count) for item in items)
        for layer in range(stage.layer_start, stage.layer_end):
            prefix = f"layer_{layer:02d}"
            layer_input_dependencies = (previous_output,) if previous_output else ()
            operators.append(
                self._elementwise(
                    "input_layernorm",
                    layer,
                    tokens,
                    model.hidden_size,
                    5.0,
                    dependencies=layer_input_dependencies,
                )
            )
            norm_name = f"{prefix}.input_layernorm"
            operators.extend(
                [
                    self._linear(
                        "q_proj",
                        layer,
                        tokens,
                        model.hidden_size,
                        model.query_width,
                        local_output_width=local_q_width,
                        dependencies=(norm_name,),
                    ),
                    self._linear(
                        "k_proj",
                        layer,
                        tokens,
                        model.hidden_size,
                        model.kv_width,
                        local_output_width=local_kv_width,
                        dependencies=(norm_name,),
                    ),
                    self._linear(
                        "v_proj",
                        layer,
                        tokens,
                        model.hidden_size,
                        model.kv_width,
                        local_output_width=local_kv_width,
                        dependencies=(norm_name,),
                    ),
                ]
            )
            operators.extend(
                [
                    ModeledOperator(
                        name=f"{prefix}.q_norm",
                        category="compute",
                        workload=OperatorWorkload(
                            flops=5.0 * tokens * local_q_width,
                            memory_bytes=2.0
                            * tokens
                            * local_q_width
                            * model.dtype_bytes,
                        ),
                        input_shape=self._shape(
                            tokens, local_q_heads, model.head_dim
                        ),
                        dependencies=(f"{prefix}.q_proj",),
                    ),
                    ModeledOperator(
                        name=f"{prefix}.k_norm",
                        category="compute",
                        workload=OperatorWorkload(
                            flops=5.0 * tokens * local_kv_width,
                            memory_bytes=2.0
                            * tokens
                            * local_kv_width
                            * model.dtype_bytes,
                        ),
                        input_shape=self._shape(
                            tokens, local_kv_heads, model.head_dim
                        ),
                        dependencies=(f"{prefix}.k_proj",),
                    ),
                    ModeledOperator(
                        name=f"{prefix}.rope",
                        category="compute",
                        workload=OperatorWorkload(
                            flops=6.0
                            * tokens
                            * (local_q_width + local_kv_width),
                            memory_bytes=3.0
                            * tokens
                            * (local_q_width + local_kv_width)
                            * model.dtype_bytes,
                        ),
                        input_shape=(
                            f"{model.dtype} Q[{tokens}, {local_q_heads}, {model.head_dim}], "
                            f"K[{tokens}, {local_kv_heads}, {model.head_dim}]"
                        ),
                        dependencies=(f"{prefix}.q_norm", f"{prefix}.k_norm"),
                    ),
                ]
            )
            operators.append(
                ModeledOperator(
                    name=f"layer_{layer:02d}.kv_cache_update",
                    category="compute",
                    workload=OperatorWorkload(
                        flops=0.0,
                        memory_bytes=2.0
                        * tokens
                        * local_kv_width
                        * model.dtype_bytes,
                    ),
                    input_shape=self._shape(2, tokens, local_kv_heads, model.head_dim),
                    dependencies=(f"{prefix}.rope", f"{prefix}.v_proj"),
                )
            )
            operators.append(
                ModeledOperator(
                    name=f"layer_{layer:02d}.attention",
                    category="compute",
                    workload=OperatorWorkload(
                        flops=4.0 * local_q_width * context_pairs,
                        memory_bytes=(
                            tokens * local_q_width
                            + 2.0 * cache_tokens * local_kv_width
                            + tokens * local_q_width
                        )
                        * model.dtype_bytes,
                    ),
                    input_shape=(
                        f"{model.dtype} Q[{tokens}, {local_q_heads}, {model.head_dim}], "
                        f"KV[B, {local_kv_heads}, L_i, {model.head_dim}]"
                    ),
                    dependencies=(f"{prefix}.rope", f"{prefix}.kv_cache_update"),
                )
            )
            operators.append(
                self._linear(
                    "o_proj",
                    layer,
                    tokens,
                    model.query_width,
                    model.hidden_size,
                    local_input_width=local_q_width,
                    dependencies=(f"{prefix}.attention",),
                )
            )
            attention_output = f"{prefix}.o_proj"
            if tp > 1:
                operators.append(
                    self._collective(
                        "attention_all_reduce", layer, tokens, (attention_output,)
                    )
                )
                attention_output = f"{prefix}.attention_all_reduce"
            operators.append(
                self._elementwise(
                    "attention_residual",
                    layer,
                    tokens,
                    model.hidden_size,
                    1.0,
                    3.0,
                    dependencies=tuple(
                        name
                        for name in (previous_output, attention_output)
                        if name
                    ),
                )
            )
            attention_residual = f"{prefix}.attention_residual"
            operators.append(
                self._elementwise(
                    "post_attention_layernorm",
                    layer,
                    tokens,
                    model.hidden_size,
                    5.0,
                    dependencies=(attention_residual,),
                )
            )
            post_norm = f"{prefix}.post_attention_layernorm"
            operators.extend(
                [
                    self._linear(
                        "gate_proj",
                        layer,
                        tokens,
                        model.hidden_size,
                        model.intermediate_size,
                        local_output_width=local_intermediate,
                        dependencies=(post_norm,),
                    ),
                    self._linear(
                        "up_proj",
                        layer,
                        tokens,
                        model.hidden_size,
                        model.intermediate_size,
                        local_output_width=local_intermediate,
                        dependencies=(post_norm,),
                    ),
                    self._elementwise(
                        "silu",
                        layer,
                        tokens,
                        local_intermediate,
                        5.0,
                        dependencies=(f"{prefix}.gate_proj",),
                    ),
                    self._elementwise(
                        "gated_mul",
                        layer,
                        tokens,
                        local_intermediate,
                        1.0,
                        3.0,
                        dependencies=(f"{prefix}.silu", f"{prefix}.up_proj"),
                    ),
                    self._linear(
                        "down_proj",
                        layer,
                        tokens,
                        model.intermediate_size,
                        model.hidden_size,
                        local_input_width=local_intermediate,
                        dependencies=(f"{prefix}.gated_mul",),
                    ),
                ]
            )
            mlp_output = f"{prefix}.down_proj"
            if tp > 1:
                operators.append(
                    self._collective("mlp_all_reduce", layer, tokens, (mlp_output,))
                )
                mlp_output = f"{prefix}.mlp_all_reduce"
            operators.append(
                self._elementwise(
                    "mlp_residual",
                    layer,
                    tokens,
                    model.hidden_size,
                    1.0,
                    3.0,
                    dependencies=(attention_residual, mlp_output),
                )
            )
            previous_output = f"{prefix}.mlp_residual"

        logits_tokens = sum(item.produces_logits for item in items)
        if stage.layer_end == model.num_layers and logits_tokens:
            operators.append(
                self._elementwise(
                    "final_norm",
                    None,
                    logits_tokens,
                    model.hidden_size,
                    5.0,
                    dependencies=(previous_output,),
                )
            )
            if model.vocab_size:
                operators.append(
                    self._linear(
                        "lm_head",
                        None,
                        logits_tokens,
                        model.hidden_size,
                        model.vocab_size,
                        local_output_width=model.vocab_size // tp,
                        dependencies=("final_norm",),
                    )
                )
        return operators

    def build(self, stage: StageConfig, items: Sequence[WorkItem]) -> OperatorWorkload:
        operators = self.build_operators(stage, items)
        return OperatorWorkload(
            flops=sum(operator.workload.flops for operator in operators),
            memory_bytes=sum(operator.workload.memory_bytes for operator in operators),
            communication_bytes=sum(
                operator.workload.communication_bytes for operator in operators
            ),
        )

    def input_shape(self, items: Sequence[WorkItem]) -> str:
        return self._shape(sum(item.token_count for item in items), self.model.hidden_size)


class RooflineModel:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.workloads = Qwen3WorkloadModel(config)

    def estimate(self, stage_name: str, items: Sequence[WorkItem]) -> PerformanceEstimate:
        stage = self.config.stage(stage_name)
        hardware = self.config.hardware[stage.resource]
        operators = self.workloads.build_operators(stage, items)
        workload = self.workloads.build(stage, items)
        peak_flops = hardware.peak_flops_tflops * 1e12 * hardware.compute_efficiency
        memory_bandwidth = hardware.hbm_bandwidth_gb_s * 1e9 * hardware.memory_efficiency
        ring_factor = 2.0 * (stage.tp_degree - 1) / stage.tp_degree
        sub_operations: list[SubOperation] = []
        compute_time = 0.0
        memory_time = 0.0
        collective_time = 0.0
        overhead = 0.0

        for operator in operators:
            if operator.category == "communication":
                duration = (
                    hardware.interconnect_latency_us * 1e-6
                    + ring_factor
                    * operator.workload.communication_bytes
                    / (hardware.interconnect_bandwidth_gb_s * 1e9)
                )
                collective_time += duration
            else:
                operator_compute = operator.workload.flops / peak_flops
                operator_memory = operator.workload.memory_bytes / memory_bandwidth
                launch = hardware.kernel_overhead_us * 1e-6
                duration = max(operator_compute, operator_memory) + launch
                compute_time += operator_compute
                memory_time += operator_memory
                overhead += launch
            sub_operations.append(
                SubOperation(
                    name=operator.name,
                    category=operator.category,
                    duration_s=duration,
                    input_shape=operator.input_shape,
                    dependencies=operator.dependencies,
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
            input_shape=self.workloads.input_shape(items),
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
            f"{self.config.model.dtype} "
            f"[{sum(item.token_count for item in items)}, {self.config.model.hidden_size}]"
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
                SubOperation("wan_serialization", "communication", serialization, shape),
                SubOperation("wan_propagation", "communication", propagation, shape),
            ),
        )
