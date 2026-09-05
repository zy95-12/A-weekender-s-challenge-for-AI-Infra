from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from .config import SimulationConfig, StageConfig
from .core import NETWORK_STAGES, OperatorWorkload, PerformanceEstimate, Phase, Stage, SubOperation, WorkItem


@dataclass(frozen=True)
class ModeledOperator:
    name: str
    category: str
    workload: OperatorWorkload
    input_shape: str
    dependencies: tuple[str, ...] = ()
    communication_kind: str | None = None


class Qwen3WorkloadModel:
    """Build a Qwen2/Qwen3 operator sequence from Hugging Face dimensions."""

    def __init__(self, config: SimulationConfig):
        self.config = config
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
            communication_kind="all_reduce",
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
        tokens = sum(item.token_count + item.recompute_tokens for item in items)
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
            rope_dependencies = (f"{prefix}.q_proj", f"{prefix}.k_proj")
            if model.architecture == "qwen3":
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
                    ]
                )
                rope_dependencies = (f"{prefix}.q_norm", f"{prefix}.k_norm")
            operators.append(
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
                    dependencies=rope_dependencies,
                )
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
            base_attention_dependencies = (f"{prefix}.rope", f"{prefix}.kv_cache_update")
            phase_groups = [
                (phase, [item for item in items if item.phase == phase])
                for phase in (Phase.PREFILL, Phase.DECODE)
                if any(item.phase == phase for item in items)
            ]
            separate = self.config.attention_backend.mode == "separate" and len(phase_groups) == 2
            attention_names: list[str] = []
            groups = phase_groups if separate else [(None, list(items))]
            for phase, group_items in groups:
                group_tokens = sum(item.token_count + item.recompute_tokens for item in group_items)
                group_pairs = sum(
                    ((item.token_count + item.recompute_tokens) * item.token_start + (item.token_count + item.recompute_tokens) * (item.token_count + item.recompute_tokens + 1) / 2)
                    if item.phase == Phase.PREFILL
                    else (item.token_count + item.recompute_tokens) * max(item.context_tokens, item.token_count)
                    for item in group_items
                )
                group_cache = sum(max(item.context_tokens, item.token_count) for item in group_items)
                label = (
                    f"{phase.value} attention"
                    if phase is not None
                    else "mix attention" if len(phase_groups) == 2 else "attention"
                )
                attention_name = f"{prefix}.{label}"
                dependencies = (
                    (attention_names[-1],)
                    if separate and attention_names
                    else base_attention_dependencies
                )
                operators.append(ModeledOperator(
                    name=attention_name,
                    category="compute",
                    workload=OperatorWorkload(
                        flops=4.0 * local_q_width * group_pairs,
                        memory_bytes=(2.0 * group_tokens * local_q_width + 2.0 * group_cache * local_kv_width) * model.dtype_bytes,
                    ),
                    input_shape=(f"{model.dtype} Q[{group_tokens}, {local_q_heads}, {model.head_dim}], KV[B, {local_kv_heads}, L_i, {model.head_dim}]"),
                    dependencies=dependencies,
                ))
                attention_names.append(attention_name)
            attention_dependency = attention_names[-1]
            operators.append(
                self._linear(
                    "o_proj",
                    layer,
                    tokens,
                    model.query_width,
                    model.hidden_size,
                    local_input_width=local_q_width,
                    dependencies=(attention_dependency,),
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
        return self._shape(sum(item.token_count + item.recompute_tokens for item in items), self.model.hidden_size)


class RooflineModel:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.workloads = Qwen3WorkloadModel(config)

    def estimate(self, stage_name: str, items: Sequence[WorkItem]) -> PerformanceEstimate:
        stage = self.config.stage(stage_name)
        resources = {stage.resource_for_phase(item.phase.value) for item in items}
        if len(resources) != 1:
            raise ValueError("one batch cannot span phase-specific resources")
        hardware = self.config.hardware[next(iter(resources))]
        pipeline_ranks = {item.pipeline_rank for item in items}
        if len(pipeline_ranks) != 1:
            raise ValueError("a batch cannot mix pipeline ranks")
        pipeline_rank = pipeline_ranks.pop()
        layer_start, layer_end = stage.pipeline_layer_range(pipeline_rank)
        local_stage = replace(
            stage,
            layer_start=layer_start,
            layer_end=layer_end,
            pp_degree=1,
        )
        operators = self.workloads.build_operators(local_stage, items)
        if pipeline_rank < stage.pp_degree - 1:
            tokens = sum(item.token_count for item in items)
            operators.append(
                ModeledOperator(
                    name=f"pp_{pipeline_rank}_send",
                    category="communication",
                    workload=OperatorWorkload(
                        0.0,
                        0.0,
                        float(
                            tokens
                            * self.config.model.hidden_size
                            * self.config.model.dtype_bytes
                        ),
                    ),
                    input_shape=self.workloads.input_shape(items),
                    dependencies=(operators[-1].name,),
                    communication_kind="p2p",
                )
            )
        workload = OperatorWorkload(
            flops=sum(operator.workload.flops for operator in operators),
            memory_bytes=sum(operator.workload.memory_bytes for operator in operators),
            communication_bytes=sum(
                operator.workload.communication_bytes for operator in operators
            ),
        )
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
                factor = ring_factor if operator.communication_kind == "all_reduce" else 1.0
                duration = (
                    hardware.interconnect_latency_us * 1e-6
                    + factor
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
        tensor_bytes = (
            sum(item.token_count for item in items)
            * self.config.model.hidden_size
            * self.config.model.dtype_bytes
            * self.config.network.activation_tensor_count
        )
        return float(tensor_bytes + self.config.network.protocol_overhead_bytes)

    def estimate(self, stage: Stage, items: Sequence[WorkItem]) -> PerformanceEstimate:
        if stage not in NETWORK_STAGES:
            raise ValueError(f"not a network stage: {stage}")
        if stage == Stage.PD_KV_TRANSFER:
            payload = float(sum(item.context_tokens for item in items) * self.config.model.num_layers * 2 * self.config.model.hidden_size * self.config.model.dtype_bytes)
            pd = self.config.scheduler.pd_disaggregation
            serialization = payload / (pd.kv_transfer_bandwidth_gb_s * 1e9)
            latency = pd.kv_transfer_latency_ms / 1000.0
            shape = f"{self.config.model.dtype} KV[{sum(item.context_tokens for item in items)}, {self.config.model.num_layers}, 2, {self.config.model.hidden_size}]"
            return PerformanceEstimate(0.0, 0.0, payload, 0.0, serialization, 0.0, latency, serialization + latency, shape, (SubOperation("pd_kv_transfer", "communication", serialization + latency, shape),))
        payload = self.payload_bytes(items)
        bandwidth_gbps = (
            self.config.network.uplink_gbps
            if stage == Stage.WAN_UP
            else self.config.network.downlink_gbps
        )
        bytes_per_second = bandwidth_gbps * 1e9 / 8.0 * self.config.network.efficiency
        serialization = payload / bytes_per_second
        propagation = self.config.network.rtt_ms / 2000.0
        sender_overhead = self.config.network.sender_overhead_ms / 1000.0
        receiver_overhead = self.config.network.receiver_overhead_ms / 1000.0
        total = sender_overhead + serialization + propagation + receiver_overhead
        tensors = self.config.network.activation_tensor_count
        shape = (
            f"{tensors} x {self.config.model.dtype} "
            f"[{sum(item.token_count for item in items)}, {self.config.model.hidden_size}]"
        )
        return PerformanceEstimate(
            flops=0.0,
            memory_bytes=0.0,
            communication_bytes=payload,
            compute_time_s=0.0,
            memory_time_s=serialization,
            collective_time_s=0.0,
            overhead_time_s=sender_overhead + propagation + receiver_overhead,
            total_time_s=total,
            input_shape=shape,
            sub_operations=(
                SubOperation("sender_staging", "communication", sender_overhead, shape),
                SubOperation("wan_serialization", "communication", serialization, shape),
                SubOperation("wan_propagation", "communication", propagation, shape),
                SubOperation("receiver_staging", "communication", receiver_overhead, shape),
            ),
        )
