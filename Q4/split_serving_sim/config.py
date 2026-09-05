from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when an experiment configuration is inconsistent."""


@dataclass(frozen=True)
class ModelConfig:
    name: str
    architecture: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int = 0
    dtype: str = "bfloat16"
    dtype_bytes: int = 2

    @property
    def query_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def parameters_per_layer(self) -> int:
        attention = (
            self.hidden_size * self.query_width
            + 2 * self.hidden_size * self.kv_width
            + self.query_width * self.hidden_size
        )
        mlp = 3 * self.hidden_size * self.intermediate_size
        norms = 2 * self.hidden_size + 2 * self.head_dim
        return attention + mlp + norms


@dataclass(frozen=True)
class StageConfig:
    name: str
    layer_start: int
    layer_end: int
    resource: str
    tp_degree: int = 1
    pp_degree: int = 1
    pp_layer_ranges: tuple[tuple[int, int], ...] = ()
    replicas: int = 1
    prefill_resource: str | None = None
    decode_resource: str | None = None

    @property
    def num_layers(self) -> int:
        return self.layer_end - self.layer_start

    def pipeline_layer_range(self, pipeline_rank: int) -> tuple[int, int]:
        if not 0 <= pipeline_rank < self.pp_degree:
            raise ValueError(f"invalid pipeline rank for {self.name}: {pipeline_rank}")
        if self.pp_layer_ranges:
            return self.pp_layer_ranges[pipeline_rank]
        layers_per_rank, remainder = divmod(self.num_layers, self.pp_degree)
        start = (
            self.layer_start
            + pipeline_rank * layers_per_rank
            + min(pipeline_rank, remainder)
        )
        width = layers_per_rank + (1 if pipeline_rank < remainder else 0)
        return start, start + width

    def resource_for_phase(self, phase: str) -> str:
        if phase == "prefill" and self.prefill_resource:
            return self.prefill_resource
        if phase == "decode" and self.decode_resource:
            return self.decode_resource
        return self.resource


@dataclass(frozen=True)
class HardwareConfig:
    count: int
    peak_flops_tflops: float
    hbm_bandwidth_gb_s: float
    memory_gb: float
    compute_efficiency: float = 0.7
    memory_efficiency: float = 0.7
    kernel_overhead_us: float = 10.0
    interconnect_bandwidth_gb_s: float = 400.0
    interconnect_latency_us: float = 5.0


@dataclass(frozen=True)
class NetworkConfig:
    uplink_gbps: float
    downlink_gbps: float
    rtt_ms: float
    efficiency: float = 1.0


@dataclass(frozen=True)
class StaticPolicyConfig:
    max_batch_size: int
    max_batched_tokens: int
    prefill_token_budget: int
    prefill_chunk_size: int
    pipeline_depth: int = 4
    max_outstanding_prefill_chunks: int = 8
    dispatch_mode: str = "eager"
    scheduler: str = "fcfs"
    continuous_batching: bool = True


@dataclass(frozen=True)
class AttentionBackendConfig:
    mode: str = "unified"


@dataclass(frozen=True)
class KVCacheConfig:
    enabled: bool = False
    block_size_tokens: int = 16
    num_blocks: int = 0
    watermark: float = 0.0
    prefix_caching: bool = False
    enable_preemption: bool = True
    preemption_mode: str = "recompute"


@dataclass(frozen=True)
class PDDisaggregationConfig:
    enabled: bool = False
    kv_transfer_bandwidth_gb_s: float = 100.0
    kv_transfer_latency_ms: float = 0.1


@dataclass(frozen=True)
class SchedulerConfig:
    policy: str = "fcfs"
    max_num_seqs: int = 256
    enable_chunked_prefill: bool = True
    kv_cache: KVCacheConfig = field(default_factory=KVCacheConfig)
    pd_disaggregation: PDDisaggregationConfig = field(default_factory=PDDisaggregationConfig)


@dataclass(frozen=True)
class RequestSpec:
    request_id: int
    arrival_time_ms: float
    input_tokens: int
    output_tokens: int
    priority: int = 0
    prefix_id: str | None = None
    prefix_tokens: int = 0


@dataclass(frozen=True)
class WorkloadConfig:
    mode: str
    num_requests: int = 0
    arrival_interval_ms: float = 0.0
    arrival_process: str = "constant"
    arrival_rate_qps: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    random_seed: int = 42
    requests: tuple[RequestSpec, ...] = ()


@dataclass(frozen=True)
class SLOConfig:
    ttft_ms: float
    tpot_ms: float


@dataclass(frozen=True)
class SimulationOptions:
    trace_enabled: bool = True
    max_time_s: float = 3600.0


@dataclass(frozen=True)
class SimulationConfig:
    model: ModelConfig
    stages: tuple[StageConfig, ...]
    hardware: dict[str, HardwareConfig]
    network: NetworkConfig
    static_policy: StaticPolicyConfig
    attention_backend: AttentionBackendConfig
    scheduler: SchedulerConfig
    workload: WorkloadConfig
    slo: SLOConfig | None = None
    simulation: SimulationOptions = field(default_factory=SimulationOptions)

    def stage(self, name: str) -> StageConfig:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(name)


def _required(data: dict[str, Any], key: str, location: str) -> Any:
    if key not in data:
        raise ConfigError(f"missing required field: {location}.{key}")
    return data[key]


def _parse_pp_layer_ranges(item: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    raw_ranges = item.get("pp_layer_ranges", [])
    if not isinstance(raw_ranges, list):
        raise ConfigError("topology.stages[].pp_layer_ranges must be a list")
    if any(not isinstance(value, list) or len(value) != 2 for value in raw_ranges):
        raise ConfigError("topology.stages[].pp_layer_ranges must contain [start, end]")
    return tuple((int(value[0]), int(value[1])) for value in raw_ranges)


def load_config(path: str | Path) -> SimulationConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    model_data = _required(data, "model", "root")
    hf_config_path = model_data.get("hf_config_path")
    if hf_config_path:
        profile_path = config_path.parent / str(hf_config_path)
        with profile_path.open("r", encoding="utf-8") as handle:
            hf_data = json.load(handle)
        # Experiment-level fields intentionally override the checked-in HF profile.
        data["model"] = {**hf_data, **model_data}
    return parse_config(data)


def parse_config(data: dict[str, Any]) -> SimulationConfig:
    model_data = _required(data, "model", "root")
    hidden_size = int(_required(model_data, "hidden_size", "model"))
    num_attention_heads = int(model_data.get("num_attention_heads", 1))
    dtype = str(model_data.get("dtype", model_data.get("torch_dtype", "bfloat16")))
    inferred_dtype_bytes = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1,
        "float8_e4m3fn": 1,
    }.get(dtype)
    if inferred_dtype_bytes is None and "dtype_bytes" not in model_data:
        raise ConfigError(f"unsupported model dtype: {dtype}")
    model = ModelConfig(
        name=_required(model_data, "name", "model"),
        architecture=str(model_data.get("model_type", model_data.get("architecture", "qwen3"))),
        num_layers=int(
            model_data["num_layers"]
            if "num_layers" in model_data
            else _required(model_data, "num_hidden_layers", "model")
        ),
        hidden_size=hidden_size,
        intermediate_size=int(model_data.get("intermediate_size", 4 * hidden_size)),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=int(model_data.get("num_key_value_heads", num_attention_heads)),
        head_dim=int(model_data.get("head_dim", hidden_size // num_attention_heads)),
        vocab_size=int(model_data.get("vocab_size", 0)),
        dtype=dtype,
        dtype_bytes=int(model_data.get("dtype_bytes", inferred_dtype_bytes or 2)),
    )

    topology_data = _required(data, "topology", "root")
    stages = tuple(
        StageConfig(
            name=str(_required(item, "name", "topology.stages[]")),
            layer_start=int(_required(item, "layer_start", "topology.stages[]")),
            layer_end=int(_required(item, "layer_end", "topology.stages[]")),
            resource=str(_required(item, "resource", "topology.stages[]")),
            tp_degree=int(item.get("tp_degree", 1)),
            pp_degree=int(item.get("pp_degree", 1)),
            pp_layer_ranges=_parse_pp_layer_ranges(item),
            replicas=int(item.get("replicas", 1)),
            prefill_resource=str(item["prefill_resource"]) if item.get("prefill_resource") else None,
            decode_resource=str(item["decode_resource"]) if item.get("decode_resource") else None,
        )
        for item in _required(topology_data, "stages", "topology")
    )

    hardware = {
        name: HardwareConfig(
            count=int(_required(item, "count", f"hardware.{name}")),
            peak_flops_tflops=float(
                _required(item, "peak_flops_tflops", f"hardware.{name}")
            ),
            hbm_bandwidth_gb_s=float(
                _required(item, "hbm_bandwidth_gb_s", f"hardware.{name}")
            ),
            memory_gb=float(_required(item, "memory_gb", f"hardware.{name}")),
            compute_efficiency=float(item.get("compute_efficiency", 0.7)),
            memory_efficiency=float(item.get("memory_efficiency", 0.7)),
            kernel_overhead_us=float(item.get("kernel_overhead_us", 10.0)),
            interconnect_bandwidth_gb_s=float(
                item.get("interconnect_bandwidth_gb_s", 400.0)
            ),
            interconnect_latency_us=float(item.get("interconnect_latency_us", 5.0)),
        )
        for name, item in _required(data, "hardware", "root").items()
    }

    network_data = _required(data, "network", "root")
    network = NetworkConfig(
        uplink_gbps=float(_required(network_data, "uplink_gbps", "network")),
        downlink_gbps=float(_required(network_data, "downlink_gbps", "network")),
        rtt_ms=float(_required(network_data, "rtt_ms", "network")),
        efficiency=float(network_data.get("efficiency", 1.0)),
    )

    policy_data = _required(data, "static_policy", "root")
    configured_batch_size = policy_data.get(
        "batch_size", policy_data.get("max_batch_size")
    )
    if configured_batch_size is None:
        raise ConfigError("missing required field: static_policy.batch_size")
    if (
        "batch_size" in policy_data
        and "max_batch_size" in policy_data
        and int(policy_data["batch_size"]) != int(policy_data["max_batch_size"])
    ):
        raise ConfigError("static_policy batch_size aliases disagree")
    if (
        "continuous_batching" in policy_data
        and "continuous_batch" in policy_data
        and bool(policy_data["continuous_batching"])
        != bool(policy_data["continuous_batch"])
    ):
        raise ConfigError("static_policy continuous batching aliases disagree")
    continuous_value = policy_data.get(
        "continuous_batching", policy_data.get("continuous_batch", True)
    )
    if isinstance(continuous_value, dict):
        continuous_value = continuous_value.get("enabled", True)
    if not isinstance(continuous_value, bool):
        raise ConfigError("continuous_batching.enabled must be boolean")
    policy = StaticPolicyConfig(
        max_batch_size=int(configured_batch_size),
        max_batched_tokens=int(
            _required(policy_data, "max_batched_tokens", "static_policy")
        ),
        prefill_token_budget=int(
            policy_data.get(
                "prefill_token_budget",
                policy_data.get("max_batched_tokens", 0),
            )
        ),
        prefill_chunk_size=int(
            _required(policy_data, "prefill_chunk_size", "static_policy")
        ),
        pipeline_depth=int(policy_data.get("pipeline_depth", 4)),
        max_outstanding_prefill_chunks=int(
            policy_data.get(
                "max_outstanding_prefill_chunks",
                policy_data.get("max_outstanding_batches", 8),
            )
        ),
        dispatch_mode=str(policy_data.get("dispatch_mode", "eager")),
        scheduler=str(policy_data.get("scheduler", "fcfs")),
        continuous_batching=continuous_value,
    )

    attention_backend = AttentionBackendConfig(
        mode=str(data.get("attention_backend", {}).get("mode", "unified"))
    )
    scheduler_data = data.get("scheduler", {})
    kv_data = scheduler_data.get("kv_cache", {})
    pd_data = scheduler_data.get("pd_disaggregation", {})
    scheduler = SchedulerConfig(
        policy=str(scheduler_data.get("policy", policy.scheduler)),
        max_num_seqs=int(scheduler_data.get("max_num_seqs", policy.max_batch_size)),
        enable_chunked_prefill=bool(scheduler_data.get("enable_chunked_prefill", True)),
        kv_cache=KVCacheConfig(
            enabled=bool(kv_data.get("enabled", False)),
            block_size_tokens=int(kv_data.get("block_size_tokens", 16)),
            num_blocks=int(kv_data.get("num_blocks", 0)),
            watermark=float(kv_data.get("watermark", 0.0)),
            prefix_caching=bool(kv_data.get("prefix_caching", False)),
            enable_preemption=bool(kv_data.get("enable_preemption", True)),
            preemption_mode=str(kv_data.get("preemption_mode", "recompute")),
        ),
        pd_disaggregation=PDDisaggregationConfig(
            enabled=bool(pd_data.get("enabled", False)),
            kv_transfer_bandwidth_gb_s=float(pd_data.get("kv_transfer_bandwidth_gb_s", 100.0)),
            kv_transfer_latency_ms=float(pd_data.get("kv_transfer_latency_ms", 0.1)),
        ),
    )

    workload_data = _required(data, "workload", "root")
    request_specs = tuple(
        RequestSpec(
            request_id=int(_required(item, "request_id", "workload.requests[]")),
            arrival_time_ms=float(
                _required(item, "arrival_time_ms", "workload.requests[]")
            ),
            input_tokens=int(_required(item, "input_tokens", "workload.requests[]")),
            output_tokens=int(_required(item, "output_tokens", "workload.requests[]")),
            priority=int(item.get("priority", 0)),
            prefix_id=str(item["prefix_id"]) if item.get("prefix_id") is not None else None,
            prefix_tokens=int(item.get("prefix_tokens", 0)),
        )
        for item in workload_data.get("requests", [])
    )
    workload = WorkloadConfig(
        mode=str(_required(workload_data, "mode", "workload")),
        num_requests=int(workload_data.get("num_requests", 0)),
        arrival_interval_ms=float(workload_data.get("arrival_interval_ms", 0.0)),
        arrival_process=str(workload_data.get("arrival_process", "constant")),
        arrival_rate_qps=(
            float(workload_data["arrival_rate_qps"])
            if "arrival_rate_qps" in workload_data
            else None
        ),
        input_tokens=int(workload_data.get("input_tokens", 0)),
        output_tokens=int(workload_data.get("output_tokens", 0)),
        random_seed=int(workload_data.get("random_seed", 42)),
        requests=request_specs,
    )

    slo_data = data.get("slo")
    slo = (
        SLOConfig(
            ttft_ms=float(_required(slo_data, "ttft_ms", "slo")),
            tpot_ms=float(_required(slo_data, "tpot_ms", "slo")),
        )
        if slo_data is not None
        else None
    )
    simulation_data = data.get("simulation", {})
    simulation = SimulationOptions(
        trace_enabled=bool(simulation_data.get("trace_enabled", True)),
        max_time_s=float(simulation_data.get("max_time_s", 3600.0)),
    )

    config = SimulationConfig(
        model=model,
        stages=stages,
        hardware=hardware,
        network=network,
        static_policy=policy,
        attention_backend=attention_backend,
        scheduler=scheduler,
        workload=workload,
        slo=slo,
        simulation=simulation,
    )
    validate_config(config)
    return config


def validate_config(config: SimulationConfig) -> None:
    if config.model.num_layers <= 0 or config.model.hidden_size <= 0:
        raise ConfigError("model dimensions must be positive")
    if config.model.dtype_bytes <= 0:
        raise ConfigError("model.dtype_bytes must be positive")
    if config.model.architecture != "qwen3":
        raise ConfigError("only model architecture 'qwen3' is supported")
    if min(
        config.model.intermediate_size,
        config.model.num_attention_heads,
        config.model.num_key_value_heads,
        config.model.head_dim,
    ) <= 0:
        raise ConfigError("Qwen3 model dimensions must be positive")
    if config.model.num_attention_heads % config.model.num_key_value_heads != 0:
        raise ConfigError("num_attention_heads must be divisible by num_key_value_heads")

    expected_names = ["edge_front", "cloud_middle", "edge_tail"]
    if [stage.name for stage in config.stages] != expected_names:
        raise ConfigError(f"topology stages must be ordered as {expected_names}")
    cursor = 0
    for stage in config.stages:
        if stage.layer_start != cursor or stage.layer_end <= stage.layer_start:
            raise ConfigError("topology layer ranges must be contiguous and non-empty")
        cursor = stage.layer_end
        stage_resources = {
            stage.resource_for_phase("prefill"),
            stage.resource_for_phase("decode"),
        }
        if stage_resources - config.hardware.keys():
            raise ConfigError(f"unknown hardware resource in stage {stage.name}")
        if stage.tp_degree <= 0 or stage.pp_degree <= 0 or stage.replicas <= 0:
            raise ConfigError(f"parallel degrees must be positive for {stage.name}")
        if stage.pp_degree > stage.num_layers:
            raise ConfigError(f"PP exceeds layer count for {stage.name}")
        if stage.pp_layer_ranges:
            if len(stage.pp_layer_ranges) != stage.pp_degree:
                raise ConfigError(f"pp_layer_ranges must match PP degree for {stage.name}")
            if (
                stage.pp_layer_ranges[0][0] != stage.layer_start
                or stage.pp_layer_ranges[-1][1] != stage.layer_end
                or any(
                    start >= end
                    for start, end in stage.pp_layer_ranges
                )
                or any(
                    left[1] != right[0]
                    for left, right in zip(
                        stage.pp_layer_ranges, stage.pp_layer_ranges[1:]
                    )
                )
            ):
                raise ConfigError(
                    f"pp_layer_ranges must be contiguous and cover {stage.name}"
                )
        required_devices = stage.tp_degree * stage.pp_degree * stage.replicas
        if any(required_devices > config.hardware[resource].count for resource in stage_resources):
            raise ConfigError(
                f"TP times PP times replicas exceeds device count for {stage.name}"
            )
        if config.model.num_attention_heads % stage.tp_degree != 0:
            raise ConfigError(f"attention heads are not divisible by TP for {stage.name}")
        if config.model.intermediate_size % stage.tp_degree != 0:
            raise ConfigError(f"intermediate size is not divisible by TP for {stage.name}")
        if (
            config.model.num_key_value_heads >= stage.tp_degree
            and config.model.num_key_value_heads % stage.tp_degree != 0
        ):
            raise ConfigError(f"KV heads are not divisible by TP for {stage.name}")
        if (
            stage.layer_end == config.model.num_layers
            and config.model.vocab_size
            and config.model.vocab_size % stage.tp_degree != 0
        ):
            raise ConfigError(f"vocabulary is not divisible by TP for {stage.name}")
    if cursor != config.model.num_layers:
        raise ConfigError("topology layer ranges must cover all model layers")

    shared_resource_layouts: dict[str, tuple[int, int, int]] = {}
    for stage in config.stages:
        layout = (stage.tp_degree, stage.pp_degree, stage.replicas)
        previous_layout = shared_resource_layouts.setdefault(stage.resource, layout)
        if previous_layout != layout:
            raise ConfigError(
                f"stages sharing {stage.resource} must use the same TP/PP/replica layout"
            )

    for name, hardware in config.hardware.items():
        if min(
            hardware.count,
            hardware.peak_flops_tflops,
            hardware.hbm_bandwidth_gb_s,
            hardware.memory_gb,
        ) <= 0:
            raise ConfigError(f"hardware values must be positive: {name}")
        if not 0 < hardware.compute_efficiency <= 1:
            raise ConfigError(f"compute_efficiency must be in (0, 1]: {name}")
        if not 0 < hardware.memory_efficiency <= 1:
            raise ConfigError(f"memory_efficiency must be in (0, 1]: {name}")
    if min(config.network.uplink_gbps, config.network.downlink_gbps) <= 0:
        raise ConfigError("network bandwidth must be positive")
    if config.network.rtt_ms < 0:
        raise ConfigError("network.rtt_ms cannot be negative")
    if not 0 < config.network.efficiency <= 1:
        raise ConfigError("network.efficiency must be in (0, 1]")

    policy = config.static_policy
    if min(
        policy.max_batch_size,
        policy.max_batched_tokens,
        policy.prefill_token_budget,
        policy.prefill_chunk_size,
        policy.pipeline_depth,
        policy.max_outstanding_prefill_chunks,
    ) <= 0:
        raise ConfigError("static policy limits must be positive")
    if policy.prefill_chunk_size > policy.max_batched_tokens:
        raise ConfigError("prefill_chunk_size cannot exceed max_batched_tokens")
    if policy.prefill_chunk_size > policy.prefill_token_budget:
        raise ConfigError("prefill_chunk_size cannot exceed prefill_token_budget")
    if policy.prefill_token_budget > policy.max_batched_tokens:
        raise ConfigError("prefill_token_budget cannot exceed max_batched_tokens")
    if policy.dispatch_mode != "eager":
        raise ConfigError("only dispatch_mode='eager' is supported in the MVP")
    if config.scheduler.policy not in {"fcfs", "priority", "shortest_prefill"}:
        raise ConfigError("scheduler.policy must be fcfs, priority, or shortest_prefill")
    if config.scheduler.max_num_seqs <= 0:
        raise ConfigError("scheduler.max_num_seqs must be positive")
    if config.attention_backend.mode not in {"unified", "separate"}:
        raise ConfigError("attention_backend.mode must be unified or separate")
    kv = config.scheduler.kv_cache
    if kv.enabled and kv.num_blocks <= 0:
        raise ConfigError("enabled KV cache requires positive num_blocks")
    if kv.block_size_tokens <= 0 or not 0 <= kv.watermark < 1:
        raise ConfigError("invalid KV block size or watermark")
    pd = config.scheduler.pd_disaggregation
    if pd.kv_transfer_bandwidth_gb_s <= 0 or pd.kv_transfer_latency_ms < 0:
        raise ConfigError("invalid PD KV transfer bandwidth or latency")
    if pd.enabled and any(
        not stage.prefill_resource
        or not stage.decode_resource
        or stage.prefill_resource == stage.decode_resource
        for stage in config.stages
    ):
        raise ConfigError("PD requires distinct prefill_resource/decode_resource per stage")

    workload = config.workload
    if workload.mode == "synthetic":
        if min(workload.num_requests, workload.input_tokens, workload.output_tokens) <= 0:
            raise ConfigError("synthetic workload sizes must be positive")
        if workload.arrival_process not in {"constant", "poisson"}:
            raise ConfigError("arrival_process must be constant or poisson")
        if workload.arrival_process == "constant" and workload.arrival_interval_ms < 0:
            raise ConfigError("arrival_interval_ms cannot be negative")
        if workload.arrival_process == "poisson" and (
            workload.arrival_rate_qps is None or workload.arrival_rate_qps <= 0
        ):
            raise ConfigError("poisson workload requires positive arrival_rate_qps")
    elif workload.mode == "trace":
        if not workload.requests:
            raise ConfigError("trace workload requires at least one request")
        ids = [request.request_id for request in workload.requests]
        if len(ids) != len(set(ids)):
            raise ConfigError("trace request IDs must be unique")
        if any(
            request.arrival_time_ms < 0
            or request.input_tokens <= 0
            or request.output_tokens <= 0
            for request in workload.requests
        ):
            raise ConfigError("trace request values are invalid")
    else:
        raise ConfigError("workload.mode must be synthetic or trace")

    # This MVP does not model paging. At minimum, each stage's sharded weights
    # must fit on one participating GPU.
    weights_by_pipeline_rank: dict[tuple[str, int], float] = {}
    for stage in config.stages:
        for pipeline_rank in range(stage.pp_degree):
            layer_start, layer_end = stage.pipeline_layer_range(pipeline_rank)
            weight_bytes = (
                (layer_end - layer_start)
                * config.model.parameters_per_layer
                * config.model.dtype_bytes
                / stage.tp_degree
            )
            if layer_start == 0:
                weight_bytes += (
                    config.model.vocab_size
                    * config.model.hidden_size
                    * config.model.dtype_bytes
                    / stage.tp_degree
                )
            if layer_end == config.model.num_layers:
                weight_bytes += (
                    config.model.vocab_size
                    * config.model.hidden_size
                    * config.model.dtype_bytes
                    / stage.tp_degree
                )
            key = (stage.resource, pipeline_rank)
            weights_by_pipeline_rank[key] = (
                weights_by_pipeline_rank.get(key, 0.0) + weight_bytes
            )
    for (resource, pipeline_rank), weight_bytes in weights_by_pipeline_rank.items():
        if weight_bytes > config.hardware[resource].memory_gb * 1e9:
            raise ConfigError(
                f"{resource} PP rank {pipeline_rank} weights do not fit in per-GPU memory"
            )
