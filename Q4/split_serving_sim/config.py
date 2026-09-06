from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
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
    attention_bias: bool = False
    tie_word_embeddings: bool = False

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
        norms = 2 * self.hidden_size
        if self.architecture == "qwen3":
            norms += 2 * self.head_dim
        biases = self.query_width + 2 * self.kv_width if self.attention_bias else 0
        return attention + mlp + norms + biases


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

    prefill_tp_degree: int | None = None
    decode_tp_degree: int | None = None
    prefill_replicas: int | None = None
    decode_replicas: int | None = None

    def for_phase(self, phase: str) -> StageConfig:
        return replace(self, resource=self.resource_for_phase(phase),
                       tp_degree=getattr(self, f"{phase}_tp_degree") or self.tp_degree,
                       replicas=getattr(self, f"{phase}_replicas") or self.replicas)

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
    activation_tensor_count: int = 1
    protocol_overhead_bytes: int = 0
    sender_overhead_ms: float = 0.0
    receiver_overhead_ms: float = 0.0


@dataclass(frozen=True)
class WanCalibrationKnot:
    activation_mib: float
    upload_components_ms: tuple[tuple[str, float], ...]
    download_components_ms: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class WanCalibrationConfig:
    enabled: bool = False
    variant: str = ""
    source: str = ""
    one_way_latency_ms: float = 0.0
    knots: tuple[WanCalibrationKnot, ...] = ()


@dataclass(frozen=True)
class DataPathConfig:
    """Host/device staging and transport options used around each WAN message."""

    enabled: bool = False
    ipc_mode: str = "copy"
    wire_fast: bool = False
    tcp_buffer_mib: float = 0.0
    d2h_bandwidth_gb_s: float = 24.0
    h2d_bandwidth_gb_s: float = 24.0
    host_memory_bandwidth_gb_s: float = 20.0
    ipc_bandwidth_gb_s: float = 12.0
    ipc_latency_ms: float = 0.05
    legacy_pack_copies: float = 2.0
    wire_fast_pack_copies: float = 1.0
    copy_ipc_copies: float = 2.0
    shm_ipc_copies: float = 0.0
    wan_calibration: WanCalibrationConfig = field(
        default_factory=WanCalibrationConfig
    )


@dataclass(frozen=True)
class HostSubmissionConfig:
    enabled: bool = False
    submit_us: float = 6.0
    submit_us_by_type: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class ExecutionConfig:
    mode: str = "pipelined"
    preserve_batch_across_stages: bool = False
    max_inflight_transactions: int = 0
    buffer_pool_mib: float = 0.0
    host_submission: HostSubmissionConfig = field(default_factory=HostSubmissionConfig)


@dataclass(frozen=True)
class ProfileSampleConfig:
    operator_type: str
    latency_ms: float
    roofline_ms: float | None = None
    signature: str | None = None
    match: tuple[tuple[str, str], ...] = ()
    source: str = ""


@dataclass(frozen=True)
class PerformanceProfileConfig:
    enabled: bool = False
    samples: tuple[ProfileSampleConfig, ...] = ()


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
class OperatorBackendConfig:
    """Physical operator plan used after framework/compiler fusion."""

    name: str = "logical"
    version: str = ""
    fused_qkv: bool = False
    fused_gate_up: bool = False
    fused_silu_mul: bool = False
    fused_add_rms_norm: bool = False


@dataclass(frozen=True)
class KVCacheConfig:
    enabled: bool = False
    allocation_mode: str = "preallocate"
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
    chunk_transfer: bool = False
    control_channel: bool = True
    control_latency_ms: float = 0.0
    admission_enabled: bool = False
    reserve_p_ms: float = 0.24
    reserve_d_ms: float = 0.11
    release_p_ms: float = 0.10
    release_d_ms: float = 0.10
    release_e_ms: float = 0.10
    local_rpc_ms: float = 1.1
    rpc_keepalive_s: float = 1.0
    prefill_window: int = 3
    decode_window: int = 2


@dataclass(frozen=True)
class SchedulerConfig:
    policy: str = "fcfs"
    max_num_seqs: int = 256
    enable_chunked_prefill: bool = True
    decode_first: bool = False
    allow_mixed_batch: bool = True
    prefer_ready_back: bool = False
    max_consecutive_decode_batches: int = 8
    max_decode_tokens_per_batch: int = 0
    max_prefill_wait_ms: float = 0.0
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
    concurrency: int = 0
    warmup_requests: int = 0
    measurement_duration_s: float = 0.0
    requests: tuple[RequestSpec, ...] = ()


@dataclass(frozen=True)
class SLOConfig:
    ttft_ms: float
    tpot_ms: float
    target_attainment: float = 0.99


@dataclass(frozen=True)
class SimulationOptions:
    trace_enabled: bool = True
    max_time_s: float = 3600.0
    max_trace_records: int = 20_000
    max_detailed_trace_records: int = 200


@dataclass(frozen=True)
class SimulationConfig:
    model: ModelConfig
    stages: tuple[StageConfig, ...]
    hardware: dict[str, HardwareConfig]
    network: NetworkConfig
    data_path: DataPathConfig
    execution: ExecutionConfig
    performance_profile: PerformanceProfileConfig
    static_policy: StaticPolicyConfig
    attention_backend: AttentionBackendConfig
    operator_backend: OperatorBackendConfig
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


def parse_performance_profile(data: dict[str, Any]) -> PerformanceProfileConfig:
    samples = tuple(
        ProfileSampleConfig(
            operator_type=str(
                _required(item, "operator_type", "performance_profile.samples[]")
            ),
            latency_ms=float(
                _required(item, "latency_ms", "performance_profile.samples[]")
            ),
            roofline_ms=(
                float(item["roofline_ms"])
                if item.get("roofline_ms") is not None
                else None
            ),
            signature=(str(item["signature"]) if item.get("signature") else None),
            match=tuple(
                sorted(
                    (str(key), str(value))
                    for key, value in item.get("match", {}).items()
                )
            ),
            source=str(item.get("source", "")),
        )
        for item in data.get("samples", [])
    )
    return PerformanceProfileConfig(
        enabled=bool(data.get("enabled", bool(samples))), samples=samples
    )


def load_performance_profile(path: str | Path) -> PerformanceProfileConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return parse_performance_profile(json.load(handle))


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
    profile_data = data.get("performance_profile", {})
    profile_path = profile_data.get("path")
    if profile_path:
        resolved_profile_path = config_path.parent / str(profile_path)
        with resolved_profile_path.open("r", encoding="utf-8") as handle:
            loaded_profile = json.load(handle)
        data["performance_profile"] = {
            **loaded_profile,
            **{key: value for key, value in profile_data.items() if key != "path"},
        }
    data_path = data.get("data_path", {})
    calibration = data_path.get("wan_calibration", {})
    calibration_path = calibration.get("path")
    if calibration_path:
        resolved = config_path.parent / str(calibration_path)
        with resolved.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        variant = str(calibration.get("variant", ""))
        variants = loaded.get("variants", {})
        if variant not in variants:
            raise ConfigError(
                f"WAN calibration variant {variant!r} not found in {resolved}"
            )
        data["data_path"] = {
            **data_path,
            "wan_calibration": {
                **variants[variant],
                **{
                    key: value
                    for key, value in calibration.items()
                    if key not in {"path"}
                },
                "source": loaded.get("source", ""),
            },
        }
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
        attention_bias=bool(model_data.get("attention_bias", False)),
        tie_word_embeddings=bool(model_data.get("tie_word_embeddings", False)),
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
            **{key: int(item[key]) if key in item else None for key in
               ("prefill_tp_degree", "decode_tp_degree", "prefill_replicas", "decode_replicas")},
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
        activation_tensor_count=int(network_data.get("activation_tensor_count", 1)),
        protocol_overhead_bytes=int(network_data.get("protocol_overhead_bytes", 0)),
        sender_overhead_ms=float(network_data.get("sender_overhead_ms", 0.0)),
        receiver_overhead_ms=float(network_data.get("receiver_overhead_ms", 0.0)),
    )
    data_path_data = data.get("data_path", {})
    calibration_data = data_path_data.get("wan_calibration", {})
    calibration_knots = tuple(
        WanCalibrationKnot(
            activation_mib=float(_required(item, "activation_mib", "data_path.wan_calibration.knots[]")),
            upload_components_ms=tuple(
                (str(name), float(value))
                for name, value in _required(item, "upload_components_ms", "data_path.wan_calibration.knots[]").items()
            ),
            download_components_ms=tuple(
                (str(name), float(value))
                for name, value in _required(item, "download_components_ms", "data_path.wan_calibration.knots[]").items()
            ),
        )
        for item in calibration_data.get("knots", [])
    )
    data_path = DataPathConfig(
        enabled=bool(data_path_data.get("enabled", False)),
        ipc_mode=str(data_path_data.get("ipc_mode", "copy")),
        wire_fast=bool(data_path_data.get("wire_fast", False)),
        tcp_buffer_mib=float(data_path_data.get("tcp_buffer_mib", 0.0)),
        d2h_bandwidth_gb_s=float(data_path_data.get("d2h_bandwidth_gb_s", 24.0)),
        h2d_bandwidth_gb_s=float(data_path_data.get("h2d_bandwidth_gb_s", 24.0)),
        host_memory_bandwidth_gb_s=float(data_path_data.get("host_memory_bandwidth_gb_s", 20.0)),
        ipc_bandwidth_gb_s=float(data_path_data.get("ipc_bandwidth_gb_s", 12.0)),
        ipc_latency_ms=float(data_path_data.get("ipc_latency_ms", 0.05)),
        legacy_pack_copies=float(data_path_data.get("legacy_pack_copies", 2.0)),
        wire_fast_pack_copies=float(data_path_data.get("wire_fast_pack_copies", 1.0)),
        copy_ipc_copies=float(data_path_data.get("copy_ipc_copies", 2.0)),
        shm_ipc_copies=float(data_path_data.get("shm_ipc_copies", 0.0)),
        wan_calibration=WanCalibrationConfig(
            enabled=bool(calibration_data.get("enabled", False)),
            variant=str(calibration_data.get("variant", "")),
            source=str(calibration_data.get("source", "")),
            one_way_latency_ms=float(
                calibration_data.get("one_way_latency_ms", network.rtt_ms / 2.0)
            ),
            knots=calibration_knots,
        ),
    )
    execution_data = data.get("execution", {})
    execution_mode = str(execution_data.get("mode", "pipelined"))
    host_data = execution_data.get("host_submission", {})
    execution = ExecutionConfig(
        mode=execution_mode,
        preserve_batch_across_stages=bool(
            execution_data.get(
                "preserve_batch_across_stages",
                execution_mode == "synchronous_rpc",
            )
        ),
        max_inflight_transactions=int(execution_data.get("max_inflight_transactions", 0)),
        buffer_pool_mib=float(execution_data.get("buffer_pool_mib", 0.0)),
        host_submission=HostSubmissionConfig(
            enabled=bool(host_data.get("enabled", False)),
            submit_us=float(host_data.get("submit_us", 6.0)),
            submit_us_by_type=tuple(
                (str(k), float(v)) for k, v in host_data.get("submit_us_by_type", {}).items()
            ),
        ),
    )
    profile_data = data.get("performance_profile", {})
    performance_profile = parse_performance_profile(profile_data)

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
    operator_backend_data = data.get("operator_backend", {})
    operator_backend_name = str(operator_backend_data.get("name", "logical"))
    vllm_defaults = operator_backend_name == "vllm"
    operator_backend = OperatorBackendConfig(
        name=operator_backend_name,
        version=str(operator_backend_data.get("version", "")),
        fused_qkv=bool(operator_backend_data.get("fused_qkv", vllm_defaults)),
        fused_gate_up=bool(
            operator_backend_data.get("fused_gate_up", vllm_defaults)
        ),
        fused_silu_mul=bool(
            operator_backend_data.get("fused_silu_mul", vllm_defaults)
        ),
        fused_add_rms_norm=bool(
            operator_backend_data.get("fused_add_rms_norm", vllm_defaults)
        ),
    )
    scheduler_data = data.get("scheduler", {})
    kv_data = scheduler_data.get("kv_cache", {})
    pd_data = scheduler_data.get("pd_disaggregation", {})
    scheduler = SchedulerConfig(
        policy=str(scheduler_data.get("policy", policy.scheduler)),
        max_num_seqs=int(scheduler_data.get("max_num_seqs", policy.max_batch_size)),
        enable_chunked_prefill=bool(scheduler_data.get("enable_chunked_prefill", True)),
        decode_first=bool(scheduler_data.get("decode_first", False)),
        allow_mixed_batch=bool(scheduler_data.get("allow_mixed_batch", True)),
        prefer_ready_back=bool(scheduler_data.get("prefer_ready_back", False)),
        max_consecutive_decode_batches=int(scheduler_data.get("max_consecutive_decode_batches", 8)),
        max_decode_tokens_per_batch=int(scheduler_data.get("max_decode_tokens_per_batch", 0)),
        max_prefill_wait_ms=float(scheduler_data.get("max_prefill_wait_ms", 0.0)),
        kv_cache=KVCacheConfig(
            enabled=bool(kv_data.get("enabled", False)),
            allocation_mode=str(kv_data.get("allocation_mode", "preallocate")),
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
            chunk_transfer=bool(pd_data.get("chunk_transfer", False)),
            control_channel=bool(pd_data.get("control_channel", True)),
            control_latency_ms=float(pd_data.get("control_latency_ms", 0.0)),
            admission_enabled=bool(pd_data.get("admission_enabled", False)),
            **{key: float(pd_data.get(key, default)) for key,default in
               (("reserve_p_ms",0.24),("reserve_d_ms",0.11),("release_p_ms",0.1),
                ("release_d_ms",0.1),("release_e_ms",0.1),("local_rpc_ms",1.1),("rpc_keepalive_s",1.0))},
            prefill_window=int(pd_data.get("prefill_window", 3)),
            decode_window=int(pd_data.get("decode_window", 2)),
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
        concurrency=int(workload_data.get("concurrency", 0)),
        warmup_requests=int(workload_data.get("warmup_requests", 0)),
        measurement_duration_s=float(
            workload_data.get("measurement_duration_s", 0.0)
        ),
        requests=request_specs,
    )

    slo_data = data.get("slo")
    slo = (
        SLOConfig(
            ttft_ms=float(_required(slo_data, "ttft_ms", "slo")),
            tpot_ms=float(_required(slo_data, "tpot_ms", "slo")),
            target_attainment=float(slo_data.get("target_attainment", 0.99)),
        )
        if slo_data is not None
        else None
    )
    simulation_data = data.get("simulation", {})
    simulation = SimulationOptions(
        trace_enabled=bool(simulation_data.get("trace_enabled", True)),
        max_time_s=float(simulation_data.get("max_time_s", 3600.0)),
        max_trace_records=int(simulation_data.get("max_trace_records", 20_000)),
        max_detailed_trace_records=int(
            simulation_data.get("max_detailed_trace_records", 200)
        ),
    )

    config = SimulationConfig(
        model=model,
        stages=stages,
        hardware=hardware,
        network=network,
        data_path=data_path,
        execution=execution,
        performance_profile=performance_profile,
        static_policy=policy,
        attention_backend=attention_backend,
        operator_backend=operator_backend,
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
    if config.model.architecture not in {"qwen2", "qwen3"}:
        raise ConfigError("model architecture must be qwen2 or qwen3")
    if min(
        config.model.intermediate_size,
        config.model.num_attention_heads,
        config.model.num_key_value_heads,
        config.model.head_dim,
    ) <= 0:
        raise ConfigError("Qwen model dimensions must be positive")
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
        for phase in ("prefill", "decode"):
            phase_stage = stage.for_phase(phase)
            if any(value is not None and value <= 0 for value in (
                getattr(stage, f"{phase}_tp_degree"), getattr(stage, f"{phase}_replicas"))):
                raise ConfigError("phase TP/replicas must be positive")
            tp = phase_stage.tp_degree
            if tp * stage.pp_degree * phase_stage.replicas > config.hardware[phase_stage.resource].count:
                raise ConfigError(f"TP times PP times replicas exceeds device count for {stage.name}/{phase}")
            if config.model.num_attention_heads % tp or config.model.intermediate_size % tp:
                raise ConfigError(f"attention heads/intermediate size not divisible by TP for {stage.name}")
            if config.model.num_key_value_heads >= tp and config.model.num_key_value_heads % tp:
                raise ConfigError(f"KV heads not divisible by TP for {stage.name}")
            if stage.layer_end == config.model.num_layers and config.model.vocab_size % tp:
                raise ConfigError(f"vocabulary not divisible by TP for {stage.name}")
    if cursor != config.model.num_layers:
        raise ConfigError("topology layer ranges must cover all model layers")

    shared_resource_layouts: dict[str, tuple[int, int, int]] = {}
    for original in config.stages:
        for phase in ("prefill", "decode"):
            stage = original.for_phase(phase)
            layout = (stage.tp_degree, stage.pp_degree, stage.replicas)
            previous_layout = shared_resource_layouts.setdefault(stage.resource, layout)
            if previous_layout != layout:
                raise ConfigError(f"stages sharing {stage.resource} must use the same TP/PP/replica layout")

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
    if config.network.activation_tensor_count <= 0:
        raise ConfigError("network.activation_tensor_count must be positive")
    if config.network.protocol_overhead_bytes < 0:
        raise ConfigError("network.protocol_overhead_bytes cannot be negative")
    if min(config.network.sender_overhead_ms, config.network.receiver_overhead_ms) < 0:
        raise ConfigError("network staging overhead cannot be negative")
    data_path = config.data_path
    if data_path.ipc_mode not in {"copy", "shm"}:
        raise ConfigError("data_path.ipc_mode must be copy or shm")
    if data_path.tcp_buffer_mib < 0 or data_path.ipc_latency_ms < 0:
        raise ConfigError("data path sizes and latency cannot be negative")
    if min(
        data_path.d2h_bandwidth_gb_s,
        data_path.h2d_bandwidth_gb_s,
        data_path.host_memory_bandwidth_gb_s,
        data_path.ipc_bandwidth_gb_s,
    ) <= 0:
        raise ConfigError("data path bandwidths must be positive")
    if min(
        data_path.legacy_pack_copies,
        data_path.wire_fast_pack_copies,
        data_path.copy_ipc_copies,
        data_path.shm_ipc_copies,
    ) < 0:
        raise ConfigError("data path copy counts cannot be negative")
    calibration = data_path.wan_calibration
    if calibration.enabled:
        if len(calibration.knots) < 2:
            raise ConfigError("enabled WAN calibration requires at least two knots")
        sizes = [knot.activation_mib for knot in calibration.knots]
        if sizes != sorted(sizes) or len(sizes) != len(set(sizes)) or sizes[0] <= 0:
            raise ConfigError("WAN calibration knot sizes must be unique and increasing")
        if calibration.one_way_latency_ms < 0:
            raise ConfigError("WAN calibration one_way_latency_ms cannot be negative")
        if any(
            value < 0
            for knot in calibration.knots
            for _, value in knot.upload_components_ms + knot.download_components_ms
        ):
            raise ConfigError("WAN calibration component latency cannot be negative")
    if config.execution.mode not in {"pipelined", "synchronous_rpc"}:
        raise ConfigError("execution.mode must be pipelined or synchronous_rpc")
    host = config.execution.host_submission
    if any(not math.isfinite(v) or v < 0 for v in [host.submit_us, *[v for _, v in host.submit_us_by_type]]):
        raise ConfigError("host submission costs must be finite and nonnegative")
    if (
        config.execution.mode == "synchronous_rpc"
        and not config.execution.preserve_batch_across_stages
    ):
        raise ConfigError("synchronous_rpc must preserve batches across stages")
    if config.execution.max_inflight_transactions < 0 or config.execution.buffer_pool_mib < 0:
        raise ConfigError("execution in-flight limits cannot be negative")
    if min(
        config.simulation.max_trace_records,
        config.simulation.max_detailed_trace_records,
    ) < 0:
        raise ConfigError("simulation trace limits cannot be negative")
    if (
        config.simulation.max_detailed_trace_records
        > config.simulation.max_trace_records
    ):
        raise ConfigError(
            "max_detailed_trace_records cannot exceed max_trace_records"
        )
    for sample in config.performance_profile.samples:
        if not sample.operator_type or sample.latency_ms < 0:
            raise ConfigError("performance profile samples require a type and non-negative latency")
        if sample.roofline_ms is not None and sample.roofline_ms <= 0:
            raise ConfigError("performance profile roofline_ms must be positive")

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
    if config.scheduler.policy not in {
        "fcfs", "priority", "shortest_prefill", "split_poc_naive", "split_poc_pd"
    }:
        raise ConfigError(
            "scheduler.policy must be fcfs, priority, shortest_prefill, or split_poc_naive"
        )
    if (
        config.scheduler.policy == "split_poc_naive"
        and config.execution.mode != "synchronous_rpc"
    ):
        raise ConfigError("split_poc_naive requires execution.mode=synchronous_rpc")
    if config.scheduler.max_num_seqs <= 0:
        raise ConfigError("scheduler.max_num_seqs must be positive")
    if config.scheduler.max_consecutive_decode_batches <= 0:
        raise ConfigError("max_consecutive_decode_batches must be positive")
    if config.scheduler.max_decode_tokens_per_batch < 0 or config.scheduler.max_prefill_wait_ms < 0:
        raise ConfigError("scheduler fairness limits cannot be negative")
    if config.attention_backend.mode not in {"unified", "separate"}:
        raise ConfigError("attention_backend.mode must be unified or separate")
    if config.operator_backend.name not in {"logical", "vllm"}:
        raise ConfigError("operator_backend.name must be logical or vllm")
    if config.operator_backend.name == "vllm" and not all(
        (
            config.operator_backend.fused_qkv,
            config.operator_backend.fused_gate_up,
            config.operator_backend.fused_silu_mul,
            config.operator_backend.fused_add_rms_norm,
        )
    ):
        raise ConfigError(
            "the current vllm backend requires the captured QKV, GateUp, "
            "Silu+Mul, and Add+RMSNorm fusion plan"
        )
    if (
        config.operator_backend.name == "vllm"
        and config.attention_backend.mode != "unified"
    ):
        raise ConfigError(
            "the current vllm physical backend requires unified attention"
        )
    kv = config.scheduler.kv_cache
    if kv.enabled and kv.num_blocks <= 0:
        raise ConfigError("enabled KV cache requires positive num_blocks")
    if kv.allocation_mode not in {"preallocate", "on_demand"}:
        raise ConfigError("kv_cache.allocation_mode must be preallocate or on_demand")
    if kv.block_size_tokens <= 0 or not 0 <= kv.watermark < 1:
        raise ConfigError("invalid KV block size or watermark")
    pd = config.scheduler.pd_disaggregation
    if pd.kv_transfer_bandwidth_gb_s <= 0 or pd.kv_transfer_latency_ms < 0:
        raise ConfigError("invalid PD KV transfer bandwidth or latency")
    if any(not math.isfinite(value) or value < 0 for value in (
        pd.reserve_p_ms,pd.reserve_d_ms,pd.release_p_ms,pd.release_d_ms,pd.release_e_ms,
        pd.local_rpc_ms,pd.rpc_keepalive_s)):
        raise ConfigError("PD lifecycle service times must be finite and nonnegative")
    if pd.admission_enabled and (not pd.enabled or config.scheduler.policy != "split_poc_pd"
                                or not config.static_policy.continuous_batching
                                or any(stage.pp_degree != 1 for stage in config.stages)):
        raise ConfigError("PD admission requires split_poc_pd, continuous batching and PP1")
    if pd.control_latency_ms < 0:
        raise ConfigError("PD control latency cannot be negative")
    if pd.prefill_window <= 0 or pd.decode_window <= 0:
        raise ConfigError("PD windows must be positive")
    cloud = config.stage("cloud_middle")
    if pd.enabled and cloud.resource_for_phase("prefill") == cloud.resource_for_phase("decode"):
        raise ConfigError("PD requires distinct cloud prefill/decode resources")
    if pd.enabled and config.execution.mode != "pipelined":
        raise ConfigError("PD requires pipelined execution")
    if config.scheduler.policy == "split_poc_pd" and (not pd.enabled or config.scheduler.allow_mixed_batch):
        raise ConfigError("split_poc_pd requires PD and unmixed batches")
    if pd.enabled and config.scheduler.kv_cache.enabled and config.scheduler.kv_cache.enable_preemption:
        raise ConfigError("PD preemption ownership is not modeled; disable enable_preemption")
    if pd.enabled and config.scheduler.kv_cache.prefix_caching:
        raise ConfigError("PD prefix cache ownership is not modeled; disable prefix_caching")

    workload = config.workload
    if workload.mode in {"synthetic", "closed_loop"}:
        if min(workload.num_requests, workload.input_tokens, workload.output_tokens) <= 0:
            raise ConfigError("generated workload sizes must be positive")
        if workload.mode == "synthetic":
            if workload.arrival_process not in {"constant", "poisson"}:
                raise ConfigError("arrival_process must be constant or poisson")
            if workload.arrival_process == "constant" and workload.arrival_interval_ms < 0:
                raise ConfigError("arrival_interval_ms cannot be negative")
            if workload.arrival_process == "poisson" and (
                workload.arrival_rate_qps is None or workload.arrival_rate_qps <= 0
            ):
                raise ConfigError("poisson workload requires positive arrival_rate_qps")
        elif (
            workload.concurrency <= 0
            or workload.num_requests - workload.warmup_requests < workload.concurrency
        ):
            raise ConfigError("closed_loop requires positive concurrency covered by num_requests")
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
        raise ConfigError("workload.mode must be synthetic, closed_loop, or trace")
    request_count = (
        len(workload.requests) if workload.mode == "trace" else workload.num_requests
    )
    if not 0 <= workload.warmup_requests < request_count:
        raise ConfigError("warmup_requests must leave at least one measured request")
    if workload.measurement_duration_s < 0:
        raise ConfigError("workload.measurement_duration_s cannot be negative")
    if workload.measurement_duration_s and workload.mode != "closed_loop":
        raise ConfigError("measurement_duration_s is only supported for closed_loop")
    if config.slo is not None and not 0 < config.slo.target_attainment <= 1:
        raise ConfigError("slo.target_attainment must be in (0, 1]")

    # This MVP does not model weight paging. At minimum, each stage's sharded
    # weights must fit on one participating GPU.
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
            tied_embedding_is_local = (
                config.model.tie_word_embeddings
                and stage.resource == config.stages[0].resource
            )
            if layer_end == config.model.num_layers and not tied_embedding_is_local:
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
