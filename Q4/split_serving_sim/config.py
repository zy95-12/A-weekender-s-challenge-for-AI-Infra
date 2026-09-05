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
    num_layers: int
    hidden_size: int
    dtype_bytes: int = 2
    params_per_layer_factor: float = 12.0


@dataclass(frozen=True)
class StageConfig:
    name: str
    layer_start: int
    layer_end: int
    resource: str
    tp_degree: int = 1

    @property
    def num_layers(self) -> int:
        return self.layer_end - self.layer_start


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
    prefill_chunk_size: int
    decode_priority: bool = True
    edge_tail_priority: bool = True
    pipeline_depth: int = 4
    max_outstanding_prefill_chunks: int = 8
    dispatch_mode: str = "eager"


@dataclass(frozen=True)
class RequestSpec:
    request_id: int
    arrival_time_ms: float
    input_tokens: int
    output_tokens: int


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


def load_config(path: str | Path) -> SimulationConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return parse_config(json.load(handle))


def parse_config(data: dict[str, Any]) -> SimulationConfig:
    model_data = _required(data, "model", "root")
    model = ModelConfig(
        name=_required(model_data, "name", "model"),
        num_layers=int(_required(model_data, "num_layers", "model")),
        hidden_size=int(_required(model_data, "hidden_size", "model")),
        dtype_bytes=int(model_data.get("dtype_bytes", 2)),
        params_per_layer_factor=float(model_data.get("params_per_layer_factor", 12.0)),
    )

    topology_data = _required(data, "topology", "root")
    stages = tuple(
        StageConfig(
            name=str(_required(item, "name", "topology.stages[]")),
            layer_start=int(_required(item, "layer_start", "topology.stages[]")),
            layer_end=int(_required(item, "layer_end", "topology.stages[]")),
            resource=str(_required(item, "resource", "topology.stages[]")),
            tp_degree=int(item.get("tp_degree", 1)),
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
    policy = StaticPolicyConfig(
        max_batch_size=int(configured_batch_size),
        max_batched_tokens=int(
            _required(policy_data, "max_batched_tokens", "static_policy")
        ),
        prefill_chunk_size=int(
            _required(policy_data, "prefill_chunk_size", "static_policy")
        ),
        decode_priority=bool(policy_data.get("decode_priority", True)),
        edge_tail_priority=bool(policy_data.get("edge_tail_priority", True)),
        pipeline_depth=int(policy_data.get("pipeline_depth", 4)),
        max_outstanding_prefill_chunks=int(
            policy_data.get(
                "max_outstanding_prefill_chunks",
                policy_data.get("max_outstanding_batches", 8),
            )
        ),
        dispatch_mode=str(policy_data.get("dispatch_mode", "eager")),
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

    expected_names = ["edge_front", "cloud_middle", "edge_tail"]
    if [stage.name for stage in config.stages] != expected_names:
        raise ConfigError(f"topology stages must be ordered as {expected_names}")
    cursor = 0
    for stage in config.stages:
        if stage.layer_start != cursor or stage.layer_end <= stage.layer_start:
            raise ConfigError("topology layer ranges must be contiguous and non-empty")
        cursor = stage.layer_end
        if stage.resource not in config.hardware:
            raise ConfigError(f"unknown hardware resource: {stage.resource}")
        if stage.tp_degree <= 0 or stage.tp_degree > config.hardware[stage.resource].count:
            raise ConfigError(f"invalid tp_degree for stage {stage.name}")
    if cursor != config.model.num_layers:
        raise ConfigError("topology layer ranges must cover all model layers")

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
        policy.prefill_chunk_size,
        policy.pipeline_depth,
        policy.max_outstanding_prefill_chunks,
    ) <= 0:
        raise ConfigError("static policy limits must be positive")
    if policy.prefill_chunk_size > policy.max_batched_tokens:
        raise ConfigError("prefill_chunk_size cannot exceed max_batched_tokens")
    if policy.dispatch_mode != "eager":
        raise ConfigError("only dispatch_mode='eager' is supported in the MVP")

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
    for stage in config.stages:
        hardware_profile = config.hardware[stage.resource]
        weight_bytes = (
            stage.num_layers
            * config.model.params_per_layer_factor
            * config.model.hidden_size
            * config.model.hidden_size
            * config.model.dtype_bytes
            / stage.tp_degree
        )
        if weight_bytes > hardware_profile.memory_gb * 1e9:
            raise ConfigError(
                f"stage {stage.name} weights do not fit in per-GPU memory"
            )
