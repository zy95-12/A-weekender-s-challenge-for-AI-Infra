"""Compile serving switches into immutable scheduler/topology/cost configuration.

The input is the baseline experiment, so disabling every serving optimization
returns it unchanged. Calibration remains a separate modeling choice.
"""
from dataclasses import dataclass, replace

from .config import SimulationConfig, PDDisaggregationConfig, validate_config


@dataclass(frozen=True)
class ServingFeatures:
    stage1: bool = False
    chunked_prefill: bool = False
    pipeline: bool = False
    decode_first: bool = False
    pd: bool = False
    pd_admission: bool = True
    prefill_replicas: int = 1
    prefill_tp: int = 1
    decode_tp: int = 1
    prefill_window: int = 3
    decode_window: int = 2
    decode_quota: int = 4
    chunk_size: int = 2048
    chunk_transfer: bool = False
    control_channel: bool = True
    control_latency_ms: float = 0.0

    @classmethod
    def optimized(cls):
        return cls(stage1=True, chunked_prefill=True, pipeline=True,
                   decode_first=True, pd=True, prefill_replicas=2)


def configure_serving(baseline: SimulationConfig, features: ServingFeatures,
                      cost_model: str = "profile") -> SimulationConfig:
    if cost_model not in {"profile", "roofline"}:
        raise ValueError("cost_model must be profile or roofline")
    if features.pd and not features.pipeline:
        raise ValueError("PD requires pipeline; use baseline preset to disable both")
    if features.chunk_transfer and not features.pd:
        raise ValueError("chunk_transfer requires PD")
    config = baseline
    if features.stage1:
        # Old baseline samples must never be reused for optimized transport.
        config = replace(config, data_path=replace(config.data_path,
            enabled=True, ipc_mode="shm", wire_fast=True, tcp_buffer_mib=16,
            wan_calibration=replace(config.data_path.wan_calibration, enabled=False)))
    if features.chunked_prefill:
        config = replace(config, static_policy=replace(config.static_policy,
            prefill_chunk_size=features.chunk_size,
            prefill_token_budget=features.chunk_size),
            scheduler=replace(config.scheduler, enable_chunked_prefill=True))
    if features.pipeline:
        config = replace(config, execution=replace(config.execution, mode="pipelined",
            preserve_batch_across_stages=True, max_inflight_transactions=features.decode_window),
            static_policy=replace(config.static_policy, pipeline_depth=features.decode_window,
                                  max_outstanding_prefill_chunks=features.decode_window),
            scheduler=replace(config.scheduler, policy="fcfs", prefer_ready_back=True))
    if features.decode_first:
        config = replace(config, scheduler=replace(config.scheduler, decode_first=True,
            max_consecutive_decode_batches=features.decode_quota))
    if features.pd:
        hardware = dict(config.hardware)
        enterprise = config.stage("edge_front").resource
        cloud = config.stage("cloud_middle").resource
        hardware[enterprise] = replace(hardware[enterprise], count=1)
        hardware["cloud_prefill"] = replace(hardware[cloud], count=features.prefill_tp * features.prefill_replicas)
        hardware["cloud_decode"] = replace(hardware[cloud], count=features.decode_tp)
        if cloud != enterprise:
            del hardware[cloud]
        stages = tuple(replace(stage, resource="cloud_prefill", tp_degree=features.prefill_tp,
                               prefill_resource="cloud_prefill", decode_resource="cloud_decode",
                               prefill_tp_degree=features.prefill_tp, decode_tp_degree=features.decode_tp,
                               prefill_replicas=features.prefill_replicas, decode_replicas=1)
                       if stage.name == "cloud_middle" else replace(stage, tp_degree=1)
                       for stage in config.stages)
        window = features.prefill_window * features.prefill_replicas + features.decode_window
        config = replace(config, hardware=hardware, stages=stages,
            execution=replace(config.execution, max_inflight_transactions=window),
            static_policy=replace(config.static_policy, max_batch_size=96,
                pipeline_depth=max(features.prefill_window, features.decode_window),
                max_outstanding_prefill_chunks=features.prefill_window * features.prefill_replicas),
            scheduler=replace(config.scheduler, policy="split_poc_pd", max_num_seqs=96,
                allow_mixed_batch=False, kv_cache=replace(config.scheduler.kv_cache, num_blocks=32768),
                pd_disaggregation=replace(config.scheduler.pd_disaggregation, enabled=True, admission_enabled=features.pd_admission,
                    prefill_window=features.prefill_window, decode_window=features.decode_window,
                    chunk_transfer=features.chunk_transfer, control_channel=features.control_channel,
                    control_latency_ms=features.control_latency_ms)))
    if cost_model == "roofline":
        config = replace(config, performance_profile=replace(config.performance_profile, enabled=False))
    validate_config(config)
    return config
