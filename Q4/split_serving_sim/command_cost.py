"""Measured synchronous forward-command service costs (not GPU kernel times).

This optional backend includes host preparation, IPC, and device staging. The
simulator must not add host submission or device/IPC staging a second time.
Samples exclude scheduler queue time; scheduling is still simulated.
"""
import json
import math
import random
import statistics
from pathlib import Path

from .core import PerformanceEstimate, SubOperation
from .performance import NetworkModel


def command_scope(config):
    return dict(model=config.model.name, dtype=config.model.dtype,
                layers=[[s.layer_start,s.layer_end] for s in config.stages],
                ipc_mode=config.data_path.ipc_mode, wire_fast=config.data_path.wire_fast,
                hardware={s.name: [config.hardware[s.resource].peak_flops_tflops,
                                   config.hardware[s.resource].hbm_bandwidth_gb_s,
                                   config.hardware[s.resource].memory_gb] for s in config.stages})


class CommandCostModel:
    includes_host_staging = True

    def __init__(self, config, profile, sampling="mean", seed=0, sampling_stages=None):
        if sampling not in {"mean", "empirical"}:
            raise ValueError("unknown command sampling mode")
        self.sampling = sampling
        self.seed = seed
        self.random = random.Random(seed)
        self.sampling_stages = set(sampling_stages) if sampling_stages is not None else {"edge_front", "cloud_middle", "edge_tail"}
        self.residual_coverage = {}
        self.config = config
        self.profile = profile
        self.coverage = {}
        self.samples = profile['samples']
        if not self.samples or any(not math.isfinite(s['latency_ms']) or s['latency_ms'] < 0
                                   or s['batch_size'] <= 0 or s['sample_count'] <= 0
                                   or s['context_min'] > s['context_max'] for s in self.samples):
            raise ValueError('invalid command cost samples')
        self.distribution_means = {}
        for sample in self.samples:
            values = sample.get("service_samples_ms")
            required = sampling == "empirical" and sample["stage"] in self.sampling_stages
            if values is None:
                if required:
                    raise ValueError("empirical mode requires service_samples_ms; rebuild profile with --include-distribution")
                continue
            if (not values or len(values) != sample["sample_count"]
                or any(not math.isfinite(v) or v < 0 for v in values)):
                raise ValueError("invalid empirical command samples")
            avg = statistics.mean(values)
            if not math.isclose(avg, sample["latency_ms"], rel_tol=1e-8, abs_tol=1e-8):
                raise ValueError("empirical samples disagree with calibrated mean")
            self.distribution_means[id(sample)] = avg
        # Aggregate costs cannot be reused across different model/serving paths.
        expected = command_scope(config)
        if profile['scope'] != expected:
            raise ValueError('command cost profile does not match model, layers, dtype, or transport')

    @classmethod
    def from_file(cls, config, path, **options):
        return cls(config, json.loads(Path(path).read_text()), **options)

    def estimate(self, stage_name, items):
        if (not items or len({(i.phase, i.token_count) for i in items}) != 1
            or any(i.pipeline_rank != 0 for i in items)):
            raise ValueError('command cost backend requires uniform queries/phases and PP1')
        stage = self.config.stage(stage_name).for_phase(items[0].phase.value)
        phase = items[0].phase.value
        emits = any(i.produces_logits for i in items)
        samples = [s for s in self.samples if s['stage']==stage_name and s['phase']==phase
                   and s['tp_degree']==stage.tp_degree and s['query_len']==items[0].token_count
                   and s['emits_logits']==emits
                   and all(s['context_min']<=i.context_tokens<=s['context_max'] for i in items)]
        counts = self.coverage.setdefault(f'{stage_name}/{phase}/tp{stage.tp_degree}',{})
        if not samples:
            raise ValueError(f'no command cost coverage for {stage_name}/{phase}; use operator backend or collect this scope')
        samples.sort(key=lambda s:s['batch_size'])
        batch = len(items)
        exact = [s for s in samples if s['batch_size']==batch]
        if exact:
            ms=exact[0]['latency_ms']; source='command_exact_batch'
        else:
            lower = [s for s in samples if s['batch_size']<batch]
            upper = [s for s in samples if s['batch_size']>batch]
            if lower and upper:
                left,right=lower[-1],upper[0]; source='command_interpolated_batch'
            elif len(samples)>=2:
                # Weighted linear fit, rather than extending the last two noisy
                # measurements. Extrapolation is always visible in coverage.
                weight=sum(s['sample_count'] for s in samples)
                x=sum(s['batch_size']*s['sample_count'] for s in samples)/weight
                y=sum(s['latency_ms']*s['sample_count'] for s in samples)/weight
                variance=sum(s['sample_count']*(s['batch_size']-x)**2 for s in samples)
                slope=max(0, sum(s['sample_count']*(s['batch_size']-x)*(s['latency_ms']-y)
                                 for s in samples)/variance)
                ms=max(0.001,y+slope*(batch-x)); source='command_extrapolated_batch'
                left=right=None
            else:
                raise ValueError('command cost table cannot interpolate/extrapolate a single batch size')
            if left is not None:
                fraction=(batch-left['batch_size'])/(right['batch_size']-left['batch_size'])
                ms=max(0.001,left['latency_ms']+fraction*(right['latency_ms']-left['latency_ms']))
        counts[source]=counts.get(source,0)+1
        if self.sampling == "empirical" and stage_name in self.sampling_stages:
            residual = min(samples, key=lambda s: (abs(s['batch_size']-batch), s['batch_size']))
            avg = self.distribution_means[id(residual)]
            ratio = self.random.choice(residual['service_samples_ms']) / avg if avg else 1.0
            ms *= ratio
            kind = "exact_batch" if residual['batch_size'] == batch else "nearest_batch_relative_residual"
            self.residual_coverage[kind] = self.residual_coverage.get(kind, 0) + 1
            source += "_empirical"
        shape=f'{phase} B{batch} Q{items[0].token_count} TP{stage.tp_degree}'
        duration=ms/1000
        return PerformanceEstimate(0,0,0,0,0,0,duration,duration,shape,
            (SubOperation('forward_command','host_and_gpu',duration,shape,profile_source=source),))


class CommandNetworkModel(NetworkModel):
    """WAN wire costs only; measured GPU commands already own device/IPC copies."""
    def estimate(self, stage, items):
        from dataclasses import replace
        estimate=super().estimate(stage,items)
        removed={'device_to_host','host_to_device','cloud_ipc'}
        kept=[op for op in estimate.sub_operations if op.name not in removed]
        operations=tuple(replace(op, dependencies=(kept[index-1].name,) if index else ())
                         for index,op in enumerate(kept))
        delta=sum(op.duration_s for op in estimate.sub_operations if op.name in removed)
        return replace(estimate, total_time_s=estimate.total_time_s-delta,
                       memory_time_s=max(0,estimate.memory_time_s-delta), sub_operations=operations)
