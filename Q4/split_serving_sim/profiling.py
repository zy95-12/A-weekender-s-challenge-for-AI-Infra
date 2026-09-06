from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .config import PerformanceProfileConfig
from .core import OperatorWorkload


@dataclass(frozen=True)
class ProfileMatch:
    duration_s: float
    source: str
    correction_factor: float
    exact: bool


def operator_type(name: str, category: str) -> str:
    leaf = name.rsplit(".", 1)[-1]
    if leaf in {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj", "lm_head",
        "qkv_proj", "gate_up_proj",
    } or name == "lm_head":
        return "mm"
    if leaf in {"attention", "mix attention", "prefill attention", "decode attention"}:
        return "flash_attention"
    if leaf.endswith("all_reduce"):
        return "collective"
    if leaf in {"input_rms_norm", "final_rms_norm"}:
        return "rms_norm"
    if leaf.endswith("add_rms_norm"):
        return "fused_add_rms_norm"
    if leaf == "rope":
        return "rotary_embedding"
    if leaf == "silu_and_mul":
        return "silu_and_mul"
    if leaf == "attention_output_alloc":
        return "tensor_allocation"
    if leaf == "embed_tokens":
        return "embedding"
    if category == "communication":
        return "communication"
    return "elementwise"


def workload_signature(
    input_shape: str,
    workload: OperatorWorkload,
) -> str:
    return json.dumps(
        {
            "input_shape": input_shape,
            "flops": round(workload.flops),
            "memory_bytes": round(workload.memory_bytes),
            "communication_bytes": round(workload.communication_bytes),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def physical_signature(**fields: Any) -> str:
    """Stable JSON signature shared by vLLM traces and the physical DAG."""

    return json.dumps(fields, sort_keys=True, separators=(",", ":"))


def network_signature(payload_bytes: float, input_shape: str) -> str:
    return json.dumps(
        {"input_shape": input_shape, "payload_bytes": round(payload_bytes)},
        sort_keys=True,
        separators=(",", ":"),
    )


class ProfilingDatabase:
    def __init__(self, config: PerformanceProfileConfig):
        self.enabled = config.enabled
        self.samples = config.samples
        self.by_type: dict[str, list[Any]] = {}
        self.cache: dict[tuple[Any, ...], ProfileMatch] = {}
        for sample in self.samples:
            self.by_type.setdefault(sample.operator_type, []).append(sample)

    def needs_signature(self, kind: str) -> bool:
        return any(
            sample.signature is not None for sample in self.by_type.get(kind, ())
        )

    def correct(
        self,
        kind: str,
        signature: str,
        roofline_duration_s: float,
        scope: dict[str, Any],
    ) -> ProfileMatch:
        if not self.enabled:
            return ProfileMatch(roofline_duration_s, "roofline", 1.0, False)
        normalized_scope = {key: str(value) for key, value in scope.items()}
        cache_key = (
            kind,
            signature,
            roofline_duration_s,
            tuple(sorted(normalized_scope.items())),
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        candidates = [
            sample
            for sample in self.by_type.get(kind, ())
            if all(normalized_scope.get(key) == value for key, value in sample.match)
        ]
        exact = [sample for sample in candidates if sample.signature == signature]
        if exact:
            duration = sum(sample.latency_ms for sample in exact) / len(exact) / 1e3
            factor = duration / roofline_duration_s if roofline_duration_s else 1.0
            sources = sorted({sample.source for sample in exact if sample.source})
            result = ProfileMatch(
                duration,
                "profile_exact" + (f":{','.join(sources)}" if sources else ""),
                factor,
                True,
            )
            self.cache[cache_key] = result
            return result
        ratios = [
            sample.latency_ms / sample.roofline_ms
            for sample in candidates
            if sample.roofline_ms is not None
        ]
        if ratios:
            factor = sum(ratios) / len(ratios)
            result = ProfileMatch(
                roofline_duration_s * factor,
                "profile_type_average",
                factor,
                False,
            )
            self.cache[cache_key] = result
            return result
        result = ProfileMatch(roofline_duration_s, "roofline", 1.0, False)
        self.cache[cache_key] = result
        return result
