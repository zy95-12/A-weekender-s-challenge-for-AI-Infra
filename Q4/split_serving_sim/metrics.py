from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .config import SLOConfig


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


@dataclass
class RequestRuntime:
    request_id: int
    arrival_time: float
    input_tokens: int
    output_tokens: int
    first_token_time: float | None = None
    finish_time: float | None = None
    token_times: list[float] = None  # type: ignore[assignment]
    prefill_queue_time: float = 0.0
    decode_queue_time: float = 0.0

    def __post_init__(self) -> None:
        if self.token_times is None:
            self.token_times = []

    def to_metrics(self) -> dict[str, Any]:
        if self.first_token_time is None or self.finish_time is None:
            raise RuntimeError(f"request did not finish: {self.request_id}")
        intervals_ms = [
            (right - left) * 1000.0
            for left, right in zip(self.token_times, self.token_times[1:])
        ]
        return {
            "request_id": self.request_id,
            "arrival_time_ms": self.arrival_time * 1000.0,
            "first_token_time_ms": self.first_token_time * 1000.0,
            "finish_time_ms": self.finish_time * 1000.0,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "ttft_ms": (self.first_token_time - self.arrival_time) * 1000.0,
            "e2e_ms": (self.finish_time - self.arrival_time) * 1000.0,
            "mean_tpot_ms": (
                sum(intervals_ms) / len(intervals_ms) if intervals_ms else None
            ),
            "p50_tpot_ms": percentile(intervals_ms, 0.50),
            "p95_tpot_ms": percentile(intervals_ms, 0.95),
            "p99_tpot_ms": percentile(intervals_ms, 0.99),
            "prefill_queue_time_ms": self.prefill_queue_time * 1000.0,
            "decode_queue_time_ms": self.decode_queue_time * 1000.0,
            "token_timestamps_ms": [value * 1000.0 for value in self.token_times],
        }


def build_summary(
    requests: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    resource_busy_time: dict[str, float],
    start_time: float,
    end_time: float,
    slo: SLOConfig | None,
) -> dict[str, Any]:
    duration = max(end_time - start_time, 0.0)
    ttfts = [request["ttft_ms"] for request in requests]
    e2es = [request["e2e_ms"] for request in requests]
    itls = [
        right - left
        for request in requests
        for left, right in zip(
            request["token_timestamps_ms"], request["token_timestamps_ms"][1:]
        )
    ]
    tpots = [
        request["mean_tpot_ms"]
        for request in requests
        if request["mean_tpot_ms"] is not None
    ]
    batch_sizes = [trace["batch_size"] for trace in traces]
    total_output_tokens = sum(request["output_tokens"] for request in requests)
    summary: dict[str, Any] = {
        "num_requests": len(requests),
        "simulation_start_ms": start_time * 1000.0,
        "simulation_end_ms": end_time * 1000.0,
        "simulation_duration_ms": duration * 1000.0,
        "observed_request_throughput_qps": len(requests) / duration if duration else None,
        "observed_output_throughput_tokens_s": (
            total_output_tokens / duration if duration else None
        ),
        "ttft_ms": {
            "mean": sum(ttfts) / len(ttfts) if ttfts else None,
            "p50": percentile(ttfts, 0.50),
            "p95": percentile(ttfts, 0.95),
            "p99": percentile(ttfts, 0.99),
        },
        "tpot_ms": {
            "mean": sum(tpots) / len(tpots) if tpots else None,
            "p50": percentile(tpots, 0.50),
            "p95": percentile(tpots, 0.95),
            "p99": percentile(tpots, 0.99),
        },
        "itl_ms": {
            "p50": percentile(itls, 0.50),
            "p95": percentile(itls, 0.95),
            "p99": percentile(itls, 0.99),
            "max": max(itls) if itls else None,
        },
        "e2e_ms": {
            "mean": sum(e2es) / len(e2es) if e2es else None,
            "p50": percentile(e2es, 0.50),
            "p95": percentile(e2es, 0.95),
            "p99": percentile(e2es, 0.99),
        },
        "average_batch_size": (
            sum(batch_sizes) / len(batch_sizes) if batch_sizes else None
        ),
        "resource_utilization": {
            resource: busy / duration if duration else 0.0
            for resource, busy in resource_busy_time.items()
        },
    }
    if slo is not None:
        ttft_p99 = summary["ttft_ms"]["p99"]
        tpot_p99 = summary["tpot_ms"]["p99"]
        compliant = []
        for request in requests:
            tpot = request["mean_tpot_ms"]
            passed = request["ttft_ms"] <= slo.ttft_ms and (
                tpot is None or tpot <= slo.tpot_ms
            )
            request["slo_pass"] = passed
            compliant.append(passed)
        compliant_requests = sum(compliant)
        attainment = compliant_requests / len(requests) if requests else 0.0
        summary["slo"] = {
            "ttft_limit_ms": slo.ttft_ms,
            "tpot_limit_ms": slo.tpot_ms,
            "target_attainment": slo.target_attainment,
            "ttft_pass": ttft_p99 is not None and ttft_p99 <= slo.ttft_ms,
            # TPOT is not applicable when every request asks for one token.
            "tpot_pass": tpot_p99 is None or tpot_p99 <= slo.tpot_ms,
            "compliant_requests": compliant_requests,
            "attainment": attainment,
            "goodput_qps": compliant_requests / duration if duration else None,
        }
        summary["slo"]["pass"] = attainment >= slo.target_attainment
    return summary
