"""Independent arrivals and fixed-window metrics shared by both execution backends."""

import random
from .config import RequestSpec
from .metrics import build_summary


def load_end(workload):
    return (
        workload.warmup_duration_s
        + workload.measurement_duration_s
        + workload.arrival_tail_s
    )


def request_specs(workload):
    if workload.requests:
        return sorted(
            workload.requests, key=lambda r: (r.arrival_time_ms, r.request_id)
        )
    rng = random.Random(workload.random_seed)
    end = load_end(workload)
    now = 0.0
    result = []
    while True:
        now += (
            rng.expovariate(workload.arrival_rate_qps)
            if workload.arrival_process == "poisson"
            else 1 / workload.arrival_rate_qps
        )
        if now >= end:
            break
        result.append(
            RequestSpec(
                len(result), now * 1000, workload.input_tokens, workload.output_tokens
            )
        )
    return result


def summary(rows, traces, start, end, slo, workload):
    """Arrival latency / completion throughput, with half-open cohorts and full drain."""
    arrivals = [r for r in rows if start * 1000 <= r["arrival_time_ms"] < end * 1000]
    completed = [r for r in rows if start * 1000 <= r["finish_time_ms"] < end * 1000]
    a = build_summary(arrivals, traces, {}, start, end, slo)
    c = build_summary(completed, [], {}, start, end, slo)
    result = dict(a)
    result.update(
        workload_mode="open_loop",
        latency_cohort="arrival",
        arrival_cohort=a,
        completion_cohort=c,
        target_arrival_rate_qps=workload.arrival_rate_qps,
        observed_arrival_rate_qps=len(arrivals) / (end - start),
        observed_request_throughput_qps=c["observed_request_throughput_qps"],
        observed_output_throughput_tokens_s=c["observed_output_throughput_tokens_s"],
    )
    if slo:
        result["slo"] = dict(a["slo"])
        result["slo"]["pass"] = a["slo"]["pass"] and c["slo"]["pass"]
        result["slo"]["goodput_qps"] = c["slo"]["goodput_qps"]
    events = []
    occupied_ms = 0.0
    for r in rows:
        lo, hi = max(start * 1000, r["arrival_time_ms"]), min(
            end * 1000, r["finish_time_ms"]
        )
        r["measured"] = start * 1000 <= r["arrival_time_ms"] < end * 1000
        r["completion_measured"] = start * 1000 <= r["finish_time_ms"] < end * 1000
        if hi > lo:
            occupied_ms += hi - lo
            events.extend([(lo, 1), (hi, -1)])
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    result["mean_inflight"] = occupied_ms / ((end - start) * 1000)
    result["peak_inflight"] = peak
    result["inflight_boundaries"] = [
        {
            "offset_s": t,
            "inflight": sum(
                r["arrival_time_ms"] <= (start + t) * 1000 < r["finish_time_ms"]
                for r in rows
            ),
        }
        for t in sorted(set([0.0, end - start] + list(range(30, int(end - start), 30))))
    ]
    result["measurement_arrivals_completed_before_load_stop"] = all(
        r["finish_time_ms"] <= load_end(workload) * 1000 for r in arrivals
    )
    result["total_requests_including_warmup_and_drain"] = len(rows)
    result["warmup_requests"] = sum(r["arrival_time_ms"] < start * 1000 for r in rows)
    result["drain_requests_excluded"] = sum(
        r["arrival_time_ms"] >= end * 1000 for r in rows
    )
    result["measurement_window"] = dict(
        mode="fixed_open_loop",
        start_ms=start * 1000,
        end_ms=end * 1000,
        duration_ms=(end - start) * 1000,
    )
    return result
