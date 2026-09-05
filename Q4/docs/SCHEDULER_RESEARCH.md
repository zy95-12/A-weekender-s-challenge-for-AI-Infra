# Scheduler alignment with vLLM V1

This note separates request ordering from resource admission. Reproducing FCFS
ordering alone is not enough to reproduce vLLM: a realistic engine step also has
a token budget, a maximum number of sequences, and KV-cache admission.

## Observed vLLM behavior

The current V1 scheduler uses a unified token view rather than two global
prefill/decode phases. It first advances running requests, then admits waiting
requests into the remaining token/sequence/KV budget. Long prefills are chunked
when the remaining per-step token budget is smaller than the prompt remainder.
Consequently, an engine step can contain both decode tokens and prefill chunks.

Core vLLM exposes two request-queue policies:

- `fcfs`: arrival order;
- `priority`: lower request priority first, with arrival time as the tie-breaker.

Resource pressure can still make observed execution depart from strict queue
order. A request that cannot obtain KV blocks can be skipped, and a running
request can be preempted. V1 uses recomputation rather than CPU swap as its
default preemption mode.

Primary references:

- <https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py>
- <https://github.com/vllm-project/vllm/blob/main/vllm/config/scheduler.py>
- <https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/>
- <https://docs.vllm.ai/en/v0.10.2/configuration/optimization.html#chunked-prefill>

## Policies we can simulate

| Policy/model | Can implement now? | Required state | Fidelity |
|---|---:|---|---|
| Static cohort | Implemented | arrival queue, cohort membership | Baseline, intentionally not vLLM |
| Eager decode-first | Implemented | ready work items, token and batch limits | Approximation of chunked-prefill behavior |
| vLLM-style FCFS | Implemented | running/waiting request queues, per-step token budget, max sequences | Good without KV pressure |
| vLLM-style priority | Implemented | FCFS state plus request priority and safe-point priority preemption | Behavioral approximation |
| Short-prefill-first | Implemented | known input length | Experimental; not core vLLM default |
| KV-aware FCFS/priority | Implemented | logical block pool, full-length reservation, watermark | One bottleneck pool rather than per-stage pools |
| Prefix-cache-aware | Implemented | shared prefix blocks, activation-time hit state, LRU eviction | Explicit-prefix model, not runtime block hashing |
| Preempt-and-recompute | Implemented | safe victim selection, released blocks, per-stage recompute tokens | Priority mode, safe points only |
| PD-disaggregated routing | Implemented | separate phase resources and serialized KV-transfer resource | One P and one D route; no replica load balancing yet |

## Recommended implementation order

1. **Done:** replace work-item phase priority with request-level `running` and `waiting`
   queues. Keep `max_batched_tokens`, add `max_num_seqs`, and schedule running
   requests before waiting requests. This produces the mixed token batch used by
   vLLM while retaining deterministic FCFS ordering.
2. **Done:** add request `priority` and a `scheduler.policy = fcfs | priority` switch. The
   batch construction logic remains shared.
3. **Done:** add a paged KV block pool. Admission uses both token budget and KV block
   availability; expose block allocation/free events in the trace.
4. **Done:** add safe-point recompute preemption, watermark, and full-input/output reservation.
5. **Done:** add explicit prefix sharing/LRU eviction and PD phase routing with a
   serialized KV-transfer event.

The next fidelity step is per-stage KV pools and multi-replica P/D load
balancing. The current logical pool intentionally represents the bottleneck
instance and should not be interpreted as byte-accurate memory placement.

## Proposed future configuration

```json
{
  "scheduler": {
    "policy": "fcfs",
    "max_num_seqs": 256,
    "max_num_batched_tokens": 8192,
    "enable_chunked_prefill": true,
    "kv_cache": {
      "enabled": false,
      "block_size_tokens": 16,
      "watermark": 0.0,
      "preemption_mode": "recompute"
    }
  }
}
```

The legacy `decode_priority` field is retained for configuration compatibility
but is no longer used by the vLLM-style scheduler. Running/waiting state and the
selected `scheduler.policy` determine request order.

