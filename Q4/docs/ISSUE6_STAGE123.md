# Issue #6 Stage 1/2/3 simulation model

This document maps the simulator controls to issue #6. The implementation is a
behavioral model: it preserves causal ordering, batching and resource pressure,
while measured profiles remain necessary for absolute latency claims.

## Stage 1: data path

When `data_path.enabled=true`, every up/down message is decomposed into device to
host copy, host packing, cloud-local IPC, WAN serialization/propagation, host
unpacking and host to device copy. The following switches change modeled work:

- `ipc_mode=shm` selects `shm_ipc_copies`; `copy` selects `copy_ipc_copies`.
- `wire_fast=true` selects `wire_fast_pack_copies`; false selects
  `legacy_pack_copies`.
- `tcp_buffer_mib>0` limits effective throughput to
  `min(link_rate, tcp_window / RTT)`, making the BDP effect explicit.

The entire direction currently owns one serialized uplink/downlink resource.
The sub-operations are visible in the Gantt detail view, but host CPU/PCIe/IPC
are not yet independently concurrent servers. This is conservative for overlap
and cannot predict CPU saturation. Exact-shape network profiles can replace the
analytical total; profile matching also receives a `data_path` variant string.

## Stage 2: chunking and fair FCFS

Chunk DAGs carry `token_start` and accumulated context, preserve KV order at
each model stage, and only the final chunk produces logits. `pipeline_depth=1`
and `max_outstanding_prefill_chunks=1` implement serial chunking.

With `scheduler.decode_first=true`, ready decode work is considered before
prefill while request order remains FCFS. Prefill is forced to the front when
either `max_consecutive_decode_batches` is reached or the oldest ready chunk
waits `max_prefill_wait_ms`. `max_decode_tokens_per_batch` bounds decode work;
the existing prefill and total-token budgets remain active. Mixed batches use
the configured unified or sequential attention backend.

This models scheduler behavior, not numerical logits. PR #11's unresolved
numerical acceptance gate therefore remains an external implementation issue.

## Stage 3: pipeline and backpressure

`execution.mode=pipelined` independently schedules edge GPU, uplink, cloud GPU,
downlink and the shared edge GPU tail. A transaction owns a buffer reservation
from its first Edge Front dispatch through its final Edge Tail completion.

- `max_inflight_transactions` bounds the number of live chunk/decode steps.
- `buffer_pool_mib` bounds their combined bidirectional activation bytes.
- Zero means unlimited for backward compatibility.

The trace records current in-flight transaction count/bytes, and the summary
records high-water marks. Edge front and tail share the same resource when their
topology `resource` names match, so the model does not assume five independent
pipeline stages.

Resource utilization is reported over the full run including warmup because
resource busy counters are maintained independently of trace retention. Batch
size and latency distributions use the measured window.

## Closed-loop evaluation and SLO

`workload.mode=closed_loop` launches `concurrency` requests and launches one
replacement when a measured request completes. Warmup requests form a separate
phase: they run first, drain completely, then the measured concurrency starts.

Summary TPOT is the distribution of per-request mean token intervals. The
separate `itl_ms` field describes all token intervals and includes its maximum.
SLO attainment is evaluated jointly per request using TTFT and mean TPOT;
`goodput_qps` counts only jointly compliant requests.

The finite run includes final drain time in observed QPS. It is suitable for
configuration A/B tests with enough requests, but is not an automatic maximum
QPS search or a fixed-duration production capacity certificate.

## Deliberate exclusions

- Prompt-lookup/speculative decoding belongs to Stage 4 and is not modeled.
- Cancellation and injected worker failures are not stochastic performance
  inputs; the simulator assumes submitted requests complete.
- Buffer ownership is byte-accurate at transaction granularity, not allocator
  or CUDA-event granular.
- Stage 1 absolute values require ordinary-run profiles for each optimization
  variant; Nsight alone is insufficient for WAN/HTTP/IPC wall time.
