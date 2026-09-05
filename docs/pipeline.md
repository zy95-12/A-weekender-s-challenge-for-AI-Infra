# Stage 3: bounded cross-request pipeline

The retained opt-in variant splits enterprise front, WAN RPC, and enterprise
back into separately scheduled tasks. A GPU worker no longer waits for HTTP.
Front/back commands remain serialized on the SAME enterprise TP group. The
cloud GPU executor is also serialized; concurrent transfers do not imply
concurrent GPU batches. Window=2 counts uploads, remote work and returned
buffers waiting for back. Window=0 preserves the synchronous path.

The measured variant uses `--ipc-mode shm --wire-fast --tcp-buffer-mib 16
--pipeline-window 2`, with prefill chunk size zero. Reusable host mailboxes now
also cover enterprise-local worker IPC in pipeline mode. Replies become owned
arrays before releasing the mailbox lock; the in-flight window bounds their
lifetime. This is not GPU zero-copy or a cross-WAN shared-memory shortcut.

Per-request front and back have separate KV positions and allocators. Cloud
execution gates expected positions and rejects replays; an out-of-order arrival
waits without holding the GPU execution lock. Admission reserves the request's
maximum possible KV blocks. Cancellation drains submitted tasks before release.
Failure marks the executor unhealthy and rejects further work; no target fallback.

## Scope and numerical gate

**Unchunked cross-request pipelining passed representative acceptance. Async
long-prompt chunking is NOT accepted:** stage 2's FP16 logits gate still fails.
This implementation and report do not claim to complete the full async-chunk
scope in issue #6. The user removed the full 96-round regression requirement;
representative correctness, lifecycle, benefit and regressions remain required.

Both initial and mailbox variants compared 132 full-vocabulary positions against
fixed native TP2 references: MAE/max-abs=0, all four 32-token greedy outputs exact.
The mailbox variant passed real 1k/4k/8k KV, C4 vs isolated outputs, 4k/8k needle
QA, streaming usage and cancellation. CPU tests cover out-of-order completion,
bounded in-flight work, cancellation drain, replay, missing predecessor and
failure closure. An active-request cloud rank0 termination produced a stream
error in 0.354 s, followed by HTTP 503 for health and new requests. The service was
restarted under owned-process controls for independent profiling.

## Representative performance

Fixed Qwen revision, 4/27/5, TP2+2, 10 Gbps per direction, 10 ms propagation RTT.
8k/256, C1/C4; three repeats, four measured requests each, separate warmup.
All measured requests succeeded and output lengths matched. These are closed-loop
completion rates, not capacity or SLO claims. Runs were sequential, not randomized.

| Variant | C | TTFT ms | TPOT ms | QPS |
|---|---:|---:|---:|---:|
| Synchronous stage-1 data path | 1 | 1018.77 | 32.18 | 0.108 |
| Initial pipeline, enterprise pipe IPC | 1 | 1351.08 | 34.06 | 0.100 |
| Pipeline + enterprise mailbox | 1 | 1035.75 | 34.13 | 0.103 |
| Synchronous stage-1 data path | 4 | 2546.69 | 39.53 | 0.317 |
| Initial pipeline, enterprise pipe IPC | 4 | 2369.04 | 39.04 | 0.321 |
| Pipeline + enterprise mailbox | 4 | 1908.17 | 38.58 | 0.337 |

The retained variant lowers C4 TTFT about 25% and improves QPS about 6.4%.
C1 TTFT increases about 1.7%, TPOT about 6%, and QPS decreases about 5.3%.
It remains opt-in; it is not a universal latency improvement. Extra enterprise
array IPC explained much of the first variant's TTFT regression; the mailbox
ablation is retained as evidence rather than silently discarding that result.

For a 512-token request already decoding when an 8k request arrives, the short
request's maximum SSE gap over three pairs was:

- Synchronous: 1034.49 / 1018.94 / 1037.22 ms.
- Retained pipeline: 581.86 / 587.34 / 620.30 ms.

The long request's TTFT stays about 1.04 s. The cloud's long GPU execution still
blocks decode; this is not chunk-level preemption or disappearance of WAN RTT.
Pooled P50/P95/P99, repeat standard deviations and sample counts are in
`results/stage3_representative/representative_summary.json`; tails are descriptive.

## Independent overlap evidence

A separate Nsight capture ran four 8k/32 requests. It is not mixed into the above
ordinary performance statistics. GPU kernels are attributed through CUDA runtime
correlation IDs and front/back or cloud-middle NVTX ranges. NCCL kernels are
excluded, and only one TP rank per side is considered. Socket upload-body and
download-body intervals exclude headers/cloud-wait ranges.

- 41 batches, 82 socket transfer ranges; union transfer duration 1415.33 ms.
- Enterprise GPU0 non-NCCL computation overlapping another batch's socket I/O:
  50.78 ms; cloud GPU0: 477.17 ms.
- Union overlap across those two independent resources: 527.95 ms. TP ranks are
  not added together as request critical-path time.
- Cloud and enterprise reports align by the same host's Nsight epoch start plus
  activity timestamps; every matched upload precedes its cloud GPU execution.

This proves actual computation/communication overlap. Socket I/O includes CPU/TCP
work and is not pure propagation. Much communication remains exposed; it does
not prove all WAN latency is hidden. `scripts/summarize_pipeline.py` reconstructs
these statistics from the SQLite exports and trace; examples include exact batch
IDs and kernel/transfer time intervals.

Serving source: initial pipeline `74b3afe`; mailbox/performance/profile source
`b26051d`. Later occupancy-counter/report edits do not alter model execution.
Small evidence is committed; full traces, NPZ, Nsight and SQLite remain local,
with sizes/SHA256 in the artifact manifest. No model weights are added.
