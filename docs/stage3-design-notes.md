# Stage 3 design notes — NOT IMPLEMENTED, stage-2 GPU gate still pending

Prepared while stage-1 full matrix runs. Do not apply runtime changes until
serial chunk correctness and stage-2 acceptance are complete.

## Worker decomposition

- Keep OFF `Runner.execute(forward)` unchanged.
- Add opt-in `front` and `back` commands; no HTTP wait inside either GPU worker.
- Front and back execute on the SAME enterprise TP group / default GPU streams.
  One orchestrator serializes GPU commands. Network transfers run independently.
- Separate front/back KV allocators and committed positions. Layer groups already
  have distinct cache tensors, so their block IDs can differ safely. Front may
  advance before back. Do not use one cursor for both and fake the ordering.
- Front returns owned activation arrays; back receives cloud outputs and emits
  only on final prompt chunk / decode. Preserve intermediate head omission.
- Release BOTH allocators, only after no in-flight work can still write them.
- Report front/back used blocks separately and compute used bytes weighted by
  layer counts, not a false single allocator occupancy formula.
- Optional enterprise mailbox can reuse existing ownership machinery, but needs
  explicit phase-3 scope: stage-1 SHM was cloud-local only. Lock and owned response
  copies are essential; no view may survive slot reuse.

## True bounded in-flight work

- Parent has shared thread-safe data client using the same WAN/buffers/wire flags.
- Window counts all outstanding tasks, including ready responses awaiting back.
- Job front_position differs from committed/output position; decode can start
  only after the final prefill back result supplies its token.
- Prefer completed back work before submitting another front to limit residence.
- Preserve per-job back order even if HTTP completions are out of order.
- Keep batch/chunk IDs, item position/query length, all stage timestamps, window
  occupancy and dependencies. Never invent five independent GPU resources.

## Cloud causality / why predecessor Future.result is insufficient

- Waiting for the previous RPC response before sending the next same-job chunk
  enforces causality but serializes upload+return, limiting single-request pipeline.
- Better: permit bounded concurrent HTTP transfer, gate GPU execution at cloud
  by expected per-request position under a Condition. Out-of-order chunks wait
  WITHOUT holding the GPU execution lock; already committed positions are replay
  errors. After GPU commit advance expected position and notify waiters.
- Requests arrive via asynchronous body receive while one GPU executor runs in
  a worker thread. Check threadpool limits exceed the bounded in-flight window.
- Add finite wait time, unhealthy/cancel notifications, cleanup of expected
  positions. Preserve ordinary OFF behavior and strict metadata validation.
- A condition serializing the cloud GPU is valid because it is one TP group;
  allowing HTTP in-flight does not mean cloud GPU batches overlap.
- Main-loop response packing can stall uploads; consider packing owned arrays in
  a worker thread. Account for CPU and memory pressure, do not call this zero-copy.

## Cancellation / failure

- Stop scheduling cancelled jobs, drain their already submitted front/RPC/back
  work safely before releasing local/cloud KV. Never release then allow a late
  RPC to resurrect or overwrite state.
- On partial execution failure fail closed, notify all pending/active clients;
  do not keep using possibly corrupted KV. Drain/cancel futures with bounded
  time and require owned-service restart as necessary.
- Shutdown must not close worker pipes concurrently with a still-running
  scheduler. Existing 5-second join is insufficient for a 30-second RPC timeout;
  provide explicit bounded close/drain semantics before Executor.close.
- Backpressure must include response buffers, not only active HTTP requests.
- Admission should also budget KV, not merely max_active: per-request maximum
  ceil((ISL+OSL-1)/16) blocks can be conservatively reserved until release.
  Front/back allocators have separate progress but each has the same block cap.
  Do not admit many long prompts then fail the executor from avoidable KV OOM.

## Measurement

- Full prefill / serial chunk+policy / async chunk pipeline are separate ablations.
- Scan chunk 256/512/1024/2048 and window 1/2/4/8 after correctness.
- Ordinary trace: front/RPC/back timestamps, U and D using shared-host monotonic
  clock, first-output critical path. SUM(chunk wall) is invalid once overlapping.
- Independent Nsight needed to prove actual GPU kernels overlap transfer ranges;
  RPC overlap alone may merely hide cloud compute, not WAN transport.
- Both enterprise front/back share GPU capacity. Preserve timing scope and do
  not sum multi-GPU kernel times as request critical path.
- Current stage-1 TP22 8k C1 ordinary step ~994 ms: U104.58+D139.87+C511.39+E238.60.
  C/E are NOT pure GPU time. Do not use old U+D694 ms to forecast pipeline gain.
- TCP16/shm/wire echo per-message total: 4 MiB26.613 ms, 16 MiB76.274 ms,
  64 MiB330.174 ms. These omit GPU attention cost and cannot predict chunk
  serving throughput by themselves.

## Stage 4 later

- Full enterprise partial target is not a draft. Additional small Qwen draft
  must be explicit, pinned, counted in GPU memory/compute/TTFT and tokenizer
  compatibility checked. No weights downloaded yet.
- Greedy verification needs all candidate positions' target argmax (full logits
  only for diagnostics). Consider GPU argmax to avoid k*full-vocab CPU copies,
  but validate tie behavior/numerical alignment with original target.
- For target cache length P and last emitted y1 not cached, verify inputs
  [y1,draft_y2,...draft_y{k+1}]. With m accepted drafts emit m+1 confirmed tokens,
  retain target KV length P+m+1; the newly emitted correction/bonus is not cached.
- Crop both target sides consistently. Draft cache must be aligned with accepted
  target prefix; rejected suffix discarded. Record actual confirmed tokens per
  round rather than equating candidate length with acceptance.
- Handle EOS, output cap, rejection at first/middle/last candidate, cancel/failure,
  teacher-forcing verification and real prompt workloads. No code yet.

### Draft cache alignment detail (for later implementation)

Known target prefix includes last emitted y1, whereas target KV length P does not.
With k proposals, a sequential draft consumes y1 and k-1 draft tokens, leaving
draft cache length P+k. Target verification consumes y1 and ALL k drafts, giving
trial target length P+k+1. If m<k accepted, crop draft to P+m+1 (within current
draft cache); next proposal consumes the uncached correction. If all k accepted,
target keep length P+k+1 is one ahead of draft cache. Do NOT grow cache by crop:
on next proposal consume missing last accepted draft PLUS the target bonus.
Track draft cached length separately from target length, and feed the known
prefix suffix on the next draft forward (one or two tokens normally).

Limit k <= remaining_output_tokens-1 so the confirmation/bonus fits. For the
last remaining output token (k=0), ordinary target decode can avoid needless
draft work and preserve the existing q=1 kernel behavior. EOS acceptance stops
emission immediately; ignore_eos baseline still runs to fixed output length.

Target argmax array z has k+1 entries. Accept draft[j]==z[j] until first mismatch
m; outputs are draft[:m]+[z[m]] (all accepted uses z[k] bonus). Truncate target
KV on BOTH sides successfully BEFORE publishing confirmed tokens. Partial
truncate failure must fail closed, not continue from inconsistent caches.

Lazy draft prefill after target TTFT is possible, but moves draft-prefill cost
into first decode ITL; report it and maximum no-output gap, not just mean TPOT.
Small draft on enterprise GPU0 shares target resources and must appear in
process/model audit, GPU memory, profiling and D(k), even if no extra GPU is used.

For integration with phase-3 pipeline, keep per-job verification exclusive:
no next same-job front until back verdict + both-side truncation. Other jobs
may remain in flight. A simple first version can synchronously obtain a draft
proposal before submitting front; this blocks parent GPU scheduling but allows
already in-flight cloud RPCs to progress. Count that contention honestly.
Do not silently disable pipeline flags in speculative mode; either integrate
and validate their combination or reject unsupported combinations explicitly.

If EOS trims a nominal accepted+bonus output list, use ACTUAL emitted M for
target keeper P+M, not blindly P+m+1. All-accepted-with-EOS may emit no bonus.
Immediate cloud rollback is simplest correct first implementation, but adds an
extra control RTT on rejection. Skip rollback when no suffix needs trimming;
measure rollback_ms and rollback roundtrips explicitly. The modeling formula
then includes D+V+RTT+rollback cost; do not hide that cost or count only forward
RPCs. Piggybacked/lazy commit is a separate protocol optimization requiring
explicit provisional state and predecessor IDs, not an implicit unsafe shortcut.
