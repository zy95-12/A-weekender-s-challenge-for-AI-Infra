# Cloud PD serving on the split-inference runtime

The opt-in PD mode adds independent cloud prefill/decode TP groups to stage
1/2/3. Enterprise still owns embedding, front/back layers, sampling and token
streaming; cloud services receive only activation tensors and opaque request
metadata. The default four-A10 allocation is E1/P2/D1, with 4/27/5 layers.

This uses vLLM 0.10.2 model layers, tensor parallelism, FlashAttention paged KV,
and `PyNcclCommunicator` with a separate `StatelessProcessGroup` for KV transfer.
It extends the existing custom partial runner and PipelineScheduler. It is not
an invocation of native V1 LLMEngine or its complete P2pNcclConnector lifecycle.

```bash
./poc up --wan --pd --max-active 16 \
  --ipc-mode shm --wire-fast --tcp-buffer-mib 16 \
  --prefill-chunk-size 1024 --scheduler-policy decode-first \
  --decode-quota 4 --pipeline-window 2
```

`--prefill-tp 1 --decode-tp 2` selects E1/P1/D2. `--split 4:30:2` moves three
enterprise tail layers to the cloud; that optional layer placement has separate
performance implications and is not the default. This launcher reserves exactly
four GPUs and supports max_active <= 16. Increasing beyond C16 is deliberately
not part of the present launcher interface or capacity claim.

## Execution and ownership

Before submitting enterprise front work, asynchronously reserve source prompt
KV at P and prompt-plus-output KV at D. Enterprise's existing KVAdmission keeps
its local capacity reserved. Pending requests, active slots, per-role forward
windows, and two concurrent handoff tasks bound work and resident buffers.

E front chunks route to P. P returns hidden/residual immediately after compute
and enqueues a handoff after its final prompt chunk. In parallel, E runs the
corresponding back chunk and emits the first token. A background P coordinator
starts D receives and P exports in one deterministic request order. This order
is essential because NCCL point-to-point transfers have no request tags.

The P copy thread waits for its compute event, gathers logical KV pages, and
transfers contiguous global-head shards through vLLM NCCL. D merges/splits heads
according to the two-KV-head model's source/destination TP degrees and scatters
them into independently allocated destination pages. Padding in the last source
block is zeroed; imported computed length includes only valid prompt tokens.

All D ranks must finish their CUDA copies before the executor commits length L
and the coordinator announces readiness. First decode consumes the first token
sampled by E, at absolute position L with prompt KV covering [0,L). There is no
last-prompt-token recomputation or second sampling operation on cloud D.

D readiness does not wait for P source cleanup. P pages remain owned until the
source CUDA work completes, then are freed. Completion/cancellation releases
E and D ownership asynchronously; admission credits are retained until remote
release returns. A one-token completion still drains an already-started handoff.

E reuses the existing ready-decode batch construction. P and D have separate
forward windows, and a request awaiting KV is absent from the runnable decode
set. Ready decode back tasks have bounded priority over ready prefill back tasks;
front selection retains the configured decode-first quota. Ready back work is
dispatched before new front work, preserving bounded response buffering. This
is an explicit policy, not kernel preemption or physical mixed batching.

## Limits and failure behavior

Three processes run in two network namespaces; P/D share the cloud namespace,
so their KV transfer is cloud-local and does not traverse enterprise WAN netem.
The current transport bootstrap and device assignment are single-host specific.
Multi-host discovery, automatic failover, and arbitrary TP degrees are not
implemented. Only this pinned model and FlashAttention KV layout are supported.

Control operations validate a launch epoch; worker reservations are idempotent
for the same UUID and lengths. Duplicate/partial import cannot advance computed
length twice. Pool memory cannot be released while a copy future is incomplete.
Transfer exceptions or timeouts poison the affected executor and surface errors;
NCCL work that cannot be canceled requires restarting the worker groups. The
implementation does not silently recompute after partial streamed output.

The old idle KV sweeper is disabled for PD-owned allocations, since a timer is
not proof that DMA has completed. Normal cleanup is coordinator-owned. A crashed
enterprise/coordinator requires the managed three-role restart rather than
transparent orphan reclamation. Service health exposes pending PD admissions,
releases and coordinator reservations.

## Diagnostics and reproducibility

`--pd-verify-kv` computes SHA256 on source shards and on actual destination KV
pages after scatter. It is expensive and is disabled for all throughput runs.
`pd_kv_trace.jsonl` records transfer completion on every rank, D readiness and P
source release; `pd_trace.jsonl` records E's first runnable decode. Existing
`split_trace.jsonl` retains absolute positions and front/back/HTTP timestamps.
Operator capture remains opt-in, with separate P/D output labels. Worker operator
capture does not automatically annotate operations on the background KV thread.

```bash
# Independent actual NCCL gate, before introducing a serving handoff:
CUDA_VISIBLE_DEVICES=1,2,3 .venv/bin/python -m scripts.pd_nccl_gate

# With PD serving started with --pd-verify-kv:
.venv/bin/python -m scripts.pd_validate

# Same-host stage123 versus stage123+PD comparison; restores original demo:
.venv/bin/python scripts/pd_compare.py

.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Experiment orchestration currently reads the fixed prompt/reference from the
existing experiment workspace and uses the existing demo path for restoration.
The benchmark itself accepts explicit prompt/reference/output paths. Production
serving has no dependency on those experiment paths. Input copies and source
hashes accompany the measured artifacts.

The validation includes current 4K greedy output, C16 exact output text and length,
per-head source/destination KV hashes, a nonaligned 257-token prompt, max_tokens=1,
and disconnects during prefill. CPU tests cover reservation atomicity, head routing,
no early import commit/reuse, failed shard behavior, cancellation drain, and
decode eligibility with a full prefill window or another request awaiting KV.

The timed comparison uses sustained closed-loop C16, at least 60 seconds and six
steady completions per client, excluding warmup/drain. SLO is per-request TTFT
<=3000 ms and average TPOT <=100 ms with >=99% attainment. Token-level ITL and
first-to-second-token gap are additional metrics, not substituted for that SLO.
This is a C16 comparison, not a search for maximum qualified concurrency/QPS.

See [C16 results](pd-c16-results.md) for measured outcomes and remaining limits.
