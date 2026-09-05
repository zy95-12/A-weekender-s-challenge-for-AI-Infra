# Prompt-lookup speculative decoding (experimental)

This opt-in draft is a CPU suffix lookup in the request's own prompt and confirmed
output. It loads no draft weights and uses no extra GPU. It is useful for copying,
repeated code/text, and has no assumed acceptance rate on ordinary questions.
It is not an independent small language model and cannot propose a continuation
when no suffix match exists. Target weights, split, and WAN remain unchanged.

`--pipeline-window 2 --speculative-tokens 2` enables up to two draft tokens.
The default is zero. Speculation requires the pipeline execution path and works
with its bounded window. Each request has at most one verification in flight;
other requests can progress. First-version verification selects a single request
at a time, so concurrent decode batching can regress and needs separate evidence.

For target cache length P, the last emitted token is not yet cached. Verification
processes that token plus k proposed tokens. Each partition executes the same
validated q=1 kernels sequentially; activations for all positions share one WAN
roundtrip. This reduces roundtrips without claiming a batched-GEMM speedup or
changing floating-point kernel shapes. The cloud sees activations and positions,
never prompt/draft token IDs or logits.

The target accepts the matching draft prefix and emits its correction/bonus.
If M tokens will actually be emitted (including EOS/length truncation), both
sides retain P+M cache positions. Excess front, cloud and back KV is truncated
before any confirmed token is published. A rejected suffix can require one
additional control roundtrip, included in the trace. Partial failure makes the
executor unhealthy; it does not resume from inconsistent state. Cancellation
first drains already submitted work and then releases both sides.

Teacher-forcing diagnostics group known forced inputs through the same verification
path, checking full-vocabulary logits against the fixed native reference. They
are separate from actual lookup/acceptance validation. Greedy checks must also
compare complete generated token sequences with speculation disabled, including
accepted/rejected proposals, EOS, length caps and cancellation.

Trace includes D(k), actual draft/accepted/emitted counts, verification RPC wall,
rollback cost/roundtrips and stage timestamps. RPC wall includes cloud compute;
do not add it again. Report total wall/TTFT/TPOT and actual SSE gap timestamps:
confirmed tokens retain one SSE event each but arrive in bursts. Average TPOT
alone does not describe the longest period without output.

CPU tests passed before GPU validation. Representative GPU results will be added
below; this text alone is not a performance or correctness acceptance report.
