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

CPU and representative GPU results are recorded below. Acceptance is scoped to
these cases; it is not a general-workload performance or SLO guarantee.

## Representative acceptance and retention

Native TP2 full-vocabulary comparison passed for k0, k2 and k4: 132 positions
per configuration over 128/1024/4096/8192 input, MAE/max-abs=0 and exact greedy.
Grouped teacher-forcing uses the same q=1 verification path; it does not silently
fall back to one WAN roundtrip per token. Real explanation, copy and code prompts
also produced exactly the same 128 target token IDs as OFF. k2/k4 passed long KV,
C4 isolation, needle QA, stream usage and cancellation. EOS and final one-token
budgets matched OFF. Unit tests exercise first/middle rejection, full acceptance,
EOS trimming, both-side rollback ordering, continuing from the cropped prefix,
and q=1 call shapes inside grouped verification.

For C1, three repeats per real prompt, fixed 128 outputs including ignore-EOS:

| Prompt | OFF TPOT ms | k2 TPOT ms | k4 TPOT ms |
|---|---:|---:|---:|
| Explanation | 32.10 | 33.38 | 35.46 |
| Copy passage | 32.29 | 29.16 | 29.61 |
| Python code | 32.24 | 35.05 | 37.55 |

k2 draft acceptance was 4/10, 57/64 and 16/35 respectively (same counts across
three repeats). Per copy request the 128 forward roundtrips became 71 forwards
plus 5 rollback roundtrips. Explanation became 124+5, and code 112+12: fewer
forward calls alone do not prove a gain after rejected work and rollback costs.
The 128-output copy includes continuation past EOS and is explicitly separate
from the following normal-EOS experiment.

The normal-EOS copy request produced the SAME 79 tokens/text in all variants:

| Variant | TPOT ms | Request wall ms | Longest SSE gap ms |
|---|---:|---:|---:|
| OFF | 32.87 | 2655.54 | 35.68 |
| k2 | 25.10 | 2044.87 | 87.56 |
| k4 | 24.20 | 1976.06 | 129.02 |

These are means of three repeats. k2 lowers normal-EOS copy TPOT about 23.6%,
but produces bursts with a substantially longer no-output interval. k4 saves
slightly more average time here, with a still longer gap and larger regressions
on other prompts. Retain k2 as an opt-in candidate for workloads with substantial
context reuse. Do not enable it globally or claim average gains on general QA
or coding. The demo default remains ordinary target inference.

For the ordinary fixed-128 copy run, 96 groups with two proposed tokens had mean
D=0.0375 ms, V_service=64.0034 ms, upload+return paths=10.7651 ms,
M=2.78125 and rollback=1.7334 ms per group. V_service is measured front-to-back
wall minus the upload/return path intervals: it includes host/IPC/GPU service,
and excludes HTTP/TCP path costs. It is not pure GPU time or a measurement of
only propagation RTT. Non-speculative rounds also contribute to request TPOT.

A separate Nsight/synchronized normal-EOS copy profile had 26 k2 verification
groups. Their mean CPU proposal D was 0.0217 ms; summed GPU-stage event time on
one rank of each sequential partition was 74.91 ms per group; mean confirmed
output M was 2.9615. This GPU measure includes collectives but excludes WAN,
CPU staging and IPC; it is not the complete V(k) wall cost. All 52 proposed tokens
were accepted in that profile; EOS caused one suffix truncation. Mean rollback
cost across groups was 0.438 ms. These profiled times are not substituted into
ordinary TPOT. Normal measurements retain actual forward and rollback counts,
D, rollback wall, TTFT/TPOT, event timestamps, percentiles and repeat dispersion.

Serving source for all ordinary variants was `5e2a40a`, clean at measurement.
`results/stage4_representative/representative_summary.json` and
`scripts/summarize_speculation.py` provide reproducible aggregation. Full logits
NPZ, profiling binaries and unfiltered traces stay local with an artifact manifest;
small per-request and filtered verification traces are included. No draft/model
weights were downloaded or added. Async long-prompt chunking remains gated by
stage 2's numerical failure; this speculative variant uses unchunked prefill.
