# Issue #5 vLLM operator calibration

This is the first calibration slice for the operator data attached to
[issue #5](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/issues/5#issuecomment-5556275523)
and the Stage 1/2/3 implementation in
[PR #14](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/pull/14).

## Source of truth

Hugging Face config remains the semantic source for layer count, hidden size,
GQA heads, intermediate size, vocabulary and dtype. It is not the physical
kernel graph. With `operator_backend.name=vllm`, cost and dependencies follow
the operators actually present in the capture:

```text
embedding -> TP all-reduce -> RMSNorm
  -> fused QKV -> RoPE -> output allocation -> unified attention
  -> O projection -> TP all-reduce -> fused residual + RMSNorm
  -> fused GateUp -> fused Silu+Mul -> Down projection -> TP all-reduce
  -> fused residual + RMSNorm for the next layer
```

The final stage also models the fused final residual/RMSNorm and TP-local LM
head. Stage boundaries carry hidden/residual state and start with the captured
fused Add+RMSNorm behavior.

For Qwen2.5-3B, TP2 and split 4/27/5, this graph reproduces the core per-rank
capture counts:

| Resource path | Linear | `pynccl.all_reduce` | fused Add+RMSNorm |
|---|---:|---:|---:|
| Edge Front (4 layers) | 16 | 9 | 7 |
| Cloud (27 layers) | 108 | 54 | 54 |
| Edge Tail (5 layers + LM head) | 21 | 10 | 11 |
| Enterprise total | 37 | 19 | 18 |

## Conversion and matching

`tools/build_issue5_profile.py` consumes the four compact operator CSV files.
For each raw invocation it:

1. maps the raw vLLM/PyTorch op to the physical DAG operator type;
2. constructs a stable signature from every input tensor shape and dtype;
3. adds request position/query length for attention and TP degree for NCCL;
4. aligns repeated invocations by batch, role, signature and ordinal;
5. reduces ordinary ops with `max(rank GPU duration)`, never their sum;
6. for NCCL, uses `max(end_rank) - max(start_rank)` from raw timestamps so an
   early rank's wait for the final rank is not charged as intrinsic latency;
7. takes the median across matching layers/invocations;
8. records the analytical Roofline value used to form the fallback correction.

CPU NVTX duration is deliberately ignored. At runtime, an exact signature uses
the measured median. A new shape uses the arithmetic mean of available
`measured / roofline` factors for the same operator type, dtype, TP and physical
backend. A missing operator type remains on Roofline.

Generated artifacts:

- `profiles/issue5_a10_tp2_baseline.json`: 4K unchunked prefill plus decode;
- `profiles/issue5_a10_tp2_stage123.json`: 1K chunk prefill plus decode;
- `profiles/issue5_mapping_report.json`: coverage, unsupported ops and aggregate
  correction factors.

The current report maps 95.4%/95.6% of GPU-bearing calls for baseline/Stage123,
covering 97.3%/98.3% of the summed captured GPU duration. All mapped core
invocations have two TP ranks. Remaining rows are predominantly broadcast,
gather, mask construction and sampling/output plumbing outside the Transformer
layer DAG.

Raw timestamp reduction was available for all 657 baseline and 876 Stage123
collective invocations. Removing early-rank arrival wait changes their summed
critical duration from 239.2 to 92.7 ms for baseline and from 285.9 to 100.4 ms
for Stage123. The resulting exact median all-reduce latency is 1.224 ms for a
4K-token activation, 0.331 ms for a 1K-token activation, and about 0.009 ms for
a one-token activation.

The mapping report also evaluates every invocation against its exact-signature
median. This is a calibration-fit/within-capture variability metric, not an
unseen-shape holdout result:

| Operator | Baseline Roofline MAPE | Baseline profiled MAPE | Stage123 Roofline MAPE | Stage123 profiled MAPE |
|---|---:|---:|---:|---:|
| GEMM / fused linear | 44.15% | 1.13% | 48.17% | 1.19% |
| Unified attention | 30.31% | 1.63% | 25.91% | 1.82% |
| TP all-reduce | 9.39% | 2.61% | 18.23% | 2.56% |
| fused Add+RMSNorm | 237.16% | 1.94% | 193.37% | 1.48% |
| fused Silu+Mul | 154.60% | 2.17% | 121.48% | 1.90% |

The exact-profile signed systematic bias for these core types is within about
one percent; this is expected because the predictor is the median of the same
capture. Layer and end-to-end accuracy must still be reported on ordinary-run
holdout data to avoid confusing calibration fit with predictive accuracy.

## Accuracy boundary and next slices

The compact CSV plus raw timestamp file is sufficient for GEMM, unified
attention, normalization, RoPE, Silu+Mul and an arrival-corrected collective
correction. It is not sufficient for the final end-to-end accuracy claim:

- NCCL timestamp comparison assumes the two ranks/devices in one Nsight report
  use the exported common trace timebase; this is true for the supplied data
  and should be revalidated for captures assembled from separate reports.
- Nsight does not measure ordinary WAN/HTTP/IPC behavior faithfully. Stage 1
  needs non-instrumented D2H, pack, IPC, TCP, unpack and H2D measurements.
- The captured data is B1, A10, FP16, TP2 and approximately 4K context. Other
  batch sizes, hardware, TP degree and contexts currently use type-average
  correction rather than direct evidence.
- Captured end-to-end stage wall time is profiling-inflated and is used only as
  an internal dependency/order check. TTFT/TPOT/QPS validation must use ordinary
  runs from the Stage 1/2/3 implementation.

The simulator now splits the network path into staging, serialized link and
overlappable propagation/receive intervals, preserves Front batch membership
end to end, and applies window=2 to whole RPC jobs. The next implementation
slice is the quantitative operator/layer/prefill/decode and end-to-end error
report against non-instrumented holdout runs.
