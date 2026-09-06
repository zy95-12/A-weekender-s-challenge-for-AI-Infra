# 4K / TP2+2 operator evidence for issue #5

Both variants use Qwen2.5-3B-Instruct FP16, four A10, TP2+2, split4/27/5, WAN10Gbps each direction and5ms each way. C1, batch1; warmed capture of one logical4096-token prefill plus8 decode steps. All9 generated IDs match the shared reference. This is operator profiling with instrumentation overhead, not an SLO/QPS benchmark or a multi-request pipeline overlap measurement.

| Variant | Prefill CSV | Decode CSV | Shape/dtype summary | Raw capture |
|---|---|---|---|---|
| Baseline, stages OFF | [prefill](baseline/prefill_operators.csv) | [decode](baseline/decode_operators.csv) | [summary](baseline/operator_summary.csv) | [ZIP](baseline/raw_capture.zip) |
| Stage1+2+3 | [prefill](stage123/prefill_operators.csv) | [decode](stage123/decode_operators.csv) | [summary](stage123/operator_summary.csv) | [ZIP](stage123/raw_capture.zip) |

The CSVs are several MB each; use GitHub's **Raw / Download raw file** if preview is limited. Raw ZIPs include original Nsight reports, kernel_instances.csv, per-worker annotation JSONL, configurations/environment, prompt/reference, output check and trace. SQLite is reproducible from the Nsight reports and is not included.

Baseline uses pipe/original wire/default TCP/unchunked prefill/no pipeline. Optimized uses shm/wire-fast/TCP16MiB/chunk1024/decode-first/quota1/window2. All four ranks cover prefill position0/query4096 in baseline, or positions0/1024/2048/3072/query1024 in optimized. Decode covers absolute positions4096–4103/query1.

Each CSV row is a framework/custom operator invocation, not an individual CUDA kernel. Columns include type/name, ordered real tensor input shapes/dtypes, tensor argument paths/device/stride, role/rank/PID, request positions, CPU NVTX duration and attributed GPU kernel count/duration sum (ns). `execution_op` is additionally recorded in optimized CSV to distinguish forward/front/back. Summary time columns are microseconds (us).

GPU kernels are joined via process/correlation ID to the CUDA launch and its innermost operator NVTX range, with no double counting. Coverage: baseline16,394 operator invocations/10,058 kernels/0 unmatched; optimized22,122 invocations/13,002 kernels/0 unmatched. Across-GPU/stream sums are not wall time. CPU-only/view operators may have zero GPU duration.

Explicit tensor arguments include weights/bias and in-place output buffers. Tensor factories such as zeros have no tensor input (`[]`); scalar factory sizes/output dtype were not collected. Attention wrapper shapes describe explicit Q/K/V/output tensors, not implicit KV state or inferred per-kernel tensor signatures. Keep `kernel_instances.csv` and request positions when extending the simulator model. Instrumentation affects TP arrival skew and timings, so these are not uninstrumented production latency estimates.

## Implementation PRs

- [#10 Stage1 data path](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/pull/10)
- [#11 Stage2 chunked prefill](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/pull/11)
- [#12 Stage3 pipeline](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/pull/12)
- [#14 missing integration fixes and opt-in operator capture](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/pull/14)

These are stacked PRs on #8/#9, not merged main. #11/#12 remain drafts. For current Stage1–3 behavior use #14's head, which contains their implementation plus the first-chunk paged-attention fix and max-active forwarding fix. Stage4 is not in this branch. Prior #11 numerical failure evidence remains historical evidence; this profiling run does not constitute comprehensive numerical acceptance.

Recording is enabled with `--profile --operator-profile`, default OFF. Reproduction and schema: [operator-profiling.md at the captured implementation commit](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/blob/b7d1b3c/docs/operator-profiling.md). Source hashes are retained per variant; baseline was captured from the earlier local Stage4 worktree with all four optimization switches OFF, optimized from the Stage1–3 follow-up branch. [SHA256SUMS](SHA256SUMS.json) covers the published data.
