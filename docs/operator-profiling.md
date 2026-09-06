# Operator profiling for the split-serving simulator

Stage1,2,3 implementations are in PR#10,#11,#12 respectively, stacked on baseline#8 and measurement#9. This follow-up adds missing local integration fixes and a reproducible operator capture switch on top of#12. The earlier numerical failure evidence remains historical evidence; this change does not declare every Stage2 numerical gate passed.

## Integration fixes

When chunking is enabled, the first prefill chunk now keeps its paged-KV block table, matching the continuation path. Previously its empty table selected the contiguous attention path. Unchunked baseline retains its original empty-table path. A regression test checks both cases and nonzero continuation positions.

`./poc up --max-active N` now forwards the requested limit to both worker roles (1–16, default8). Previously the concurrency experiments initially recorded16 in the launcher while workers still received their default8. Published final high-concurrency measurements were rerun after this fix; their evidence is linked in issue#6.

## Opt-in recording

```bash
./poc up --wan --tp 2 --profile --operator-profile \
  --ipc-mode shm --wire-fast --tcp-buffer-mib 16 \
  --prefill-chunk-size 1024 --scheduler-policy decode-first \
  --decode-quota 1 --pipeline-window 2 --max-active 16

.venv/bin/python scripts/capture_operators.py \
  --root results/operator_stage123_4k \
  --prompt /path/to/prompt.json --reference /path/to/reference.json \
  --variant stage123

./poc down
```

`--operator-profile` requires `--profile`, defaults OFF and enables the worker-side `SPLIT_OPERATOR_CAPTURE=1` instrumentation. With it disabled no dispatcher interception is installed. Prompt format is `{"prompt_ids": [...]}` with4096 IDs; reference format is `{"greedy_ids": [...]}` with at least9 IDs. Both exact files are published with the issue#5 evidence.

Use `--variant baseline` and omit Stage1/2/3 switches for the baseline. Warmup precedes capture. Both variants generate9 token IDs: one first token and8 decode steps. The debug request uses the live scheduler; Stage3 front/RPC/back and chunk scheduling remain active. Concurrency1 provides comparable operator shapes, not cross-request throughput evidence.

Stop the diagnostic service to flush its Nsight reports. Copy the raw files from the directory named in `source_results.txt` to the selected output root. Export `enterprise.nsys-rep` and `cloud.nsys-rep` with Nsight Systems2025.3.1 `nsys export --type sqlite --output <role>.sqlite <role>.nsys-rep`, then run:

```bash
python3 scripts/export_operator_capture.py --root results/operator_stage123_4k
```

The service can then be restarted with its original configuration.

## CSV contract and limits

`prefill_operators.csv` and `decode_operators.csv` contain actual operator invocations: role/rank/PID, phase, `execution_op` (forward/front/back), batch ID, request absolute positions/query lengths, operator type, ordered input tensor shapes/dtypes, full tensor metadata with argument paths/device/stride, CPU NVTX duration and GPU kernel count/duration sum. GPU times ending `_ns` are nanoseconds.

`operator_summary.csv` groups role/rank/phase/type/input shapes/dtypes, with calls and total/mean/min/max GPU duration in microseconds. `kernel_instances.csv` retains kernel names/timestamps/streams and the operator ID for joins.

TorchDispatch annotates real tensor arguments without reading tensor values. PyNCCL methods are wrapped separately. Export joins GPU process/correlation ID to CUDA runtime launch, then the innermost operator range containing that launch. CPU range time is not GPU duration. Each kernel is attributed once; sums across streams or GPUs are not wall time. View/CPU-only operators may have zero GPU kernels.

Shapes are TP-local explicit tensor arguments, including weights/bias and in-place output buffers. Tensor factories such as `aten.zeros` have no input tensor and are recorded with `[]`; scalar factory sizes/output dtype are not collected. Attention wrapper records its explicit Q/K/V/output arguments, not implicit forward-context KV state. Individual kernel input signatures are not fabricated from operator metadata. Kernel-level mappings and request positions are retained for further modeling.

Instrumentation and phase synchronization perturb timings, especially TP communication arrival skew; these profiles are not SLO/QPS benchmarks or proof of multi-request pipeline speedup.

## Representative validation

4K, TP2+2, four A10, FP16, split4/27/5, WAN10Gbps each direction/5ms each way, C1. Optimized configuration uses chunk1024/window2/quota1 and Stage1 data-path switches. No Stage4 code is part of this branch.

- Baseline capture:16,394 operator invocations,10,058 kernels,0 unmatched.
- Stage1+2+3 capture:22,122 operator invocations,13,002 kernels,0 unmatched.
- Four ranks each cover prefill positions0/1024/2048/3072 (query1024), followed by decode positions4096–4103 (query1). Baseline prefill is position0/query4096.
- Warmup and captured9-token outputs match the shared reference. Both captures release active/waiting/KV state.51 CPU tests pass, including chunk metadata and mixed-dtype operator-capture checks.

CSV evidence and related PR links are published in issue#5. Stage1–3 runtime/scheduler behavior and the previously published continuous-concurrency/decode-quota measurements are complementary evidence; profiles alone do not specify a calibrated maximum-QPS simulator.
