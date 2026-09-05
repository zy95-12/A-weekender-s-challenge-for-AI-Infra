# PR #8 baseline validation

This report compares the uncalibrated Q4 analytical model with the measured
Qwen2.5-3B split-inference baseline in PR #8. The source data is the 32-point
normal-run `baseline_summary.csv` and the independent Nsight
`profile_summary.csv` / CUDA kernel summaries at PR #8 commit `09dbe653`.

## Method

- Reproduce all 4 enterprise/cloud TP combinations, 4 ISL/OSL pairs and
  client concurrency 1/4: 32 end-to-end points.
- Use `split_poc_naive + synchronous_rpc`, whole Prefill and all-active Decode.
- Represent a homogeneous closed-loop concurrency group as one simultaneous
  wave. In this scheduler all equal-length requests finish together, so its
  steady cycle throughput is the wave throughput. The real run still has more
  repetitions, warmup and variance than this deterministic model.
- Compare client TTFT/TPOT/QPS only with normal runs. Compare GPU stages and
  operators only with the independent profiling runs; the datasets are not mixed.
- No measured value is used to fit simulation parameters. Sender/receiver
  staging overhead remains zero.

The evaluator runs every matrix point in an isolated process with trace
disabled and a 768 MiB address-space limit. The parent is intended to run under
a 1.2 GiB shell limit. Results are checkpointed after each point.

```bash
ulimit -v 1258291
python tools/validate_pr8_baseline.py \
  --repo .. --pr8-ref origin/pr8-review \
  --memory-limit-mb 768 \
  --output outputs/validation/pr8_accuracy.json
```

Observed maximum RSS was 31.0 MiB. A separate worst-case 8192/1024,
concurrency-4 run with trace enabled used about 59 MiB after trace capping.

## Accuracy summary

Positive bias means the simulator predicts a value larger than measurement;
negative bias means it is optimistic.

| Level | Metric | Points/topologies | MAPE | Mean bias | Aggregate error |
|---|---|---:|---:|---:|---:|
| Operator | FlashAttention Prefill | 4 | 9.43% | +9.43% | — |
| Operator | FlashAttention Decode | 4 | 5.50% | -4.93% | — |
| Operator | Matrix multiplication | 4 | 37.13% | +37.13% | — |
| Operator | TP collective (TP configs) | 3 | 84.45% | -84.45% | — |
| Stage | Edge Front | 64 | 31.95% | -4.93% | +18.05% |
| Stage | Cloud Middle / 27 homogeneous layers | 64 | 25.01% | +3.30% | +15.31% |
| Stage | Edge Back | 64 | 27.68% | +8.58% | +25.43% |
| Phase GPU | Prefill GPU stages | 96 | 30.27% | +25.46% | +20.06% |
| Phase GPU | Decode GPU stages | 96 | 26.15% | -20.82% | -20.48% |
| Phase step | Prefill end-to-end step | 32 | 42.05% | -42.05% | -45.69% |
| Phase step | Decode end-to-end step | 32 | 21.49% | -21.49% | -22.28% |
| Client | TTFT mean | 32 | 33.49% | -33.49% | -42.94% |
| Client | TPOT mean | 32 | 13.84% | -10.50% | -11.44% |
| Client | QPS | 32 | 20.21% | +18.09% | +17.87% |

Across the 32 points, Pearson correlation is 0.990 for TTFT, 0.978 for QPS,
and only 0.468 for TPOT. The model captures TTFT and throughput ordering well,
but is not accurate enough for absolute SLO/capacity claims.

### Layer/stage phase split

| Stage | Prefill MAPE / bias | Decode MAPE / bias |
|---|---:|---:|
| Edge Front | 33.74% / +20.11% | 30.15% / -29.96% |
| Cloud Middle | 23.03% / +22.23% | 26.99% / -15.62% |
| Edge Back | 34.04% / +34.04% | 21.31% / -16.88% |

Cloud Middle contains only the 27 homogeneous transformer layers, so its stage
percentage is the best available layer-level validation. Front and Back also
contain embedding/final norm/LM head and are not pure transformer-layer errors.

For TP=2+2, concurrency 1, representative Cloud Middle per-layer numbers are:

| Phase / shape | Measured ms/layer | Simulated ms/layer | Error |
|---|---:|---:|---:|
| Prefill 512 | 1.292 | 1.202 | -7.0% |
| Prefill 2048 | 3.801 | 4.387 | +15.4% |
| Prefill 8192 | 17.006 | 18.845 | +10.8% |
| Decode 512 context | 0.675 | 0.375 | -44.4% |
| Decode 2048 context | 0.672 | 0.377 | -43.9% |
| Decode 8192 context | 0.673 | 0.385 | -42.8% |

### Network split

| Direction/phase | MAPE | Bias |
|---|---:|---:|
| Prefill upload | 69.11% | -69.11% |
| Prefill download | 71.78% | -71.78% |
| Decode upload | 8.13% | -8.13% |
| Decode download | 7.54% | -7.54% |

Decode tensors are tiny and dominated by configured 5 ms one-way propagation,
so the simple model is close. Large Prefill transfers expose missing D2H,
NumPy/HTTP serialization, copies and receiver staging costs.

### End-to-end topology split

| Enterprise+Cloud TP | TTFT MAPE | TPOT MAPE | QPS MAPE |
|---|---:|---:|---:|
| 1+1 | 25.2% | 7.6% | 7.4% |
| 1+2 | 35.6% | 17.0% | 24.5% |
| 2+1 | 31.6% | 4.0% | 8.6% |
| 2+2 | 41.5% | 26.7% | 40.3% |

The worst workload is 8192/256: TTFT MAPE 48.7% and QPS MAPE 31.2%. The
largest TTFT error is TP=2+2, 8192/256: 1833.18 ms measured versus 823.65 ms
simulated (-55.1%).

## Main error sources

1. **RPC staging is absent.** PR #8 timers include D2H, NumPy/HTTP packing,
   TCP, unpacking and H2D. Q4 models payload serialization and propagation;
   configured fixed staging inputs are zero. This dominates long Prefill TTFT.
2. **TP collective is too optimistic.** The PCIe model underpredicts measured
   NCCL AllReduce occupancy by 80.2%–87.3%, making TP=2 look too beneficial.
3. **Logical MM is not a kernel lookup model.** Q4 applies Roofline and launch
   overhead to seven logical projections per layer. vLLM/CUTLASS selects and
   fuses different GEMM/GEMV paths by shape. Aggregate MM is overpredicted by
   28.8%–48.2% depending on TP.
4. **Runtime residual is missing.** KV/cache kernels, block-table work,
   CUDA/Python synchronization, bookkeeping, sampling/logits, IPC and HTTP
   metadata are outside logical operators. Thus Decode stages remain low even
   when modeled MM is high: missing collective and residual costs are larger.
5. **Operator profiling is aggregate.** PR #8 has per-batch stage timings but
   CUDA kernel names only as aggregate Nsight summaries. FA/MM validation sums
   the profiling workload and divides summed TP kernel time by TP degree; it is
   not a per-shape kernel latency validation.
6. **The simulator is deterministic.** It has no OS/TCP jitter or run-to-run
   variance and uses one homogeneous concurrency wave rather than the measured
   repeated harness. This affects tail latency and some QPS bias.

## Validity conclusion

The model is valid as a **behavioral and trend simulator**: DAG order, batch
membership, blocking, activation bytes, TTFT ordering and QPS ordering agree
with the POC. It is not yet a calibrated **absolute performance simulator**.
TTFT is typically too low by one third and QPS too high by about one fifth;
TPOT is closer in mean error but has weak cross-configuration correlation.

Calibration should use a held-out protocol: fit operator-class lookup or
efficiencies, size-dependent RPC staging, NCCL latency/bandwidth and fixed
per-step residual on part of PR #8, then evaluate untouched TP/workload points.
Fitting all 32 points and reporting the same points would not be validation.
