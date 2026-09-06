# Issue #6 TTFT/TPOT–QPS curve validation

This validation reproduces the measurement protocol from Issue #6: closed-loop
concurrency 1/2/4/8/12/16, immediate replacement after completion, and the
actual completion rate on the x-axis. Warmup and final drain are excluded from
the fixed 60-second simulation window.

## Calibration inputs

- GPU operators use the Issue #5 vLLM physical/fused operator captures.
- WAN stages use the independent Issue #5 activation-size sweep. Each direction
  is reconstructed from D2H, pack, measured HTTP path, IPC/unpack, and H2D
  components. Piecewise linear interpolation is used from 8 KiB to 128 MiB.
- The measured HTTP component already includes the 5 ms one-way propagation.
  The simulator therefore subtracts propagation before placing the remainder on
  the serialized link and does not add RTT or staging costs a second time.
- Outside the measured byte range, the simulator falls back to the analytical
  data-path model instead of extrapolating.

The Issue #6 curves are holdout data and are not used to fit any parameter.

## Run

From `Q4`:

```bash
PYTHONPATH=. python tools/simulate_issue6_qps_curve.py \
  --config configs/issue6_qps_sweep.json
```

All sweep inputs, including the concurrency list, are in
`configs/issue6_qps_sweep.json`.

The runner applies a 768 MiB address-space limit and executes points serially.
It writes a checkpoint after every point. The measured peak RSS for the full
12-point run was about 95 MiB. Outputs are:

- `outputs/validation/issue6_qps_curve/report.json`
- `outputs/validation/issue6_qps_curve/points.csv`
- `outputs/validation/issue6_qps_curve/curves.svg`

![Measured and simulated curves](https://raw.githubusercontent.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/63fe626a617a49cee891dcc9ceca68efdb836757/Q4/outputs/validation/issue6_qps_curve/curves.svg)

## Quantitative result

| Variant | Metric | MAPE | Signed bias |
|---|---|---:|---:|
| Baseline | completed QPS | 23.58% | +23.58% |
| Baseline | mean TTFT | 60.62% | -60.62% |
| Baseline | P99 TTFT | 69.92% | -69.92% |
| Baseline | mean TPOT | 27.61% | +9.01% |
| Baseline | P99 TPOT | 22.88% | -22.88% |
| Stage 1+2+3 | completed QPS | 64.82% | +64.82% |
| Stage 1+2+3 | mean TTFT | 40.73% | -40.73% |
| Stage 1+2+3 | P99 TTFT | 36.52% | -36.52% |
| Stage 1+2+3 | mean TPOT | 39.47% | -39.47% |
| Stage 1+2+3 | P99 TPOT | 40.22% | -40.22% |

The simulator reproduces the qualitative Stage 1+2+3 shape: QPS saturates as
concurrency rises, TTFT remains mostly flat, and TPOT rises. It is consistently
too fast, however. At C4 it predicts 1.600 versus 0.933 QPS, 319.6 versus
544.7 ms mean TTFT, and 27.74 versus 45.93 ms mean TPOT.

The Baseline now permits forward-level interleaving while keeping a one-slot WAN
path. This reproduces per-request Prefill and batched Decode instead of allowing
one request to monopolize all 79 decode iterations. Its C4 mean TPOT is 48.61 ms
versus 48.48 ms measured. The remaining major failure is TTFT queue growth: the
simulated mean stays near 728 ms while the measured value rises from 806 ms at
C1 to 6331 ms at C16. A request-level legacy lock/queueing state is still absent.

## What the WAN calibration improved

At C4, the Baseline QPS prediction moves from 1.317 without calibration to
0.867 with calibration (actual 0.666), and mean TPOT moves from 33.65 to
48.61 ms (actual 48.48). For Stage 1+2+3, QPS moves from 1.733 to 1.600
(actual 0.933), while mean TPOT moves from 25.71 to 27.74 ms (actual 45.93).

The remaining Stage 1+2+3 residual is about 12 ms per decode step at C1 and
roughly 176 ms of TTFT. The WAN echo probe already includes transport and data
movement, while the operator profile uses GPU time. The most likely missing
term is CPU-side vLLM scheduling, launch/worker synchronization, and request
bookkeeping around a real forward. It should be measured independently rather
than fitted from the holdout curve.

## Validity boundary

The WAN capture is C1 and cannot identify multi-message CPU/HTTP contention.
The operator capture is also C1; unseen batch shapes use an operator-type mean
correction. Consequently this version supports behavioral curve simulation and
quantifies its error, but is not yet an accurate SLO capacity predictor.
