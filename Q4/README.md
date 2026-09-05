# Q4：Split-LLM Serving 系统建模

本目录是 [Issue #5](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/issues/5)
的第一阶段实现：使用 analytical Roofline、依赖 DAG、naive continuous batching
和离散事件资源模型，模拟一个 Split-LLM Serving 实例的端到端行为。

本版本的目标是验证调度、资源竞争和请求间流水逻辑，不进行最大 QPS 搜索，也不承诺未经
profiling 校准的绝对性能精度。

## 系统路径

Prefill chunk 和每次 decode iteration 都经过：

```text
Edge Front -> WAN Up -> Cloud Middle -> WAN Down -> Edge Tail
```

## 快速演示

需要 Python 3.11 或更高版本，没有第三方运行时依赖。从仓库根目录执行：

```bash
cd Q4

python -m split_serving_sim \
  --config configs/example.json \
  --output-dir outputs/example
```

示例配置位于 [`configs/example.json`](configs/example.json)，其中包含：

- Qwen3-32B analytical model profile；
- Edge 6 层、Cloud 56 层、Edge Tail 2 层；
- Edge TP=1、Cloud TP=4；
- 10 Gbps 上下行和 10 ms RTT；
- BS=8、prefill chunk=512、每批 prefill budget=512、decode priority；
- 16 个请求，每 500 ms 到达一个，每个输入 4096 tokens、输出 32 tokens。

命令输出：

- `summary.json`：整体 TTFT、TPOT、E2E、吞吐和利用率；
- `requests.jsonl`：逐请求指标和 token 时间戳；
- `trace.jsonl`：逐 batch/resource 的流水区间；
- `gantt.html`：独立、可交互的流水甘特图。
- `gantt.svg`：不依赖 JavaScript、可在 GitHub 直接预览的静态总览。

## 已提交的示例结果

- [`outputs/demo/summary.json`](outputs/demo/summary.json)
- [`outputs/demo/requests.jsonl`](outputs/demo/requests.jsonl)
- [`outputs/demo/trace.jsonl`](outputs/demo/trace.jsonl)
- [`outputs/demo/gantt.html`](outputs/demo/gantt.html)
- [`outputs/demo/gantt.svg`](outputs/demo/gantt.svg)

本次 demo 的关键结果为：

```text
Requests                 16
Average batch size       3.18
Observed throughput      1.87 requests/s
P99 TTFT                 467.99 ms
P99 TPOT                 127.51 ms
Cloud GPU utilization    82.8%
Edge GPU utilization     45.9%
SLO                      Fail
```

Observed throughput 只是指定输入流量下的观测结果，不代表最大 SLO-compliant QPS。

## 代码入口

```text
config.py          JSON schema、解析和拓扑/显存校验
performance.py     Dense workload、Roofline、TP 和 WAN 模型
dag.py             Prefill chunk DAG 与 lazy decode DAG
scheduler.py       Naive decode-priority eager scheduler
simulator.py       Event queue、资源占用和 batch execution
metrics.py         TTFT、TPOT、E2E 和利用率
visualization.py   独立 HTML/SVG 甘特图
cli.py             命令行入口和结果文件输出
```

核心执行链：

```text
JSON Config
    -> Request Arrival
    -> Lazy Execution DAG
    -> Naive Scheduler / Batch Formation
    -> Roofline or Network Duration
    -> Resource-aware Event Loop
    -> Metrics + Trace + Gantt
```

## 调度语义

- `batch_size` 是上限，eager dispatch 可以执行更小的 batch；
- `max_batched_tokens` 同时限制一个 batch 的 token 数；
- `prefill_token_budget` 限制单次调度注入的 prefill token，避免 prefill 洪峰长期阻塞 decode；
- decode 默认优先，然后按 ready time 和 request ID 排序；
- 一个 GPU batch 只包含同一 stage，但可以混合 ready 的 prefill/decode work；
- Edge Front 和 Edge Tail 共享 topology 中配置的 Edge GPU；
- prefill chunk 在每个模型 stage 上保持因果顺序，同时允许跨 stage overlap；
- decode DAG 在上一个 token ready 后懒生成；
- WAN uplink/downlink 是两个独立的单服务器资源；
- `pipeline_depth` 限制单请求在途 prefill chunk；
- `max_outstanding_prefill_chunks` 提供全局 backpressure。

## Workload 模式

Synthetic workload 支持固定间隔或 Poisson 到达，也可以传入显式 trace：

```json
{
  "workload": {
    "mode": "trace",
    "requests": [
      {"request_id": 0, "arrival_time_ms": 0, "input_tokens": 128, "output_tokens": 8},
      {"request_id": 1, "arrival_time_ms": 5, "input_tokens": 64, "output_tokens": 4}
    ]
  }
}
```

## 甘特图

GitHub Raw 会使用 CSP 禁止内联 JavaScript，因此在 GitHub 上请直接预览
[`outputs/demo/gantt.svg`](outputs/demo/gantt.svg)。HTML 本身也包含静态首屏，不再因脚本
被禁用而显示空白。

需要缩放、拖拽和展开子流时，在本地打开 `outputs/demo/gantt.html`，或执行：

```bash
python -m http.server 8000 --directory outputs/demo
```

然后访问 `http://localhost:8000/gantt.html`。

- 支持鼠标滚轮或按钮缩放，并可按住图表水平拖动；
- 每一条主流表示 `edge_gpu`、`wan_up`、`cloud_gpu` 或 `wan_down` 资源；
- 点击资源左侧的 `▶` 只展开两条子流：`计算 / Compute` 和 `通信 / Communication`；
- 同一行的色块不会重叠，表示资源互斥；
- 不同行同时执行表示请求或 chunk 正在跨 stage 流水；
- 粗边框表示 batch 包含 decode work；
- 计算子流色块直接标注 attention projection、attention、MLP、kernel overhead 等算子名；
- 通信子流色块直接标注 TP collective、WAN serialization、WAN propagation 等操作名；
- 子流色块会显示 input shape，悬停或点击可查看完整详情；
- 点击主流 batch 可查看 request IDs、phase、输入形状、算子耗时和通信耗时。
- 键盘 `←`/`→` 平移视窗，`↑` 放大，`↓` 缩小；
- 点击主流 batch 后，绿色箭头显示它依赖的前驱 batch，红色箭头显示依赖它的后继 batch；
- 窗口外的直接依赖会用指向时间窗口边界的箭头表示，详情区同时列出完整 batch ID。

## 测试

```bash
python -m unittest discover -s tests -v
```

当前包含 16 个行为测试，覆盖配置校验、batch-aware Roofline、WAN batching、chunk
因果关系、lazy decode、共享 Edge GPU、依赖时序、BS batching、operator/TP 子流和甘特图输出。

## 当前边界

暂未实现 QPS 搜索、GPU profiling 校准、EP/MoE/MLA、PD Separation、speculative
decode、KV paging、prefix cache、CUDA stream overlap 和随机 WAN jitter。
