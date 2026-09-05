# Q4：Split-LLM Serving 系统建模

本目录是 [Issue #5](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/issues/5)
的第一阶段实现：使用 analytical Roofline、依赖 DAG、可切换 continuous batching
和离散事件资源模型，模拟一个 Split-LLM Serving 实例的端到端行为。

本版本的目标是验证调度、资源竞争和请求间流水逻辑，不进行最大 QPS 搜索。模型结构和
shape 来自 Hugging Face Qwen3 配置，耗时仍是未经 profiling 校准的 analytical Roofline。

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

- 仓内保存的 Hugging Face Qwen3-32B 原始结构配置；
- Edge 6 层、Cloud 56 层、Edge Tail 2 层；
- Edge TP=1、Cloud TP=4×PP=2；
- 10 Gbps 上下行和 10 ms RTT；
- BS=8、prefill chunk=512、每批 prefill budget=512、FCFS continuous batching；
- 4 个请求，每 100 ms 到达一个，每个输入 512 tokens、输出 4 tokens。

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
Requests                 4
Average batch size       1.60
Observed throughput      7.33 requests/s
P99 TTFT                 152.42 ms
P99 TPOT                 101.64 ms
Cloud PP0 utilization    40.6%
Cloud PP1 utilization    40.6%
Edge GPU utilization     38.0%
SLO                      Fail
```

Observed throughput 只是指定输入流量下的观测结果，不代表最大 SLO-compliant QPS。

## 代码入口

```text
config.py          JSON schema、解析和拓扑/显存校验
performance.py     Qwen3 逐层算子 workload、Roofline、TP 和 WAN 模型
dag.py             Prefill chunk DAG 与 lazy decode DAG
scheduler.py       FCFS scheduler 与 batch capacity/backpressure
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
    -> FCFS Scheduler / Batch Formation
    -> Roofline or Network Duration
    -> Resource-aware Event Loop
    -> Metrics + Trace + Gantt
```

## 调度语义

- `batch_size` 是上限，eager dispatch 可以执行更小的 batch；
- `max_batched_tokens` 同时限制一个 batch 的 token 数；
- `prefill_token_budget` 限制单次调度注入的 prefill token，避免 prefill 洪峰长期阻塞 decode；
- Scheduler 严格按 ready time、request ID、work-item ID 做 FCFS，不提供 decode 插队；
- 一个 GPU batch 只包含同一 stage，但可以混合 ready 的 prefill/decode work；
- Edge Front 和 Edge Tail 共享 topology 中配置的 Edge GPU；
- prefill chunk 在每个模型 stage 上保持因果顺序，同时允许跨 stage overlap；
- decode DAG 在上一个 token ready 后懒生成；
- WAN uplink/downlink 是两个独立的单服务器资源；
- `pipeline_depth` 限制单请求在途 prefill chunk；
- `max_outstanding_prefill_chunks` 提供全局 backpressure。

### Continuous batching 开关

`static_policy.continuous_batching=true` 时，系统维护最多 `batch_size` 个 active sequence
slots；请求先进入 FCFS admission queue，有空 slot 时立刻补入。每次资源空闲都根据当前
ready work items 重新形成 batch，因此 batch 成员可以在 decode iteration 之间变化，完成的
sequence slot 会由等待请求补入。

关闭时，请求按 FCFS 形成最多 `batch_size` 个请求的静态 cohort。cohort 内所有请求完成前
不会 admission 后续请求，也不会用后续请求填补提前结束的 slot：

```json
{
  "scheduler": "fcfs",
  "continuous_batching": false
}
```

无论开关状态如何，正在执行的 kernel/batch 都不会在中途改变成员；重组发生在资源下一次
dispatch 时。

配置解析也接受等价的短名称 `continuous_batch`，README 和示例统一使用
`continuous_batching`。

## Qwen3 cost model

`configs/example.json` 通过 `hf_config_path` 引用
[`models/qwen3_32b_config.json`](models/qwen3_32b_config.json)。该文件摘自
[`Qwen/Qwen3-32B config.json`](https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json)，
保存模型的结构字段，包括
hidden/intermediate size、Q/KV heads、head dim、vocabulary、layer 数和 dtype；不下载模型权重。

每个 DecoderLayer 自下而上展开为：

```text
input RMSNorm
  -> Q/K/V projections
  -> Q norm / K norm -> RoPE
  -> KV-cache update
  -> attention
  -> O projection -> TP all-reduce
  -> residual + post-attention RMSNorm
  -> gate/up projections -> SiLU -> gated multiply
  -> down projection -> TP all-reduce
  -> residual
```

模型入口另有 embedding，出口按需要执行 final RMSNorm 和 LM head。每个算子单独计算
FLOPs、权重/activation/KV 访存量，并使用：

```text
latency = max(FLOPs / effective_peak_flops,
              bytes / effective_hbm_bandwidth) + kernel_launch_overhead
```

Prefill attention 使用 causal triangle 的 token pair 数，decode 使用当前 KV context。Qwen3
的 GQA 会分别按照 query heads 和 KV heads 计算 attention 与 KV-cache 流量。

### 并行语义

- `tp_degree` 表示单个推理 replica 内的 Tensor Parallel；
- `pp_degree` 把该模型 stage 的 layer range 均衡切给多个 Pipeline Parallel rank；
- 可选 `pp_layer_ranges` 可以覆盖默认均分，显式配置每个 PP rank 的连续 layer range；
- Q/K/V、gate/up 按 column parallel 推导 local shape；
- O projection、down projection 按 row parallel 推导，并在每一层原位插入 all-reduce；
- 当 TP 不大于 KV head 数时对 KV heads 分片；TP 大于 KV head 数时复制 KV heads；
- 相邻 PP rank 之间建立真实 work-item DAG，并计入 hidden-state P2P 传输；prefill chunks
  可以在 PP ranks 上形成对角流水；
- `replicas` 表示该 stage 的独立推理实例数，设备需求为
  `tp_degree * pp_degree * replicas`；
- 当前 dynamic routing 使用 `request_id % replicas`，保证请求及其 KV cache 对 replica sticky；
- 每个 replica 有独立队列和资源占用状态，可以并行处理不同请求。

例如：

```json
{
  "name": "cloud_middle",
  "layer_start": 6,
  "layer_end": 62,
  "resource": "cloud_gpu",
  "tp_degree": 4,
  "pp_degree": 2,
  "pp_layer_ranges": [[6, 34], [34, 62]],
  "replicas": 2
}
```

需要至少 16 张 `cloud_gpu`。本阶段尚未实现 replica 间动态负载均衡。

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
- 计算子流按 layer 展开，色块直接标注 `layer_06.q_proj`、`layer_06.attention` 等算子名；
- 通信子流在每层正确位置显示 `attention_all_reduce`、`mlp_all_reduce`，并显示 WAN 操作；
- 色块本身只标算子名；悬停或点击后显示 dtype 和 input shape；
- 点击主流 batch 可查看 request IDs、phase、输入形状、算子耗时和通信耗时。
- 键盘 `←`/`→` 平移视窗，`↑` 放大，`↓` 缩小；
- 点击主流 batch 后，绿色箭头显示它依赖的前驱 batch，红色箭头显示依赖它的后继 batch；
- 点击算子色块后，会额外显示该算子的 tensor 前驱和后继算子箭头；
- 窗口外的直接依赖会用指向时间窗口边界的箭头表示，详情区同时列出完整 batch ID。

## 测试

```bash
python -m unittest discover -s tests -v
```

当前包含 28 个行为测试，覆盖 Hugging Face profile、Qwen3/GQA shape、逐层 TP collective、
PP 对角流水、replica sticky routing、continuous/static batching、FCFS、WAN 和甘特图输出。

## 当前边界

暂未实现 QPS 搜索、GPU profiling 校准、SP/CP、动态 replica 负载均衡、EP/MoE/MLA、
PD Separation、speculative decode、KV paging、prefix cache、CUDA stream overlap 和随机 WAN jitter。
