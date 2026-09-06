# Q4：Split-LLM Serving 系统建模

本目录是 [Issue #5](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/issues/5)
的第一阶段实现：使用 analytical Roofline、依赖 DAG、可切换 continuous batching
和离散事件资源模型，模拟一个 Split-LLM Serving 实例的端到端行为。

本版本的目标是验证调度、资源竞争和请求间流水逻辑，不进行最大 QPS 搜索。模型结构和
shape 来自 Hugging Face Qwen2/Qwen3 配置，耗时仍是未经 profiling 校准的 analytical Roofline。

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

### Issue #6 Stage 1+2+3 演示

[`configs/issue6_stage123.json`](configs/issue6_stage123.json) 使用 Qwen2.5-3B、
4/27/5 split、端云 TP2+2，打开数据路径优化、1024-token chunked prefill、
有界 decode-first FCFS、跨请求流水和闭环并发 4：

```bash
python -m split_serving_sim \
  --config configs/issue6_stage123.json \
  --output-dir outputs/issue6-stage123
```

对应控制项为：

- `data_path.ipc_mode=copy|shm`、`wire_fast` 和 `tcp_buffer_mib`；
- `scheduler.decode_first`、`max_consecutive_decode_batches`、
  `max_decode_tokens_per_batch` 和 `max_prefill_wait_ms`；
- `execution.max_inflight_transactions`、`buffer_pool_mib`；
- `workload.mode=closed_loop`、`concurrency` 和 `warmup_requests`；
- `slo.target_attainment`，输出逐请求联合达标率与 goodput。

数据路径参数是可解释的分析模型；示例中的 PCIe、内存和 IPC 带宽不是 PR #9/#10
实测拟合结果。用于容量结论前应通过 `performance_profile` 导入对应开关、payload 和
并发下的普通运行数据。完整语义与边界见
[`docs/ISSUE6_STAGE123.md`](docs/ISSUE6_STAGE123.md)。

### 复现 PR #8 baseline 行为

以下配置对应 [PR #8](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/pull/8)
的 Qwen2.5-3B、双侧 A10 TP=2、同步 HTTP RPC 和 naive scheduler：

- [`configs/pr8_split_1_3.json`](configs/pr8_split_1_3.json)：4 / 27 / 5；
- [`configs/pr8_split_1_1.json`](configs/pr8_split_1_1.json)：9 / 18 / 9；
- [`configs/pr8_split_3_1.json`](configs/pr8_split_3_1.json)：13 / 9 / 14。

```bash
python -m split_serving_sim \
  --config configs/pr8_split_1_3.json \
  --output-dir outputs/pr8-split-1-3
```

baseline 使用 `execution.mode=synchronous_rpc` 和
`scheduler.policy=split_poc_naive`：一次 batch 必须以相同成员顺序完成 Front、上传、Cloud、
下载和 Back，期间不能插入另一个 batch；每轮最多接纳一个新请求，有未完成 Prefill 时只运行
一个 Prefill，否则把所有 active 请求组成 Decode batch。

这些配置不会移除优化能力。需要实验 chunked prefill、mixed continuous batching 或跨 stage
流水时，只需切换相应参数：

```json
{
  "execution": {"mode": "pipelined"},
  "static_policy": {"continuous_batching": {"enabled": true}},
  "scheduler": {"policy": "fcfs", "enable_chunked_prefill": true},
  "attention_backend": {"mode": "unified"}
}
```

示例配置位于 [`configs/example.json`](configs/example.json)，其中包含：

- 仓内保存的 Hugging Face Qwen3-32B 原始结构配置；
- Edge 6 层、Cloud 56 层、Edge Tail 2 层；
- Edge TP=1、Cloud TP=4×PP=2；
- 10 Gbps 上下行和 10 ms RTT；
- BS=8、prefill chunk=512、每批 prefill budget=512、FCFS continuous batching；
- 4 个请求，每 10 ms 到达一个，每个输入 512 tokens、输出 8 tokens；该到达模式会产生
  可供甘特图检查的 mixed prefill/decode batch。

命令输出：

- `summary.json`：整体 TTFT、请求级 TPOT、逐 token ITL、吞吐、SLO goodput 和利用率；
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
Average batch size       1.70
Observed throughput      8.05 requests/s
P99 TTFT                 252.66 ms
P99 TPOT                 120.76 ms
Mixed batches            4
Cloud PP0 utilization    58.7%
Cloud PP1 utilization    57.1%
Edge GPU utilization     50.8%
SLO                      Fail
```

Observed throughput 只是指定输入流量下的观测结果，不代表最大 SLO-compliant QPS。

## 代码入口

```text
config.py          JSON schema、解析和拓扑/显存校验
performance.py     Qwen2/Qwen3 逐层算子 workload、Roofline、TP 和 WAN 模型
dag.py             Prefill chunk DAG 与 lazy decode DAG
scheduler.py       vLLM-style running-first 调度、FCFS/priority 排序与 backpressure
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
    -> vLLM-style Scheduler / Batch Formation
    -> Roofline or Network Duration
    -> Resource-aware Event Loop
    -> Metrics + Trace + Gantt
```

## 调度语义

- `batch_size` 是上限，eager dispatch 可以执行更小的 batch；
- `max_batched_tokens` 同时限制一个 batch 的 token 数；
- `prefill_token_budget` 限制单次调度注入的 prefill token，避免 prefill 洪峰长期阻塞 decode；
- Scheduler 先推进 running requests，再按 `fcfs`、`priority` 或实验性的
  `shortest_prefill` 顺序接纳 waiting requests；`split_poc_naive` 用于复现 PR #8；
- `decode_first=true` 在 FCFS 内优先选择 ready decode，并由连续 decode batch 数、
  decode token 上限和 prefill 最大等待时间保证长 prompt 获得推进机会；
- 一个 GPU batch 只包含同一 stage，但可以混合 ready 的 prefill/decode work；
- Edge Front 和 Edge Tail 共享 topology 中配置的 Edge GPU；
- prefill chunk 在每个模型 stage 上保持因果顺序，同时允许跨 stage overlap；
- decode DAG 在上一个 token ready 后懒生成；
- WAN uplink/downlink 是两个独立的单服务器资源；
- `pipeline_depth` 限制单请求在途 prefill chunk；
- `max_outstanding_prefill_chunks` 提供全局 backpressure。
- `max_inflight_transactions` 和 `buffer_pool_mib` 从 Edge Front 到 Edge Tail
  限制所有 prefill/decode step 的在途数量与激活内存。

### Continuous batching 开关

`static_policy.continuous_batching.enabled=true` 时，系统维护最多 `scheduler.max_num_seqs` 个 active sequence
slots；请求先进入 FCFS admission queue，有空 slot 时立刻补入。每次资源空闲都根据当前
ready work items 重新形成 batch，因此 batch 成员可以在 decode iteration 之间变化，完成的
sequence slot 会由等待请求补入。

关闭时，请求按 FCFS 形成最多 `batch_size` 个请求的静态 cohort。cohort 内所有请求完成前
不会 admission 后续请求，也不会用后续请求填补提前结束的 slot：

```json
{
  "continuous_batching": {"enabled": false}
}
```

无论开关状态如何，正在执行的 kernel/batch 都不会在中途改变成员；重组发生在资源下一次
dispatch 时。

配置解析也接受等价的短名称 `continuous_batch`，README 和示例统一使用
`continuous_batching`。

## Attention backend 与 Scheduler/KV

```json
{
  "attention_backend": {"mode": "unified"},
  "scheduler": {
    "policy": "fcfs",
    "max_num_seqs": 8,
    "enable_chunked_prefill": true,
    "kv_cache": {
      "enabled": false,
      "allocation_mode": "preallocate",
      "block_size_tokens": 16,
      "num_blocks": 0,
      "watermark": 0.0,
      "prefix_caching": false,
      "enable_preemption": true,
      "preemption_mode": "recompute"
    }
  }
}
```

- `unified`：mixed batch 每层只有一个 `mix attention`，对应 FlashAttention 风格；
- `separate`：mixed batch 每层顺序执行 `prefill attention`、`decode attention`，对应
  FlashInfer 风格，并产生两次 kernel launch；
- 点击 mixed batch 后，详情区分别列出 Prefill requests 和 Decode requests；
- KV block pool、watermark、prefix sharing/LRU eviction 会影响 admission；
- `allocation_mode` 可选择 admission 时完整 `preallocate`，或随 position 增长的 `on_demand`；
- priority 模式可在安全点抢占低优先级请求，恢复时在各 GPU stage 计入 recompute；
- PD 模式可以通过每个 stage 的 `prefill_resource`、`decode_resource` 分离计算资源，
  并在首 token 前插入显式 `pd_kv_transfer`。

完整调研、配置和精度边界见 [`docs/SCHEDULER_RESEARCH.md`](docs/SCHEDULER_RESEARCH.md)。

## Qwen2/Qwen3 cost model

### Profiling 修正接口

可选的 `performance_profile` 按以下顺序修正 Roofline：同类型且 signature
完全一致时直接采用 profiling 延迟；没有相同 signature 时使用同类型、同作用域样本的平均
`latency_ms / roofline_ms`；没有同类型样本时保持原 Roofline。

```json
{
  "performance_profile": {
    "path": "../profiles/pr8_a10.json",
    "enabled": true
  }
}
```

每个 sample 可以包含 `operator_type`、`signature`、`latency_ms`、
`roofline_ms` 和 `match`。`match` 可限制 `phase`、`tp_degree`、`stage`、
`dtype`、网络方向或 `data_path` 变体。trace 会记录最终使用的 profile source 和修正系数。

PR #8 profile 可通过以下命令重新生成和验证：

```bash
python tools/build_pr8_profile.py --repo .. --pr8-ref origin/pr8-review
python tools/validate_pr8_baseline.py \
  --repo .. --pr8-ref origin/pr8-review \
  --profile profiles/pr8_a10.json \
  --output outputs/validation/pr8_accuracy_profiled.json
```

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
的 GQA 会分别按照 query heads 和 KV heads 计算 attention 与 KV-cache 流量。Qwen2 路径使用
相同的 GQA/RoPE/SwiGLU 主体，但不会生成 Qwen3 特有的 Q/K Norm 算子。

PR #8 配置引用固定 revision 对应的
[`models/qwen2_5_3b_instruct_config.json`](models/qwen2_5_3b_instruct_config.json)，并用实验级
override 将 dtype 设置为 FP16。

### 网络与 RPC staging

`network.activation_tensor_count` 控制每个方向传输的 activation tensor 数。普通 split 模型
可以使用 1；PR #8 同时传输 hidden states 和 residual，因此设置为 2。payload 为：

```text
tokens × hidden_size × dtype_bytes × activation_tensor_count
  + protocol_overhead_bytes
```

`sender_overhead_ms` 和 `receiver_overhead_ms` 可用于加入 D2H、NumPy/HTTP pack、unpack/H2D
等固定 staging 开销。默认是 0，避免在没有实测校准时伪造精度。

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

长输出会产生大量逐算子对象。`simulation` 提供两级内存保护：

```json
{
  "simulation": {
    "trace_enabled": true,
    "max_trace_records": 20000,
    "max_detailed_trace_records": 200
  }
}
```

- `max_trace_records` 限制 timeline batch 总数；
- `max_detailed_trace_records` 限制携带逐算子子流的 batch 数；
- 超过明细限制的 batch 仍保留 stage 总耗时，并标记
  `sub_operations_omitted=true`；
- 不需要甘特图的矩阵实验应设置 `trace_enabled=false`。

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

当前包含 39 个行为测试，覆盖 Hugging Face profile、Qwen2/Qwen3 GQA shape、逐层 TP collective、
PP 对角流水、replica sticky routing、continuous/static batching、FCFS、PR #8 同步 RPC、
双 tensor WAN、按需 KV、trace 内存保护和甘特图输出。

PR #8 baseline 的量化验证、受限内存复现命令和误差分解见
[`docs/PR8_VALIDATION.md`](docs/PR8_VALIDATION.md)，机器可读结果见
[`outputs/validation/pr8_accuracy.json`](outputs/validation/pr8_accuracy.json)。

## 当前边界

暂未实现 QPS 搜索、GPU profiling 校准、SP/CP、动态 replica 负载均衡、EP/MoE/MLA、
speculative decode、逐 stage 独立 KV pool、CUDA stream overlap 和随机 WAN jitter。
