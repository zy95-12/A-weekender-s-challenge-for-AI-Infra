# POC 实测报告

测试服务器：4 × NVIDIA A10（每卡 23 GB），GPU 0/1 为 Enterprise TP=2，GPU 2/3 为 Cloud TP=2。
模型为 Qwen2.5-3B-Instruct，固定 revision 与 SHA256 见 `split_poc/__init__.py`。

## 当前状态（2026-09-05）

真实模型 POC 功能与精度验收通过，机器可一键启动并演示。完整流水线结果：
`results/validation_final/validation.json`。这是 POC 验收 PASS，不是性能 SLO PASS。
零权重测试只用于开发期验证协议、NCCL、KV 管理与故障处理，不作为以下模型精度或性能证据。

本 PR 包含可审阅的历史实测快照：

- [完整验收结果](../results/validation_final/validation.json)及该目录内的 JSON/JSONL、日志、GPU/DCGM 数据。
- [WAN 短压测汇总](../results/validation_final/benchmark_wan/summary.json)和[本地短压测汇总](../results/validation_final/benchmark_local/summary.json)。
- [企业 Nsight 报告](../results/20260905-122434/enterprise.nsys-rep)和[云端 Nsight 报告](../results/20260905-122434/cloud.nsys-rep)，以及同目录 kernel 汇总 CSV。
- 首次并发断言失败的日志及原始输出，用于核对本文的排查说明。

不提交模型权重、虚拟环境、原始 logits NPZ、可由 Nsight 报告重建的 SQLite，以及临时服务状态。
这些原始文件仍留在当前服务器；本 PR 的快照不是重新运行的结果。
`results/` 默认忽略，仅本次明确选择的历史报告进入版本控制。

## 精度与功能实测

原生 vLLM TP=2、FP16、FlashAttention、NCCL、eager，128/1024/4096/8192 四种输入长度，
每种长度独立重复两次；每次保存完整词表的 Prefill + 256 步 Decode logits。

| 切分 Front / Cloud / Back | 比较位置数 | 最大 MAE / 最大绝对误差 | Top-1 / Top-5 | 功能验收 |
|---|---:|---:|---:|---|
| 4 / 27 / 5 | 1028 | 0 / 0 | 100% / 100% | PASS |
| 9 / 18 / 9 | 1028 | 0 / 0 | 100% / 100% | PASS |
| 13 / 9 / 14 | 1028 | 0 / 0 | 100% / 100% | PASS |

所有位置均逐值一致；FP32 cosine 计算最小值为 0.99999988。
三种配置均通过独立 greedy 前 32 个 token-ID 对照，以及：

- 1024/4096/8192 输入 + 固定 256 输出的增量 KV 推理。
- 4 并发与逐条执行的正常回答 token-ID 精确比较。
- 4096/8192 token 的信息检索问答，都正确返回 `WILLOW-7319`。
- 32 token 流式事件与 usage 核对、客户端取消后的 KV 回收。
- WAN 下额外检查四种输入长度、每种 Prefill + 32 Decode，全部通过。

原始文件：`results/validation_final/reference/`、`correctness_*/`、`acceptance_*.json`。

并发测试排查记录：首次断言将已经输出 EOS 的回答强制延长为 32 token，数学题正常回答均为 `42`，
但 EOS 后模型臆造的下一轮对话不同。最终并发断言比较到首个 EOS（含）或第 32 步，以先到者为准；
所有强制延长输出仍保存在 `acceptance_*.isolation.json`，首次失败记录在 `results/validation/`。
没有放宽原生对照的数值阈值，也没有缩短其 256 步 Decode 验证。

## 短压测结果：非正式容量结论

官方 `vllm bench serve`，每点 20 请求，Poisson 到达，固定 4096/256，ignore_eos，未启用 profiler。

| 链路 | Offered QPS | 成功率 | P99 TTFT | P99 TPOT | 本测点阈值 |
|---|---:|---:|---:|---:|---|
| 本地无限制 | 0.5 | 100% | 2021 ms | 38.89 ms | 满足，但样本不足 |
| WAN：双向各 5 ms / 10 Gbps | 0.5 | 100% | 4225 ms | 56.51 ms | 不满足 TTFT |
| WAN：双向各 5 ms / 10 Gbps | 1.0 | 100% | 16111 ms | 56.32 ms | 不满足 TTFT |
| WAN：双向各 5 ms / 10 Gbps | 2.0 | 100% | 23689 ms | 56.35 ms | 不满足 TTFT |

原始结果：`results/validation_final/benchmark_local/` 与 `benchmark_wan/`。
每个测点有官方客户端逐请求数据、GPU/DCGM 采样、运行配置和 `split_trace.jsonl`。
`max_slo_offered_qps` 明确为 null；尚未执行每点 1000 请求的正式 SLO 容量搜索。
当前简单调度器没有 prefill/decode 重叠和流水线优化，高负载下的 TTFT 排队是后续性能工作的重点。

## 故障与恢复

真实权重服务空闲时，仅停止已核对 PID 身份的本项目 Cloud 进程组。
等待 1 秒后发送请求，约 56.6 ms 返回 HTTP 503，健康检查也返回 503；没有生成回答或本地完整模型 fallback。
原始记录：`results/validation_final/fault_check.json`。随后 `./poc up` 成功恢复服务并通过真实聊天测试。

该测试会中断演示，复现时先确保无人使用：

```bash
.venv/bin/python scripts/fault_check.py --stop-owned-cloud
./poc up --wan
```

## GPU 时间线证据

独立真实模型 4096/32 采样（与压测分开）：`results/20260905-122434/`。
`enterprise.nsys-rep` 与 `cloud.nsys-rep` 均正常导出；`profile_request.json` 保存真实响应。
对应 `*_cuda_gpu_kern_sum.csv` 中，Enterprise 有 1216 次 NCCL AllReduce，Cloud 有 3456 次；
两侧各 128 次 NCCL Broadcast，并存在真实 FlashAttention/GEMM 核。
这些计数是两张 GPU 聚合的 kernel instance 数，不是跨企业—云的 NCCL 通信次数。
两侧只通过 HTTP 传输激活，NCCL 各自在本地 TP 组内执行。

采样期间 Nsight 注入的 NCCL 库版本可能与普通运行不同；该目录不能作为正式性能结果。
首次关闭被停服打断的记录另存于 `results/20260905-122221/`，不作为最终操作流程验收。

```bash
./poc up --profile --wan
.venv/bin/python scripts/profile_request.py  # 等待脚本退出后再停服
./poc down
./poc up --wan
```

## 可复现验收

```bash
.venv/bin/python scripts/validate_poc.py
```

该命令先停止现有演示，检查模型哈希，再执行原生 vLLM 基准、三种切分的精度及功能检查、WAN 检查和短压测。
只有全部通过才写入 PASS，并保留 WAN 演示服务运行。任一阶段失败会停止流水线并保留日志。

正式性能实验应独立运行至少 1000 请求/测点；默认验收流水线每点 20 请求，仅为链路冒烟测试。

## 实现边界

- Split 使用自定义调度器和执行器，模型层、注意力、Paged KV 与 TP 运算来自 vLLM 0.10.2；不是未经修改的原生 vLLM Scheduler。
- 仅本服务器 A10 实测，不能外推为 T4/L20 的结果。
- 企业—云是同机独立 network namespaces，WAN 使用专用 veth 的 tc 整形，不是两台物理服务器。
- 支持 FP16、greedy、单候选及最长 16384 token 总上下文；不支持量化、工具调用或非零 temperature。
- 不上传 token ID、文本或 logits 不等同于隐状态具有密码学隐私保障。
- 入口仅绑定 localhost；无认证接口不应直接暴露公网。
