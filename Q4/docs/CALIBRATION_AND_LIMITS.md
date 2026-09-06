# 校准、硬编码与未解决问题

本文件是当前交付的适用范围清单。**可迁移的是机制和接口，不是当前数值表。** 已验证的主要目标是 Qwen2.5-3B-Instruct / FP16 / 4-27-5 层 / A10 / 4K 输入、79 输出。解析示例与 demo 的 Qwen3-32B 不等于已经经过实测校准。

## 1. 已校准或带实验依据的成本

| 项目 | 当前值/来源 | 使用方式与迁移要求 |
|---|---|---|
| 算子 profile | `issue5_a10_tp2_baseline.json`、`issue5_a10_tp2_stage123.json`，Issue #5 算子记录 | 精确签名优先，缺失 shape 同类型修正/解析回退；改模型、融合方式、kernel、dtype、TP 或硬件需重新采集 |
| 原 PR8 profile | `pr8_a10.json` | 历史校准，不能用于当前最优 PD 的精度声明 |
| command profile | `issue6_pd_tp1_c24/c40_commands.json` 和 `c40_empirical_commands.json` | 来自完整调用 wall time，不是 kernel-only；phase/TP/batch/query/logits/context 限定覆盖范围，换环境需重新标定 |
| 耗时分布 | empirical 表保留逐命令样本；seed 17/29/43 用于当前验证 | 独立抽样；未保留跨阶段/时间相关性。稀疏 B33 只有一个样本，插值/外推和借用邻近 batch 波动并非泛化保证 |
| baseline CPU submission | 默认 6 μs/子操作，MM 7.5 μs、attention 18 μs | 来自 launch 典型值和融合子操作内 kernel 数量近似；不是每个算子的严格独立 CPU 标定，也不是通用硬件常数 |
| 企业 CPU post-back | `issue6_c40_serving_host.json`；新 C40 短测 decode 平均 3.636 ms/batch，其中 trace 3.282 ms | 模型返回后到 complete_back 函数体结束；含 GIL/I/O/线程等待。按请求行分摊只是内部时序近似；换 CPU、日志、软件或 I/O 需重新测量 |
| WAN profile | `issue5_wan_calibration.json`，baseline/stage1，C1、TP2+2、10 Gbps、单向 5 ms，8 KiB–128 MiB | 按 payload 分段插值，范围外解析回退；不能独立识别高并发 HTTP/CPU 排队，也不能直接跨网络/主机搬运路径复用 |
| reserve 成本 | P 0.24 ms、D 0.11 ms；C1 executor 获锁后调用近似 | 含 IPC，队列等待另由事件产生。`local_rpc_ms=1.1` 是同次采集的近似拆分，并非普适常数 |

command 模式不重复叠加 host submission，并移除已包含的 D2H/H2D/cloud IPC 部分。post-back 成本在 command 结束之后。禁止把 TTFT 总缺口再作为一次 command overhead 塞进去。

`command_scope` 检查模型名、dtype、分层、transport 和部分硬件规格，sample 检查 TP/shape/context。它**没有完整校验 CPU、驱动、vLLM、Torch、频率等环境指纹**，校验通过不代表可跨环境复用。host profile 也存在同样限制。

## 2. 暂定成本与解析默认值：不是测量结论

| 参数/机制 | 当前值或假设 | 风险与修改位置 |
|---|---|---|
| PD KV 传输 | 默认 100 GB/s + 0.1 ms | `PDDisaggregationConfig` 中的解析占位值，不能称为目标硬件实测 NCCL 带宽；应按拓扑校准 |
| KV control latency | 默认 0 ms | 控制操作仍有依赖，但零值不意味着真实控制免费 |
| release 成本 | 企业/P/D 各 0.10 ms | 暂定值，未独立校准；CPU post-back 打点不包含后续 release |
| 控制 HTTP | RTT 来自 network；新连接额外一个 RTT，keepalive 1 秒；local RPC 分为两个半程 | 简化连接模型，无完整 TCP 状态、状态查询、重试和线程池争用 |
| 解析搬运 | D2H/H2D 24 GB/s、host memory 20 GB/s、IPC 12 GB/s + 0.05 ms 等默认值 | 见 `DataPathConfig`，只是默认假设；复制次数、有效带宽随路径与硬件改变 |
| Roofline 硬件参数 | A10 示例 125 TFLOPS、600 GB/s；baseline efficiency 0.45/0.65，launch 10 μs | 峰值与经验 efficiency 不是所有模型的实测效率；interconnect 24 GB/s、8 μs 也需校准 |
| WAN 解析参数 | 示例 RTT 10 ms、10 Gbps、efficiency 0.8、2 个激活张量、384 bytes 协议开销 | 可配置；协议、实际带宽和序列化流程变化时重标定 |

这些值暴露在 config 或源码默认值中，并未通过拟合端到端指标证明正确。开关关闭某项并不消除其他成本或依赖。

## 3. 硬编码的业务/结构假设

| 位置 | 当前固定假设 | 迁移时的处理 |
|---|---|---|
| `presets.py` optimized | 企业 TP1、max_active 96、KV blocks 32768、禁止 mixed batch；默认双 P TP1+D TP1、chunk2048、P/D window3/2、quota4 | 这是实验预设，会覆盖部分输入 JSON；TP/window/chunk/quota 有 CLI 开关，其他值需修改预设或直接构造目标配置 |
| `vendor/serving/manifest.json` | 调度源码固定到 `34c809c...`，包含真实 1 ms wait/sleep 行为 | serving 改动后需同步快照、哈希与决策测试；当前没有自动跟随 main |
| 虚拟 serving | PP1，1–2 个 P replica，固定长度 closed-loop，warmup=0 或 concurrency；成功路径 | 新 PP 拓扑、异构请求、取消/错误恢复需扩展执行接口，不能仅换 cost JSON |
| KV 与模型占位 | 真实 KVAdmission 默认 16-token block；虚拟 token 为 0，KV 使用计数部分为占位输出 | 更改 block 大小时要同步原逻辑；仿真不执行模型、不验 logits，不模拟每个角色的真实 KV 张量 |
| command 校准工具 | decode context 4096–4175；本轮采集/回归为 4K/79 | 这是工具中的工况限定；换长度要重新构建范围和数据，不可把现表当成长上下文通用表 |
| 模型算子图 | Qwen2/Qwen3 类 dense decoder 及已支持融合/attention 形式 | 新架构需要核对算子图；MoE/MLA 等未实现不能只换模型名 |
| 并发和 stream | 简化 worker 串行 lane、critical-rank CPU 提交、有限 WAN 资源 | 多 stream 重叠、完整多 rank 到达与 cloud thread scheduling 未完整复现 |

公开 Python API 的配置空间比真实源码后端支持范围大。不能把 behavioral 中可运行的 PP/mixed-batch 能力，等同于 serving 后端已验证。

## 4. Demo 的固定数据和未校准部分

完整 demo 保留原有六个页面区域。安全攻击按钮重放已有证据；baseline 启动/对话是**假数据交互**，不调用根目录真实 serving 服务。性能模拟 API 会调用 Q4，不伪造成功曲线。

当前 demo 只支持 Qwen3-32B/A10/9 卡、云 TP4×PP2 + 企业 1 卡。A10 参数硬编码为 125 TFLOPS、600 GB/s、24 GB、efficiency 0.55/0.65；未对该模型实测校准。并发白名单 1/2/4/8、输入 32–2048、输出 2–64；128 请求、5 秒窗、batch/active 上限 8。其他预留模型/硬件返回 `422 not_implemented`。demo 并未接入当前 Qwen2.5-3B PD 校准预设。

因此：演示可运行不等于硬件容量准确；假对话不能作为模型正确性证明。范围说明也保留在 [demo README](../../demo/README.md) 和页面中。

## 5. 当前未解决的问题

1. **云端接口时间边界**：真实 `cloud_send_ns` 后仍持锁执行 `after_forward/pack`；仿真较早开始下载并释放 HTTP 门。condition/worker/HTTP 锁完整交接未复现。
2. **入口等待**：旧 trace 无法完整拆开请求处理、reserve、future 观察、ready front 排队；未计时的 CPU/线程调度仍可能存在。
3. **精度残差**：当前 C40 QPS +3.05%、TTFT −10.64%、TPOT −2.80%；是校准工况对照，不是独立泛化验收。baseline 小并发仍有显著误差。
4. **成本相关性**：empirical 是独立抽样；原子成本包含部分内部等待，仍有重复/遗漏风险。不能用总数接近掩盖分项抵消。
5. **PD 执行模型**：没有完整 NCCL status/重试、线程池容量、各角色独立 KV pool、真实云 PDControl 代码执行；release 和 KV 参数仍有占位值。
6. **覆盖面**：无自动最大 SLO QPS 搜索、MoE/MLA、SP/CP/EP、运行中 replica 重平衡、投机推理；serving 无 mixed/preemption/PP。
7. **输出/资源**：serving 汇总的通用 utilization/average batch 为空；其完整 trace/decisions 暂未按配置上限裁剪，大规模任务仍需控制内存。

## 6. 迁移与重新校准顺序

先确认目标架构和调度协议受支持 → 更新模型/拓扑配置 → 校准算子或 command、CPU 收尾、搬运/网络及 PD 成本 → 检查精确/回退覆盖率 → 使用独立请求/工况验证分项和总指标。

可复用的是队列规则、事件依赖、数据量公式和成本接口。**排队时间应由新成本和资源竞争产生，不应复制当前 28 ms/38 ms 等残差为新硬件常数。** 结构依赖漏建时先修实现，不能只重新 profiling。
