# 开环 TTFT / TPOT 误差定位

2026-09-07。使用已归档实测数据，重放完全相同的 Poisson 到达时间；没有重新拟合 cost model，没有新跑 GPU 压测。配置为 Qwen2.5-3B FP16、4K 输入、79 输出 token、原有 WAN 设置。

数据源：`results/open-loop-validation-repeat-20260907`；预测基准：`results/sim-open-loop-repeat-20260907`。原始数据亦在 `demo/evidence/simulation-validation-raw.zip`。本次重放 trace 在 `results/error-localization-20260907`，分项结果保存在 [components.csv](validation/error_localization/components.csv)。

## 对齐与统计口径

只统计 measurement 窗口内计划到达的请求，TTFT 从 scheduled_start 算起。实测 client_request_id 数字后缀对应仿真 request_id。PD 逐请求核验两个 prefill chunk 的绝对位置 0 / 2048、query_len 2048，以及 78 个 decode position 4096–4173。重放每条请求 TTFT 与原存档预测差异小于 1e-6 ms。

PD TTFT 将两个 chunk 的分段时长相加，减去两个执行区间的重叠，并计入首轮 front 前等待、chunk 间空隙和客户端交付。TPOT 由首个 token 的 back_end 到最后一个 token 的 back_end 分段，除以 78，再校正首尾客户端交付偏差。逐请求分项与客户端指标的重构误差小于 1e-4 ms。ns 字段使用 monotonic 时钟，未混用 time_ns 的墙钟时间。

下表均为请求均值，差值为仿真减实测，单位 ms。

| 系统 / 到达率 / seed | 请求数 | TTFT 实测 → 仿真 | TPOT 实测 → 仿真 |
|---|---:|---:|---:|
| Baseline / 0.75 / 29 | 41 | 1180.06 → 1124.35 | 81.24 → 51.42 |
| PD / 1.00 / 29 | 56 | 633.97 → 513.55 | 36.62 → 47.52 |
| PD / 4.44 / 43 | 575 | 976.55 → 872.43 | 78.11 → 86.98 |

## PD：TTFT 低估主要落在通信区间和首轮等待

| TTFT 分项差值 | λ=1.00 | λ=4.44 |
|---|---:|---:|
| front 前等待（含准入、调度、接收/提交） | +8.92 | -58.73 |
| front 后打包/提交 + 上传 + 下载 | -270.01 | -80.87 |
| cloud queue | +8.89 | +15.98 |
| cloud command | +11.90 | +4.05 |
| enterprise front + back command | -5.21 | -0.61 |
| 返回后等 back | -4.22 | +6.37 |
| 重叠抵扣的差异 | +129.80 | +11.64 |
| 交付及 chunk 间空隙 | -0.49 | -1.95 |
| 总误差 | -120.43 | -104.12 |

低负载通信区间总和实测 363.83 ms、仿真 93.82 ms；不能把 270.01 ms 全算成 TTFT，因为实测还多隐藏了 129.80 ms 的 chunk 重叠。高负载通信总和实测 176.43 ms、仿真 95.56 ms，同时首轮等待实测 366.74 ms、仿真 308.02 ms。

这里的“通信”是时间戳划定的应用端到端区间，包含打包、收包、排队、CPU 调度和可能的拷贝，不是纯链路传输。cloud_send 在 cloud 打包/后处理前打点，部分服务端收尾落在 download 中。不能由这些 trace 直接把误差全部归因于带宽、D2H 或 kernel launch。首轮等待也不能全部叫 PD admission，需要额外边界才能继续分拆。

## PD：TPOT 高估主要落在 enterprise command

| TPOT 分项差值 | λ=1.00 | λ=4.44 |
|---|---:|---:|
| enterprise front | +8.10 | -0.38 |
| enterprise back | +2.58 | +7.54 |
| 上一轮 back 结束到下一轮 front 开始 | +0.24 | +2.44 |
| cloud command | +0.70 | +0.89 |
| 其他合计 | -0.73 | -1.62 |
| 总误差 | +10.90 | +8.86 |

低负载 front + back 多估 10.68 ms，占净误差约 98%；高负载 back 多估 7.54 ms，占净误差约 85%。这些是带 CPU / IPC / staging 的命令耗时，不是纯 GPU 算子耗时。高负载首二 token 的 back_end 间隔实测 90.29 ms、仿真 89.21 ms；首二间隔的净误差很小，整体 TPOT 的偏差持续发生在后续 decode。首二间隔各分项仍有正负抵消。

### 同 batch 比较发现成本表缺口

按 batch_id 去重，统计对应整次运行（含 warmup/tail）的 decode command；这部分是 batch 加权，区别于上面的 arrival-cohort 请求加权。

| 场景 / batch / command | 实测样本数 | 仿真样本数 | 实测 ms | 仿真 ms |
|---|---:|---:|---:|---:|
| λ=1 / B2 / front | 795 | 472 | 2.73 | 11.37 |
| λ=4.44 / B24 / back | 109 | 57 | 12.18 | 22.98 |
| λ=4.44 / B32 / back | 98 | 121 | 14.42 | 29.30 |

`issue6_pd_tp1_c40_empirical_commands.json` 的 enterprise decode 仅有 B1、B2、B33–38，B3–32 全靠插值。B2 front 11.742 ms 仅 7 个样本；B33 back 30.094 ms 仅 1 个样本，却成为整个 B2–33 区间的插值端点。上述同 batch 比较表明，偏差不能只解释为 scheduler 改变了 batch 分布；稀疏且不代表当前运行的成本样本是一个明确问题。但这是观测定位，尚未通过替换 profile 的控制实验量化其全部因果影响。

## Baseline：TPOT 的服务时间和等待都偏低

| TPOT 分项 | 实测 | 仿真 | 差值 |
|---|---:|---:|---:|
| enterprise 本地 | 8.61 | 3.22 | -5.39 |
| cloud | 15.37 | 7.76 | -7.62 |
| 上传 + 下载 | 11.05 | 11.80 | +0.75 |
| 等待 | 45.97 | 28.64 | -17.33 |
| 未分解余项 | 0.22 | 0.00 | -0.22 |
| 总计 | 81.24 | 51.42 | -29.82 |

实测 baseline 没有 front/back 的独立时间戳：本地取 step_wall - rpc_wall；cloud 为 cloud_received → cloud_send；等待为 trace.queue_ms。仿真本地为 front + tail，等待为客户端总时长减 stage duration。两者存在 CPU/staging 归属差异，因此本地、cloud、通信之间的细分不是完全相同的实现边界。合并服务区间仍低估约 12.26 ms，等待低估 17.33 ms；不能把后者当成独立的 scheduler bug，服务过快本身就会减少排队。

Baseline TTFT 总误差仅 -55.71 ms，却包含本地 -54.49、cloud -149.97、通信 +180.81、等待 -28.94、余项 -3.12 ms 的抵消。这部分尤其受 staging 归属影响，不宜因 TTFT 总误差较小就宣称每个 cost model 都准确。

## 下一步

优先补企业端 decode 的典型 batch profile（B2/8/16/24/32），避免单样本端点插值；保留独立 seed 作为验证集。其次在 prefill pack/收包/拷贝/云端收尾处补边界，以确定通信区间缺口归属；最后在服务成本修正后再分析剩余准入/调度等待。当前结论定位到均值，不代表 P99 尾部已完成因果归因，也未修改现有仿真精度展示或重新校准成本。
