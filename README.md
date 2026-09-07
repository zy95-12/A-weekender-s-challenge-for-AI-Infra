# Split-Infer 优化/建模 Demo 演示

[Web Demo](demo/README.md)把本项目从安全分析、功能实现、系统优化到性能建模的研究结果放在同一页面中。它不生成假进度或假性能数据：交互操作调用仓库中的真实代码，历史实验则明确读取归档证据。

```bash
python3 demo/server.py --host 127.0.0.1 --port 8088
```

本机访问 <http://127.0.0.1:8088>；远程机器可用 `ssh -L 8088:127.0.0.1:8088 <user>@<server>` 转发端口。UI 和仿真需要 Python 3.11+；启动真实推理还需要 Linux/root、4 × A10、模型权重和已准备好的 vLLM 环境。服务、攻击、仿真和对话在 UI 后端串行执行，以免实验相互竞争。

### 摘要

大模型的安全推理是企业上云关注的关键问题之一。本项目围绕大模型安全推理的 SplitInfer 方案，开展了安全性分析、功能实现、系统优化和性能建模。实验表明，加深企业端部署层数可以降低 token 被明文恢复的成功率；在特定 SLO（4K 上下文输入，TTFT ≤ 3s、TPOT ≤ 100ms）下，按本次最高已测合格点比较，优化系统吞吐达到 baseline 的约 **15.1 倍（0.317 QPS → 4.775 QPS）**。项目同时实现了面向多种硬件配置与模型的理论性能建模框架；在已校准的 Qwen2.5-3B / A10 配置下，19 组开环实验中 TTFT、TPOT 均值的平均绝对相对误差分别为 **12.9% 和 23.4%**，为 SplitInfer 方案的落地提供参考。

恢复率结论针对本次数据与攻击方法，不代表绝对安全。15.1 倍：采样具有随机性，这是最高已测合格点之比。上述建模误差仅适用于已校准配置，其他模型和硬件的预测精度尚未实测验证。

### 01 背景

**大模型已进入企业核心业务，数据安全上云成为基础设施的关键问题。** 私有输入随模型使用进入云端。TLS 保护传输链路，但在服务端终止后，明文和中间状态仍可能暴露给运行时。本项目从安全攻击、功能实现、系统优化、性能建模四个方面分析 Split-Infer 的能力与边界。

### 02 大模型安全推理的业界洞察

当前判断是：**TEE / Confidential Computing 应当成为 AI-Factory 的投入重点，Split-Infer 是端云协同的核心方向，FHE / MPC 保持长期关注。** 星级是指定威胁模型下的定性比较，不是安全认证或实测分数；“性能影响”星数越多代表额外成本越高。

| 技术方向 | 主要价值 | 安全性 | 推理性能负面影响 |
|---|---|---:|---:|
| TEE / Confidential Computing | 用硬件隔离、远程证明和按证明释放密钥建立运行时可信边界 | ★★★★☆ | ★★☆☆☆ |
| 数据最小化 | 本地删除、脱敏、假名化和筛选，只上传完成任务所需的信息 | ★★☆☆☆ | ★☆☆☆☆ |
| Split-Infer | 企业和云端分担模型层，联合设计切分、激活传输、KV 与调度 | ★★★☆☆ | ★★★☆☆ |
| FHE | 云端直接计算密文，适合先评估范围明确的高敏子任务 | ★★★★★ | ★★★★★ |
| MPC | 通过秘密共享等协议限制任何单方取得完整数据 | ★★★★★ | ★★★★☆–★★★★★ |

完整威胁模型、技术边界和投入建议见[大模型安全推理技术洞察](docs/01_tech_insight.md)。

### 03 Split-Infer 方案与安全性

[题目原始方案（Issue #1）](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/issues/1)将模型拆成企业侧 Front/Back 与云侧 Middle：企业侧将 token 转为 hidden states，通过 WAN 交给 Infra 小组运行中间层，再取回激活完成尾层和 logits。这样可以减少直接上传 token 的暴露，但 hidden state 本身并不安全。

PR #15 的攻击表明，embedding 边界的 FP16 hidden state 可以用公开 embedding 表做全词表 FP32 余弦最近邻检索，几乎完整恢复 token。Demo 的“生成激活”和“恢复原文”按钮运行的是 [Q1/demo_attack.py](Q1/demo_attack.py)；它是暴露与反演实验，不是密码学加密/解密，也不能据此声称能够直接反演经过 4 层计算的 serving 激活。

按钮对应的 CLI 如下。首次运行先执行 `./poc setup` 准备与 serving 相同版本的模型：

```bash
mkdir -p results/demo-attack
printf '%s\n' '{"text":"Security inference should protect private prompts."}' \
  > results/demo-attack/input.json
.venv/bin/python Q1/demo_attack.py --model models/qwen \
  --output results/demo-attack --phase encode
.venv/bin/python Q1/demo_attack.py --model models/qwen \
  --output results/demo-attack --phase recover
```

`hidden.npy` 是实际生成的 FP16 矩阵，`encode.json` 记录 shape、dtype、摘要和耗时，`recover.json` 记录逐位置恢复结果。简单 MLP 的归档实验同时显示，增加企业侧前层深度可以降低当前攻击下的 token 恢复率，但结果与数据分布、攻击器和训练预算强相关，不能作为绝对安全证明。后续性能实验把 **Split 4 / 27 / 5** 作为研究起点；定向噪声可能进一步提升隐私，但本项目尚未实现或测量该防护。数据见 [security-depth.json](demo/evidence/security-depth.json)，攻击实现和局限见 [Demo 说明](demo/README.md)。

企业端深度与 MLP token 恢复率（同 demo 的归档数据）：

| 企业前层数 | 4K 样本恢复率 | 独立公开测试恢复率 |
|---|---:|---:|
| 4 | 53.98% | 82.71% |
| 9 | 40.41% | 79.96% |
| 18 | 23.83% | 76.03% |
| 27 | 20.79% | 75.43% |
| 36 | 9.59% | 60.33% |

### 04 功能实现

Baseline 使用 Qwen2.5-3B-Instruct FP16 权重：企业侧运行首尾层，云侧运行中间层，并使用 vLLM partial runner、FlashAttention、Paged KV 和独立 TP 通信组。默认切分为 4/27/5，企业与云各 TP=2，部署在 4 × A10 上，对外提供 OpenAI-like API。

页面“启动 Demo”按钮等价于：

```bash
./poc up --wan
```

页面对话框把请求转发到真实 SSE；命令行可直接调用：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-3B-Instruct","messages":[{"role":"user","content":"为什么安全推理很重要？"}],"temperature":0,"max_tokens":128,"stream":true}'
```

浏览器显示的 TTFT/TPOT 按收到的 token 事件计时，因此包含网络、代理和浏览器开销。精度验证固定 8 道 GSM8K 输入，对齐 token 和绝对位置，以 teacher forcing 比较原生单卡完整模型与 Split TP2+2 的 logits cosine、top1 一致率及 top-k overlap；它不比较自由生成答案是否完全一致。详见[原生 TP1 精度报告](docs/gsm8k-native-tp1.md)和[完整 serving 实现](docs/integrated-serving.md)。

原生完整模型 TP1 与 Split 4/27/5、TP2+2 的精度实测：

| 阶段 | 位置数 | Logits cosine 均值 / 最低 | Top1 一致 | Top-5 overlap 均值 / 最低 | Top-10 overlap 均值 / 最低 | Top-20 overlap 均值 / 最低 |
|---|---:|---:|---:|---:|---:|---:|
| Prefill | 8 | 0.99999665 / 0.99999465 | 8/8 | 100.00% / 100.00% | 100.00% / 100.00% | 100.00% / 100.00% |
| Decode | 56 | 0.99999659 / 0.99999193 | 56/56 | 100.00% / 100.00% | 99.29% / 90.00% | 99.64% / 95.00% |

### 05 真实系统优化

**C16 Baseline 实测耗时**：4K 输入 / 79 输出、TP2+2，96条完成请求、7488个 decode 步骤。下表为均值，来自同一批请求的 trace。

| 实测区间 | TTFT 拆解（ms） | TPOT 拆解（ms） |
|---|---:|---:|
| 上行 RPC 区间 | 115.63 | 5.52 |
| 下行 RPC 区间 | 154.76 | 5.68 |
| 云侧处理区间 | 341.75 | 14.69 |
| 企业侧处理及本步其余开销 | 122.83 | 10.08 |
| 本步以外等待及交付残差 | 5596.44 | 71.01 |
| 端到端合计 | 6331.41 | 106.98 |

这张表来自历史 C16 闭环测量，仅用于分析耗时来源；下方吞吐曲线统一使用开环实测。RPC 区间包含主机处理，云侧区间不等于纯 GPU kernel 时间；步外残差包含调度等待、其他请求执行和交付开销，不能全部视为纯排队。[原始请求与 trace](demo/evidence/baseline-c16-raw.zip)可运行 `python3 scripts/check_demo_evidence.py` 独立复算。

**分析结论：**

1. **通信开销与云侧处理同量级，需要优化通信路径。** SplitInfer 引入了显著的端云通信开销，往返 RPC 耗时与云侧处理时间相当，远高于仅按“通信量 / 带宽”估算的时间。差额包含固定链路时延，以及序列化、内存拷贝、IPC 等额外成本；应结合 profiling 削减可避免的操作，不能将差额全部视为冗余传输。
2. **通信瓶颈持续存在，还需要用流水隐藏等待。** 减少通信路径中的额外开销后，端云传输与固定链路时延仍然存在。需要让不同请求的计算、传输和 KV 迁移尽可能重叠，减少通信阻塞计算的时间。
3. **大并发下，请求等待是主要开销，需要端云协同调度。** C16 下，请求在计算步骤之外的等待远大于单步执行时间。应协同安排企业前后层、云侧 prefill/decode 与 KV 就绪时机，减少排队和相互阻塞，才能将局部加速转化为 SLO 下的吞吐提升。

**已实现的优化与推荐配置：**

| 优化 | 已实现的行为 | 当前配置 |
|---|---|---|
| 通信路径优化：SHM / wire fast / TCP buffer | 共享内存 IPC、减少打包开销、16 MiB socket buffer；仍有 D2H/H2D，不是跨 WAN 零拷贝 | 开启 |
| 分块预填充与解码优先调度 | 2048-token chunk，decode quota=4；不是 mixed P/D batch | 开启 |
| 计算与通信异步流水 | 多请求计算和传输在途，D window=2；不保证任意并发都有收益 | 开启 |
| PD 与双 P 副本 | 企业 TP1 + 2 × P TP1 + D TP1，P window=3 | 开启 |
| 独立控制通道 | readiness 使用独立 worker pipe；reserve/release 仍有普通 executor 路径 | 开启 |
| Chunk KV 提前迁移 | 能力已实现，但实测未证明整体更优；推荐整段 prompt 迁移 | 关闭，可选 |

连续 batching 已有；混合 prefill/decode batch 与动态 P/mix/D 角色切换未实现，不计入收益；投机推理未包含。

```bash
./poc up --preset optimized --wan
./poc up --preset baseline --wan
```

**Baseline 与优化系统的开环实测：**

同一模型、四张 A10、4K 输入 / 79 输出，WAN 每方向10Gbps、单向5ms。Baseline 为 TP2+2，优化系统为企业 TP1 + 两个 TP1 P + TP1 D；两者统一 max_active=96、kv_blocks=32768。同一目标到达率使用相同泊松请求计划，客户端不限制并发。

横轴为实测完成 QPS，不是设定到达率。延迟按测量窗口内到达请求统计，TTFT 包含客户端发包延误，TPOT 是每条请求的平均 token 间隔。合格要求到达、完成两个 cohort 均有至少99%请求同时满足 TTFT≤3s、TPOT≤100ms，错误计失败。达标 QPS 只计算满足联合 SLO 的完成请求。

| 系统 | 目标 / 实际到达率（请求/s） | 完成 / 达标 QPS | TTFT 均值/P99 ms | TPOT 均值/P99 ms | 到达 / 完成 SLO % | 平均 / 峰值在途 | 到达请求数 | 测量 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.500 / 0.300 | 0.317 / 0.317 | 886.0 / 1508.9 | 44.13 / 62.00 | 100.00 / 100.00 | 1.4 / 4 | 18 | 60 |
| Baseline | 0.750 / 0.550 | 0.533 / 0.533 | 978.1 / 2015.7 | 56.08 / 114.84 | 96.97 / 100.00 | 2.8 / 7 | 33 | 60 |
| Baseline | 1.000 / 0.883 | 0.717 / 0.450 | 1179.2 / 2317.7 | 120.63 / 269.06 | 52.83 / 62.79 | 9.1 / 26 | 53 | 60 |
| 优化系统 | 0.500 / 0.300 | 0.317 / 0.317 | 667.2 / 713.0 | 32.39 / 34.58 | 100.00 / 100.00 | 1.0 / 4 | 18 | 60 |
| 优化系统 | 1.000 / 0.883 | 0.883 / 0.883 | 633.8 / 735.4 | 35.24 / 38.90 | 100.00 / 100.00 | 3.1 / 7 | 53 | 60 |
| 优化系统 | 1.500 / 1.533 | 1.433 / 1.433 | 616.2 / 821.8 | 40.01 / 48.29 | 100.00 / 100.00 | 5.4 / 13 | 92 | 60 |
| 优化系统 | 2.000 / 1.983 | 2.033 / 2.033 | 614.7 / 1146.2 | 42.48 / 54.81 | 100.00 / 100.00 | 7.9 / 17 | 119 | 60 |
| 优化系统 | 3.000 / 3.067 | 3.167 / 3.167 | 661.5 / 1147.5 | 51.23 / 61.90 | 100.00 / 100.00 | 14.4 / 25 | 184 | 60 |
| 优化系统 | 4.136 / 4.433 | 4.425 / 4.425 | 885.2 / 2008.2 | 69.40 / 89.74 | 100.00 / 100.00 | 28.1 / 47 | 532 | 120 |
| 优化系统 | 4.440 / 4.700 | 4.775 / 4.775 | 998.9 / 2307.0 | 76.90 / 95.84 | 100.00 / 100.00 | 33.1 / 51 | 564 | 120 |
| 优化系统 | 4.744 / 5.042 | 5.042 / 4.508 | 1121.8 / 2756.8 | 86.44 / 106.30 | 89.42 / 89.42 | 39.6 / 58 | 605 | 120 |

优化系统本次最高已测合格点完成 **4.775 QPS**，目标到达率4.440/s、实际4.700/s；更高目标4.744/s的点联合达标率为89.42%，主要触及TPOT限制。相比 baseline 最高已测合格点0.317 QPS，约为15.1倍；baseline 该点仅18条到达请求，不能据此推断精确容量提升倍数。

每点预热30秒、粗扫测量60秒（高负载点120秒）、继续到达30秒后排空；每点只有一个 seed=17 的随机轨迹。实际到达率会偏离目标值；短窗口结果不表示长期生产容量或精确最大QPS。

![开环 TTFT–QPS：Baseline 与优化系统](demo/evidence/open-loop-ttft-qps.png)

![开环 TPOT–QPS：Baseline 与优化系统](demo/evidence/open-loop-tpot-qps.png)

[CSV](demo/evidence/open-loop-sweep.csv) · [原始请求、计划、trace和环境](demo/evidence/open-loop-raw.zip) · [开环实测报告](demo/evidence/open-loop-report.md) · [完整验证报告](docs/live-demo-validation.md)

#### 优化分析结论与下一步

当前瓶颈是资源分工与端云流水，尚不能认定 GPU 计算已经整体打满。在最高已测合格点 4.775 QPS 下，企业 E、云 P0/P1、云 D 的计算命令覆盖率分别为 75.5%、78.2% / 80.6%、26.6%。继续加压至完成 5.042 QPS 时，TPOT P99 从 95.84 ms 升至 106.30 ms，D 的命令覆盖率仍只有 25.7%，说明 SLO 在 D 持续计算打满前就已触顶。

按 QPS容量 ≈ minᵢ(QPS实测 / 命令覆盖率ᵢ) 折算，在 batch 效率、路由比例和每请求服务成本不变的假设下，当前拓扑的命令服务容量约为 5.9～6.0 QPS；4.775 QPS 约为该估算的 80%。这是局部容量估算，不是已验证的 SLO 上限；随机到达产生的排队会使延迟提前超标。

下一步一：调整节点分工，提高容量上限。让 D 承接部分 prefill，以利用 P/D 负载差异；采用 decode 优先、prefill 分块和 token 预算限制，避免新任务阻塞 decode。P 压力缓解后，企业 E 可能成为下一处瓶颈，其按当前成本折算的容量约为 6.32 QPS。

下一步二：优化企业端 CPU、数据搬运和提交路径。优先减少正常推理中的完整 logits 回传 CPU、重复元数据构造，以及响应到计算提交的交接开销。这既可能缩短每轮关键路径、使 SLO 吞吐更接近容量，也可能降低服务成本、进一步提高容量上限。两项优化应分别验证，再测叠加效果，沿用相同开环到达序列对比 TTFT / TPOT 尾延迟和最高合格 QPS。

计算命令覆盖率包含命令内 CPU / IPC / 拷贝，已扣除云侧命令排队，不等于纯 GPU kernel 利用率。企业端两段等待合计 21.91 ms/token，混合了排队、CPU 处理和依赖等待，不能全部算作可回收收益。上述下一步方案尚未实现或验证，不计入当前性能提升。

[开环节点服务时间数据](demo/evidence/open-loop-service-check.json)

复现开环扫描（独占四卡服务，结束后恢复默认 baseline WAN）：

```bash
.venv/bin/python scripts/open_loop_sweep.py --out results/my-open-loop
.venv/bin/python scripts/package_open_loop.py --source results/my-open-loop
python3 scripts/plot_open_loop.py
```

绘图需 matplotlib；启动命令和开关详见 [serving 文档](docs/integrated-serving.md)。

### 06 推理性能建模

[Q4 仿真器](Q4/README.md)从模型结构构建计算/通信 DAG，使用 **Roofline + profiling 修正**估算成本，通过 **event-driven** 引擎产生资源竞争、排队和流水重叠。WAN 使用实测曲线修正拷贝、序列化、IPC 和传输成本；优化 PD 直接复用真实 Scheduler 源码，baseline 使用行为调度模型。

Demo 第六部分提供开环到达率输入和硬件/模型下拉项，每个点实际执行仿真、不缓存结果：

| 选项 | 支持范围 |
|---|---|
| 硬件 | A10、L20 48GB、H20 96GB、Ascend910B 64GB |
| 模型 | Qwen2.5-3B、Qwen3-32B、DeepSeek V4-Flash |
| 成本 | A10/3B使用原operator或command/host/WAN校准；其他组合仅公开参数Roofline，关闭A10校准表 |
| 部署 | 企业首4/尾5层，自动按权重+96请求KV容量选择TP，显示实际总卡数；不是固定4卡比较 |
| 负载 | 4K输入/79输出，独立泊松到达，预热30s、测量60/120s、继续发包30s后排空 |

V4-Flash包含路由/共享专家MoE、低秩投影、SWA/CSA/HCA压缩注意力和mHC；当前按 **BF16反量化后的架构场景**建模，不声称FP4/FP8加速、EP或实际推理服务支持。硬件公开参数、HF固定版本和局限见[理论模式说明](Q4/docs/PUBLIC_ROOFLINE.md)。新设备/模型的预测不纳入实测精度表。

<!-- SIMULATION_ACCURACY_START -->
Qwen2.5-3B / 4×A10 / 4K输入、79输出；19组开环实测对照（原11组 seed17 + 新8组 seed29/43）。精确回放实测到达计划，成本抽样固定seed17；未使用这些结果重新拟合成本。

经过算子profiling、command/host及WAN等修正后的误差，单位为百分比；MAPE为实验点等权的平均绝对相对误差，偏差为有符号均值。

| 系统 | 组数 | QPS MAPE / 偏差 | TTFT均值 MAPE / 偏差 | TPOT均值 MAPE / 偏差 | TTFT P99 MAPE | TPOT P99 MAPE |
|---|---:|---:|---:|---:|---:|---:|
| Baseline · 算子/CPU/WAN修正 | 5 | 4.5 / +0.02 | 4.3 / -4.26 | 34.5 / -34.48 | 4.4 | 25.3 |
| 优化 PD · command/host profiling修正 | 14 | 0.7 / +0.00 | 16.0 / -15.96 | 19.4 / +19.39 | 14.6 | 14.6 |
| 全部实验（点间等权） | 19 | 1.7 / +0.01 | 12.9 / -12.88 | 23.4 / +5.21 | 11.9 | 17.4 |

① 计算与数据搬运的成本模型存在误差。Roofline 和带宽估计难以完整描述实际执行时间，因此如没有可靠的算子性能校准，则仿真结果只能做定性分析；即使经过校准，遇到未覆盖的 shape、batch 或不同执行状态，插值、外推仍可能产生偏差。本次优化系统在 4.44 QPS 下，TPOT 高估 8.86 ms/token，其中企业端尾层命令高估 7.54 ms/token，已发现稀疏采样和单样本插值端点的问题。

② Serving 的通信与等待关系需要更细粒度建模。同一场景下，TTFT 低估 104.12 ms，主要落在 prefill 通信区间和首次计算前等待。通信区间包含 CPU、收发包、拷贝等，不能只用通信量／带宽描述，同时需要正确模拟这些工作的重叠与资源竞争。等待应由调度、依赖和资源占用推导出来，而不是简单补一个固定 overhead。

[逐点CSV](demo/evidence/simulation-accuracy.csv) · [完整统计JSON](demo/evidence/simulation-accuracy.json) · [补测原始记录与预测](demo/evidence/simulation-validation-raw.zip)
<!-- SIMULATION_ACCURACY_END -->

详细实测分项、时间线对齐方法与成本表覆盖问题见[TTFT / TPOT 误差定位报告](Q4/docs/ERROR_LOCALIZATION.md)。

页面“运行仿真”调用 [demo/simulate.py](demo/simulate.py)。单点复现：

```bash
printf '%s\n' '{"variant":"optimized","workload_mode":"open_loop","arrival_rate_qps":0.5,"model":"qwen3-32b","hardware":"h20"}' > /tmp/split-sim-input.json
python3 demo/simulate.py /tmp/split-sim-input.json /tmp/split-sim-output.json
cat /tmp/split-sim-output.json
```

误差来源与校准范围见[校准与局限](Q4/docs/CALIBRATION_AND_LIMITS.md)、[开环验证](Q4/docs/OPEN_LOOP_VALIDATION.md)。可迁移的是结构、事件依赖和调度；kernel、CPU/锁、WAN搬运、互联有效带宽仍需按环境重新校准。

### 页面交互与代码入口

| 页面交互 | 等价命令或实现 |
|---|---|
| 生成激活 / 恢复 Token | 上述 `Q1/demo_attack.py --phase encode/recover`；[实现](Q1/demo_attack.py) |
| 启动 Demo | `./poc up --wan`；停止使用 `./poc down` |
| 发送对话 | 上述 `/v1/chat/completions` curl；[代理实现](demo/server.py) |
| 运行开环仿真 | 上述 `demo/simulate.py`，或按 [Q4 CLI](Q4/README.md)输出完整 trace/Gantt |
| 下载实验结果 | [Demo 证据目录](demo/evidence)和[验收报告](docs/live-demo-validation.md) |

前端实际使用的异步接口、任务产物位置和浏览器端到端验证方法统一记录在 [demo/README.md](demo/README.md)。
