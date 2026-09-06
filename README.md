# Split-Infer 优化/建模 Demo 演示

[Web Demo](demo/README.md)把本项目从安全分析、功能实现、系统优化到性能建模的研究结果放在同一页面中。它不生成假进度或假性能数据：交互操作调用仓库中的真实代码，历史实验则明确读取归档证据。

```bash
python3 demo/server.py --host 127.0.0.1 --port 8088
```

本机访问 <http://127.0.0.1:8088>；远程机器可用 `ssh -L 8088:127.0.0.1:8088 <user>@<server>` 转发端口。UI 和仿真需要 Python 3.11+；启动真实推理还需要 Linux/root、4 × A10、模型权重和已准备好的 vLLM 环境。服务、攻击、仿真和对话在 UI 后端串行执行，以免实验相互竞争。

### 01 背景

大模型已经进入企业核心业务，私有输入也随之进入云端。TLS 保护传输链路，但在服务端终止后，明文和中间状态仍可能暴露给运行时。本项目从安全攻击、功能实现、系统优化、性能建模四个方面分析 Split-Infer 的能力与边界。

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

### 05 真实系统优化

C16 baseline 的 96 请求、4K 输入/79 输出 trace 显示：平均 TTFT 为 6331.412 ms，其中 prefill step 为 734.972 ms、其余等待与交付残差为 5596.440 ms；平均 TPOT 为 106.975 ms，其中 decode step 为 35.966 ms、残差为 71.009 ms。RPC 区间还包含打包和 IPC，不能全部解释为纯 WAN；企业/云区间也不是纯 GPU kernel 时间。可下载[原始 trace](demo/evidence/baseline-c16-raw.zip)，或运行 `python3 scripts/check_demo_evidence.py` 独立复算。

当前推荐配置依次处理数据路径开销、长 prefill 阻塞、流水气泡以及 P/D 资源耦合：

| 优化 | 已实现的行为 | 当前配置 |
|---|---|---|
| Stage1 数据路径 | SHM IPC、wire fast、16 MiB socket buffer；仍有 D2H/H2D | 开启 |
| Chunked prefill / decode-first | 2048-token chunk，decode quota=4；不是 mixed P/D batch | 开启 |
| 异步流水 | 多请求计算和传输在途，D window=2 | 开启 |
| PD 与双 P 副本 | Enterprise TP1 + 2 × P TP1 + D TP1，P window=3 | 开启 |
| 独立控制通道 | readiness 使用独立 worker pipe | 开启 |
| Chunk KV 提前迁移 | 能力已实现，但实测未证明整体更优 | 关闭，可选 |

真实四卡 closed-loop 扫描采用 4K/79 workload；联合 SLO 要求至少 99% 请求同时满足 TTFT ≤ 3 s、请求平均 TPOT ≤ 100 ms，并同时检查完成与到达 cohort。最高已测合格点是 **C44 / 5.365 QPS**；C46 和 C48 不合格，主要触及 TPOT 限制。由于未测 C45，也未做长期生产压力验证，这不是精确的全局最大容量。

![优化后 TTFT-QPS 曲线](demo/evidence/optimized-ttft-qps.png)

![优化后 TPOT-QPS 曲线](demo/evidence/optimized-tpot-qps.png)

[CSV](demo/evidence/optimized-sweep.csv) · [原始请求、trace 和环境](demo/evidence/optimized-sweep-raw.zip) · [完整验证报告](docs/live-demo-validation.md)

复现实测扫描：

```bash
./poc up --preset optimized --wan
.venv/bin/python scripts/demo_sweep.py --output results/my-demo-sweep
python3 scripts/package_demo_evidence.py --sweep results/my-demo-sweep \
  --serving-results "$(cat run/current_results)" --output results/my-demo-evidence
```

### 06 推理性能建模

[Q4 仿真器](Q4/README.md)使用 DAG + event-driven 执行，支持 baseline behavioral scheduler 和固定版本的真实 PD scheduler 源码。当前 Demo **只开放** Qwen2.5-3B、A10 × 4、Split 4/27/5、4096 输入/79 输出：optimized 使用 C40 empirical command/host profile，baseline 使用 operator/host cost。每个并发点都会实际执行，不读取结果缓存；除校准点外均是预测值。

页面“运行仿真”会对输入的并发点逐一调用 [demo/simulate.py](demo/simulate.py)。单点 CLI 复现如下：

```bash
printf '%s\n' '{"variant":"optimized","concurrency":40}' > /tmp/split-sim-input.json
python3 demo/simulate.py /tmp/split-sim-input.json /tmp/split-sim-output.json
cat /tmp/split-sim-output.json
```

归档的 C40 seed17 结果为 5.233 QPS、平均 TTFT 723.99 ms、平均 TPOT 87.47 ms；这是仿真预测，不是 GPU 实测。算子覆盖、系统偏差、外推边界和未实现能力见[校准与局限](Q4/docs/CALIBRATION_AND_LIMITS.md)、[Q4 验证](Q4/docs/VALIDATION.md)和[C40 回放结果](demo/evidence/simulation-c40-validation.json)。完整配置、trace、逐请求时间线和可交互甘特图的运行方法见 [Q4 README](Q4/README.md)。

### 页面交互与代码入口

| 页面交互 | 等价命令或实现 |
|---|---|
| 生成激活 / 恢复 Token | 上述 `Q1/demo_attack.py --phase encode/recover`；[实现](Q1/demo_attack.py) |
| 启动 Demo | `./poc up --wan`；停止使用 `./poc down` |
| 发送对话 | 上述 `/v1/chat/completions` curl；[代理实现](demo/server.py) |
| 运行并发仿真 | 上述 `demo/simulate.py`，或按 [Q4 CLI](Q4/README.md)输出完整 trace/Gantt |
| 下载实验结果 | [Demo 证据目录](demo/evidence)和[验收报告](docs/live-demo-validation.md) |

前端实际使用的异步接口、任务产物位置和浏览器端到端验证方法统一记录在 [demo/README.md](demo/README.md)。
