# Split-vLLM 企业—云推理 POC

当前集成入口：**[可回退的 serving、开关清单与 GSM8K 验证](docs/integrated-serving.md)**。

```bash
./poc up --preset optimized --wan   # 当前最佳已验证配置
./poc up --preset baseline --wan    # 全关，回退到 split baseline 执行模式
./poc validate-gsm8k --output results/gsm8k-new
```


本项目对应 [issue #4](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/issues/4)。
Qwen2.5-3B-Instruct 的 Embedding、前后层、Final Norm 与 LM Head 在企业侧；中间层在独立 Cloud 进程执行。
两侧使用 vLLM 0.10.2 的真实模型层、FlashAttention、Paged KV 与独立 NCCL TP group。
调度使用项目内的简单执行器，非原生 vLLM Scheduler；原生 LLMEngine 用于独立精度基准。

## 一键启动

当前服务器：4 × A10，Ubuntu，Python 3.12，Docker/NVIDIA 驱动和 CUDA 工具已安装。
默认 Enterprise 使用 GPU 0,1；Cloud 使用 GPU 2,3；均 TP=2。

```bash
cd /root/A-weekender-s-challenge-for-AI-Infra
./poc up
```

初次启动会准备虚拟环境和固定版本模型；启动完成前会进行真实 Chat 冒烟测试。
需要 root 创建项目专用 network namespaces；需要 `ip`、`tc`、`curl`、`rg`、`python3-venv`。
公网或包镜像下载速度决定首次准备时间。

浏览器访问 `http://127.0.0.1:8000`。远程使用 SSH 转发：

```bash
ssh -L 8000:127.0.0.1:8000 root@<服务器地址>
```

然后在自己电脑打开 `http://127.0.0.1:8000`。

```bash
./poc status
./poc demo
./poc down
./poc up --split 1:1
./poc up --split 3:1 --wan
./poc up --enterprise-tp 1 --cloud-tp 2  # 两侧 TP 可独立配置
./poc wan 5                 # 双向各 5ms + 各 10Gbps
./poc wan --delay-ms 10 --bandwidth-gbps 1
./poc up --wan --delay-ms 5 --bandwidth-gbps 10
./poc network-check --output run/network_check.json
./poc local                 # 恢复专用链路无限制
```

只有项目的 `split-e` / `split-c` 链路接受 tc 配置；不修改管理网卡。
`wan/local` 会核验实际 qdisc 并同步 `run/launch.json` 与 `run/network.json`；
`up` 即使复用服务，也重新应用并核验所请求的网络配置。
`network-check` 需要 ping、iperf3，除命令成功外，还要求零丢包、平均 RTT 与双向配置延迟之和相差不超过 2 ms、
实测 TCP 吞吐在配置带宽的 80%–110% 内。可通过 `--rtt-tolerance-ms`、`--min-bandwidth-ratio` 调整独立检查的容差；
benchmark 使用上述默认值。这是实验条件检查，不是推理 SLO 门槛。
入口只绑定主机 loopback；不要直接将无认证的演示接口暴露公网。

### 测量证据与 warmup

每个新 benchmark 点先执行网络检查，再保存测量前后的 `network.json` / `network_after.json`。
配置不符、读取失败或前置 ping/iperf 验证失败会终止实验；只有 `measurement_validation.json` 为 `PASS` 的新测量点才通过测量检查。
使用新的输出目录，脚本拒绝覆盖已存在的测量点/汇总。通过本项目命令进行的网络修改、网络检查与 benchmark 互斥；
不要在测量中直接运行外部 tc、修改源码或发送无关请求。前后快照不能排除外部程序在中途修改再恢复配置。

`environment.json` 在测量时重新采集，含源码 SHA-256、Git commit 与 dirty 状态；
服务启动时的环境另存为 `launch_environment.json`，两者不可混为一谈。
`config.json` 包含完整命令、实验 ID、网络意图与实测 RTT/吞吐。
`split_trace.jsonl` 只保留正式请求 ID 对应的记录，带 `experiment_id/is_measured/is_warmup`；
其余记录另存 `excluded_trace.jsonl`，标记为 `warmup_or_unattributed`，`is_warmup=null`，
因为不能将其他客户端流量误认成 warmup。`trace_scope.json` 记录过滤计数。
TTFT/TPOT/QPS 仍由官方客户端统计；遥测覆盖整个客户端进程（含 warmup），并非纯正式请求窗口。

历史 `validation_final` 和 `baseline_20260905` 数据保持原样，不补造新证据，也不声称由修复后的脚本生成。

## API

### 实验性数据路径开关（issue #6）

以下开关默认关闭；只改变无损传输实现，不改变模型、层切分或 WAN 参数：

```bash
./poc up --wan                               # 原路径：pipe IPC + 原打包
./poc up --wan --wire-fast                   # 仅减少打包副本
./poc up --wan --ipc-mode shm                # 仅云侧本机共享缓冲区
./poc up --wan --ipc-mode shm --wire-fast    # 两者组合
./poc up --wan --ipc-mode shm --wire-fast --tcp-buffer-mib 16
```

`shm` 仅连接云侧服务进程和本侧 GPU worker，端云之间仍走 HTTP/TCP 与 tc 整形。
`--tcp-buffer-mib` 默认 0（系统默认），可选 1–64 MiB；只配置 POC 数据连接两端的 socket，
在 connect/listen 前设置以协商窗口缩放，需要 Linux CAP_NET_ADMIN，不修改全局 sysctl。
当前同步 executor 的锁保护单槽缓冲区；响应在锁释放前复制为独立数组，避免后续请求覆盖。
`/health` 的 `optimizations`、两侧配置和测量记录保留开关值；正式正确性报告必须匹配开关配置。
这是正在验证的实验特性，不预先承诺推理性能提升。进展见 [实施记录](docs/optimization-progress.md)。

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-3B-Instruct","messages":[{"role":"user","content":"什么是 KV Cache？"}],"temperature":0,"max_tokens":128,"stream":true}'
```

支持 `/v1/chat/completions`、`/v1/completions`、`/v1/models`、`/health`、`/metrics`。
首版只支持 greedy (`temperature=0`)、单候选、最长总上下文 16384 token、最长输出 1024 token。
Completions 支持文本或单个 token-ID 数组，以及 `ignore_eos`。
尚未实现 stop strings、logprobs、tools、非零 temperature；请求不支持的这些已识别参数会报错。

## 运行与数据路径

| 配置 | Front | Cloud | Back |
|---|---:|---:|---:|
| 1:3 | 4 | 27 | 5 |
| 1:1 | 9 | 18 | 9 |
| 3:1 | 13 | 9 | 14 |

层数配置实际从 `configs/split_*.json` 加载；可调整 Front/Cloud/Back，三段之和必须为 36。
两侧启动握手会核对实际层边界。

`split-enterprise` 和 `split-cloud` 是两个独立网络命名空间；各有两个 GPU worker。
企业前后段共享企业 TP group。Cloud 没有 tokenizer、Embedding、Norm 或 LM Head 实例。
RPC 传输 FP16 `hidden_states`、`residual` 和白名单执行元数据；不使用 pickle、共享 GPU memory 或跨侧 NCCL。
Cloud 从逻辑 position 构建自己的物理 KV block table。Decode 每步只执行新增 token，并按轮次对活跃请求组成 batch。
两侧 KV 按 request ID 隔离，结束/取消时释放；Cloud 清理过期请求。
网络、执行或状态错误会使 executor 不健康并要求重启，禁止本地完整模型 fallback。

隐状态并非加密数据，本项目不证明中间表示不可反演。

## 精度验证

先停止 Split，使用原生 vLLM 保存完整 vocabulary logits；默认每种长度重复两遍，含 Prefill + 256 Decode：

```bash
./poc down
CUDA_VISIBLE_DEVICES=0,1 VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  .venv/bin/python -m split_poc.reference
./poc up --split 1:3
.venv/bin/python scripts/correctness.py --steps 257 --output results/correctness_1_3
```

对 1:1、3:1 重复最后两条命令，并使用独立输出目录。
Reference 使用原生 LLMEngine、固定模型和相同 FP16/Attention/TP/NCCL 配置。
原生模型 `compute_logits` 的只读 hook 仅复制 logits，返回原 tensor；没有修改模型计算。
Teacher-forcing 固定原生 token 序列；另测 greedy 前 32 个 token 的精确 ID。
阈值遵循 issue，Top-5 定义为各位置集合交集比例的均值。
诊断接口仅用于本机验证；正式性能运行不调用它们。

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## 性能与观测

```bash
./poc wan 5
.venv/bin/python scripts/benchmark.py --rates 0.5,1,1.5,2 --requests 1000 \
  --correctness-report results/correctness_1_3/summary.json
.venv/bin/python scripts/telemetry.py --output results/telemetry --seconds 300
```

基准使用官方 `vllm bench serve` 的客户端数据；主 workload 为 4096/256。
详细 profiler 必须单独运行，不能把启用 Nsight 的数字作为正式 SLO 结果。
低样本 smoke benchmark 只验证脚本，不证明 P99 SLO。

```bash
./poc down
./poc up --profile --wan
.venv/bin/python scripts/profile_request.py  # 等待采样关闭完成
./poc down
./poc up --wan
```

Nsight 报告保存在此次启动的结果目录。Nsight 可能注入自身的 NCCL 包装库，因此需要与正式环境分开记录。
DCGM 采集需要本机 `nv-hostengine` 正常运行；`dcgmi discovery -l` 可检查连接。

`run/*.log` 保存服务日志；`run/current_results` 指向当前实验目录；`results/<启动时间>/` 保存配置和 `split_trace.jsonl`。
Trace 中 GPU/传输耗时以 batch 为单位，同一 batch 内各请求共享，不应将这些重复行相加计算总 GPU 时间。
`rpc_wall_ms` 包含传输、Cloud 队列和计算；不能全部解释为纯 WAN 延迟。
上传/下载分段时间使用同一台服务器的 monotonic 时钟，包含 HTTP、序列化及 CPU 开销；迁移到两台主机时需要先校准时钟，不能直接沿用单机的一程时间计算。
大文件和模型默认不进入 Git；固定模型 revision 位于 `split_poc/__init__.py`。

## ISL/OSL 与 TP 性能基线

[2026-09-05 实测报告](docs/baseline-2026-09-05.md)：32 个配置单元、96 轮、1,152 个请求；
另有四种 TP 拓扑的同版本 Nsight/DCGM profiling。这里报告真实基线，不做 SLO 达标声明。

新一轮基线以真实测量为目标，不以 SLO 达标作为成功条件：

- 固定 Qwen2.5-3B-Instruct FP16、4/27/5 层，10 Gbps / 单向 5 ms WAN。
- 端/云 TP 组合：1+1、1+2、2+1、2+2；每侧 TP 数等于该侧实际卡数。
- ISL/OSL：512/128、2048/256、8192/256、2048/1024。
- 闭环客户端并发 1、4，每点三次重复；QPS 是该并发下的实测完成率，不是最大容量。
- 普通性能运行不做同步 GPU 阶段计时；`--phase-profile` 或 `--profile` 才启用详细阶段计时。
- 每种 TP 拓扑另行运行 Nsight + DCGM，覆盖相同 workload 和并发，禁止用采样运行的吞吐替代性能基线。

```bash
./poc down
CUDA_VISIBLE_DEVICES=0 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
  .venv/bin/python -m split_poc.reference --tp 1 --steps 33 --output results/baseline_native_tp1
CUDA_VISIBLE_DEVICES=0,1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
  .venv/bin/python -m split_poc.reference --tp 2 --steps 33 --output results/validation_final/reference
.venv/bin/python scripts/baseline_matrix.py --output results/baseline_matrix --resume
.venv/bin/python scripts/summarize_baseline.py results/baseline_matrix
.venv/bin/python scripts/verify_baseline_traces.py results/baseline_matrix
```

汇总生成 `REPORT.md`、`baseline_summary.csv`、`profile_batches.csv`、`profile_summary.csv`
和 `dcgm_profile_samples.csv`；逐请求长度与传输量核对结果保存在 `trace_verification.json`。
部分结果会明确标记未完成。混合 TP 没有单一匹配的原生 TP 参考，数值差异单独记录为
`CROSS_TP_OBSERVATION`，不冒充同 TP 的精度验收通过。

这是固定长度的合成 workload：普通测试使用请求编号加重复单 token 文本，详细采样使用
同一单 token 的精确 ID 序列；实际执行全部模型层和完整自回归输出，不是语义质量测试。
`ignore_eos` 保证 OSL，不启用 prefix cache、投机解码或提前结束。
本轮并行变量限于单卡/TP=2 及两侧非对称 TP，不包含 DP 或新增流水并行调度。

长时间多卡采样建议使用 Nsight Systems 2025.3.1：本机 2024.6.2 在 NCCL
`cudaEventRecord` 的 CUPTI 路径可重复卡住，关闭新版工具的
`--cuda-event-trace=false` 后重试。启动脚本会在支持时自动加上这个开关，并优先使用
本机隔离安装的 2025.3.1；其他机器可通过 `SPLIT_NSYS_BIN=/absolute/path/to/nsys`
指定工具。该设置只影响 profiler，不改变普通推理的软件版本或性能测量。
NVIDIA 关于事件追踪额外依赖的说明见
[CUDA Event completion trace](https://forums.developer.nvidia.com/t/device-side-cuda-event-completion-with-multiple-cuda-streams/326600)。

## 验收状态

当前服务器已通过真实权重的三种切分精度与功能验收，支持一键启动演示。
见 [实测报告](docs/poc-validation.md)：三种切分各 1028 个位置与原生 logits 逐值一致，
长上下文、4 并发、流式输出、故障恢复和真实 Nsight 采样均已执行。
短压测不等同于正式 SLO 达标；WAN 测点的首 token 延迟未达到目标，尚未完成正式容量搜索。

完整复现（会停止当前演示，完成后保留 WAN 服务）：

```bash
.venv/bin/python scripts/validate_poc.py
```

## Split-inference 仿真与演示

[Q4 仿真入口](Q4/README.md)提供 baseline 回退与当前 PD 模型；[完整 Web demo](demo/README.md)保留安全实验回放和真实 Q4 性能模拟。
校准值、硬编码假设、迁移要求及未解决问题统一见 [仿真交付说明](docs/split-inference-simulator.md)。
