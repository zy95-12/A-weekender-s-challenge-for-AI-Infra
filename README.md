# Split-vLLM 企业—云推理 POC

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
./poc network-check         # ping 与 iperf3，需要已安装 iperf3
./poc local                 # 恢复专用链路无限制
```

只有项目的 `split-e` / `split-c` 链路接受 tc 配置；不修改管理网卡。
入口只绑定主机 loopback；不要直接将无认证的演示接口暴露公网。

## API

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

## 验收状态

当前服务器已通过真实权重的三种切分精度与功能验收，支持一键启动演示。
见 [实测报告](docs/poc-validation.md)：三种切分各 1028 个位置与原生 logits 逐值一致，
长上下文、4 并发、流式输出、故障恢复和真实 Nsight 采样均已执行。
短压测不等同于正式 SLO 达标；WAN 测点的首 token 延迟未达到目标，尚未完成正式容量搜索。

完整复现（会停止当前演示，完成后保留 WAN 服务）：

```bash
.venv/bin/python scripts/validate_poc.py
```
