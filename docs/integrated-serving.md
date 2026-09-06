# 可回退的 split serving 集成

本分支面向 main，整合 #8–#12、#14、#16–#21。没有 Stage4 投机推理代码。旧 PR 用于追溯各阶段实验；当前实现与配置以本文为准。历史 results/ 快照仍保留在旧 PR 中，不在此 PR 重复收录；docs/evidence/ 保留已发布的 PD 和 WAN 图表、审计及归档。

## 一条命令启动

环境：Linux、root（network namespaces/tc）、四张 A10 或满足显存要求的四卡、可用 NVIDIA 驱动、Python 3.12、ip/tc/curl/rg。首次启动自动创建虚拟环境并下载固定 revision 的 Qwen2.5-3B-Instruct；准备时间不计入推理性能。推荐配置经本机四张 A10 验证，每个 TP1 P 约需 18.5GiB 显存。

```bash
./poc up --preset optimized --wan
curl -N --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"split-qwen","messages":[{"role":"user","content":"请用一句话介绍你自己。"}],"temperature":0,"max_tokens":64,"stream":true}'
```

启动包含健康检查与真实请求冒烟，返回 SSE `data:` 事件并以 `[DONE]` 结束。还支持 `/v1/completions`、非流式请求、`/health`、`./poc status`、`./poc down`。入口绑定 loopback；本项目是研究 POC 服务，不含生产鉴权。

回退：

```bash
./poc up --preset baseline --wan
```

同一工作目录内切换 preset 会先关闭其拥有的进程，再按新配置启动。其他 checkout 的服务应先在其自己的目录执行 `./poc down`；多个 checkout 共享 GPU 和 namespace，不能同时启动。

baseline 是未优化执行模式：企业 TP2、共享云 TP2、企业前4/云27/企业后5层，pipe IPC、原 wire、默认 TCP buffer、完整 prefill、legacy 调度、无异步流水、无 PD。保留正确性修复及故障诊断；不保证在不同软件环境或负载下复现历史计时的每个数值。

## 开关与依赖

参数显式值覆盖 preset，不受参数前后顺序影响；不修改已测模型和权重。

| 特性 | 关闭 / 回退 | optimized |
|---|---|---|
| Stage1 本机 IPC | `--ipc-mode pipe` | `--ipc-mode shm` |
| Stage1 wire 打包 | `--no-wire-fast` | `--wire-fast` |
| Stage1 TCP buffer | `--tcp-buffer-mib 0` | `--tcp-buffer-mib 16` |
| Stage2 prefill 切块 | `--prefill-chunk-size 0` | 2048 |
| Stage2 decode 调度 | `--scheduler-policy legacy` | decode-first、`--decode-quota 4` |
| Stage3 在途流水 | `--pipeline-window 0`（非 PD） | D 全局窗口 2 |
| 云 PD 分离 | `--no-pd`（并取消 PD 专属配置）；全关用 baseline preset | `--pd` |
| P 副本拓扑 | `--prefill-replicas 1 --prefill-tp 2` | 两个 TP1 P，共享一个 TP1 D |
| 独立 P 窗口 | `--pd-prefill-window 0` 继承全局窗口 | 每 P 3 |
| PD 独立 readiness 控制通道 | `--no-pd-control-channel` | 开启 |
| chunk KV 提前迁移 | `--no-pd-chunk-transfer`，整段 prompt 迁移 | 关闭；可选 `--pd-chunk-transfer` |
| 算子记录 | 不加 profiling 参数 | 关闭；`--profile --operator-profile` 开启 |
| WAN 标定端点 | 不设置 `SPLIT_WAN_PROBE=1` | 默认关闭 |

PD 必须有非零 pipeline window。双 P 需要 P TP1/D TP1；单 P 支持 P2/D1 或 P1/D2，企业 TP1，总计四卡。依赖不满足时拒绝启动，不静默更改用户参数。控制连接异常恢复、KV 所有权和释放顺序校验属于可靠性保护，不作为可关闭的性能特性。

```bash
# 查看有效参数，不启动进程：
python3 scripts/manage.py up --preset optimized --print-config
# 同样的优化，仅回到单 TP2 P：
./poc up --preset optimized --prefill-replicas 1 --prefill-tp 2 --wan
# Stage1/2/3 开启但不启用 PD：
./poc up --preset baseline --wan --ipc-mode shm --wire-fast --tcp-buffer-mib 16 \
  --prefill-chunk-size 2048 --scheduler-policy decode-first --decode-quota 4 --pipeline-window 2
```

## GSM8K 功能与 logits 验证

先准备环境/模型（`./poc setup`，已启动过可跳过），确保其他 checkout 的服务已停止：

```bash
# 固定 seed=42 的 8 道 GSM8K test 问题：
./poc validate-gsm8k --output results/gsm8k-logits-new
# 可扩展至全部 1319 题；默认不要求跑完整矩阵：
./poc validate-gsm8k --limit 0 --output results/gsm8k-logits-full
```

验收分为两个独立部分，不评价回答题目的准确率或答案一致性：

1. **功能**：GSM8K 问题通过公共 `/v1/completions` 接口生成非空文本；检查 HTTP 状态、输入/输出 token 计数和请求排空。baseline 与 optimized 都执行。
2. **数值正确性**：每题对比 prefill 最后一个输入 position 的全词表 logits，以及后续 7 个 decode position 的全词表 logits。单体先生成固定 token 序列，split 通过 teacher forcing 使用相同序列；绝不按各自自由生成的后续输入直接比较。

输入采用 train 前16题作为固定上下文，再接 test 问题，让代表场景跨越 2048-token chunk 边界；test 参考答案不进入 prompt，也不参与验收。模型 revision、FP16、输入 token IDs 固定。baseline 对照原生单体 vLLM TP2/full prefill；optimized 对照原生单体 vLLM TP1/chunk2048。分别匹配 TP 和 attention 路径，以隔离 split/传输/KV 引入的误差，不把 TP 或原生切块引起的浮点差异误判为 split 错误。原生模型保持完整36层，不是另一个 split 实例。

原生使用 vLLM V0，只读观察实际 sampler `selected_token_indices`，不改变 LM head 的输入或返回 logits。split 记录实际送入 LM head 的 hidden-state rows。每行保存 `token_id`、`absolute_position`、`num_computed_tokens`、`query_len`、hidden/logits row index、top1 token/logit。脚本首先验证绝对位置、输入 token、context/query 长度，再计算误差。中间 prefill chunk 未产生输出 logits 的位置不伪造记录；prefill 验收范围是最后一个 prompt position，不声称 dump 了全 prompt 的所有位置。

固定数值门槛沿用本项目精度阈值：逐位置 MAE≤0.01、RMSE≤0.02、最大绝对误差≤0.1、cosine≥0.9999；prefill/decode 分开汇总，任一位置失败即返回非零。另记录 temperature=1 的稳定 softmax TV、最大概率差及 top1 是否一致，后两者不作为生成答案一致性门槛。未对 GSM8K 调整数值阈值。

数据来自固定 commit 的 openai/grade-school-math，保存下载 SHA256、样本 indices、prompt IDs/hash、完整 logits NPZ、原始服务输出、有效启动参数及模型版本。输出目录不可覆盖已有结果。脚本依次启动两种原生配置、split baseline、split optimized，成功后保留 optimized 服务，失败后关闭本目录服务。不会停止其他工作目录的服务；有在途请求时拒绝开始。

单阶段也可通过 `scripts/gsm8k_validate.py --phase prepare|native|split|compare` 分别运行。native 使用 `VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN CUDA_VISIBLE_DEVICES=0,1`，并以 `--variant baseline|optimized` 选择配置；独立复算 compare 不需要 GPU。

结果见 [GSM8K 验证报告](gsm8k-validation.md)。此前做过的64题答案对比仅为历史附加观察，不是本次验收条件，也不代替 logits 检查。

## 性能范围

此前同模型、4K/79-token、WAN 每方向10Gbps/单向5ms 的 C40 测量为 5.161 QPS，TTFT 平均/P99 794/1353ms，请求平均 TPOT 的平均/P99 89.28/93.84ms。两个 cohort 都通过“至少99%请求同时 TTFT≤3s、请求平均 TPOT≤100ms”。本次集成不重复全并发扫描，不将 GSM8K 的不同输入/输出长度套入该 QPS。推荐 preset 对应此测量配置；不是已证明的最大容量。
