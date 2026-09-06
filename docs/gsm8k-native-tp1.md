# 单卡原生完整模型对 Split 4/27/5：cosine 与 token 排名对照

使用2026-09-07新采集的同一批真实 logits，离线重新计算指标，无需重复GPU推理。原生完整模型为单卡TP1，Split为企业TP2+云TP2、4/27/5层；Qwen2.5-3B-Instruct、FP16、vLLM0.10.2 V0、eager、FlashAttention、full-prefill。模型revision为 `aa8e72537993ba99e69dfaafa59ed015b17504d1`。

固定8道GSM8K test输入，indices为51、209、228、285、457、501、563、1309，长度3268–3319。原生与split独立采集，decode使用原生生成的token做teacher forcing，逐绝对位置比较。每题比较最后一个prefill位置和7个decode位置；诊断逐请求执行，不代表C16/C44动态batch场景。

## 当前指标

| 阶段 | 位置数 | Logits cosine平均 | Logits cosine最低 | Top1一致 |
|---|---:|---:|---:|---:|
| Prefill末位置 | 8 | 0.99999665 | 0.99999465 | 8/8（100%） |
| Decode | 56 | 0.99999659 | 0.99999193 | 56/56（100%） |

| 阶段 | Top5 overlap平均 / 最低 | Top10 overlap平均 / 最低 | Top20 overlap平均 / 最低 |
|---|---:|---:|---:|
| Prefill末位置 | 100% / 100% | 100% / 100% | 100% / 100% |
| Decode | 100% / 100% | 99.2857% / 90% | 99.6429% / 95% |

Decode的56个位置中，4个位置top10集合有1个token不同，4个位置top20集合有1个token不同，其余集合完全一致。Top-k集合相同不代表集合内排名相同，也不代表概率值逐值相同。Top1一致只描述这64个受测位置，不是完整自由回答一致性结论。

## 口径

- Cosine：对同一位置的151936维原始logits计算点积除以两个L2范数的乘积，不先做softmax或中心化。高维向量cosine很高不自动保证前列token一致，因此同时报告top1和top-k。
- Top1：比较最大logit对应的token ID，同时保留原生与split的ID。
- Top-k overlap：`|TopK(native) ∩ TopK(split)| / k`，不是Jaccard，也不考虑集合内顺序。逐位置计算，再按阶段统计平均与最低值。
- 同分：logit降序，相等时token ID升序，保证FP16同分情形可复现。最低重叠率反映集合边界差异，不作为额外失败门槛。

**当前使用描述性指标，不沿用MAE/绝对误差门槛，也未自行设定新的通过阈值。** 比较程序正常退出表示采集数据有效且完成计算，不等价于安全性或精度认证。之前的绝对误差报告保留为历史记录，不影响当前评价。

## 证据与复算

[当前汇总](../demo/evidence/accuracy-tp1-ranking.json) · [逐位置cosine、top1 ID、top20 ID及overlap](../demo/evidence/accuracy-tp1-ranking-positions.json) · [完整原始logits、输入、配置与采集日志](../demo/evidence/accuracy-tp1-raw.zip) · [SHA256](../demo/evidence/SHA256SUMS.json)。原始ZIP沿用此前采集归档，其内汇总和复算源码是历史绝对指标版本；用当前仓库脚本可对其中相同NPZ重新计算排名指标。

将ZIP解压至新目录，运行当前脚本：

```bash
.venv/bin/python scripts/gsm8k_validate.py --phase compare \
  --compare-variants baseline --metric-mode ranking --output /path/to/extracted
```

`ranking`是默认模式。只有显式指定`--metric-mode absolute`时才复算历史绝对误差与旧门槛。此次没有修改、量化或替换原始logits。

若重新采集，先在空输出目录执行`--phase prepare`，然后停止本目录split服务，用 `CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN` 执行 `--phase native --variant baseline --native-tp 1`；再 `./poc up --wan`，执行 `--phase split --variant baseline`，最后按上面命令compare。各阶段使用同一个 `--output`。
