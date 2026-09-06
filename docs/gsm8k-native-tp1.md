# 单卡原生完整模型对 Split 4/27/5 的新精度对照

2026-09-07 重新进行 GPU 采集。原生完整模型改为 **单卡 TP1**；Split 保持企业 TP2 + 云 TP2、4/27/5 层、WAN 单向5ms/10Gbps。模型为固定 revision `aa8e72537993ba99e69dfaafa59ed015b17504d1` 的 Qwen2.5-3B-Instruct，FP16、vLLM0.10.2 V0、eager、FlashAttention，两边均为完整 prefill。

复用先前固定的8道 GSM8K test 输入，indices 为51、209、228、285、457、501、563、1309，输入长度3268–3319。此次先停止 split，设置 `CUDA_VISIBLE_DEVICES=0` 采集原生 TP1，再启动真实 Split TP2+2，使用新原生输出 token 做 teacher forcing，独立重新采集 split logits。没有复用旧 split logits，也没有将原生 logits 传入 split。公共API功能测试为C8，logits诊断逐请求执行；两者的并发范围不同。

## 结果

下表 MAE/RMSE 是每个阶段中最差位置的全词表误差；概率指标用 temperature=1 的稳定softmax计算。

| 阶段 | 位置数 | 最大逐位置 MAE | 最大逐位置 RMSE | 最大绝对误差 | 最大 softmax TV | 最大单token概率差 | 未通过位置 |
|---|---:|---:|---:|---:|---:|---:|---:|
| prefill末位置 | 8 | 0.01045594 | 0.01314045 | 0.078125 | 0.00425808 | 0.00348155 | 1/8 |
| decode | 56 | 0.01624706 | 0.02020044 | 0.09765625 | 0.01012980 | 0.01012806 | 7/56 |

全位置平均 MAE 分别为0.00741936、0.00742412。64/64位置的top1 token一致；不将其解释为所有自由回答一致。最大单token概率变化分别约0.348和1.013个百分点。

**原数值门槛未通过，比较命令退出码为1。** 门槛保持逐位置MAE≤0.01、RMSE≤0.02、最大绝对误差≤0.1、cosine≥0.9999，没有针对本次结果放宽。prefill最小cosine为0.99999465，decode为0.99999193；失败来自部分位置的MAE/RMSE。功能生成正常不等于数值门槛通过。

原有“原生TP2对Split TP2+2逐值一致”是另一组对照，仍保留在[原验证报告](gsm8k-validation.md)。此次改成原生TP1后，计算中的分片、矩阵形状和归约路径发生变化，测得非零误差；不能仅凭这组实验把误差全部归因于切分或全部归因于TP。两次试验都只检查最后一个prefill位置与7个decode位置，不覆盖全prompt每个位置，也不代表高并发动态batch下的误差范围。

## 原始证据与复算

[最新汇总](../demo/evidence/accuracy-tp1.json) · [逐绝对位置记录](../demo/evidence/accuracy-tp1-positions.json) · [完整原始 logits、输入、配置和日志](../demo/evidence/accuracy-tp1-raw.zip) · [SHA256](../demo/evidence/SHA256SUMS.json)。NPZ中原生与split均保存独立捕获的全词表logits，FP16计算结果转FP32保存；不是FP32原生模型对照。

解压ZIP到新目录后，可无GPU复算：

```bash
.venv/bin/python scripts/gsm8k_validate.py --phase compare \
  --compare-variants baseline --output /path/to/extracted
# 预期退出码1：按原门槛正确报告本次不通过。
```

重新采集须使用空输出目录，先按既有 `--phase prepare` 准备同一组8题，然后依次执行：

```bash
./poc down
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
  .venv/bin/python scripts/gsm8k_validate.py --phase native --variant baseline \
  --native-tp 1 --output results/new-tp1
./poc up --wan
.venv/bin/python scripts/gsm8k_validate.py --phase split --variant baseline \
  --output results/new-tp1
.venv/bin/python scripts/gsm8k_validate.py --phase compare \
  --compare-variants baseline --output results/new-tp1
```

`--native-tp` 只改变原生完整模型的TP；默认行为仍沿用此前匹配各split配置的对照。此次显式选择TP1、baseline/full-prefill，未开启原生chunked prefill。页面现在显示本次结果，并标注原数值门槛“未通过”。
