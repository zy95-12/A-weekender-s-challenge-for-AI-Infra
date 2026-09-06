# 公开硬件与其他模型：理论 Roofline 模式

Demo 06 的模型、硬件下拉项直接改变 Q4 的算子计算量、访存量、通信载荷、容量和成本。
只有 Qwen2.5-3B/A10 使用既有实测校准；其他组合关闭 A10 operator、command、host submission、
post-back 和 WAN 表，使用 `max(FLOPs/有效算力, bytes/有效带宽) + launch` 与解析传输。
其他组合采用logical架构算子图，不将vLLM的CUDA融合实现视作Ascend的真实kernel。
这些选项是理论仿真，不下载权重、不启动其他模型推理服务，也没有跨硬件精度承诺。

## 硬件参数与出处

配置：[public_roofline.json](../hardware/public_roofline.json)，检索日期2026-09-07。
单位为稠密 FP16/BF16 TFLOPS、十进制 GB/s 和 GB，不使用稀疏算力替代稠密算力。

| 设备 | 算力 | 显存带宽 | 容量 | 单向互联假设 | 来源 |
|---|---:|---:|---:|---:|---|
| L20 | 119.5 | 864 | 48 | 32 | [MegaScale-Infer Table 3](https://arxiv.org/html/2504.02263v2)；PCIe4×16互联为理论假设 |
| H20 | 148 | 4000 | 96 | 450 | [AReaL-Hex §4.4](https://arxiv.org/html/2511.00796v1)、[NVIDIA 96GB SKU](https://docs.nvidia.com/ai-enterprise/release-6/6.7/infra-software/vgpu/reference/hopper.html) |
| Ascend910B | 320 | 1600 | 64 | 196 | [910B2 GEMM作者实测](https://ascend-rs.org/en/ch09-performance.html)、[公开仿真参数及验证](https://github.com/rainbay001-dotcom/xPU-simulator)、[64GB实验环境](https://arxiv.org/html/2604.09752v1) |

910B有多个SKU，320T/1600GB/s是此模型选用的公开研究参数，不是所有910B的统一官方规格。
HCCS单向196GB/s也只是拓扑假设；不声称复现CANN、MTE或vector流水。
算力与访存效率均暂取0.7、launch10μs、collective latency8μs，未在新增设备上标定。
TP>8 时互联改为暂定25GB/s跨节点链路，不假定所有设备共享节点内峰值互联。
A10沿用原配置125T/600GB/s/23GB；新模型运行在A10时同样不复用3B的实测成本表。

## 模型来源与算子图

- Qwen3-32B：[官方 HF config](https://huggingface.co/Qwen/Qwen3-32B/blob/9216db5781bf21249d130ec9da846c4624c16137/config.json)。
  64层、hidden5120、Q64/KV8、head_dim128、FFN25600；使用已有Qwen3含Q/K norm的算子图。
- DeepSeek V4选择 **V4-Flash**，不是V4-Pro：[官方 HF config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/config.json)。
  43层、hidden4096、Q64、共享K=V512、256路由专家/top6+1共享专家、FFN2048。
  参考 [HF实现](https://github.com/huggingface/transformers/blob/a353632607c59463e6ced86a44c2de3c2cd62d5e/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py)，
  新增低秩Q/分组O、SWA/CSA/HCA压缩、indexer/top-k、hash/learned router、MoE和mHC。
  使用主干43层对应的compression schedule；不运行额外MTP/投机路径。

V4采用 **BF16反量化后的架构场景**，所有专家驻留、只有选中专家计算。
不假定原checkpoint的FP4专家/FP8权重能在这些设备上获得特定加速。
均匀独立路由下，batch含T个token预计触及专家数为 `E × (1 − (1 − k/E)^T)`；
访存随触及专家数变化，计算随 `T×(k+shared)` 变化，容量仍按全部专家计算。
专家采用TP分片并计入all-reduce，不实现EP/all-to-all或专家负载不均衡。
SWA窗口128；CSA对4倍压缩KV选top512；HCA使用128倍压缩KV；访存包括复制的K=V缓存及压缩缓冲。
mHC按4路残差流计算WAN载荷，不能继续使用dense模型单hidden宽度。

这些是分解后的架构级近似，不是HF逐kernel回放：mHC融合、FP32向量计算效率、压缩器边界开销、
稀疏attention实际kernel、quant/dequant、索引器缓存/路由相关性仍未校准。
理论网络沿用10Gbps/RTT10ms场景及搬运/PD默认参数。reserve P/D为0.24/0.11ms、release各0.10ms、local RPC1.1ms、KV传输100GB/s+0.1ms；这些是保留的主机/协议假设，不是新增硬件上的公开实测。改变设备不会自动消除WAN限制。

## 容量与卡数

企业首4层、尾5层；Qwen3为4/55/5，V4-Flash为4/34/5。
理论模式按96条活动请求、4175token的权重+KV，预留20%容量空间，选择可整除attention heads的最小2次幂TP。
baseline最低企业/云TP2；PD包含一个企业实例、两个完整P副本和一个D副本。
输出 `catalog.capacity_plan/stages/total_devices` 和每rank权重/KV需求。卡数不是固定4张，不能做同卡数排名。
容量检查不是实际部署可行性认证：allocator碎片、workspace及全部分布式后端约束未逐项复现。

## 运行

```bash
printf '%s\n' '{"variant":"optimized","workload_mode":"open_loop","arrival_rate_qps":0.5,"model":"qwen3-32b","hardware":"h20"}' > /tmp/sim-input.json
python3 demo/simulate.py /tmp/sim-input.json /tmp/sim-output.json
```

换模型/硬件必须重新做算子与通信采集后，才能把理论趋势升级为精度承诺。Demo中的精度表仅来自A10/3B校准路径。
