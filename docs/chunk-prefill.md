# 阶段 2 草案：串行 chunk 与基础调度

已做真实 GPU 验证：greedy 一致，但长输入 logits 超过既定门槛。当前不能作为通过正确性验收的优化发布。

## 开关

```bash
./poc up --wan --ipc-mode shm --wire-fast --tcp-buffer-mib 16 \
  --prefill-chunk-size 512 --scheduler-policy decode-first --decode-quota 1
```

chunk size 默认 0，保持完整 prefill；policy 默认 legacy，保持 prefill 优先。
两者是独立开关。decode-first 在等待 prefill 存在时，最多执行 quota 轮 decode，
然后强制推进一个 FIFO prefill chunk。chunk size 限制每轮 prefill token 预算；
decode batch 仍受 max_active 限制。它不是基于成本模型的自适应调度算法。

非零 position 的 chunk 使用已有 KV 块表继续 attention；两侧按实际 query_len 追加。
中间 chunk 不执行最终 norm/lm head、不增加生成 token 数、不向用户发出 token。
最后一个 prefill chunk 才生成首 token。正常 decode 仍为 query_len=1。

该路径仍同步往返，既没有多在途请求，也没有 GPU/WAN 流水。
8k/512 的 16 次往返意味着传播 RTT 项约 160 ms，而不是完整 prefill 的约 10 ms。
是否改善 TTFT/TPOT 必须实测，特别应保留单请求回退结果。

## 数据与验收

```bash
.venv/bin/python scripts/chunk_sweep.py --output results/new_chunk_scan \
  --tp 2 --ipc-mode shm --wire-fast --tcp-buffer-mib 16
```

候选大小为 0/256/512/1024/2048，分别 legacy/decode-first；每个候选先执行同 TP、
预先固定标准的 logits/greedy 正确性验证。默认每候选扫描 8k/256 C1/C4 各一轮，
用于代表场景验收；按用户更新要求不再强制完整矩阵。

另以 `mixed_arrival.py` 先启动 512-token 输入的 decode，请求输出 8 个 token 后注入 8k prefill。
记录两请求的 TTFT/TPOT 与逐 SSE 内容事件 P50/P95/P99、最大间隔；当前协议每 token 一个事件，
即使文本为空也计数，并与 usage 严格核对。此口径不应自动沿用到未来投机成批输出。
样本较少，尾分位数只作描述；pair completion QPS 不是稳定系统容量。

后续仍需：真实数值、取消/异常回归，各 chunk 上下文位置的独立 profiling，
候选消融及代表工况重复，不再要求完整 TP/长度/并发矩阵。

trace 新增 `position_start`、`query_len`、`emits_token`，中间 chunk 的 token_idx=-1。
校验器兼容历史单 prefill，检查分块 query 总和=ISL、输出总数=OSL、位置连续及激活字节。
TTFT 拆解必须对同请求的所有 prefill chunk 累加等待/执行，不能把一个 chunk 当成完整 prefill。
普通 trace 与详细 GPU profiling 始终分开统计。

## 代表场景验收结果

按用户更新要求，以代表场景验收，不要求 96 轮矩阵。固定 Qwen revision、
4/27/5 切分、10 Gbps / 10 ms RTT、shm+wire+TCP16，33 个 full-vocabulary
位置 × 128/1024/4096/8192 输入，另每长度 32-token greedy：

| TP | chunk | logits | 最大逐位置 MAE | 最大绝对误差 | greedy |
|---|---:|---|---:|---:|---|
| 2+2 | 256 | FAIL | 0.534747 | 3.232422 | 四组一致 |
| 2+2 | 512 | FAIL | 0.170266 | 0.958008 | 四组一致 |
| 2+2 | 1024 | FAIL | 0.468083 | 2.714844 | 四组一致 |
| 2+2 | 2048 | FAIL | 0.184347 | 1.059570 | 四组一致 |
| 1+1 | 1024 | FAIL | 0.405327 | 2.998047 | 四组一致 |

原标准保持 MAE≤0.01、RMSE≤0.02、max-abs≤0.1、cosine≥0.9999、
top1 一致及平均 top5≥0.99。不能仅凭 greedy 一致替代 logits 验收。
128/1024 不实际切块时 TP2+2 与原生逐值一致，较长输入出现超限；
执行形状改变可能涉及数值路径，但目前尚未用逐层比较确认根因。
另一个使用较早 reference 的失败尝试仅留本地；正式表使用固定的
validation_final/reference（TP2）和 baseline_native_tp1（TP1）。

TP1+1 chunk1024 的真实中途取消 PASS：观察到首个非最终 chunk 后断开，
共完成两个 chunk，两侧 active/KV 均归零，取消前没有输出 token。
CPU 元数据、位置、异常失败关闭测试通过，但不抵消数值门槛失败。

因此未继续对这些失败配置跑收益测试，也不默认启用。阶段2仍有明确
数值阻塞；阶段3先验证不切块的跨请求流水，不能声称通过串行 chunk 验收。
精简证据在 results/stage2_representative/，完整 logits NPZ 仅留本机。

后续补测 256/512 仍均失败；没有以增加性能采样替代数值问题处理。

## 逐层诊断（非性能运行）

另用 8192 个 apple token、6 个输出，比较完整 prefill 与 chunk1024。
在 36 层选取固定全局位置采样 hidden/residual；两个运行的 greedy token 一致。
首个分块的 position=0、layer=0 已有 hidden 最大差 0.00146484、
residual 最大差 0.00195313；到 layer35 的 position0，最大差分别为 0.25/0.296875。
这说明执行形状变化的数值差异在非零位置 KV 续接之前就存在，不能只归因于
续接游标。它尚未定位到具体 GEMM/attention kernel，也不能排除其他问题。

临时诊断还尝试了 prefill 线性层 FP32 计算（权重来源不变、输出回到 FP16、
decode 保持原 q=1 路径），固定原生参考仍 FAIL：最大 MAE 0.304121、
max-abs 1.734375，greedy 一致。该方案增加了临时显存，未进入正式实现，
没有测量或宣称性能收益，也没有修改参考或放宽门槛。

插桩补丁和采样入口在 docs/diagnostics/，仅供隔离诊断工作树使用。
补丁用 `git apply --unidiff-zero docs/diagnostics/chunk-numerics.patch` 应用。
插桩同步 GPU 并复制样本，耗时不能作为 benchmark。精简比较、失败报告和
本地完整样本校验清单在 results/stage2_representative/layer_diagnostic/。
阶段2仍需要找到能满足既定数值门槛的分块执行路径；当前全部候选不予保留为
已验收优化。阶段3的异步长 prompt 分块也因此仍未验收。
