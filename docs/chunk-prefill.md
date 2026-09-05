# 阶段 2 草案：串行 chunk 与基础调度

当前仅 CPU 单元测试通过，尚未真实 GPU 验收，不作为已完成优化发布。

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
用于筛选，不替代保留变体所需的三次重复完整矩阵。

另以 `mixed_arrival.py` 先启动 512-token 输入的 decode，请求输出 8 个 token 后注入 8k prefill。
记录两请求的 TTFT/TPOT 与逐 SSE 内容事件 P50/P95/P99、最大间隔；当前协议每 token 一个事件，
即使文本为空也计数，并与 usage 严格核对。此口径不应自动沿用到未来投机成批输出。
样本较少，尾分位数只作描述；pair completion QPS 不是稳定系统容量。

后续仍需：真实数值、取消/异常回归，各 chunk 上下文位置的独立 profiling，
候选消融、代表工况三次重复、保留配置的完整 TP/长度/并发矩阵。

trace 新增 `position_start`、`query_len`、`emits_token`，中间 chunk 的 token_idx=-1。
校验器兼容历史单 prefill，检查分块 query 总和=ISL、输出总数=OSL、位置连续及激活字节。
TTFT 拆解必须对同请求的所有 prefill chunk 累加等待/执行，不能把一个 chunk 当成完整 prefill。
普通 trace 与详细 GPU profiling 始终分开统计。
