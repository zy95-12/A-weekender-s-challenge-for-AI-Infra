# Issue #6 系统优化实施记录

关联：https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/issues/6

## 状态

| 阶段 | 状态 | 备注 |
|---|---|---|
| 0：归因与 anchor | 完成 | 单/多在途 WAN echo、GPU copy、TP 2+2 的 12 轮未优化 anchor 均通过 |
| 1：数据路径 | 待验证 | 已根据数据选择 shared-memory IPC 与单次 join 打包，尚未在推理服务启用 |
| 2：chunked prefill＋基础调度 | 未开始 | 独立开关与未优化路径必须保留 |
| 3：异步流水 | 未开始 | 在阶段 2 正确性通过后进行 |
| 4：投机推理 | 未开始 | 必须计入 draft 资源、接受率和 KV 回退 |
| 5：RDMA/GDR | 暂缓 | 按用户要求，前四阶段结束后再评估 |

## 阶段 0 初步证据（不是优化收益）

固定 10 Gbps、单向 5 ms 的真实 TCP echo，客户端/云侧分别运行于 POC 的两个 namespace。
两张 FP16 tensor 合计为表中单方向字节量；每种大小和 IPC 条件先 warmup 一次，正式重复 10 次。
以下为客户端 HTTP wall 和嵌套的云侧 IPC wall 均值；没有模型计算。

| 单方向激活量 | 无 IPC 的 HTTP wall ms | 有 IPC 的 HTTP wall ms | 其中 IPC roundtrip ms |
|---|---:|---:|---:|
| 8 KiB | 10.76 | 10.79 | 0.15 |
| 32 KiB | 10.68 | 10.85 | 0.15 |
| 4 MiB | 37.37 | 47.52 | 10.40 |
| 16 MiB | 131.92 | 170.08 | 47.38 |
| 64 MiB | 608.80 | 935.17 | 303.76 |

原始数据：`results/optimization_stage0/transport_v2/measurement.json`，网络前置检查与测量后快照同目录。
`transport_v1` 是服务恢复期间 qdisc 未就绪而被前置检查拒绝的失败尝试，不进入统计。
v2 的 TCP snapshot 在连接关闭后采集，不能用于分析活跃连接窗口；后续版本已将采集点移至连接关闭前，须补测。

隔离 GPU copy 微基准（A10、每条件 20 次、两种模式均复用目标 buffer）中，64 MiB 的均值：

| 条件 | D2H ms | pack ms | unpack ms | H2D ms |
|---|---:|---:|---:|---:|
| pageable buffer | 6.47 | 104.02 | 19.97 | 6.47 |
| pinned buffer | 2.60 | 104.90 | 20.20 | 2.88 |

原始数据：`results/optimization_stage0/gpu_copy/measurement.json`。
H2D 使用原 host buffer，codec roundtrip 另外做逐元素正确性校验；这不是完整 GPU→网络→GPU 延迟。
这些同步微基准不与正在推理的 GPU 争用，也不混入 serving TTFT/TPOT。

## 阶段 1 决策

证据优先支持减少大数组 IPC 和打包副本，而不是先将全部时间投入 pinned memory 或更换网络协议。
计划分别以 `--ipc-mode shm` 和 `--wire-fast` 开关验证；默认 `pipe` 和旧 pack，组合与单项都要保留对照。
共享内存只限云侧服务进程与本侧 GPU worker，端云间仍使用真实 HTTP/TCP 和相同 WAN 整形。
禁止把微基准差值直接当作真实 Qwen 的端到端收益；启用前须做代码测试、模型正确性和重复 A/B。

## 阶段 0 完成验收

`transport_v3` 补齐活跃 TCP snapshot 和 2/4 消息在途实验。64 MiB 单消息、有 IPC 的 HTTP wall 为 973.97 ms，
其中 IPC 往返 331.21 ms；无 IPC 为 620.16 ms。与 v2 相比有波动，但大数组 IPC 为主要成本之一的结论一致。
64 MiB、2/4 消息在途时，平均组完成时间分别 2173.97/3401.00 ms，双向 payload 合计吞吐分别 0.988/1.264 Gbps。
此吞吐计数同时包含上传和返回，不是单方向 iperf 线速。增加在途数没有让当前 CPU/HTTP/IPC 路径接近线速，不能预设流水一定消除这些成本。

TP 2+2 未优化 anchor：`results/optimization_stage0/anchor_tp22/`，每点 3 次、共 144 正式请求，全部成功。
执行脚本 commit 为 `e730c6e`；模型执行路径保持 PR #8 未优化实现。TTFT/TPOT 为均值，QPS 为各次实际吞吐均值：

| ISL/OSL | 并发 | TTFT ms | TPOT ms | QPS |
|---|---:|---:|---:|---:|
| 512/128 | 1 | 99.52 | 32.16 | 0.239 |
| 512/128 | 4 | 228.63 | 34.27 | 0.873 |
| 8192/256 | 1 | 1809.03 | 33.13 | 0.098 |
| 8192/256 | 4 | 4504.49 | 43.93 | 0.255 |

该 anchor 是后续优化的直接对照，不替换历史 96 轮矩阵，也不是全 TP/全 workload 的新充分基线。
各点保留官方 benchmark、逐请求指标、实际网络前后状态、启动/测量环境及源码哈希。
18 项既有单元测试通过。阶段 0 没有修改模型算法，没有开启任何优化。

小型 JSON/日志证据随阶段 0 PR 交付；anchor 完整 trace 和生成 workload 只留在服务器，权重不入 Git。

## 可复现入口

```bash
./poc up --wan
.venv/bin/python scripts/transport_microbench.py run --output results/new_transport --repeats 10
.venv/bin/python scripts/gpu_copy_microbench.py --output results/new_gpu_copy --repeats 20
.venv/bin/python scripts/optimization_anchor.py --output results/new_anchor --repeats 3
```

三者应顺序运行，使用新输出目录。anchor 不修改服务部署，仅测当前配置，包含 512/128、8192/256、并发 1/4。
本文件会在各阶段验证完成后更新；未完成的实验不能标记 PASS。
