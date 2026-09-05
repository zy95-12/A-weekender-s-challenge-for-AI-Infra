# Issue #6 系统优化实施记录

关联：https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/issues/6

## 状态

| 阶段 | 状态 | 备注 |
|---|---|---|
| 0：归因与 anchor | 完成 | 单/多在途 WAN echo、GPU copy、TP 2+2 的 12 轮未优化 anchor 均通过 |
| 1：数据路径 | 代表场景验收完成 | 根据用户更新的验收范围收尾；收益与回退见下方收敛结论 |
| 2：chunked prefill＋基础调度 | 本地实现待 GPU 验收 | 串行 chunk、基础调度及验证脚本已有本地提交 |
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

## 阶段 1 代表工况结果（完整验收尚未结束）

三个无损开关为 `--ipc-mode shm`、`--wire-fast`、`--tcp-buffer-mib 16`。
前两者分别减少云侧本机 IPC 和打包临时副本；第三个只在数据连接建立前增加 socket 缓冲区，
不改 sysctl、模型或 WAN。所有开关默认关闭，HTTP 仍为数据面，未实施 RDMA。

### 单项消融：真实 WAN echo

每方向 64 MiB、一次未计入 warmup、10 次正式重复；各列为均值 ms。
客户端 total = pack + HTTP wall + unpack；IPC 和云 pack 嵌套在 HTTP 内，不能再相加。

| 变体 | client pack | HTTP wall | client unpack | client total | 其中云 IPC | 其中云 pack |
|---|---:|---:|---:|---:|---:|---:|
| OFF | 80.52 | 953.75 | 8.12 | 1042.39 | 303.75 | 102.06 |
| wire | 33.65 | 905.43 | 15.77 | 954.85 | 308.68 | 33.33 |
| shm | 81.20 | 672.11 | 8.46 | 761.77 | 31.35 | 102.92 |
| shm＋wire | 30.37 | 624.23 | 11.79 | 666.38 | 32.67 | 32.66 |
| shm＋wire＋TCP16 | 32.92 | 278.77 | 18.49 | 330.17 | 29.43 | 32.60 |

证据目录：`results/optimization_stage1/micro_{off,wire,shm,both,both_tcp16}/`。
每个目录还有 8/32 KiB、4/16 MiB、无 IPC 对照和多在途实验，均通过网络检查。
TCP snapshot 在连接仍建立时采集，但不是整轮持续采样：默认接收缓冲区约 6 MiB、
发送缓冲区约 4 MiB，已有连接出现 rwnd/sndbuf 限制；TCP16 的实际内核缓冲区为 32 MiB、
协商窗口约 16 MiB。配置 RTT 仍为 10 ms，不能把此收益解释为降低传播 RTT。

### 真实 Qwen，TP 2+2

每点三次重复，C1 每次 8 请求、C4 每次 16 请求；各变体共 144 正式请求。
单元格依次为 TTFT ms / TPOT ms / QPS，均为各次均值的平均（每次样本数相同）。

| ISL/OSL | 并发 | OFF anchor | shm＋wire | shm＋wire＋TCP16 |
|---|---:|---:|---:|---:|
| 512/128 | 1 | 99.51 / 32.16 / .2390 | 88.90 / 31.89 / .2416 | 71.86 / 32.78 / .2361 |
| 512/128 | 4 | 228.63 / 34.27 / .8732 | 204.57 / 34.13 / .8811 | 160.43 / 34.23 / .8873 |
| 8192/256 | 1 | 1809.03 / 33.13 / .0975 | 1337.63 / 32.08 / .1051 | 1018.70 / 33.11 / .1057 |
| 8192/256 | 4 | 4504.49 / 43.93 / .2547 | 3260.74 / 40.72 / .2932 | 2530.05 / 40.04 / .3140 |

全开时四行 TTFT 重复间样本标准差分别为 0.20/0.54/6.31/25.33 ms，
TPOT 为 0.147/0.298/0.026/0.271 ms；QPS 为 .0011/.0075/<.0001/.0023。
逐点 P50/P95/P99 见各 `summary.json`，逐请求原始值见 `qps_inf/requests.jsonl`。
样本量小，尾分位数只作描述，不作 SLO/容量结论。这些运行顺序执行，尚非随机交错 A/B。

8k 的 TTFT 下降约 44%，C4 吞吐增加约 23%；但 512/C1 的 TPOT 从 32.16 升至 32.78 ms、
QPS 从 .2390 降至 .2361。不能宣称全部工况的吞吐提高，也不能用长输入收益掩盖短输入回退。
单项 wire/shm 目前只有微基准消融，真实模型测了 OFF、两者组合、再加 TCP 三档；
尚不能将组合的端到端收益完全分摊给单个开关。

### 为后续 chunk/流水建模提供归因

8k/C1 普通 trace，按正式 client ID 排除 warmup，每档 24 次 prefill，均值 ms：

| 互斥区间 | OFF | shm＋wire | 再加 TCP16 |
|---|---:|---:|---:|
| 上传路径 U | 331.09 | 302.03 | 104.58 |
| 返回路径 D | 363.23 | 284.11 | 139.87 |
| 云服务 C = RPC−U−D | 807.62 | 504.14 | 511.39 |
| 端侧其余 E = step−RPC | 283.43 | 222.79 | 238.60 |
| step = U＋D＋C＋E | 1785.37 | 1313.07 | 994.44 |

U/D 含 HTTP/CPU，C/E 含计算、拷贝、IPC 等，均不是纯 GPU 时间；排队和客户端边界差值不在 step 中。
共享 IPC＋打包主要减少云服务区间；随后 TCP16 使 U＋D 从 586.14 降至 244.45 ms。
不能继续用原始 694.31 ms 上传/返回成本预估后续流水收益。
新的纯 GPU 时间线由完整矩阵的独立 profiling 提供，不拼接进普通运行 TTFT。

作为下一阶段候选大小的输入，TCP16＋shm＋wire 的独立 echo（含 IPC）还测得：

| 对应 query token 数 | 单方向激活 | 每消息 client total 均值 ms |
|---|---:|---:|
| 1 | 8 KiB | 10.825 |
| 4 | 32 KiB | 10.864 |
| 512 | 4 MiB | 26.613 |
| 2048 | 16 MiB | 76.274 |
| 8192 | 64 MiB | 330.174 |

仅将隔离消息均值做算术求和，16×512 对应约 425.8 ms，4×2048 对应约 305.1 ms；
这不是已跑出的 chunk 模型性能，也没有包括 GPU attention 随上下文增长的成本。
它说明不能只按“块越小越容易重叠”选尺寸：每块固定开销与大消息效率需要一起扫描。
8/32 KiB 消息仍约 10.8 ms，传播 RTT 没有因本阶段优化消失。

### 正确性、保留决策与剩余门槛

- `correctness_both_tp22`、`correctness_all_tp22`：128/1024/4096/8192 输入，
  每种 33 个位置，共 132 次 full-vocabulary 比较，MAE/max-abs 均为 0；四组 greedy 32 token 完全一致。
- 两种配置的 acceptance 均通过：真实 KV、C4 隔离、4k/8k needle QA、流式 usage、取消/释放。
- 24 项单元测试通过；SHM 明确所有权与容量，响应在释放 executor 锁前独立复制。
- pinned micro 已证明单方向 64 MiB copy 约从 6.47 降至 2.6–2.9 ms，
  当前不引入异步 CUDA 生命周期复杂度；异步 copy 未实现、没有伪装为已验证。
- 保留现有 HTTP，暂不抽离二进制通道：先用已测出的 IPC/pack/window 改善作为新起点。
- 完整 32 工况 × 3 次与独立 Nsight 运行中；TP 1+1、混合 TP、异常故障注入的新增开关回归仍待验收。

测量代码：组合 anchor 为 `ccf3e42`，全开 anchor 为 `f2dcb97`；
`docs/summarize_stage1.py` 可从完整本地 trace 重建 `representative_summary.json`，
包含所有代表工况的互斥时间拆解、合并样本 P50/P95/P99 和重复间标准差。
完整 trace 共约 78.9 MB（含阶段 0 对照），不上传 Git；路径、大小与 SHA256 见
`representative_large_artifacts.json`。小型逐请求证据随报告提交。
完整矩阵以 `e831e65` 启动（仅恢复 OFF 的数组生命周期、对齐监听 backlog、补文档）。
原始证据保留各轮源码哈希及启动/测量环境；`anchor.json` 必须为 PASS，不能只凭零散成功点宣布完成。

```bash
.venv/bin/python scripts/baseline_matrix.py \
  --output results/optimization_stage1/full_matrix_all \
  --ipc-mode shm --wire-fast --tcp-buffer-mib 16
```

## 完整矩阵中间进展：TP 1+1 已完成

本节只是已完成拓扑的中间结果，不代表整个矩阵完成。TP 1+1 共 24 轮、288 个正式请求，
同 TP 正确性 PASS；20 个独立 profiling 请求均完成，两侧 Nsight/CSV 已生成。

| ISL/OSL | C | TTFT ms | TPOT ms | QPS |
|---|---:|---:|---:|---:|
| 512/128 | 1 | 75.24 | 28.60 | .2697 |
| 512/128 | 4 | 172.02 | 30.99 | .9736 |
| 2048/256 | 1 | 282.36 | 28.78 | .1312 |
| 2048/256 | 4 | 666.12 | 32.40 | .4480 |
| 8192/256 | 1 | 1335.25 | 29.54 | .1128 |
| 8192/256 | 4 | 3340.32 | 40.68 | .2917 |
| 2048/1024 | 1 | 282.29 | 28.72 | .0337 |
| 2048/1024 | 4 | 666.67 | 31.18 | .1228 |

独立 profiling 的 8k/C1 样本：prefill GPU 前段 86.43 ms、云中段 661.36 ms、后段 115.83 ms；
同步计时后的云 D2H 为 34.68 ms，和普通运行中含 GPU 等待的同名字段含义不同。
该 prefill 只有 1 个 profile 样本，不作为稳定延迟均值；255 个 decode batch 的 GPU 前/中/后段均值
为 2.46/13.92/4.26 ms，上传＋返回路径 10.80 ms。其 profiled step 33.40 ms，
高于普通 TPOT 29.54 ms，且两者边界不同；禁止将这些 GPU 分项直接拼进普通运行性能表。

原始来源：`results/optimization_stage1/full_matrix_all/tp_1_1/`。
此时完整矩阵仍在测试其余拓扑，所有 profile 二进制继续仅保留服务器。

## 阶段 1 收敛结论（验收范围更新）

用户明确取消阶段性 96 轮全矩阵要求：后续阶段以典型场景的功能、正确性、
生命周期与可重复收益验收；保留重要回退，不做全工况或 SLO 保证。

据已提交 TP2+2 代表场景三次重复、真实模型正确性/取消验收与独立微基准，
阶段 1 保留 shm、wire-fast、TCP16 三个 opt-in 开关，完成本阶段交付。
8k C1/C4 TTFT 降低约 44%，8k C4 QPS 增加约 23%；512 C1 TPOT/QPS 小幅回退，
因此不默认开启，也不声称所有场景受益。wire/shm 单项端到端归因仍未证明；
pinned 异步拷贝与独立二进制通道不纳入本次保留方案。

完整矩阵停留在 58/96 个完成轮次：TP1+1 和 TP1+2 各 24 轮，TP2+1 为 10 轮；
共 688 正式请求，已完成轮次成功率/固定输出长度校验通过。前两组有独立
Nsight 产物；矩阵没有 completion.json，目前无运行进程，不再续跑剩余工况。
这些额外本地产物不冒充完整矩阵，也不改变混合 TP 的 CROSS_TP_OBSERVATION 口径。
上文“运行中”“仍待完整矩阵”等为历史中间记录，由本节取代。

尚未完成新增开关的完整故障注入覆盖；已通过的真实取消/资源回收与组件测试
构成本次代表场景证据，不能据此声称所有故障情形已验收。后续 chunk/流水会
针对新增状态转换单独验证取消、错误传播和释放，不继承未经验证的结论。
