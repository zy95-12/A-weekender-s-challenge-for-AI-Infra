# Issue #6：Stage 1/2/3 叠加 PD 的 C16 验证

2026-09-06。**当前 4K 输入、79-token 输出下，PD 叠加 Stage 1/2/3 有明确的
C16 收益：QPS 从 2.089 提升到 3.560（+70.4%），平均 TPOT 降低 41.6%。**
这是固定 C16 的一次配对实验，不是最大 SLO QPS 或跨工作负载结论。

## 实验条件与结果

两组都是四张 A10、Qwen2.5-3B-Instruct 同一 revision、FP16、4/27/5 层切分、
企业—云每方向 10 Gbps/单向 5 ms。保留 shm、wire-fast、TCP 16 MiB、
prefill chunk 1024、decode-first quota=4。关闭 Stage 4、profiling 和 KV hash
诊断。PD 为企业 TP1、云 P TP2、云 D TP1；对照为企业 TP2、共享云 TP2。
PD 的 P/D 各有独立 window=2；对照使用原有全局 window=2。

持续闭环 16 个客户端，完成即补请求。每个点预热至所有客户端至少完成一次，
然后至少测 60 秒、每客户端至少完成 6 次稳态请求。QPS 排除预热和排空。
延迟 cohort 为测量窗口内完成的请求，同时检查窗口内发起请求的 SLO。

SLO：每请求 TTFT ≤3000 ms、首 token 后**平均 TPOT** ≤100 ms，至少 99% 请求达标。
这个定义不要求每一个 token 间隔都小于 100 ms。

| 指标 | Stage 1/2/3，TP2+2 | Stage 1/2/3 + PD，E1/P2/D1 |
|---|---:|---:|
| 正式请求数 | 126 | 214 |
| 测量窗口，秒 | 60.327 | 60.117 |
| QPS | 2.089 | **3.560** |
| TTFT 平均 / P99，ms | 922.91 / 2239.00 | **512.23 / 685.79** |
| TPOT 平均 / P99，ms | 87.41 / 95.00 | **51.04 / 53.47** |
| 平均端到端，ms | 7741.37 | **4493.63** |
| 完成 cohort SLO 达标 | 126/126 | 214/214 |
| 发起 cohort SLO 达标率 | 100% | 100% |
| 首二 token 间隔平均 / P99，ms | 51.56 / 67.76 | **75.47 / 146.51** |
| 所有 token 间隔合并后的 P99，ms | 129.37 | 96.95 |
| 每请求最大 token 间隔的 P99，ms | 134.02 | **223.78** |

两组正式请求均满足当前 SLO，因此本点的完成 QPS 与 goodput 相同。包括预热、
排空的全部 404 个请求，输出文本和 79-token 长度均与参考一致；两组独立的
greedy 诊断也逐 token ID 匹配此前 79-token 参考。

**收益不是没有代价：PD 的首二 token 间隔和少量请求的最大间隔变差。**
如果把 SLO 改成单个 token 间隔上限，这组数据不能沿用“100% 达标”的结论。
平均 TPOT 的改善不能掩盖交接处的停顿。

## 实际 KV 交接，而非理想重叠估计

对 PD 的 214 个正式请求，按 UUID 联结企业 front/back、KV source/destination
和 D-ready 时间线。所有请求均满足：

* D 全部接收完成之后才确认 KV ready；
* 企业首次 decode front 提交不早于 KV ready；
* P 源页回收不早于源传输完成和 D ready；
* prompt positions 为 0/1024/2048/3072，随后 decode 为 4096..4173，
  无缺失、重复或错位。

53.7% 的请求在企业首 token 就绪前已经完成 KV 交接。剩余请求仍有等待；
从企业最终 prefill back 完成计算起计，尚未隐藏的 KV-ready 等待平均为
20.83 ms、P99 为 90.94 ms（已隐藏的请求记为零）。

D 接收线程的平均 wall duration 为 33.54 ms，接收结束到 KV ready 平均为
6.36 ms。前者包含等待发送、通信、scatter 和运行时竞争，不能当成纯 PCIe
传输时间，也不能用此前隔离微基准的 7–8 ms 替代。这些可见交接成本是后续
优化首二 token 间隔的直接依据。

## 正确性与功能验证

独立开启 `--pd-verify-kv` 的诊断运行验证了实际 source KV 与 scatter 后的
destination KV，而非只检查接收缓冲区。E1/P2/D1 的 21 次迁移及 E1/P1/D2 的
6 次迁移，两个 global KV heads 的 SHA256 均一致。

覆盖当前 4K prompt、C16 正确输出、独立 257-token 非 block 对齐 prompt、
max_tokens=1、两种 prefill 期间断开连接，以及企业/P/D 的最终 KV 计数归零。
反向 E1/P1/D2 只做了功能验证，未做 C16 性能比较。4/30/2 为可选配置，本轮
PD serving 未评价它的端到端性能。

另以固定 teacher-forcing 序列，对比 E1/P2/D1 与企业 TP1/共享云 TP2：
每个 prompt 8 个 logits 位置，显式对齐为 `L-1 .. L+6`。第一行 prefill 的
logits 在两种 prompt 上均逐值一致；后续 decode TP 不同，不能要求逐值相等。

| 对齐的 prompt | logits 最大绝对差 | logits MAE | 最大 softmax 概率差 | top1 一致 |
|---|---:|---:|---:|---:|
| 当前 4K | 0.0703125 | 0.006503 | 3.02×10⁻⁸ | 8/8 |
| 独立 257-token | 0.0371094 | 0.004197 | 0.002624 | 8/8 |

第二组概率最大差约 0.2624 个百分点。这是小样本、跨 decode TP 的观测，不能
泛化为所有文本均无输出变化；KV 字节一致性和实际生成验证才分别对应迁移
正确性与当前工作负载正确性。

58 个 CPU 测试通过，包含 reservation 原子性、TP head 路由、未完成迁移
不得 commit/release、失败 shard、取消排空，以及 P window 满或另一个请求
等待 KV 时仍可调度已就绪 decode。NCCL 通信验证使用的是实际 vLLM 通信器。
跨进程硬故障的行为是显式报错并重启 worker groups，未声称自动恢复。

## 数据与复现

* [比较 CSV](evidence/issue-6/pd-c16/comparison.csv)
* [逐请求/时序审计](evidence/issue-6/pd-c16/audit.json)
* [逐绝对位置 logits/概率误差](evidence/issue-6/pd-c16/logits_comparison.json)
* [完整实验数据 ZIP](evidence/issue-6/pd-c16/raw_results.zip)
* [实现、启动参数与限制](pd-serving.md)

ZIP 包括逐请求 token 间隔、完整执行 trace、原始 logits NPZ、迁移哈希、配置、
网络检查、源文件哈希和诊断结果。未包含模型权重或 Nsight 文件。本轮结束后
恢复原有 Stage 1 演示服务。
