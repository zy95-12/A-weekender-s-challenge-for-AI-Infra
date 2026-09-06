# PD prefill 窗口与 chunk 大小实验

在 `--pd --pipeline-window 2` 下，新增 `--pd-prefill-window N` 只覆盖 P 在途窗口；D 保持 2。默认 0 继承原窗口，保持旧行为。允许 1..8；非 PD 模式拒绝此覆盖参数。

在途任务从企业端 front 开始到 back 完成才释放位置。它不是 GPU 数量、请求并发数或混合 GPU batch。HTTP 传输线程数同步调整为 P+D 窗口之和。KV 顺序、取消时先排空、D 等待 KV ready 的规则保持原样。

本轮模型 Qwen2.5-3B-Instruct FP16，4096 输入、自然结束 79 输出、并发 C24、E1/P2/D1、企业前 4/云 27/企业后 5 层。WAN 每方向 10Gbps、5ms。stage 1/2/3、独立控制通道开启，KV 按完整 prompt 迁移。max-active 96，KV blocks 32768。D 窗口 2，decode quota 4。

SLO：至少 99% 请求同时满足 TTFT <= 3000ms 和请求平均 TPOT <= 100ms，开始/完成两个 cohort 都检查。此实验是固定 C24 对比，不能直接声明最大 SLO QPS。

单独 Nsight 捕获用于诊断，开启 CUDA event 同步的阶段计时；所有性能点关闭 profiling。窗口 2/3/4 各至少 90 秒、每客户端至少 4 个正式周期；在最好通过 SLO 的窗口上测试 chunk 2048。必要时只对基准与候选做 180 秒确认，不做全工况矩阵。

运行：`.venv/bin/python -m scripts.pd_window_experiment --only profile`，随后 `--only performance`。结果在 `results/pd_window/`，脚本拒绝覆盖已有实验目录。每批结束恢复原工作目录的 Stage1 demo。

## Profiling 结论

在 C24 正式测量窗口内按 batch_id 关联 Nsight NVTX，P 两个 rank 各匹配 296 个 1024-token chunk。进一步按发起线程与 CUDA correlation ID 关联 GPU 工作，排除仅在时间上重叠的后台 KV 迁移；不能只按 GPU 时间落入 NVTX 区间来归属内核。

| 项目 | 每 chunk 平均 |
|---|---:|
| P forward 服务墙钟（排除 HTTP 排队） |60.04ms|
| cloud_middle_prefill CUDA event 区间 |53.08ms|
| cloud_receive CUDA event 区间 |1.28ms|
| 等待中间层完成后的 D2H 墙钟 |0.745ms|
| 实际 D2H CUDA memcpy（rank 0，8MiB） |0.445ms|
| GEMM 内核时间之和，各 rank |30.12–30.17ms|
| NCCL 内核时间之和，各 rank |18.30–18.99ms|
| 全部内核执行区间并集，各 rank |52.71–53.43ms|

所以旧 trace 的 `cloud_d2h_ms≈39ms` 大部分不能归因于实际设备到主机拷贝：非 profiling 模式前序 GPU 工作异步发出，`.cpu()` 才等待其完成。

CUDA event 阶段区间包括阶段内的执行间隙与 TP 同步；NCCL 内核可能包含等待，不是纯有效通信。内核时间之和可能跨 stream 重叠，也不可把两个 rank 相加当作延迟。阶段 event、服务墙钟和 kernel 分类是不同口径。未被这些 stage 字段覆盖的服务开销不再武断拆分为 IPC 或元数据。

该机 GPU 间拓扑是 PHB（经过 PCIe host bridge），没有 NVLink 链路；只作为测量环境记录，不据此声称已经确定 NCCL 开销的根因。捕获只覆盖 E 和 P；D 未启动 Nsight capture，本轮分析目标为 P。

## C24 首轮消融（profiling 关闭）

| P 窗口 / chunk | QPS | TTFT 均值 / P99 ms | TPOT 均值 / P99 ms | 双 cohort SLO 达标率 |
|---|---:|---:|---:|---:|
| 2 / 1024（本轮对照） |3.666|2405 / 2716|53.05 / 55.82|100% / 100%|
| 3 / 1024 |3.805|2208 / 2409|52.64 / 55.09|100% / 100%|
| 4 / 1024 |3.805|2171 / 2378|53.17 / 55.84|100% / 100%|
| 3 / 2048 |3.991|1750 / 2034|54.60 / 62.27|100% / 100%|

P 窗口 2→3：QPS +3.78%；3→4 没有可见的吞吐增益。P=3 下 chunk 1024→2048：QPS +4.90%。最终候选相对原配置 QPS +8.87%、平均 TTFT -27.25%、平均 TPOT +2.92%。这些是首轮数据，较长确认独立列出，不择优覆盖。

| P 窗口 / chunk | 第一次 front 前等待 ms | 开始后的 prefill 流水线跨度 ms | 每 chunk P 服务 ms | 每请求 P 服务成本估算 ms |
|---|---:|---:|---:|---:|
| 2 / 1024 |2050|354|60.67|242.66|
| 3 / 1024 |1801|406|61.44|245.78|
| 4 / 1024 |1711|460|61.66|246.64|
| 3 / 2048 |1258|492|116.85|233.70|

扩大窗口主要减少进入流水线前的等待，单请求进入后的跨度反而增长；chunk 2048 将每请求从 4 次云 forward 减至 2 次，按平均 chunk 服务耗时估算的每请求 P 服务成本下降约 4.9%，与额外吞吐收益方向及量级一致。此估算不把重叠阶段简单相加当端到端延迟。

chunk 2048 的逐 token ITL P99 是 89.98ms，原配置为 69.89ms；首二 token 间隔 P99 为 136.66ms，原配置为 84.08ms。当前 SLO 是请求平均 TPOT，不能据此声称所有单 token 间隔都低于 100ms。

## 独立重启后的 180 秒确认

| 配置 | 完成请求数 | QPS | TTFT 均值 / P99 ms | TPOT 均值 / P99 ms | 双 cohort SLO 达标率 |
|---|---:|---:|---:|---:|---:|
| P 窗口 2 / chunk 1024 |651|3.613|2667 / 2956|50.99 / 54.00|99.69% / 99.69%|
| P 窗口 3 / chunk 2048 |724|4.019|1503 / 1883|57.29 / 61.23|100% / 100%|

确认轮 QPS +11.23%、平均 TTFT -43.63%、平均 TPOT +12.34%。两个配置都满足联合 SLO，但候选以更高 TPOT 换取更低 TTFT 和更高吞吐。候选逐 token ITL P99 为 94.61ms（原配置 77.78ms），首二 token 间隔 P99 为 151.51ms（原配置 95.56ms）。

两轮方向一致：候选相对原配置 QPS 提升约 8.9%～11.2%；TTFT 降低，TPOT 增加。推荐限定于本次 C24、4K/79-token 工作负载及既定 SLO；尚未重新扫描该配置的最大 SLO QPS，不扩展为任意并发或输入分布的结论。

四个首轮性能点、两个确认点及一个独立 profiling 点，共 3162 条请求（含预热和排空）通过输出、绝对位置、KV ready 顺序、联合 SLO 复算及排空检查；63 项 CPU 测试通过。性能点均关闭 profiling，正式测量源代码哈希核对通过。

## 复算与原始证据

- `scripts/pd_window_experiment.py`：启动、固定网络检查、预热、持续负载、排空、配置和 trace 归档、恢复 demo。
- `scripts/audit_pd_window.py`：校验原始输出、79-token 长度、绝对位置、KV ready 顺序、联合 SLO 两个 cohort、服务排空、健康采样及测量源代码哈希，导出 comparison.csv。
- `scripts/analyze_pd_nsys.py`：从 cloud_prefill.sqlite 关联 NVTX、CUDA launch、kernel 和 memcpy，导出 kernel_summary.csv 和 nsys_breakdown.json。
- `scripts/plot_pd_window.py`：使用独立 matplotlib 环境生成首轮比较图。
- `scripts/package_pd_window.py`：原始数据和复算源码归档，保存 SHA256、校验 ZIP CRC。原始 SQLite 可从 nsys-rep 再导出，不重复归档。

原始文件目录：`results/pd_window/`。导出命令：`nsys export --type sqlite --output results/pd_window/profile-p2-ch1024/cloud_prefill.sqlite results/pd_window/profile-p2-ch1024/cloud_prefill.nsys-rep`。随后运行 `.venv/bin/python scripts/analyze_pd_nsys.py` 和 `.venv/bin/python scripts/audit_pd_window.py`。

ZIP 内 `raw/` 是原始数据，`source/` 是代码；独立复算时将 raw 放到 source/results/pd_window，使用相同 Python 依赖。重新运行 GPU 实验还需原模型权重及四张 A10，权重不包含在归档中。

## 候选配置

```bash
./poc up --wan --pd \
  --ipc-mode shm --wire-fast --tcp-buffer-mib 16 \
  --prefill-chunk-size 2048 --scheduler-policy decode-first --decode-quota 4 \
  --pipeline-window 2 --pd-prefill-window 3 \
  --max-active 96 --kv-blocks 32768
```

这是 E1/P2/D1，P/D GPU 数量保持 2/1；stage 1/2/3 开启、独立控制通道开启、整段 KV 迁移、stage4 关闭。P=3 在这里仅指在途窗口。运行脚本最终恢复原 Stage1 demo，不自动将此实验配置长期部署。

证据：[首轮对比图](evidence/issue-6/pd-window/comparison.png)、[全部测量 CSV](evidence/issue-6/pd-window/comparison.csv)、[审计结果](evidence/issue-6/pd-window/audit_summary.json)、[内核汇总](evidence/issue-6/pd-window/kernel_summary.csv)、[原始证据 ZIP](../results/pd_window/evidence.zip)、[ZIP 哈希](evidence/issue-6/pd-window/archive.json)。
