# PD 推荐配置与 baseline：固定 SLO 并发扫描

目标是在同一 4K/79-token 负载下，区分吞吐趋平的位置与 SLO 的并发上边界。复用 PR #17 的 serving，不修改调度策略或模型计算。

## 两组系统

| 设置 | Baseline | 推荐配置 |
|---|---|---|
| GPU | 企业 TP2＋共享云 TP2 | 企业 TP1＋P TP2＋D TP1 |
| 模型 | Qwen2.5-3B-Instruct，FP16，4/27/5 切层 | 相同 |
| Stage 1 | 关闭 | shm、wire-fast、TCP 16 MiB |
| Stage 2 | 整段 prefill、legacy 调度 | chunk1024、decode-first、quota4 |
| Stage 3 | 关闭 | window2，P/D 独立窗口 |
| 独立控制通道 | 无 PD | 开启 |
| KV chunk 迁移、Stage 4 | 关闭 | 关闭 |
| Operator profiling、KV 哈希 | 关闭 | 关闭 |
| 活跃请求上限 / KV 页数 | 96 / 32768 | 96 / 32768 |

模型 revision 为 `aa8e72537993ba99e69dfaafa59ed015b17504d1`；4×NVIDIA A10。网络双向各 10 Gbps、单程延迟 5 ms；PD 的 KV 在云内 NCCL 传输。

原启动器仅允许 max-active≤16，默认 8192 个 KV 页也不足以支撑高并发。此实验只扩展启动器容量参数：允许 max-active 到 128，并透传已有 server `--kv-blocks` 参数；默认值保持不变。实际两组均固定上限 96、32768 页，每页 16 tokens。4K 输入、max_tokens128 的 96 个请求最多预留 25344 页，不会因本次 KV 容量上限而提前排队。实际运行使用低于该上限的客户端并发；扩容没有改变企业端调度、P/D 窗口或每批 token 预算。

## 负载与 SLO

使用此前同一 4096-token prompt，正常 EOS 产生 79 tokens。每个并发点重新验证 79 个 greedy token IDs，所有持续请求核对完整文本及 token 数。禁用 prefix caching。

客户端采用持续闭环：每个客户端收到一个完整响应后立即发下一条请求。逐步增加客户端数量，记录真实完成 QPS。SLO 定义为 **至少 99% 的请求同时满足 TTFT≤3s 且请求平均 TPOT≤100ms**。TPOT 为该请求首、末 token 时间差除以 78；不是单个 token 间隔上限。报告另外保留逐 token 间隔与每请求最大间隔。

- 粗扫：每个客户端至少完成一次预热；正式窗口至少 45 秒、每客户端至少 3 个完成周期。
- 边界细化：在最大通过 C 与下一个失败 C 间细化到相邻整数，至少 60 秒、4 个周期。
- 边界确认：候选最大通过并发、候选最高通过 QPS 及紧邻失败并发，至少 180 秒、6 个周期。若更长窗口推翻短窗口结果，再重复一次有争议的边界：重复通过才采用该 C；重复失败则标为不稳定、采用下一档已确认通过 C。先前窗口全部保留。
- QPS 分母是正式窗口时长，分子是窗口内完成请求。延迟也按该完成 cohort 统计；另检查窗口内开始的请求，全部等完成后核对 SLO。两种 cohort 都必须≥99% 才算通过。
- 每点保存所有请求的原始时间、文本、逐 token 间隔、每 0.5 秒的活跃/KV/排队观测。每点前后检查网络整形，所有任务完成后确认 KV 清空。

图中每个并发只显示一个点：确认窗口优先，其次细化窗口，再次粗扫；同一 C 有重复确认时采用最后一次，不挑 QPS 更高的窗口。先前窗口在 TTFT/TPOT 图中用淡色标记保留。所有重复测量均保留在 `all_points.csv`，不会把失败点删掉。星标是确认测量中满足 SLO 的最高 QPS；吞吐曲线中的更高失败点不能算作 SLO 容量。

这是有限持续窗口下、当前重复 4K/79-token 负载的实测上限，不是任意负载的容量保证，也不是开放到达流量或统计置信度认证。

## 复现

在本机本工作树下，原主工作树仍用于实验后恢复演示服务，基准 prompt/reference 来自先前 PD 测量。脚本会保存启动配置并检查计算源文件前后 SHA256 不变：

```bash
.venv/bin/python -m scripts.pd_slo_sweep --variant optimized
.venv/bin/python -m scripts.pd_slo_sweep --variant baseline
.venv/bin/python -m scripts.pd_slo_report
# 使用独立安装 matplotlib 的 Python；依赖版本随证据发布。
/root/pd-slo-plot-env/bin/python -m scripts.plot_pd_slo
.venv/bin/python -m scripts.package_pd_slo
```

每次 sweep 拒绝覆盖已有同名目录；若换机器，需要修改 runner 中主工作树与历史 prompt/reference 路径。运行两组必须串行，不能同时占用 GPU 和网络。原始 trace、网络验收、配置、SHA256、复算和绘图脚本均包含在证据包中。
