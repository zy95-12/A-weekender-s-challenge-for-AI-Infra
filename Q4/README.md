# Split-inference serving simulator

第一版 split inference 研究仿真模型：输出完成 QPS、TTFT、TPOT、逐请求 token 时间线和执行 trace。支持通用行为调度，以及直接执行本仓真实 PD 调度源码的虚拟后端。

当前主要校准场景为 Qwen2.5-3B-Instruct、FP16、4/27/5 分层、A10、4K 输入/79 输出。优化 PD 为企业 TP1 + 两个 P TP1 + 一个 D TP1。尚未通过跨模型/硬件泛化或精确 SLO 容量验收。

- [架构与配置边界](docs/ARCHITECTURE.md)
- [校准、硬编码、迁移要求和未解决问题](docs/CALIBRATION_AND_LIMITS.md)
- [验证结果与复现](docs/VALIDATION.md)
- [开环实测回放与误差报告](docs/OPEN_LOOP_VALIDATION.md)
- [完整 Web demo](../demo/README.md)

## 快速运行

Python 3.11+，仿真和 demo 无第三方运行依赖，无需 GPU、模型权重或 vLLM。

```bash
cd Q4
python3 -m split_serving_sim --config configs/example.json --output-dir outputs/example
```

仓库根目录也可以 `pip install ./Q4` 后使用 `split-serving-sim`。安装包包含调度源码快照；模型描述、配置和成本表使用本仓 Q4 下的文件。

### baseline 回退

```bash
python3 -m split_serving_sim \
  --config configs/issue6_baseline_host.json --preset baseline \
  --scheduler-backend behavioral --cost-model profile \
  --concurrency 8 --num-requests 512 --warmup-requests 8 \
  --measurement-duration-s 60 --output-dir outputs/baseline
```

baseline 预设保留输入配置，成本开关也保留；上例仍启用历史 CPU submission/WAN/operator 校准。回退不等于禁用所有成本修正。

### 当前 PD 模型

```bash
python3 -m split_serving_sim \
  --config configs/issue6_baseline_host.json --preset optimized \
  --scheduler-backend serving --cost-model command \
  --command-profile profiles/issue6_pd_tp1_c40_empirical_commands.json \
  --command-sampling empirical --cost-seed 17 \
  --serving-host-profile profiles/issue6_c40_serving_host.json \
  --concurrency 40 --num-requests 1024 --warmup-requests 40 \
  --measurement-duration-s 90 --output-dir outputs/pd-c40
```

输出 `summary.json`、`metrics.json`、`resolved_config.json`、`serving_features.json`、`requests.jsonl`、`trace.jsonl`。serving 另输出 `serving_trace.jsonl` 和 `decisions.jsonl`；`serving_request_id` 用于连接请求指标与原 trace。配置 `simulation.trace_enabled=true` 时生成 `gantt.html/svg`。

仿真使用占位 token，不验证模型输出正确性。真实模型 GSM8K/logits 验证使用仓库根目录 serving 工具。

## 开关

| 开关 | 作用 |
|---|---|
| `--preset baseline/optimized` | 基准/当前优化预设 |
| `--scheduler-backend behavioral/serving` | 行为模型/固定版本真实 PD 调度 |
| `--cost-model roofline/profile/command` | 解析算子/算子校准/完整 command 成本 |
| `--stage1`、`--chunked-prefill`、`--pipeline` | 数据路径、分块 prefill、异步流水 |
| `--decode-first`、`--decode-quota` | decode 优先与连续轮数 |
| `--pd`、`--prefill-replicas`、`--prefill-tp`、`--decode-tp` | PD 布局 |
| `--prefill-window`、`--decode-window` | 在途窗口 |
| `--chunk-transfer`、`--control-channel` | KV 提前迁移与独立控制 lane |
| `--pd-admission` | reserve/release 生命周期成本 |
| `--command-sampling mean/empirical`、`--cost-seed` | 均值/实测分布抽样 |
| `--serving-host-profile FILE` | CPU 收尾成本；不传即关闭 |

布尔开关支持 `--no-*`。serving 要求 PD+pipeline、PP1、闭环或开环、1–2 个 P replica，无 mixed batch/preemption。关闭 PD/pipeline 回退时选 behavioral。`--no-pd-admission` 在 serving 将 reserve/release 成本置零，仍保留真实准入顺序，不删除依赖。

## 验证和校准工具

在 Q4 目录运行，避免与根目录真实 serving 的 `tests` 包混淆：

```bash
python3 -m unittest discover -s tests -q
python3 tools/validate_release.py
```

第二条覆盖 baseline C1/C8/C16 和 PD 三个固定种子，无 GPU/外部目录依赖。验证结果不是重新证明模型预测精度，而是确保收敛未改变已记录结果。

`tools/build_issue5_profile.py` / `build_issue5_wan_calibration.py` 转换算子/WAN 数据；`build_pd_command_profile.py --include-distribution` 保留逐命令样本；`build_serving_host_profile.py` 提取 CPU 收尾成本。原始采集输入通过参数提供。

最终证据在 `docs/validation/`，测试 oracle 在 `tests/fixtures/`。历史排查及中间数据保留于提交 `061f10b` 所在分析分支，不是主线运行依赖。已有 ISSUE*/PR8* 文档为历史研究记录，当前状态以本页链接的三个文档为准。

## 开环负载与实测回放

开环按独立时间表到达，完成请求不触发补发；服务端活动窗口继续限制实际准入。
在上面的 PD 命令中，将 `--concurrency/--num-requests/--warmup-requests` 替换为：

```bash
--workload-mode open_loop --arrival-process poisson --arrival-rate-qps 4.440035098628993 \
--random-seed 17 --warmup-duration-s 30 --measurement-duration-s 120 --arrival-tail-s 30
```

`num_requests` 在开环下不控制请求数。也可在 JSON 的 `workload.requests` 提供精确的
`request_id/arrival_time_ms/input_tokens/output_tokens` 时间表，此时优先回放时间表。
`concurrency` 和 `warmup_requests` 必须为 0；CLI 切换到开环时自动清除继承的闭环值。
请求数由到达率与持续时间决定，过载不会降低发包率。停止发包后继续排空，超过
`simulation.max_time_s` 则报错，不将未完成请求丢出统计。

统计窗口采用 `[warmup, warmup + duration)`：顶层 QPS 来自窗口内完成请求，
TTFT/TPOT 来自窗口内到达请求（包含窗口后才完成的慢请求）。`arrival_cohort`、
`completion_cohort` 分别保留完整指标，两者联合 SLO 达标率均 ≥99% 才通过。
TTFT 从计划到达时刻起算；另输出平均/峰值在途请求数和边界积压。

从仓库根目录复现实测的 11 个开环点，无需 GPU：

```bash
python3 -m zipfile -e demo/evidence/open-loop-raw.zip /tmp/measured-open-loop
python3 Q4/scripts/compare_open_loop.py \
  --measured-root /tmp/measured-open-loop --output-dir /tmp/predicted-open-loop
```

可用 `--point optimized/refine-4.440` 只回放一个点。比较器固定沿用旧成本表，
将两系统的服务端活动上限设为实测的 96、KV blocks 设为 32768；不拟合本次结果。
输出逐请求预测、解析后配置、双 cohort 汇总、带符号相对误差 CSV/JSON。
Demo 06 的并发输入仍是闭环接口；开环入口为此 CLI 和回放工具。
