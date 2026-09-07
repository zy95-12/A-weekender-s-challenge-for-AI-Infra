# 当前验证与精度边界

## 功能/回归

从 Q4 运行 `python3 -m unittest discover -s tests -q`。当前 108 项 CPU 测试覆盖配置、DAG、batch/窗口、PD admission、原源码决策 oracle、乱序响应、CPU/GPU 提交重叠、分布采样及 post-back CPU 工作。真实模块与源码快照使用相同事件序列的 oracle 在 `tests/fixtures/serving_decisions.json`。

`python3 tools/validate_release.py` 在干净 checkout、无 GPU 和外部 `/root/...` 目录依赖下复现 baseline C1/C8/C16 以及 PD seed 17/29/43。该检查是数值回归，不是重新采集硬件数据。输出保存在 `outputs/release-validation`。

## 当前优化 PD

Qwen2.5-3B、FP16、4/27/5、A10，企业 TP1+双 P TP1+D TP1，4K/79、C40、90 秒测量。stage1/2/3、P/D window3/2、quota4；KV control channel 开，chunk migration 关。

| 指标 | 原 90 秒实测 | 当前仿真，3 seed 均值 | 相对误差 |
|---|---:|---:|---:|
| 完成 QPS | 5.1613 | 5.3185 | +3.05% |
| TTFT ms | 793.913 | 709.452 | −10.64% |
| TPOT ms | 89.282 | 86.783 | −2.80% |

固定 command 均值、未加 CPU 收尾时为 5.8667 / 1060.322 / 73.822；仅恢复耗时分布时为 5.6000 / 778.670 / 81.324。不能把 CPU 收尾毫秒数直接线性加到 TTFT/TPOT，调度反馈改变了完成顺序。

当前三 seed：

| seed | QPS | TTFT ms | TPOT ms |
|---|---:|---:|---:|
| 17 | 5.2889 | 716.980 | 87.048 |
| 29 | 5.3333 | 716.417 | 86.418 |
| 43 | 5.3333 | 694.959 | 86.884 |

完整输出：[校准与对照数据](validation/serving_cost_distribution/)。GPU command 表来自原 90 秒 run，CPU 收尾来自随后 30 秒 probe；不是独立验证集，也不是最大 SLO QPS 搜索。

## TTFT 残差

| 主要边界 | 实测−仿真 ms |
|---|---:|
| 请求进入→首个 front | 28.504 |
| 首轮上传 | 13.722 |
| 最后一轮下载 | 24.707 |
| 两轮 P 调用之间 | 8.908 |
| 首轮 P 入口排队 | 6.733 |

总差 84.461 ms；四段模型调用合计高估 1.043 ms。上传/下载含主机处理，不等于纯 WAN。详细逐位置分解：[CSV](validation/serving_ttft/components.csv)。新 30 秒 probe TTFT 709.807 ms 看似接近模型，但仍有上下行少算约 32.59 ms 与 P 排队多算约 30.16 ms 等抵消，不能只看总数。

CPU 收尾 probe 为 30.216 秒、152 请求，输出全部与 reference 一致；352 个 decode batch 的模型返回后工作均值 3.636 ms，其中 trace 3.282 ms、token events 0.170 ms、其他 0.184 ms。数据在 `post_back_samples.csv` 和 `post_back_probe.jsonl.gz`。探针代码提交 `fe3e4c5`；没有新增 GPU 同步，也没有额外无打点对照来分离本次探针影响。

## Baseline 回退 ≠ 精度通过

C1/C8/C16 的当前回归值：

| 并发 | QPS | TTFT ms | TPOT ms |
|---|---:|---:|---:|
| C1 | 0.416667 | 726.953809 | 20.996945 |
| C8 | 0.933333 | 3271.292142 | 56.187717 |
| C16 | 1.066667 | 6179.107380 | 96.111258 |

相对历史仿真缓存差值均为零。原实测 baseline C1 TPOT 32.433 ms、C8 67.889 ms、C16 106.975 ms，因此 TPOT 误差仍分别约 −35.26%、−17.24%、−10.16%。不能用“回归不变”替代准确性声明。

## Demo 验证与边界

根目录运行 `python3 -m unittest demo/test_demo.py -v`；启动 `python3 demo/server.py --host 127.0.0.1 --port 8088`，调用 README 的 `/api/simulate` 示例。这会跑真实 Q4 解析计算。未支持模型/硬件返回 422；假对话与安全回放不是新 GPU 实验。

根目录真实 serving 的完整测试需要其安装环境；Q4 本身不依赖这些包。主线发布应分别运行 Q4、demo、根目录 serving 测试，避免两个名为 tests 的包相互遮蔽。

## 数据与历史

原始多 GB Nsight 和全部中间试验没有纳入本 PR。部分 profile provenance 保留采集机绝对路径，是来源标签，不是仿真运行依赖。需要重新构建 profile 时通过工具参数传入对应原始 trace。

原始分析完整记录保留于 `061f10b` 及其祖先提交。当前工具/数据整理不改变已校准的成本和数值；新增模型/硬件仍按 [校准清单](CALIBRATION_AND_LIMITS.md) 独立验证。

## 本次主线整理验证

Q4 99 项、demo 2 项、根目录 serving 78 项测试通过；JavaScript 语法检查通过。demo HTTP 实际调用 Q4 的 C1/C2 点返回成功，未支持模型返回 422，结果见 [smoke 数据](validation/demo_api_smoke.json)。wheel 构建后在独立 venv、仓库外目录执行 serving+empirical+host 路径成功；快照资源包含在安装包中。

## 开环独立验证

已回放 11 个真实开环到达计划，结果和复现见 [开环误差报告](OPEN_LOOP_VALIDATION.md)。沿用原成本表，不能将旧闭环校准精度直接视为开环容量预测精度。

新增8组GPU重复验证及19组MAPE/偏差汇总见[重复验证报告](REPEATED_OPEN_LOOP.md)。公开硬件与模型选项已做容量隔离和两后端prefill/decode功能检查，尚无跨设备GPU实测精度。
