# 架构与运行边界

## 目录

- `split_serving_sim/`：标准库运行时、配置、算子/通信成本、调度和指标。
- `configs/`、`models/`：实验描述与 HF 模型结构 JSON，无权重。
- `profiles/`：带来源的校准表，适用范围见 [校准清单](CALIBRATION_AND_LIMITS.md)。
- `tools/`：数据转换、真实源码决策对照、可移植发布回归。
- `tests/fixtures/`：从真实 serving 模块生成的决策 oracle。
- `docs/validation/`：最终校准和误差证据，不随运行覆盖。
- `outputs/`：每次运行生成，git 忽略。
- 仓库根目录 `demo/`：现有完整 Web 演示，调用通用 Q4 后端。

## 两个调度后端

`behavioral` 通过 DAG 和离散事件实现分区执行、资源排队、TP/PP/replica、批处理与 KV 管理，用于通用实验、baseline 回退及历史对照。它不是原生 vLLM 完整内部调度器。mixed batch 等研究能力属于这个后端。

`serving` 加载 `vendor/serving` 中固定版本的真实 `PDScheduler`、`PipelineScheduler.run()`、`choose()`、`KVAdmission` 方法体：不改写方法内部决策。加载器仅选择类/函数定义，跳过应用依赖导入、GPU/HTTP 启动和线程初始化。每次运行有独立 globals，不 monkeypatch 全局标准库。

快照版本 `34c809ca88b8b796f07cccfe55066db0b74cf7f8` 与 main 已合入的 serving 对应，文件 SHA256 记录于 manifest。快照允许只安装 Q4 后仍能运行；本仓 serving 未来修改时，不会自动偷偷改变仿真。同步流程：更新快照文件和 manifest → 运行真实模块/快照决策对照 → 运行测试和发布回归。

虚拟执行接口负责时钟、worker 串行 lane、RPC Future、WAN 数据路径、KV 传输与 CPU 收尾。真实企业调度仍按以下顺序工作：

```text
admit / observe futures → release → ready back → eligible front → wait / sleep
```

阻塞 front/back 调用推进时钟，云侧事件可并行完成；企业调度在返回后才能继续观察 Future。CPU 收尾在模型返回后推进时间，不能再加到 kernel 成本中。

这是企业业务调度的复用，不是执行真实云 HTTP server、PDControl 线程代码或 GPU kernel。云端锁交接尚有已知差异。

## 三种成本后端

- `roofline`：按模型结构、融合算子和硬件配置估算计算/内存/collective 成本。
- `profile`：同签名采用算子校准，其他 shape 使用同类修正或解析回退，输出覆盖率。
- `command`：完整 forward 调用耗时，包括主机准备、IPC、搬运和部分内部等待；因此关闭重复的 host submission 计费，并移除通信模型里的 D2H/H2D/cloud IPC 重复项。

command 的 `mean` 固定均值；`empirical` 按实测分布采样，保持每个 shape 的期望值。缺失 batch 沿用原插值/外推均值，用相邻 batch 的归一化波动作为近似，单独报告 residual coverage。独立随机数和显式 seed 保证复现。

## 配置

JSON 顶层字段为 `model`、`topology`、`hardware`、`network`、`data_path`、`execution`、`static_policy`、`attention_backend`、`operator_backend`、`scheduler`、`workload`、`slo`、`simulation`；完整字段和合法性检查在 `config.py`，可运行例子见 `configs/example.json`。

预设不是自动优化器：`configure_serving` 将 CLI 开关编译成布局和调度参数。optimized 的若干值会覆盖输入配置，详见校准清单，不能假设所有 JSON 值都被原样保留。解析后的配置写入 `resolved_config.json`，开关写入 `serving_features.json`。

serving 当前要求固定长度 closed-loop、PD+pipeline、PP1、一或两个 P replica、warmup=0 或 concurrency；没有 mixed batch/preemption。TP 成本能否使用由 sample 的 TP 和范围决定，提供的当前最佳实测表主要为 TP1。关闭 PD/pipeline 时应选 behavioral 回退。

## 指标与 trace

TTFT 从请求产生到首 token；TPOT 为首 token 后的平均间隔。固定测量窗按完成请求选择 cohort，warmup 和 drain 不计入测量集合。输出 QPS 是完成吞吐，不自动代表满足某个 SLO 的最大 QPS；目前没有自动最大 QPS 搜索。

serving 中的 token 为占位符，验证的是事件与时间，不是语言模型正确性。trace 的 command 时间不是 GPU kernel-only；host/worker 等待可能包含在其中。serving 的通用资源利用率/平均 batch 汇总目前为空，应使用执行和 serving trace 做同口径分析。

`simulation.trace_enabled` 控制 Gantt 生成；serving 后端仍收集完整执行/决策记录，尚未实现按 trace 数量上限裁剪，因此大任务需要注意内存。demo 单独限制地址空间并串行执行请求，但这不是完整资源隔离系统。
