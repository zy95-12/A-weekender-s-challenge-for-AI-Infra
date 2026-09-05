# PR #8 测量可信度修复

针对 [review 评论](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/pull/8#issuecomment-5550778241)。
本轮只修复测量流程，不优化推理性能，不重新生成或修改历史 baseline。

## 改动

- `up` 复用 GPU 服务前重新应用并检查网络；`wan/local` 检查成功后同步当前网络与启动记录。
- WAN 支持 `--delay-ms`、`--bandwidth-gbps`，保留 `poc wan 5` 和 `up --delay 5` 的兼容性。
- `network-check` 检查两侧实际 qdisc、ping 丢包/RTT、iperf TCP 吞吐；失败退出并保存 FAIL 证据。
  默认 RTT 容差为 2 ms，吞吐范围为配置值的 80%–110%。这不涉及推理 SLO。
- 每个 benchmark 点独立执行前置网络检查，并核验测量后 qdisc/网络意图；
  本项目网络修改、检查、benchmark 之间采用互斥锁。缺少网络意图或无法读取 qdisc 时拒绝继续。
- `validate_poc.py` 总体 PASS 之前必须通过网络检查；新运行先将总体状态置为 RUNNING，避免遗留 PASS。
- 启动环境与测量环境分别保存，测量环境重新采集当前源码 SHA-256、Git commit/dirty 状态和软件环境。
- 正式请求 trace 按精确请求 ID 过滤；非正式记录单独保存，不猜测其一定来自 warmup。
  官方客户端性能统计不变；遥测仍包含 warmup，已明确标记采样范围。
- 不覆盖已有实验点或汇总；新实验必须使用新目录。

## 回归

18 项单元测试通过（新增 11 项，原有协议/KV 测试 7 项），同时通过 Python 编译、shell 语法与 diff 空白检查。
网络验证阻断总体 PASS 的测试使用 mock，不代表重新执行了三种切分的完整模型验收。

真实服务器回归使用原有 Qwen2.5-3B-Instruct、4/27/5 层切分、TP 2+2，未重启 GPU 服务：

1. `poc local → poc up --wan`：服务复用时恢复两侧 TBF/netem。
2. `poc wan --delay-ms 10 --bandwidth-gbps 1`：实测 RTT 20.023 ms、TCP 吞吐 0.938577 Gbps，通过检查。
3. 故意通过外部 tc 删除企业侧 root qdisc，保留 WAN 意图：benchmark 非零退出，
   `network_check.json` 为 FAIL（实际 noqueue），未生成性能汇总。
4. 恢复默认 10 Gbps、单向 5 ms；真实 `(ISL, OSL)=(512,128)`、2 请求、并发 1 smoke 通过。
   首轮正式 trace 256 行，非正式记录另存 128 行；实测 RTT 10.015 ms、TCP 吞吐 9.417484 Gbps。
5. 完成代码修改后，另以 `wan_smoke_final` 目录重跑相同 smoke，避免将中间版本的源码哈希当作最终版本证据。
   结果 PASS，2/2 请求成功，正式/排除 trace 分别为 256/128 行；RTT 10.016 ms、TCP 吞吐 9.414417 Gbps。
   测量环境中的 26 个源码/入口文件哈希与最终本地文件逐一核对一致。

本地证据目录为 `results/review_fix_20260905/`：

- `network_1gbps_20ms.json`：非默认参数的实际网络检查。
- `rejected_mismatch/qps_inf/network_check.json`：故障注入的失败证据。
- `wan_smoke/`：第一轮真实推理回归。
- `wan_smoke_final/`：最终源码版本的真实推理回归，包含完整客户端命令、测量/启动环境、网络检查及前后快照、请求指标和分离的 trace。

本次 PR 纳入 `wan_smoke_final/` 的完整小规模文本证据、`network_1gbps_20ms.json` 和
`rejected_mismatch/qps_inf/network_check.json`；中间版本的 `wan_smoke/` 仅留在服务器。
测量环境如实保留测量时的 Git commit 和 dirty 状态；源码 SHA-256 对应本次提交的代码，未改写历史 provenance。
权重和大型 profiling 文件未纳入本次改动。
两请求仅用于修复回归，不作为新的充分性能 baseline。历史 `validation_final` 缺少的网络证据未补造；
原有 96 轮 baseline 未重跑，其记录对应当时版本而非本次修复版本。

## 边界

互斥锁只约束本项目命令，不能阻止外部管理员直接操作 tc。前后快照无法证明中途从未发生过改变再恢复。
测量时应独占测试服务与网络，避免外部流量或源码修改。
非正式 trace 的 `is_warmup=null` 表示无法区分初始预热和其他客户端流量，不能纳入正式请求建模。
`measurement_validation.json` 的 PASS 是测量检查通过，不表示推理 SLO 达标或重新通过完整模型正确性验收。
