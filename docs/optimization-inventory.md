# 当前优化清单（2026-09-06）

本分支已整合为面向 main 的 serving 交付，包含下表各阶段实现及 WAN 标定工具；投机推理排除。当前启动、baseline 回退及 GSM8K 同位置 logits 验收以 [集成说明](integrated-serving.md) 和 [验证报告](gsm8k-validation.md) 为准。下文 PR 链保留为历史开发索引。

已创建 PR 不等于已合并。当前这些优化采用依赖式 PR；本分支包含 #8→#9→#10→#11→#12→#14→#16→#17→#18→#19 的实现。Stage 4 位于 #13 的独立分支，不包含在当前 PD 运行路径。

| 特性 | 行为及当前选择 | PR |
|---|---|---|
| Stage 1 数据路径 | shm 减少进程间大数组复制；wire-fast 减少打包复制；TCP buffer 16 MiB。开启 | [#10](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/pull/10)、[#14](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/pull/14) |
| Stage 2 切块与基础调度 | chunk 2048；decode-first、quota 4；按轮连续接纳 decode 并合批。开启。首块 block table 修正已进入 #14 | [#11](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/pull/11)、[#14](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/pull/14) |
| Stage 3 异步流水 | 跨请求重叠企业前层、WAN/云计算、企业后层；有界在途任务及顺序保护。开启 | [#12](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/pull/12)、[#14](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/pull/14) |
| Stage 4 投机推理 | prompt-lookup 草稿、精确 q=1 验证；批量验证优化未完成。当前关闭，不声称已集成 PD | [#13](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/pull/13) |
| 云侧 PD 分离 | P/D 独立实例、请求调度、KV 预留/迁移/ready/释放；复用 vLLM 通信组件。开启 | [#16](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/pull/16) |
| 独立控制通道 | readiness 控制不排在正常 GPU forward 命令后。开启 | [#17](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/pull/17) |
| chunk KV 提前迁移 | 支持按已完成页提前传输。当前关闭，使用整段 prompt KV 迁移 | [#17](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/pull/17) |
| 独立 P 窗口 | 每个 P 在途窗口 3，D 全局窗口 2。窗口不是 GPU batch 大小 | [#19](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/pull/19) |
| P 副本扩展 | 两个 TP1 P，按剩余 prefill token 负载分配；请求固定 P；独立 KV 迁移通道，共享 D KV pool。开启 | [#20](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/pull/20) |
| 控制连接恢复 | 控制/数据连接池分离，明确空闲期限；仅幂等控制操作在连接异常时新建连接重试一次；forward 不重放 | [#20](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/pull/20) |
| 算子 profiling | input shape、dtype、算子及时间采集开关；性能测试关闭 | [#14](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/pull/14) |
| SLO 测量 | 开始/完成 cohort、逐请求 TTFT/TPOT、输出与绝对位置审计 | [#18](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/pull/18)、#19、本分支 |

WAN 曲线探针及 baseline/Stage1 测量数据已单独提交 [#21](https://github.com/zy95-12/A-weekender-s-challenge-for-AI-Infra/pull/21)，基于 #14，为可选仿真标定工具，不是 PD 运行依赖。当前 PR 整理重新运行双 P 分支 71 项 CPU 测试、WAN 分支 51 项 CPU 测试，均通过；没有重复 GPU 压测。

## 当前推荐配置及实测边界

Qwen2.5-3B-Instruct FP16；四张 A10；企业前4/云27/企业后5层；E1 + P0(TP1) + P1(TP1) + D(TP1)。WAN 每方向 10 Gbps、单向 5 ms。

```bash
./poc up --wan --pd --prefill-replicas 2 --prefill-tp 1 --decode-tp 1 \
  --ipc-mode shm --wire-fast --tcp-buffer-mib 16 \
  --prefill-chunk-size 2048 --scheduler-policy decode-first --decode-quota 4 \
  --pipeline-window 2 --pd-prefill-window 3 --max-active 96 --kv-blocks 32768
```

4K 输入、79-token 输出、C40：5.161 QPS；TTFT 平均/P99 794/1353 ms；请求平均 TPOT 的平均/P99 89.28/93.84 ms。两个 cohort 的联合 SLO 达标率均为 100%，门槛是至少 99% 请求同时 TTFT≤3s、请求平均 TPOT≤100ms。它不是逐 token 间隔限制，也没有证明 5.161 是最大容量。

相对本轮同 SLO 下一个 TP2 P 的已通过点 C24 4.004 QPS，提升 28.9%；该对照已经开启 stage 1/2/3 和 PD，不是原始未优化 baseline。详细版本差异、失败窗口排除、测试及数据见 [双 P 报告](pd-replicas.md) 和 [窗口报告](pd-window.md)。原始大 ZIP 当前留在实验机；仓库包含图表、CSV、审计、哈希和复算代码。

实验后已恢复原 Stage 1 demo；推荐配置通过显式参数启用，不等于已长期部署。

## 尚未实现

P 改 Mix 的真实 prefill/decode 混合 batch、企业端对应混合执行、按 SLO 动态调节 Mix token 预算，均仍是方案。当前每个物理 batch 只执行一个 phase。Stage 4 多请求批量验证、RDMA/GDR 也尚未完成，不计入已验证收益。
