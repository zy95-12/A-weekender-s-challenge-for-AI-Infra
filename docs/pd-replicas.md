# 两个独立 TP1 prefill 实例，共用一个 TP1 decode 实例

四卡布局为 E1 + P0(TP1) + P1(TP1) + D(TP1)。每个 P 独立保存云侧 27 层完整权重和两个 KV heads；同一请求全部 prefill chunk 固定路由到同一个 P，所有请求 decode 仍可在同一 D 合批。

企业端以尚未完成的 prefill token 数分配新请求，统计已接纳和正在预留的请求，等负载时轮转。P 窗口是每实例独立上限，D 窗口为全局上限。取消或完成后，先排空提交的任务和迁移，再释放 D 的全局 KV 页及企业端 admission credits。

D 通过两个独立 vLLM StatelessProcessGroup / PyNcclCommunicator 连接 P0/P1。每条链路拥有自己的 copy stream、工作队列和 NCCL 收发顺序；D 统一管理物理 KV 页，并记录每条请求的来源实例。跨来源 reserve/start/status/commit/release 被拒绝。模型 TP 组仍与 KV 迁移组分离。实例失败沿用 fail-closed 语义，不对已产生 KV 的请求静默改路由。

启用参数：`--pd --prefill-replicas 2 --prefill-tp 1 --decode-tp 1`。默认 `--prefill-replicas 1` 保留原 TP2 P 布局。管理脚本支持启动、健康检查和停止两个 P；日志、配置、KV trace 和可选算子捕获按实例分开保存。

本轮测试使用 Qwen2.5-3B-Instruct FP16、企业前4/云27/企业后5层，4096-token 输入、自然 EOS 79-token 输出、C24，WAN 每方向 10Gbps/5ms。stage1/2/3 开启，chunk 2048、每 P 窗口3、D窗口2、decode quota4；整段 KV 迁移，独立控制通道开启。max-active96、KV32768页。性能测试关闭 profiling 和 KV 哈希。

SLO：至少99%的请求同时满足 TTFT≤3000ms、请求平均 TPOT≤100ms，开始和完成两个 cohort 都验收。固定 C24 的实验不等于重新测出最大 SLO QPS。

## 正确性

71 项 CPU 测试通过，包括负载分配、固定路由、一个 P 窗口满时另一 P 与 D 仍可推进、共享 D pool 不同来源页互不重叠、错误来源被拒绝、独立迁移完成与释放顺序。

GPU 验证：C24 的24条请求，P0/P1各承担12条，均输出原参考的79个token；31次KV迁移（两个来源都覆盖）逐head SHA256完全匹配；257-token非页对齐输入分别通过两个P，两份8-position teacher-forcing logits逐元素一致；max_tokens=1以及两种提前取消时序均正常排空，E、两个P和D的KV/active计数均归零。

完整实验入口：`.venv/bin/python -m scripts.pd_replicas_experiment --mode validation`，随后 `--mode performance`；只对两种布局做90秒代表性对比，需要时用 `--mode confirm` 各追加180秒。`scripts/audit_pd_replicas.py` 复算性能和检查逐请求路由、绝对位置、KV ready 顺序及源码哈希。


## 持续负载发现与修复

第一版在双 TP1 连续负载下完成226条正确请求后，企业端的一条释放控制请求收到 `RemoteProtocolError: Server disconnected without sending a response`，触发 fail-closed，后续24条在途请求被终止。云进程日志未显示 CUDA 异常；该次没有完成规定测量窗口，不作为有效 QPS 结果。原始数据、日志和当时源代码保存在 `results/pd_replicas_initial/`。

修复将企业端的 P 控制与数据连接池分离，控制/数据 P 客户端及 P→D 控制客户端使用1秒空闲复用期限。自定义 TCP transport 原先没有转交 `limits`，同时修正了这一点。对于已验证幂等的 PD reserve/wait/start/status/commit/release，遇到连接异常后使用新连接重试一次；forward 不允许重放，HTTP 错误和超时不自动重试。

新增回归测试在迁移 start 已生效后模拟响应丢失，确认第二次调用不会重复入队 NCCL copy，并验证重试次数上限、forward 禁止重试、TCP transport 尊重连接池期限。修复后重新完成全套 GPU 正确性验证，再重新测量两种布局；不将失败窗口截短后当作通过结果。


C40 的 TP2 对照仍发生内部 HTTP 断开；检查发现 D 数据连接仍沿用5秒期限。原异常没有记录URL，因此不能仅凭原日志确定具体连接。最终将所有 P/D 客户端的空闲复用期限统一为1秒、服务端空闲关闭期限设为60秒，避免同时关闭/复用的边界；新增故障 URL 和完整异常记录。相应回归测试纳入上述71项。C24完整窗口保存在 `pd_replicas_v2/`（包含其源代码快照），C40最终窗口保存在 `pd_replicas/`；二者计算、KV和分流策略相同，连接期限设置有上述差异。同一并发下两种布局使用相同版本；不混入两次中断窗口。

## C24 完整窗口

| P 布局 | QPS | TTFT 均值 / P99 ms | TPOT 均值 / P99 ms | 双 cohort 联合 SLO |
|---|---:|---:|---:|---:|
| 一个 TP2 实例 |4.004|1521 / 1931|57.31 / 63.80|100% / 100%|
| 两个 TP1 实例 |4.163|843 / 1200|62.82 / 66.76|100% / 100%|

C24 的 QPS 只增加3.97%，平均 TTFT 降低44.61%，平均 TPOT 增加9.61%。实际两个P都在工作：正式窗口内第一段prefill分配188/187条请求；P0/P1每chunk服务墙钟约168.1/170.0ms，各自服务覆盖率约69.8%/70.5%。TP2每chunk约115.7ms、覆盖率92.1%。这些是服务区间覆盖率，不是GPU利用率。

C24下两个P尚未充分利用，因此只补C40这一个高并发点，检验能否把剩余P处理能力转成SLO吞吐，没有扫描全部并发。


## C40 最终完整窗口

| P 布局 | QPS | TTFT 均值 / P99 ms | TPOT 均值 / P99 ms | 双 cohort 联合 SLO |
|---|---:|---:|---:|---:|
| 一个 TP2 实例 |4.064|5441 / 5950|56.81 / 62.71|0% / 0%|
| 两个 TP1 实例 |5.161|794 / 1353|89.28 / 93.84|100% / 100%|

在同一SLO口径下，已验证的可用吞吐从 TP2 的 C24 4.004 QPS 提高到双 TP1 的 C40 5.161 QPS，+28.91%。TP2 的 C40 不满足SLO，不能作为其可用容量。这里只测试C24/C40两个代表点，没有证明5.161是最大SLO QPS。

双 TP1 在 C40 的P0/P1服务覆盖率达到86.1%/89.6%，企业端front/back命令覆盖率80.7%，D服务覆盖率25.5%；decode batch平均33.74条（C24双TP1为20.14）。这些都是墙钟区间指标，不是硬件利用率。增加并发后，P的剩余能力确实转化成更大的decode工作人口和更高吞吐。

代价：C40双TP1的逐token ITL P99为158.07ms，首二token间隔P99为260.03ms。当前验收的是请求平均TPOT≤100ms，不能解读为每个token间隔≤100ms。低并发下TP1单请求prefill也可能慢于TP2，本次没有重扫低并发。

最终保留的4个完整性能窗口合计1823条请求（含预热/排空），均输出79个token且文本与参考一致，绝对位置、实例固定路由、KV ready顺序和排空检查通过。C40双TP1的545条请求分配为P0 272条/P1 273条。两个C40窗口无连接异常、无需触发控制重试。GPU功能验证的31次哈希匹配及跨实例logits证据来自v2，最终仅再调整连接期限及故障诊断，计算/KV算法未改动；所有版本的源代码快照都保留并核对哈希。

## 使用

```bash
./poc up --wan --pd --prefill-replicas 2 --prefill-tp 1 --decode-tp 1 \
  --ipc-mode shm --wire-fast --tcp-buffer-mib 16 \
  --prefill-chunk-size 2048 --scheduler-policy decode-first --decode-quota 4 \
  --pipeline-window 2 --pd-prefill-window 3 --max-active 96 --kv-blocks 32768
```

P0/P1分别使用GPU1/GPU2，D使用GPU3，企业端使用GPU0。两个P分别监听8001/8003，D监听8002。增加的权重副本使每个P的显存占用从TP2时约9.4GiB提升到约18.5GiB（验证时），本场景在A10显存容量内通过验证。

实验结束已恢复原工作目录的Stage1 demo。新布局以显式开关提供，不改旧启动参数的默认拓扑。

[对比图](evidence/issue-6/pd-replicas/comparison.png) · [全部完整窗口CSV](evidence/issue-6/pd-replicas/comparison.csv) · [逐请求审计汇总](evidence/issue-6/pd-replicas/audit_summary.json) · [原始证据ZIP](../results/pd_replicas/evidence.zip)
