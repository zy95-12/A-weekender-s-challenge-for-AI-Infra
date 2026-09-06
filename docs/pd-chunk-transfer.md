# PD 按 prefill chunk 提前迁移 KV

本改动叠加 PR #16 的 PD serving 与 Stage 1/2/3，新增 `--pd-chunk-transfer` 开关。不改变模型切层、attention 实现、企业端 decode-first/quota=4 策略。默认仍整段迁移；所有 PD 配置使用独立的 worker 控制通道。

## 调度与数据路径

1. P worker 在实际 forward CUDA stream 上记录完成事件及已计算前缀。控制线程不会用自己的默认 stream 冒充 forward 完成事件。
2. 每次 prefill forward 返回后，协调器将新完成的完整 KV 页加入迁移队列；非最终边界向下对齐 16 tokens，最终边界可以不对齐。非最终页以后不会再被 P 写入。
3. 复用 vLLM `PyNcclCommunicator`，按全局 KV head 将逻辑页范围迁移到 D 预留的物理页。支持 P2/D1 和 P1/D2；没有增加 CPU staging。最终页未使用槽位清零。
4. 同一请求的范围按序完成，跨请求 send/recv 启动次序由同一 dispatch 锁约束。重复的相同范围幂等；拒绝缺口、重叠和超过已计算前缀的导出。
5. P 保留全部 prompt KV，供后续 prefill attention 使用。D 只有在 `[0,prompt_len)` 全部范围及所有 rank 完成后才能 commit，之后企业端才允许第一个 decode front。
6. `pd_start/status/commit` 使用每个 GPU worker 的独立 Pipe 与控制线程，避免排在 GPU forward 后。分配、释放仍由主执行通道串行处理。请求取消必须等待全部已排队范围完成，再释放 P/D KV。

4K、27 个云层、FP16、两个 KV heads 的逻辑 KV 总量仍为 108 MiB。chunk=1024 时分为四次各 27 MiB；减少的是最终交接剩余数据，不是总字节数。迁移可能与 P 计算及 D 的其他请求竞争 GPU/PCIe 资源，收益以实测为准。

## 使用

```bash
./poc up --wan --pd --max-active 16 \
  --ipc-mode shm --wire-fast --tcp-buffer-mib 16 \
  --prefill-chunk-size 1024 --pipeline-window 2 \
  --scheduler-policy decode-first --decode-quota 4 \
  --pd-chunk-transfer
```

增加 `--prefill-tp 1 --decode-tp 2` 切换 E1/P1/D2。`--pd-verify-kv` 校验每段每个 head 的源张量与实际 D KV 页哈希；该校验包含额外 D2H，只用于正确性验证，不用于性能测试。health 与保存的启动配置显示 chunk 开关。

`pd_kv_trace.jsonl` 每个完成请求记录所有 chunk 的范围、源/目标复制时间及字节数，并保留最终 KV-ready/source-release 时间。`split_trace.jsonl` 记录绝对 token 位置及 E/P/D 时序。接收线程时间含排队、NCCL 等待和 scatter，不能当作纯链路传输时间。

## 验证与限制

- CPU 测试覆盖连续范围、非对齐边界、幂等、未完成前缀导出拒绝、最终 commit、取消回收，以及原有 PD/调度测试。
- GPU 验证覆盖 C16、独立 257-token 输入、max_tokens=1、prefill 取消、逐段实际 KV 页哈希及绝对位置 logits。
- 性能实验比较原实现的两种 GPU 分配，再在 P2/D1 下比较仅独立控制通道与增加 chunk 迁移。4K/C16、WAN、模型与所有 Stage 1/2/3 参数保持一致。
- 仍使用静态 P/D GPU 分配及预留 D 全请求容量；没有提前释放 P 前缀页，也没有跨请求物理 mixed batch。控制协调器沿用有限线程池；更高并发的控制开销没有在本次证明。

运行 `python -m scripts.pd_chunk_audit` 从原始请求与迁移轨迹重算指标并检查位置、范围覆盖和交接因果顺序。实测结果见 [C16 报告](pd-chunk-results.md)。
