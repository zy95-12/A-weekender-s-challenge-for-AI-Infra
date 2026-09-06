# Split-inference 仿真与 demo

当前交付包含 Q4 仿真模型和完整 Web demo。主线真实 serving 代码继续位于根目录 `split_poc/`，Q4 是独立的标准库仿真包，不执行模型权重。

- [运行仿真、开关及 baseline 回退](../Q4/README.md)
- [校准项、硬编码参数、可迁移部分和未解决问题](../Q4/docs/CALIBRATION_AND_LIMITS.md)
- [结构与后端](../Q4/docs/ARCHITECTURE.md)
- [当前精度、证据和验证命令](../Q4/docs/VALIDATION.md)
- [启动完整 demo](../demo/README.md)

当前 PD C40 校准对照误差为 QPS +3.05%、TTFT −10.64%、TPOT −2.80%，尚不是跨模型/硬件泛化或精确 SLO 容量验收。demo 的性能区调用 Q4 解析模型；安全区是证据回放，baseline 启动/对话是假数据展示，当前 Qwen3-32B/A10 演示参数未做实测校准。
