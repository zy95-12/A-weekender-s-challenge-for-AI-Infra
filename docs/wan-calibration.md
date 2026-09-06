# WAN 延迟曲线与探针

此 PR 将此前实验目录中的 WAN 探针、测量脚本和数据纳入版本管理，基于 Stage 1/2/3 集成 PR #14。默认不安装探针；仅设置 `SPLIT_WAN_PROBE=1` 时增加实验端点。它复用实际 serving transport、IPC、D2H/H2D 和 TP broadcast，跳过模型计算。

[测量报告](evidence/issue-5/wan-calibration/README.md) · [曲线 CSV](evidence/issue-5/wan-calibration/curves.csv) · [正式样本 CSV](evidence/issue-5/wan-calibration/samples.csv) · [拟合节点](evidence/issue-5/wan-calibration/fit_models.json) · [原始测量 ZIP](evidence/issue-5/wan-calibration/raw_measurements.zip)

报告中的 baseline/、stage1/、source/、配置和 SVG/PDF 文件位于原始 ZIP 内，解压后保留原相对路径。SHA256SUMS.json 是原始实验清单：仓库展示文件及 ZIP 内其余文件均已逐项校验，ZIP CRC 通过。原始报告未改写，测量代码与实验源快照保持一致。

测量范围：单条在途消息、TP2+2、WAN 双向 10Gbps/单向 5ms，单向 activation 净数据量 8KiB–128MiB。两种配置共 528 次往返（含预热），384 个正式样本。返回数组均逐元素匹配。32MiB 消息完整 GPU-ready RTT 均值由 baseline 456.12ms 降至 Stage1 164.59ms；这是边界通信成本，不是模型推理 TTFT。

建议按线性字节坐标做相邻测量点的分段线性插值。完整 GPU-ready RTT 已包含复制/IPC/打包及传输；不得再重复叠加同一成本。不将 C1 的曲线直接推广为并发场景结论。

运行 `.venv/bin/python scripts/wan_calibration.py`；脚本启动两组配置并恢复原 demo，结果写入 results/wan_calibration，拒绝覆盖已有目录。运行 GPU 实验需原模型与四张 A10；本次 PR 整理不重复 GPU 压测，保留原 528 次测量及校验结果。当前分支另跑完整 CPU 回归测试，日志见证据目录的 pr-tests.log。
