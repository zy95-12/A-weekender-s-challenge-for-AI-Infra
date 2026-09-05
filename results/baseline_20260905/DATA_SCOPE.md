# Baseline 数据交付范围

本目录在 Git 中是精简快照，不是服务器实验目录的完整镜像。

## 随 PR 提交

- 根目录报告、基线和逐 batch/阶段/DCGM CSV、矩阵配置、完成记录、覆盖与执行审计结果。
- 各拓扑的模型审计、网络检查，以及精度比较 JSON；不包含原始 logits 数组。
- 全部 96 轮的 summary.json，以及 qps_inf 下的 benchmark.json、requests.jsonl、config.json、network.json、environment.json。
- 四组 profiling 的配置、环境、完整请求清单，以及 kernel、NVTX、CUDA 内存拷贝时间/大小 CSV。
- ARTIFACT_MANIFEST.json：服务器原始目录文件的相对路径、大小、SHA256 和本次是否纳入 Git。

这些文件可以直接用于查看客户端统计、重复间差异、逐 batch 建模和环境溯源。

## 仅在服务器保留

- 原始 .nsys-rep、完整 split_trace.jsonl、原始遥测和运行日志。
- 原始精度数组 .npz，以及可从 .nsys-rep 重新导出的 .sqlite。
- profiling_nsys2024_6 旧版本采样。失败采样位于报告所列的独立目录，不纳入本次清单。

本机完整目录：`/root/A-weekender-s-challenge-for-AI-Infra/results/baseline_20260905`。
**外部归档尚未上传，当前没有 Release/对象存储下载链接。** 清单不是下载地址，也不代表已完成异地备份。
模型权重不在本目录中，不纳入交付或清单。

## 复现注意事项

报告中的 trace 审计和 profiling 覆盖 JSON 是在完整原始数据上生成的已保存结果。
从 Git 克隆后直接运行 summarize_baseline.py 会缺少原始 profiling trace；不要覆盖已提交的汇总。
verify_baseline_traces.py 也需要完整原始 trace，不能仅用精简快照重验。
完整重建请按 README 准备模型与原生参考，并用 baseline_matrix.py 输出到一个新目录。
已有精简快照不是可直接 --resume 的完整实验目录。

ARTIFACT_MANIFEST.json 不包含自身和此说明文件的哈希，避免自引用；两者由 Git 提交版本管理。
