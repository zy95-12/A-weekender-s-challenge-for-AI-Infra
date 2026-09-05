"""Aggregate repeat-level client baselines and unique GPU profile batches.

Incomplete matrices are labelled partial. P99 is retained in raw client output
but is not promoted to a capacity claim from these small repeated samples.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics
import numpy as np


def csv_output(path, rows):
    if not rows:
        return
    with path.open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def network_summary(log):
    """Read both structured fail-closed checks and historical ping+iperf logs."""
    match = re.search(r"rtt min/avg/max/mdev = [\d.]+/([\d.]+)/", log)
    rtt = match.group(1) if match else "未采集"
    if "{" not in log:
        return rtt, "未采集"
    data = json.loads(log[log.index("{"):])
    if "measured" in data:
        if data.get("result") != "PASS":
            raise ValueError("Cannot summarize a failed network check as valid")
        return f"{data['measured']['rtt_ms']:.3f}", f"{data['measured']['throughput_gbps']:.3f}"
    return rtt, f"{data['end']['sum_received']['bits_per_second'] / 1e9:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root)
    groups = {}
    for path in sorted(root.glob("tp_*/isl_*_r_*/summary.json")):
        match = re.fullmatch(r"isl_(\d+)_osl_(\d+)_c_(\d+)_r_(\d+)", path.parent.name)
        if not match:
            continue
        isl, osl, concurrency, repeat = map(int, match.groups())
        _, enterprise_tp, cloud_tp = path.parent.parent.name.split("_")
        key = (int(enterprise_tp), int(cloud_tp), isl, osl, concurrency)
        summary = json.loads(path.read_text())
        point = summary["points"][0]
        raw = json.loads((path.parent / "qps_inf/benchmark.json").read_text())
        groups.setdefault(key, []).append((repeat, point, raw))
    rows = []
    for key, records in sorted(groups.items()):
        enterprise_tp, cloud_tp, isl, osl, concurrency = key
        qps = [point["achieved_qps"] for _, point, _ in records]
        ttft = [t * 1000 for _, _, raw in records for t in raw["ttfts"]]
        tpot = [sum(itls) * 1000 / (length - 1) for _, _, raw in records
                for itls, length in zip(raw["itls"], raw["output_lens"]) if length > 1]
        rows.append({"enterprise_tp": enterprise_tp, "cloud_tp": cloud_tp,
            "total_gpus": enterprise_tp + cloud_tp, "isl": isl, "osl": osl,
            "client_concurrency": concurrency, "repeats": len(records), "requests": len(ttft),
            "ttft_mean_ms": statistics.mean(ttft), "ttft_p50_ms": float(np.percentile(ttft, 50)),
            "ttft_p95_ms": float(np.percentile(ttft, 95)), "ttft_p99_ms": float(np.percentile(ttft, 99)),
            "ttft_repeat_mean_std_ms": statistics.stdev(point["mean_ttft_ms"] for _, point, _ in records) if len(records)>1 else 0,
            "tpot_mean_ms": statistics.mean(tpot),
            "tpot_p50_ms": float(np.percentile(tpot, 50)), "tpot_p95_ms": float(np.percentile(tpot, 95)),
            "tpot_p99_ms": float(np.percentile(tpot, 99)),
            "tpot_repeat_mean_std_ms": statistics.stdev(point["mean_tpot_ms"] for _, point, _ in records) if len(records)>1 else 0,
            "qps_repeat_mean": statistics.mean(qps), "qps_repeat_std": statistics.stdev(qps) if len(qps) > 1 else 0,
            "qps_repeat_min": min(qps), "qps_repeat_max": max(qps),
            "success_rate": sum(raw["completed"] for _, _, raw in records) / len(ttft)})
    csv_output(root / "baseline_summary.csv", rows)
    batches = []
    metrics = ["enterprise_front_prefill_ms", "enterprise_front_decode_ms", "cloud_middle_prefill_ms",
        "cloud_middle_decode_ms", "enterprise_back_prefill_ms", "enterprise_back_decode_ms",
        "upload_bytes", "download_bytes", "upload_ms", "download_ms", "rpc_wall_ms",
        "cloud_queue_ms", "enterprise_stage_pack_ms", "enterprise_receive_ms",
        "cloud_receive_ms", "cloud_d2h_ms", "step_wall_ms"]
    for path in sorted(root.glob("tp_*/profiling/split_trace.jsonl")):
        seen = set()
        for line in path.read_text().splitlines():
            record = json.loads(line)
            client_id = record.get("client_request_id", "")
            match = re.match(r"profile-isl-(\d+)-osl-(\d+)-c-(\d+)-i-", client_id)
            if not match or record["batch_id"] in seen:
                continue
            capture_first_prefill = not seen and record["phase"] == "prefill"
            seen.add(record["batch_id"])
            isl, osl, concurrency = map(int, match.groups())
            batches.append({"topology": path.parent.parent.name, "isl": isl, "osl": osl,
                "requested_concurrency": concurrency, "actual_batch_size": record["batch_size"],
                "batch_id": record["batch_id"], "phase": record["phase"],
                "capture_first_prefill": capture_first_prefill,
                "representative_token_idx": record["token_idx"],
                "representative_context_len": record["context_len"],
                "representative_queue_ms": record["queue_ms"],
                "raw_activation_bytes_each_direction": 2 * 2048 * 2 * (record.get("query_len", isl) if record["phase"] == "prefill" else record["batch_size"]),
                "emits_token": record.get("emits_token", True),
                **{metric: record.get(metric) for metric in metrics}})
    csv_output(root / "profile_batches.csv", batches)
    profile_groups = {}
    for batch in batches:
        key = tuple(batch[k] for k in ("topology", "isl", "osl", "requested_concurrency", "actual_batch_size", "phase"))
        profile_groups.setdefault(key, []).append(batch)
    profile_rows = []
    for key, group in sorted(profile_groups.items()):
        row = dict(zip(("topology", "isl", "osl", "requested_concurrency", "actual_batch_size", "phase"), key))
        row["unique_batches"] = len(group)
        for metric in metrics:
            values = [batch[metric] for batch in group if batch[metric] is not None]
            row[metric + "_mean"] = statistics.mean(values) if values else None
        profile_rows.append(row)
    csv_output(root / "profile_summary.csv", profile_rows)
    dcgm_rows = []
    dcgm_fields = ("dcgm_1001_gr_active", "dcgm_1002_sm_active", "dcgm_1005_dram_active",
        "dcgm_1009_pcie_tx_bytes", "dcgm_1010_pcie_rx_bytes", "dcgm_155_power",
        "dcgm_100_sm_clock", "dcgm_101_mem_clock", "dcgm_203_gpu_util", "dcgm_204_mem_util", "dcgm_252_fb_used")
    for path in sorted(root.glob("tp_*/profiling/dcgm_timestamped.jsonl")):
        _, enterprise_tp, cloud_tp = path.parent.parent.name.split("_")
        for line in path.read_text().splitlines():
            record = json.loads(line)
            parts = record["line"].split()
            if len(parts) != 13 or parts[0] != "GPU":
                continue
            index = int(parts[1])
            role = "enterprise" if index in range(int(enterprise_tp)) else "cloud" if index in range(2, 2 + int(cloud_tp)) else "unused"
            values = []
            for value in parts[2:]:
                try:
                    values.append(float(value))
                except ValueError:
                    values.append(None)
            dcgm_rows.append({"topology": path.parent.parent.name, "observed_time_ns": record["observed_time_ns"],
                              "gpu_index": index, "role": role, **dict(zip(dcgm_fields, values))})
    csv_output(root / "dcgm_profile_samples.csv", dcgm_rows)
    configuration = json.loads((root / "matrix_config.json").read_text())
    expected_cells = len(configuration["pairs"].split(",")) * len(configuration["workloads"]) * len(configuration["concurrency"].split(","))
    profile_coverage = []
    for path in sorted(root.glob("tp_*/profiling/profile_cases.json")):
        cases = json.loads(path.read_text())["cases"]
        environment = json.loads((path.parent / "profile_environment.json").read_text())
        profile_coverage.append({"topology": path.parent.parent.name, "requests": len(cases),
            "full_lengths_valid": all(case["usage"]["prompt_tokens"] == case["isl"] and
                case["usage"]["completion_tokens"] == case["osl"] for case in cases),
            "profiler": environment.get("profiler", {}).get("version", "legacy capture; see environment and launch logs"),
            "nsight_reports_present": all((path.parent / f"{role}.nsys-rep").exists() for role in ("enterprise", "cloud"))})
    (root / "profile_coverage.json").write_text(json.dumps(profile_coverage, indent=2))
    completed = ((root / "completion.json").exists() and len(rows) == expected_cells and
        all(row["repeats"] == configuration["repeats"] for row in rows) and
        (configuration["skip_profile"] or (len(profile_coverage) == len(configuration["pairs"].split(",")) and
         all(row["requests"] == 20 and row["full_lengths_valid"] and row["nsight_reports_present"] for row in profile_coverage))))
    lines = ["# Qwen Split Inference 性能基线", "", f"状态：{'矩阵完成' if completed else '部分结果，矩阵尚未完成'}。", "",
        f"固定层切分配置 {configuration['split']}，FP16、A10、10 Gbps / 单向 5 ms。TP=1 为单卡执行。", "",
        "QPS 是闭环固定客户端并发下的实际完成请求率，不是 offered QPS、最大 QPS 或 SLO 结论。", "",
        "TTFT 从真正发送 HTTP 请求开始计算，不包含客户端并发信号量之前的等待。", "",
        "客户端 TPOT 是 (请求总生成时间 - TTFT)/(OSL - 1)，包含生成期间的调度等待。",
        "当前调度器的其他请求 Prefill 可阻塞已有请求的 Decode，因此 TPOT 不等于独立 Decode batch 的 GPU 时长。", "",
        f"每点 {configuration['repeats']} 次独立运行；每次请求数为 max(8, 4×并发)，均另有官方客户端 warmup。", "",
        f"已采集 {len(rows)} 个配置单元、{sum(row['repeats'] for row in rows)} 轮、{sum(row['requests'] for row in rows)} 个请求。", "",
        "表内 TTFT/TPOT 为全部请求的均值；QPS 为重复测量吞吐的均值。重复间样本标准差及合并样本 P50/P95/P99 见 CSV；小样本 P99 仅作描述。", "",
        "| 端 TP | 云 TP | ISL | OSL | 并发 | 次数 | TTFT ms | TPOT ms | QPS |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for key in ("enterprise_tp", "cloud_tp", "isl", "osl", "client_concurrency", "repeats")) +
            f" | {row['ttft_mean_ms']:.2f} | {row['tpot_mean_ms']:.2f} | {row['qps_repeat_mean']:.4f} |")
    lines += ["", "## 实际模型、网络与数值核对", "",
              "Qwen2.5-3B-Instruct 真实 FP16 权重；端侧负责 Embedding、首尾层及输出头，云侧仅执行中间层。",
              "每个解码步都传输 hidden states 和 residual；不是分别部署两份完整模型，也不是仅传文本的代理。",
              "WAN 由独立 network namespace / veth 上的 tbf + netem 实现，真实 TCP 流量经过整形链路。",
              "这是单服务器内的可控 WAN 仿真，不是跨地域公网实测；未注入随机抖动、丢包或 TLS。", "",
              "| 拓扑 | ping 平均 RTT ms | iperf 接收 Gbps | 数值核对 | 最大 MAE | 最大绝对误差 | 32-token 自由生成匹配 |",
              "|---|---:|---:|---|---:|---:|---|"]
    for path in sorted(root.glob("tp_*/correctness/summary.json")):
        correctness = json.loads(path.read_text())
        network_path = path.parent.parent / "network_check.log"
        network = network_path.read_text() if network_path.exists() else ""
        rtt, bandwidth = network_summary(network)
        free = correctness.get("free_running", [])
        matching = f"{sum(item['exact_match'] for item in free)}/{len(free)} 个输入长度"
        lines.append(f"| {path.parent.parent.name} | {rtt} | {bandwidth} | {correctness['result']} | "
                     f"{correctness['max_mae']:.6f} | {correctness['max_abs_error']:.6f} | {matching} |")
    lines += ["", "## 建模数据", "", "- `baseline_summary.csv`：客户端重复测量聚合。",
              "- `profile_batches.csv`：独立 Nsight 运行的逐 batch 数据，已去除同一 batch 对多个请求的重复计数。",
              "- `profile_summary.csv`：按拓扑、workload、实际 batch size、phase 聚合。",
              "- `dcgm_profile_samples.csv`：带主机读取时间戳、GPU 编号和端/云角色的 DCGM 原始字段值，字段名保留 DCGM ID；不可用值留空。",
              "- `tp_*/profiling/`：两侧 Nsight、kernel/NVTX 汇总与原始 trace。",
              "- `tp_*/model_audit.json`：实际 worker 层归属、参数量、GPU 进程、模型哈希。",
              "- `profile_coverage.json`：每种拓扑的完整输出覆盖、Nsight 文件及 profiler 版本。",
              "- `tp_*/profiling/*cuda_gpu_mem_*_sum.csv`：CUDA 内存拷贝时间/字节数汇总；不是 WAN 链路吞吐。",
              "", "详细 profile 含同步阶段计时和 Nsight 开销，不能混入正常运行的吞吐表。",
              "上传/下载计时包含 CPU、HTTP 和序列化开销，不等于纯传播延迟。",
              "RPC wall 已包含上行、云侧排队/计算/拷贝、下行，不能再与这些子项相加。",
              "可先用 step ≈ front + stage_pack + RPC + enterprise_receive + back + residual 建模；",
              "residual 包含元数据、进程间通信、logits 回传与 CPU 处理等未独立归因部分，并非新的实测字段。",
              "front 包含 Embedding，back 包含末层、最终 Norm 和输出头；Nsight 多卡 kernel 时间之和不是请求关键路径时长。",
              "普通性能运行中的 cloud_d2h_ms 可能包含等待尚未完成的 GPU 计算；阶段归因只使用独立详细采样。",
              "capture_first_prefill 标记每次 Nsight capture 的首个 Prefill，可能有采样启动开销；拟合时应单独处理。",
              "profile_batches.csv 的 representative_* 字段属于该 batch 中一个请求，完整逐请求队列数据保留在原始 trace。",
              "KV 使用量为已分配逻辑 blocks；nvidia-smi 显存包含预分配 KV 池，不能将两者等同。",
              "RPC 携带 hidden states + residual 两个 FP16 张量：原始激活每 token、每方向 8192 bytes，另加协议头。",
              "1:3 配置按层数划分；端侧独有 Embedding/输出头，不能按 1:3 估算参数量或 GPU 工作量。",
              "混合 TP 的数值差异单独保存为 CROSS_TP_OBSERVATION，不冒充同 TP 原生精度验收。",
              "当前仍为自定义 vLLM partial runner 与简单调度器，不代表原生 vLLM serving 性能。", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"completed": completed, "cells": len(rows), "repetitions": sum(row["repeats"] for row in rows),
                      "profile_batches": len(batches)}, indent=2))


if __name__ == "__main__":
    main()
