"""Audit request-level open-loop evidence and publish the two-preset comparison."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def close(a, b):
    assert abs(a - b) < 1e-6, (a, b)


def package(source, output):
    summaries = json.loads((source / "summaries.json").read_text())
    points = []
    plans = {}
    groups = {}
    reference = json.loads((source / "current_reference.json").read_text())
    for s in summaries:
        folder = source / s["path"]
        rows = json.loads((folder / "requests.json").read_text())
        plan = json.loads((folder / "schedule.json").read_text())
        launch = json.loads((folder / "launch.json").read_text())
        assert (
            launch["preset"] == s["system"]
            and launch["max_active"] == 96
            and launch["kv_blocks"] == 32768
        )
        assert not any(
            launch[k] for k in ["profile", "phase_profile", "operator_profile"]
        )
        assert (
            json.loads((folder / "network_check.json").read_text())["result"] == "PASS"
        )
        assert len(rows) == len(plan["offsets_s"]) and len(
            {r["request_id"] for r in rows}
        ) == len(rows)
        plan_key = (s["target_arrival_rate"], s["duration_s"])
        if plan_key in plans:
            assert plans[plan_key] == plan
        plans[plan_key] = plan
        start, end = s["measurement_start"], s["measurement_end"]
        duration = end - start
        arrivals = [r for r in rows if start <= r["scheduled_start"] < end]
        completion = [r for r in rows if start <= r["end"] < end]

        def passed(r):
            return (
                r["error"] is None
                and r["tokens"] == 79
                and r["scheduled_ttft_ms"] is not None
                and r["tpot_ms"] is not None
                and r["scheduled_ttft_ms"] <= 3000
                and r["tpot_ms"] <= 100
            )

        assert s["all_outputs_match"] == all(
            r["error"] is None and r["tokens"] == 79 and r["text"] == reference["text"]
            for r in rows
        )
        for r in rows:
            assert passed(r) == r["slo_pass"]
            if r["ttft_ms"] is not None:
                close(
                    r["scheduled_ttft_ms"],
                    r["ttft_ms"] + (r["start"] - r["scheduled_start"]) * 1000,
                )
        for cohort, key in [
            (arrivals, "arrival_cohort"),
            (completion, "completion_cohort"),
        ]:
            assert len(cohort) == s[key]["requests"]
            close(sum(passed(r) for r in cohort) / len(cohort), s[key]["joint_slo"])
            for metric in ["scheduled_ttft_ms", "tpot_ms"]:
                values = [r[metric] for r in cohort if r[metric] is not None]
                close(float(np.mean(values)), s[key]["mean_" + metric])
                close(float(np.percentile(values, 99)), s[key]["p99_" + metric])
        close(len(arrivals) / duration, s["actual_arrival_rate"])
        close(
            sum(r["error"] is None and r["tokens"] == 79 for r in completion)
            / duration,
            s["successful_completed_qps"],
        )
        close(sum(passed(r) for r in completion) / duration, s["goodput_qps"])
        close(
            sum(max(0, min(end, r["end"]) - max(start, r["start"])) for r in rows)
            / duration,
            s["mean_inflight"],
        )
        for name in ["initial_health.json", "final_health.json"]:
            h = json.loads((folder / name).read_text())
            assert (
                all(h[k] == 0 for k in ["active", "waiting", "kv_used_blocks"])
                and not h.get("pd_reserving")
                and not h.get("pd_releasing")
            )
        a = s["arrival_cohort"]
        b = s["completion_cohort"]
        h = json.loads((folder / "health.json").read_text())
        hh = [r for r in h if start <= r["time"] < end]
        point = {
            "system": s["system"],
            "target_arrival_rate": s["target_arrival_rate"],
            "actual_arrival_rate": s["actual_arrival_rate"],
            "completed_qps": s["successful_completed_qps"],
            "goodput_qps": s["goodput_qps"],
            "mean_ttft_ms": a["mean_scheduled_ttft_ms"],
            "p99_ttft_ms": a["p99_scheduled_ttft_ms"],
            "mean_tpot_ms": a["mean_tpot_ms"],
            "p99_tpot_ms": a["p99_tpot_ms"],
            "arrival_slo": a["joint_slo"],
            "completion_slo": b["joint_slo"],
            "slo_pass": min(a["joint_slo"], b["joint_slo"]) >= 0.99,
            "duration_s": s["duration_s"],
            "mean_inflight": s["mean_inflight"],
            "peak_inflight": s["peak_inflight"],
            "requests": len(arrivals),
            "ttft_failures": a["ttft_failures"],
            "tpot_failures": a["tpot_failures"],
            "errors": a["errors"],
            "p99_dispatch_lag_ms": a["p99_dispatch_lag_ms"],
            "inflight_boundaries": s["inflight_boundaries"],
            "max_waiting_sampled": max(r.get("waiting", 0) for r in hh),
            "all_outputs_match": s["all_outputs_match"],
            "measurement_arrivals_completed_before_load_stop": s[
                "measurement_arrivals_completed_before_load_stop"
            ],
            "raw_path": s["path"],
        }
        points.append(point)
        groups.setdefault(s["system"], []).append(point)
    points.sort(key=lambda r: (r["system"], r["target_arrival_rate"]))
    winners = {
        system: max(
            (r for r in rows if r["slo_pass"]),
            key=lambda r: r["completed_qps"],
            default=None,
        )
        for system, rows in groups.items()
    }
    text = []
    for system, name in [("baseline", "Baseline"), ("optimized", "优化系统")]:
        r = winners[system]
        text.append(
            f"{name}最高已测合格点：完成 {r['completed_qps']:.3f} QPS（目标到达率 {r['target_arrival_rate']:.3f}/s）"
            if r
            else f"{name}尚无已测合格点"
        )
    conclusion = "；".join(text) + "。这是本次有限窗口结果，不是精确最大容量。"
    evidence = {
        "source": "measured",
        "mode": "open-loop-poisson",
        "seed": 17,
        "model": "Qwen2.5-3B-Instruct",
        "input_tokens": 4096,
        "output_tokens": 79,
        "latency_cohort": "scheduled-arrival",
        "slo": {
            "ttft_ms": 3000,
            "request_mean_tpot_ms": 100,
            "joint_attainment": 0.99,
            "cohorts": ["arrival", "completion"],
        },
        "conclusion": conclusion,
        "points": points,
        "winners": winners,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "open-loop-sweep.json").write_text(json.dumps(evidence, indent=2))
    keys = [k for k in points[0] if k != "inflight_boundaries"]
    with (output / "open-loop-sweep.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(points)
    lines = [
        "# Baseline / 优化系统开环对比",
        "",
        conclusion,
        "",
        "同一 Qwen2.5-3B-Instruct FP16、4×A10、Split 4/27/5、4096 输入 / 79 输出。Baseline 为 TP2+2；优化系统为企业 TP1 + 两个 TP1 P + TP1 D，stage1/2/3、chunk2048、quota4、P窗口3/D窗口2、整prompt KV迁移。WAN 每方向10Gbps、单向5ms。两套系统均 max_active=96、kv_blocks=32768，无 profiling。",
        "",
        "泊松到达 seed17，相同目标到达率/时长的请求计划逐值相同。客户端无并发上限，不等响应再补请求。每点预热30秒、粗扫测量60秒（λ≥4时120秒）、继续到达30秒后排空。遇到首个失效点后只补一个中间负载点。每点一个随机轨迹，无多种子重复或精确边界搜索。",
        "",
        "延迟按测量窗口内计划到达请求统计；TTFT 包括客户端发包延误，TPOT 是请求内平均token间隔。QPS=窗口成功完成数/窗口时长；goodput仅计满足联合SLO的完成请求。SLO要求到达与完成cohort各至少99%请求同时TTFT≤3s、TPOT≤100ms；错误计失败。实际到达率会偏离设定值。",
        "",
        "| 系统 | 目标/实际到达率 | 完成QPS | TTFT均值/P99 ms | TPOT均值/P99 ms | 到达/完成SLO | 平均/峰值在途 | 窗口s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in points:
        lines.append(
            f"| {r['system']} | {r['target_arrival_rate']:.3f}/{r['actual_arrival_rate']:.3f} | {r['completed_qps']:.3f} | {r['mean_ttft_ms']:.1f}/{r['p99_ttft_ms']:.1f} | {r['mean_tpot_ms']:.2f}/{r['p99_tpot_ms']:.2f} | {r['arrival_slo']:.2%}/{r['completion_slo']:.2%} | {r['mean_inflight']:.1f}/{r['peak_inflight']} | {r['duration_s']} |"
        )
    lines += [
        "",
        "## 数据验收",
        "",
        f"已从逐请求记录复算 {len(points)} 个点的QPS、均值、P99、两个cohort联合SLO和平均在途并发；相同负载点到达计划一致、网络验证通过、各点首尾排空。所有阶段输出正确：{all(r['all_outputs_match'] for r in points)}；所有测量到达请求均在停止持续负载前完成：{all(r['measurement_arrivals_completed_before_load_stop'] for r in points)}。",
        "",
        "在途轨迹、错误数、TTFT/TPOT失败数、发包延误等见JSON。health采样的waiting仅表示admission等待，不能代替内部调度排队或证明长期稳定。低请求数的单次窗口只能作为探索性证据，不作99%达标率的统计置信保证。",
        "",
        "复现：`./.venv/bin/python scripts/open_loop_sweep.py --out results/NEW_DIRECTORY`；打包：`./.venv/bin/python scripts/package_open_loop.py --source results/NEW_DIRECTORY`。运行器测试结束恢复默认baseline WAN服务。",
        "",
        "[CSV](open-loop-sweep.csv) · [JSON](open-loop-sweep.json) · [原始数据](open-loop-raw.zip)",
        "",
        "![TTFT](open-loop-ttft-qps.png)",
        "![TPOT](open-loop-tpot-qps.png)",
    ]
    (output / "open-loop-report.md").write_text("\n".join(lines) + "\n")
    with zipfile.ZipFile(output / "open-loop-raw.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in source.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(source))
        z.write(__file__, "package_open_loop.py")
    print(conclusion)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "demo/evidence")
    a = p.parse_args()
    package(a.source, a.output)
