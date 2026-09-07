"""Publish equal-weight per-run MAPE, signed bias and source-linked evidence."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["qps", "ttft_mean_ms", "tpot_mean_ms", "ttft_p99_ms", "tpot_p99_ms"]


def publish(old, new, raw):
    rows = []
    for source, path in [("original_seed17", old), ("repeat_seeds29_43", new)]:
        for r in json.loads(path.read_text()):
            rows.append(dict(r, experiment_source=source))
    if len(rows) != 19 or len(json.loads(new.read_text())) != 8:
        raise ValueError("Expected original 11 and repeated 8 complete comparisons")
    measured = json.loads((raw / "summaries.json").read_text())
    if len(measured) != 8 or not all(
        r["all_outputs_match"] and r["measurement_arrivals_completed_before_load_stop"]
        for r in measured
    ):
        raise ValueError("Incomplete or invalid measured repetitions")
    groups = []
    for system, label in [
        ("baseline", "Baseline · 算子/CPU/WAN修正"),
        ("optimized", "优化 PD · command/host profiling修正"),
        ("all", "全部实验（点间等权）"),
    ]:
        rr = [r for r in rows if system == "all" or r["point"].startswith(system + "/")]
        g = dict(system=system, label=label, count=len(rr))
        for k in METRICS:
            errors = [r[k + "_error_pct"] for r in rr]
            g[k] = dict(
                mape=statistics.mean(abs(e) for e in errors),
                bias=statistics.mean(errors),
                max_absolute_error=max(abs(e) for e in errors),
            )
        g["slo_disagreements"] = sum(
            r["measured_pass"] != r["simulated_pass"] for r in rr
        )
        groups.append(g)
    findings = [
        "① 计算与数据搬运的成本模型存在误差。Roofline 和带宽估计难以完整描述实际执行时间，因此如没有可靠的算子性能校准，则仿真结果只能做定性分析；即使经过校准，遇到未覆盖的 shape、batch 或不同执行状态，插值、外推仍可能产生偏差。本次优化系统在 4.44 QPS 下，TPOT 高估 8.86 ms/token，其中企业端尾层命令高估 7.54 ms/token，已发现稀疏采样和单样本插值端点的问题。",
        "② Serving 的通信与等待关系需要更细粒度建模。同一场景下，TTFT 低估 104.12 ms，主要落在 prefill 通信区间和首次计算前等待。通信区间包含 CPU、收发包、拷贝等，不能只用通信量／带宽描述，同时需要正确模拟这些工作的重叠与资源竞争。等待应由调度、依赖和资源占用推导出来，而不是简单补一个固定 overhead。"
    ]
    report = dict(
        description=f"Qwen2.5-3B / 4×A10 / 4K输入、79输出；{len(rows)}组开环实测对照（原11组 seed17 + 新8组 seed29/43）。精确回放实测到达计划，成本抽样固定seed17；未使用这些结果重新拟合成本。",
        groups=groups,
        findings=findings,
        points=rows,
        metric_definition="MAPE=mean(abs(100*(sim/measured-1))); bias=mean(100*(sim/measured-1)); equal weight per run, not per request; no confidence interval or cross-device accuracy claim.",
    )
    evidence = ROOT / "demo/evidence"
    (evidence / "simulation-accuracy.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    with (evidence / "simulation-accuracy.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with zipfile.ZipFile(
        evidence / "simulation-validation-raw.zip", "w", zipfile.ZIP_DEFLATED
    ) as z:
        for p in sorted(raw.rglob("*")):
            if p.is_file() and (
                p.suffix in (".json", ".py", ".txt") or p.name.endswith("trace.jsonl")
            ):
                z.write(p, Path("measured") / p.relative_to(raw))
        for p in sorted(new.parent.rglob("*.json")):
            z.write(p, Path("predicted") / p.relative_to(new.parent))
    md = (
        report["description"]
        + "\n\n经过算子profiling、command/host及WAN等修正后的误差，单位为百分比；MAPE为实验点等权的平均绝对相对误差，偏差为有符号均值。\n\n| 系统 | 组数 | QPS MAPE / 偏差 | TTFT均值 MAPE / 偏差 | TPOT均值 MAPE / 偏差 | TTFT P99 MAPE | TPOT P99 MAPE |\n|---|---:|---:|---:|---:|---:|---:|\n"
    )
    for g in groups:
        vals = [f"{g[k]['mape']:.1f} / {g[k]['bias']:+.2f}" for k in METRICS[:3]]
        md += (
            f"| {g['label']} | {g['count']} | "
            + " | ".join(vals)
            + f" | {g['ttft_p99_ms']['mape']:.1f} | {g['tpot_p99_ms']['mape']:.1f} |\n"
        )
    md += (
        "\n"
        + "\n\n".join(findings)
        + "\n\n[逐点CSV](demo/evidence/simulation-accuracy.csv) · [完整统计JSON](demo/evidence/simulation-accuracy.json) · [补测原始记录与预测](demo/evidence/simulation-validation-raw.zip)\n"
    )
    readme = ROOT / "README.md"
    text = readme.read_text()
    start = text.index("<!-- SIMULATION_ACCURACY_START -->") + len(
        "<!-- SIMULATION_ACCURACY_START -->"
    )
    end = text.index("<!-- SIMULATION_ACCURACY_END -->")
    readme.write_text(text[:start] + "\n" + md + text[end:])
    (ROOT / "Q4/docs/REPEATED_OPEN_LOOP.md").write_text(
        "# 开环重复验证与平均误差\n\n" + md.replace("](demo/", "](../../demo/")
    )
    print(json.dumps(groups, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--old",
        type=Path,
        default=ROOT / "Q4/docs/validation/open_loop/comparison.json",
    )
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    a = p.parse_args()
    publish(a.old, a.new, a.raw)
