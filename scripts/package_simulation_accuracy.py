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
    old_boundary = next(r for r in rows if r["point"] == "optimized/rate-4.744")
    baseline = [r for r in rows if r["point"].startswith("baseline/")]
    optimized = [r for r in rows if r["point"].startswith("optimized/")]
    false_pass = [r for r in rows if r["simulated_pass"] and not r["measured_pass"]]
    repeat_base = [
        r
        for r in rows
        if r["point"].startswith("baseline/") and abs(r["target_rate"] - 0.75) < 0.001
    ]
    btp = [r["tpot_mean_ms_measured"] for r in repeat_base]
    bslo = [100 * r["arrival_cohort_slo_measured"] for r in repeat_base]
    findings = [
        f"① Decode与排队仍有残差：Baseline {len(baseline)}组 TPOT 均值 MAPE={groups[0]['tpot_mean_ms']['mape']:.1f}%，平均有符号偏差={groups[0]['tpot_mean_ms']['bias']:.1f}%。原 λ=1 组的平均在途请求实测9.07、仿真5.95（−34.4%）；仅完成QPS接近并不代表延迟准确。",
        f"② 尾部与相关性：{len(rows)}组中有{len(false_pass)}组被仿真误判为SLO通过。原优化 λ=4.744 组 TPOT 均值误差仅{old_boundary['tpot_mean_ms_error_pct']:+.1f}%，但P99实测{old_boundary['tpot_p99_ms_measured']:.2f}ms、仿真{old_boundary['tpot_p99_ms_simulated']:.2f}ms，实际达标率89.42%，预测100%。独立成本抽样未保留阶段间/时间上的相关性；这是待验证的机制原因，不能仅凭总指标完成归因。",
        "③ Profiling覆盖不等于全部精确命中：原优化 λ=4.440 的每个decode stage共2259次成本查询，精确batch占30.9%、插值62.9%、外推6.2%；host查询中70.1%借用邻近batch按请求分摊。C40成本表用于低负载有外推风险。",
        f"④ 到达波动与有限样本：Baseline λ=.75 的三个实测种子，TPOT均值范围{min(btp):.1f}–{max(btp):.1f}ms，达标率{min(bslo):.1f}%–{max(bslo):.1f}%。每组仿真精确回放同组实测时间表，因此这个波动不是该组仿真误差的借口，但说明单种子不能代表长期SLO容量。",
        "优化系统的边界重复结果也需注意：λ=4.440 的 seed43 完成4.725 QPS，但达标率94.26%、TPOT P99=104.76ms。原15.1倍是seed17最高观测通过点的比较，不是多种子稳定容量提升。",
        "⑤ 可迁移的是DAG、事件依赖和调度规则；算子/command/CPU收尾、WAN搬运、互联有效带宽与融合方式需重新校准。L20/H20/Ascend910B及其他模型仅提供公开参数理论预测，不纳入上述实测精度表。",
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
