"""Render the measured fixed-SLO comparison without conflating failed QPS."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/evidence/issue-6/pd-slo'


def main():
    rows=list(csv.DictReader((OUT/'curve.csv').open()))
    selection=json.loads((OUT/'selection.json').read_text())
    best={name:selection[name]['highest_confirmed_slo_qps'] for name in selection}
    b,o=best['baseline'],best['optimized'];ratio=o['completed_qps']/b['completed_qps']
    practical=min((r for r in rows if r['system']=='optimized' and r['slo_pass']=='True'
                   and float(r['completed_qps'])>=.95*o['completed_qps']),key=lambda r:int(r['concurrency']))
    text=f'''# 固定 SLO 下的最大实测 QPS 与并发拐点

当前 4096 输入 / 79 输出 tokens、4×A10、WAN 10 Gbps / 单程 5 ms 的持续闭环负载下，推荐配置在 **C{o['concurrency']} 测得 {o['completed_qps']:.3f} QPS**，baseline 在 **C{b['concurrency']} 测得 {b['completed_qps']:.3f} QPS**，最高确认合规吞吐约 **{ratio:.2f} 倍**。

SLO：至少 99% 请求**同时**满足 TTFT≤3s、请求平均 TPOT≤100ms；完成与开始 cohort 都检查。以下容量来自至少 180 秒的确认窗口。超过 SLO 的点仍画在曲线上，但不计入合规容量。

| 系统 | 确认并发 | 正式请求数 | 窗口 s | QPS | TTFT 均值 / P99 ms | TPOT 均值 / P99 ms | 完成 / 开始 cohort 达标率 |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
    for name,label in [('baseline','Baseline（保守）'),('optimized','推荐配置')]:
        r=best[name]
        text+=f"| {label} | {r['concurrency']} | {r['requests']} | {r['duration_s']:.1f} | {r['completed_qps']:.3f} | {r['mean_ttft_ms']:.1f} / {r['p99_ttft_ms']:.1f} | {r['mean_tpot_ms']:.2f} / {r['p99_tpot_ms']:.2f} | {r['slo_attainment']:.2%} / {r['start_cohort_slo_attainment']:.2%} |\n"
    if 'repeated_boundary' in selection['baseline']:
        previous=list(csv.DictReader((OUT/'all_points.csv').open()))
        boundary=[r for r in previous if r['system']=='baseline' and int(r['concurrency'])==4]
        text+='\nBaseline C4 的所有窗口如下；最后一次是因为前后结果矛盾而额外重复，未选择 QPS 更高的窗口覆盖它。\n\n| 窗口 | 秒数 | QPS | TTFT P99 ms | 完成 / 开始 cohort 达标率 |\n|---|---:|---:|---:|---:|\n'
        for r in boundary:
            text+=f"| {r['variant']} | {float(r['duration_s']):.1f} | {float(r['completed_qps']):.3f} | {float(r['p99_ttft_ms']):.1f} | {float(r['slo_attainment']):.2%} / {float(r['start_cohort_slo_attainment']):.2%} |\n"
        if not selection['baseline']['repeated_boundary']['latest']['slo_pass']:
            single=max(float(r['completed_qps']) for r in boundary if r['slo_pass']=='True')
            text=text.replace(f'最高确认合规吞吐约 **{ratio:.2f} 倍**',f'相对 baseline 最佳单次合规窗口 **{single:.3f} QPS**，提升 **{o["completed_qps"]/single:.2f} 倍**；若按保守稳定点 C3 比较，则为 **{ratio:.2f} 倍**')
            text+=f"\nC4 的一次长窗口曾通过、重复未通过，因此 C4 标为不稳定，采用 C3 作为保守容量。**若以 baseline 最佳单次通过窗口 {single:.3f} QPS 比较，优化版提升为 {o['completed_qps']/single:.2f} 倍**；不能只报保守 baseline 分母对应的更大倍数。\n"
        else:
            text+='\nC4 两次长窗口均通过，但粗扫曾失败且 TTFT 接近 3s，属于贴边容量；需要更大延迟余量时 baseline 应退至 C3。\n'
    text+='\n## 拐点与运行建议\n\n'
    for name,label in [('baseline','Baseline（保守）'),('optimized','推荐配置')]:
        s=selection[name];fail=s['next_concurrency_confirmed_failure']
        if fail is not None:
            r=next(r for r in rows if r['system']==name and int(r['concurrency'])==fail)
            text+=f"- {label}：采用的通过并发 C{s['largest_confirmed_passing_concurrency']}；紧邻的 C{fail} 确认失败，其 TTFT P99 为 {float(r['p99_ttft_ms']):.1f} ms、TPOT P99 为 {float(r['p99_tpot_ms']):.2f} ms，完成 cohort 达标率 {float(r['slo_attainment']):.2%}。\n"
    text+=f"\n在已测点中，**C{practical['concurrency']} 已达到最高确认合规 QPS 的 {float(practical['completed_qps'])/o['completed_qps']:.1%}**，但平均 TTFT 仅 {float(practical['mean_ttft_ms']):.1f} ms。这是更稳妥的运行点；把并发推到 SLO 边缘换来的吞吐收益很小。这个判断区分了吞吐开始趋平与 SLO 上边界，不把它们当成同一个拐点。\n"
    text+='\n优化版各点的首次 prefill 等待与计算后流水跨度：\n\n| 并发 | QPS | 首次 prefill 开始前等待均值 ms | 开始后 prefill 流水跨度均值 ms | 请求步加权 decode batch 大小 |\n|---|---:|---:|---:|---:|\n'
    for r in rows:
        if r['system']=='optimized' and int(r['concurrency']) in {8,16,24,32,48,o['concurrency']}:
            text+=f"| C{r['concurrency']} | {float(r['completed_qps']):.3f} | {float(r['mean_first_prefill_wait_ms']):.1f} | {float(r['mean_prefill_pipeline_span_ms']):.1f} | {float(r['request_weighted_decode_batch_size']):.2f} |\n"
    text+='''
这些是请求与调度 trace 的时间分解，首次等待包含请求进入系统至第一次企业 front 的全部时间。健康采样同时保留 active、waiting 与 PD reservation 数，可检查是否撞到容量上限。它们可以定位延迟堆积的位置，不能单凭这些时间就把瓶颈归因为某个 GPU 算子。

## 曲线

横轴 QPS 均为窗口内总完成吞吐；圆点表示联合 SLO 通过，叉号表示失败，星标表示最高确认合规 QPS。TTFT 与 TPOT 都同时画均值和 P99；并发标签标出主要采样点。每个 C 优先采用最后一次确认窗口，否则采用细化或粗扫窗口；TTFT/TPOT 图中淡色标记保留先前窗口，原始重复测量不丢弃。

![TTFT–QPS](evidence/issue-6/pd-slo/ttft_qps.png)

![TPOT–QPS](evidence/issue-6/pd-slo/tpot_qps.png)

![Concurrency–QPS](evidence/issue-6/pd-slo/concurrency_qps.png)

## 全部曲线点

延迟单元格为均值 / P99，单位 ms；达标率取两种 cohort 中较小者。

| 系统 | C | 窗口类型 | QPS | TTFT | TPOT | 达标率 | SLO |
|---|---:|---|---:|---:|---:|---:|---|
'''
    for r in rows:
        text+=f"| {r['system']} | {r['concurrency']} | {r['stage']} | {float(r['completed_qps']):.3f} | {float(r['mean_ttft_ms']):.1f} / {float(r['p99_ttft_ms']):.1f} | {float(r['mean_tpot_ms']):.2f} / {float(r['p99_tpot_ms']):.2f} | {min(float(r['slo_attainment']),float(r['start_cohort_slo_attainment'])):.2%} | {'PASS' if r['slo_pass']=='True' else 'FAIL'} |\n"
    text+='''
## 配置、验证与证据

推荐配置是 E1/P2/D1＋Stage 1/2/3＋独立控制通道，**关闭 KV chunk 迁移**；baseline 为 E2＋共享云 TP2、Stage 1/2/3 关闭。两组使用同一模型、切层、prompt、网络，活跃上限均 96、KV 页数均 32768，避免原来的 C16 启动器限制或 KV 容量制造假拐点。默认启动参数没有改变，扩容仅在本实验显式指定。

62 个已有 CPU 测试通过。原始请求逐个核对 79-token 输出及绝对位置；PD 另外检查整段迁移范围、首次 decode 不早于 KV-ready、源 KV 释放晚于传输完成；每点结束无残留 KV。请求平均 TPOT 与全部逐 token 间隔、两个 cohort 的 SLO 均从原始数据复算。

本报告给出有限持续闭环窗口内的**最大实测合规吞吐**，不把它当作任意工作负载、开放到达率或未来请求分布的容量保证。尤其短窗口粗扫的 P99 是描述性值，边界取更长的确认结果。模型输出长度改变后需重新测量。

[完整实验方法与复现命令](pd-slo-method.md) · [全部重复测量 CSV](evidence/issue-6/pd-slo/all_points.csv) · [绘图 CSV](evidence/issue-6/pd-slo/curve.csv) · [审计结果](evidence/issue-6/pd-slo/audit.json)

[Baseline 原始数据](evidence/issue-6/pd-slo/baseline_raw.zip) · [优化版原始数据](evidence/issue-6/pd-slo/optimized_raw.zip) · [复算代码](evidence/issue-6/pd-slo/reproduce.zip) · [SHA256](evidence/issue-6/pd-slo/SHA256SUMS.json)
'''
    (ROOT/'docs/pd-slo-results.md').write_text(text)
    print('Wrote docs/pd-slo-results.md')


if __name__=='__main__':main()
