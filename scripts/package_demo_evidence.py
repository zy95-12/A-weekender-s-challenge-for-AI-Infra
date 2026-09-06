"""Package measured serving points; never synthesize benchmark observations."""

import argparse, csv, hashlib, json, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", type=Path, required=True)
    p.add_argument("--serving-results", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "demo/evidence")
    a = p.parse_args()
    a.output.mkdir(exist_ok=True, parents=True)
    summaries = sorted(
        a.sweep.glob("*/summary.json"),
        key=lambda x: (json.loads(x.read_text())["concurrency"], x.parent.name),
    )
    points = []
    for path in summaries:
        s = json.loads(path.read_text())
        rows = json.loads((path.parent / "requests.json").read_text())
        measured = [
            r for r in rows if s["measurement_start"] <= r["end"] < s["measurement_end"]
        ]
        arrivals = [
            r
            for r in rows
            if s["measurement_start"] <= r["start"] < s["measurement_end"]
        ]
        assert len(measured) == s["requests"] and measured and arrivals
        assert all(r["error"] is None and r["tokens"] == 79 for r in rows)
        assert all(
            r["slo_pass"] == (r["ttft_ms"] <= 3000 and r["tpot_ms"] <= 100)
            for r in rows
        )
        assert abs(len(measured) / s["duration_s"] - s["completed_qps"]) < 1e-10
        for metric in ("ttft_ms", "tpot_ms"):
            values = sorted(r[metric] for r in measured)
            assert abs(sum(values) / len(values) - s["mean_" + metric]) < 1e-7
            ix = (len(values) - 1) * 0.99
            lo = int(ix)
            p99 = values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (
                ix - lo
            )
            assert abs(p99 - s["p99_" + metric]) < 1e-7
        assert (
            abs(
                sum(r["slo_pass"] for r in measured) / len(measured)
                - s["slo_attainment"]
            )
            < 1e-12
        )
        assert (
            abs(
                sum(r["slo_pass"] for r in arrivals) / len(arrivals)
                - s["start_cohort_slo_attainment"]
            )
            < 1e-12
        )
        s["joint_slo_pass"] = (
            min(s["slo_attainment"], s["start_cohort_slo_attainment"]) >= 0.99
        )
        s["artifact"] = path.parent.name
        points.append(s)
    # Prefer a longer confirmation run for the same C; retain all raw runs.
    selected = {}
    for s in points:
        if (
            s["concurrency"] not in selected
            or s["duration_s"] > selected[s["concurrency"]]["duration_s"]
        ):
            selected[s["concurrency"]] = s
    curve = [selected[c] for c in sorted(selected)]
    passing = [s for s in curve if s["joint_slo_pass"]]
    best = max(passing, key=lambda s: s["completed_qps"]) if passing else None
    failures = [s for s in curve if not s["joint_slo_pass"]]
    conclusion = (
        f"本次扫描最高合格吞吐：C{best['concurrency']}，{best['completed_qps']:.3f} QPS。"
        if best
        else "尚无合格点。"
    ) + (
        f" 首个已测失效点 C{failures[0]['concurrency']}；未声明全局最大容量。"
        if failures
        else " 尚未测到失效边界。"
    )
    report = {
        "source": "measured",
        "model": "Qwen2.5-3B-Instruct",
        "layer_split": [4, 27, 5],
        "input_tokens": 4096,
        "output_tokens": 79,
        "slo": {
            "ttft_ms": 3000,
            "request_mean_tpot_ms": 100,
            "joint_attainment": 0.99,
            "cohorts": ["completion", "arrival"],
        },
        "conclusion": conclusion,
        "points": curve,
        "all_runs": points,
    }
    (a.output / "optimized-sweep.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    fields = [
        "concurrency",
        "completed_qps",
        "mean_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "p99_tpot_ms",
        "slo_attainment",
        "start_cohort_slo_attainment",
        "joint_slo_pass",
        "requests",
        "duration_s",
        "artifact",
    ]
    with (a.output / "optimized-sweep.csv").open("w") as f:
        w = csv.DictWriter(
            f, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        w.writeheader()
        w.writerows(curve)
    with zipfile.ZipFile(
        a.output / "optimized-sweep-raw.zip", "w", zipfile.ZIP_DEFLATED
    ) as z:
        for path in a.sweep.rglob("*"):
            if path.is_file():
                z.write(path, "sweep/" + str(path.relative_to(a.sweep)))
        for path in a.serving_results.glob("*"):
            if path.is_file() and path.suffix in (".json", ".jsonl"):
                z.write(path, "serving/" + path.name)
        z.write(a.sweep / "launch.json", "launch.json")
        for name in [
            "scripts/pd_benchmark.py",
            "scripts/demo_sweep.py",
            "scripts/package_demo_evidence.py",
            "Q1/data/current_prompt.json",
            "Q1/data/current_reference.json",
        ]:
            z.write(ROOT / name, name)
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in a.output.iterdir()
        if p.is_file() and p.name != "SHA256SUMS.json"
    }
    (a.output / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2))
    print(conclusion)


if __name__ == "__main__":
    main()
