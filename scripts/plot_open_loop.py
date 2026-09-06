"""Export open-loop preset comparison using the same evidence as chapter 05."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "demo/evidence"
rows = json.loads((OUT / "open-loop-sweep.json").read_text())["points"]
for metric, slo in [("ttft", 3000), ("tpot", 100)]:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for system, color in [("baseline", "tab:orange"), ("optimized", "tab:green")]:
        rr = sorted(
            [r for r in rows if r["system"] == system],
            key=lambda r: r["target_arrival_rate"],
        )
        for stat, style in [("mean", "-"), ("p99", "--")]:
            ax.plot(
                [r["completed_qps"] for r in rr],
                [r[f"{stat}_{metric}_ms"] for r in rr],
                style,
                color=color,
                label=f"{system} {stat}",
            )
            for r in rr:
                ax.plot(
                    r["completed_qps"],
                    r[f"{stat}_{metric}_ms"],
                    "o",
                    color=color,
                    markerfacecolor=color if r["slo_pass"] else "white",
                )
        for r in rr:
            ax.annotate(
                f"λ={r['target_arrival_rate']:.2f}",
                (r["completed_qps"], r[f"mean_{metric}_ms"]),
                xytext=(4, 6),
                textcoords="offset points",
                fontsize=7,
                color=color,
            )
    ax.axhline(slo, color="tab:red", linestyle=":", label=f"SLO {slo} ms")
    ax.set(
        xlabel="Measured successful completed QPS",
        ylabel=f"{metric.upper()} (ms), arrival cohort",
        title="Open-loop Poisson arrivals / 4K + 79 tokens / 4 A10",
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.text(
        0.5,
        0.015,
        "Hollow markers: joint SLO fails. One seed per point; 60s or 120s measurement.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    for ext in ["png", "svg"]:
        fig.savefig(OUT / f"open-loop-{metric}-qps.{ext}", dpi=160)
    svg = OUT / f"open-loop-{metric}-qps.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
