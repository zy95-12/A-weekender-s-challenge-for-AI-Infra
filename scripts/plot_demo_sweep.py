"""Export the same measured curve shown by the UI as standalone PNG/SVG."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo/evidence"
rows = json.loads((OUT / "optimized-sweep.json").read_text())["points"]
for metric, slo in [("ttft", 3000), ("tpot", 100)]:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for stat, marker in [("mean", "o"), ("p99", "s")]:
        ax.plot(
            [r["completed_qps"] for r in rows],
            [r[f"{stat}_{metric}_ms"] for r in rows],
            marker=marker,
            label=stat,
        )
    for r in rows:
        ax.annotate(
            f"C{r['concurrency']}",
            (r["completed_qps"], r[f"mean_{metric}_ms"]),
            xytext=(4, 7),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(slo, color="tab:red", linestyle="--", label=f"SLO {slo} ms")
    ax.set(
        xlabel="Measured completed QPS",
        ylabel=metric.upper() + " (ms)",
        title="Qwen2.5-3B / 4 A10 / optimized dual-P PD / 4K + 79 tokens",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"optimized-{metric}-qps.{ext}", dpi=160)
    svg = OUT / f"optimized-{metric}-qps.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    plt.close(fig)
