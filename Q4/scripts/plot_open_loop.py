"""Plot compare_open_loop.py output; optional dependency: matplotlib."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("comparison", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
rows = json.loads(a.comparison.read_text())
fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
for system, color in [("baseline", "#b04a35"), ("optimized", "#166b96")]:
    points = sorted(
        [r for r in rows if r["point"].startswith(system + "/")],
        key=lambda r: r["target_rate"],
    )
    for metric, ax, title in zip(
        ["qps", "ttft_mean_ms", "tpot_mean_ms"],
        axes,
        ["Completed QPS", "Arrival TTFT mean (ms)", "Arrival TPOT mean (ms)"],
    ):
        for source, style in [("measured", "o-"), ("simulated", "x--")]:
            ax.plot(
                [r["target_rate"] for r in points],
                [r[metric + "_" + source] for r in points],
                style,
                color=color,
                label=f"{system}: {source}",
            )
        ax.set(title=title, xlabel="Offered arrival rate (requests/s)")
        ax.grid(alpha=0.2)
axes[0].legend(fontsize=8)
fig.suptitle("Open-loop replay: identical arrivals, unchanged cost profiles")
fig.tight_layout()
fig.savefig(a.output, dpi=170)
