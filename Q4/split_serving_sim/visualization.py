from __future__ import annotations

import html
from pathlib import Path
from typing import Any


STAGE_COLORS = {
    "edge_front": "#2563eb",
    "wan_up": "#06b6d4",
    "cloud_middle": "#7c3aed",
    "wan_down": "#14b8a6",
    "edge_tail": "#f59e0b",
}


def render_gantt_html(traces: list[dict[str, Any]], destination: Path) -> None:
    """Render resource intervals as a standalone, dependency-free SVG page."""
    if not traces:
        destination.write_text("<html><body><p>No trace data.</p></body></html>\n")
        return

    preferred = ["edge_gpu", "wan_up", "cloud_gpu", "wan_down"]
    discovered = {str(row["resource"]) for row in traces}
    resources = [name for name in preferred if name in discovered]
    resources.extend(sorted(discovered - set(resources)))

    start_ms = min(float(row["start_time_ms"]) for row in traces)
    end_ms = max(float(row["end_time_ms"]) for row in traces)
    duration_ms = max(end_ms - start_ms, 1e-9)
    label_width = 130
    plot_width = 1400
    top = 70
    lane_height = 64
    bar_height = 34
    svg_width = label_width + plot_width + 30
    svg_height = top + lane_height * len(resources) + 70

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Split-serving pipeline Gantt</title>",
        "<style>",
        "body{font-family:ui-sans-serif,system-ui;margin:24px;color:#172033}",
        ".viewport{overflow-x:auto;border:1px solid #d9dfeb;border-radius:10px}",
        "svg{background:#fff}.lane{fill:#f8fafc}.grid{stroke:#dce3ed;stroke-width:1}",
        ".bar{stroke:#172033;stroke-width:.5}.decode{stroke-width:1.7}",
        ".legend{display:flex;gap:16px;flex-wrap:wrap;margin:12px 0}",
        ".swatch{display:inline-block;width:12px;height:12px;margin-right:5px;border-radius:2px}",
        "code{background:#f1f5f9;padding:2px 5px;border-radius:4px}",
        "</style></head><body>",
        "<h1>Split-serving pipeline</h1>",
        (
            f"<p>Timeline: <code>{start_ms:.3f} ms</code> – "
            f"<code>{end_ms:.3f} ms</code>; bold outline means the batch contains decode work. "
            "Hover a bar for batch/request details.</p>"
        ),
        "<div class='legend'>",
    ]
    for stage, color in STAGE_COLORS.items():
        parts.append(
            f"<span><i class='swatch' style='background:{color}'></i>{stage}</span>"
        )
    parts.extend(
        [
            "</div><div class='viewport'>",
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{svg_width}' height='{svg_height}' "
            f"viewBox='0 0 {svg_width} {svg_height}'>",
        ]
    )

    for index, resource in enumerate(resources):
        y = top + index * lane_height
        parts.append(
            f"<rect class='lane' x='0' y='{y}' width='{svg_width}' height='{lane_height}'/>"
        )
        parts.append(
            f"<text x='12' y='{y + 27}' font-size='14' font-weight='600'>"
            f"{html.escape(resource)}</text>"
        )

    tick_count = 10
    for tick in range(tick_count + 1):
        ratio = tick / tick_count
        x = label_width + ratio * plot_width
        value = start_ms + ratio * duration_ms
        parts.append(
            f"<line class='grid' x1='{x:.2f}' y1='48' x2='{x:.2f}' "
            f"y2='{top + lane_height * len(resources)}'/>"
        )
        parts.append(
            f"<text x='{x:.2f}' y='38' text-anchor='middle' font-size='11'>"
            f"{value:.1f} ms</text>"
        )

    resource_index = {name: index for index, name in enumerate(resources)}
    for row in sorted(traces, key=lambda value: (value["start_time_ms"], value["batch_id"])):
        stage = str(row["stage"])
        start = float(row["start_time_ms"])
        end = float(row["end_time_ms"])
        x = label_width + (start - start_ms) / duration_ms * plot_width
        width = max((end - start) / duration_ms * plot_width, 0.8)
        y = top + resource_index[str(row["resource"])] * lane_height + 10
        phases = list(row["phases"])
        css_class = "bar decode" if "decode" in phases else "bar"
        color = STAGE_COLORS.get(stage, "#64748b")
        tooltip = html.escape(
            " | ".join(
                [
                    f"batch={row['batch_id']}",
                    f"stage={stage}",
                    f"phase={','.join(phases)}",
                    f"requests={row['request_ids']}",
                    f"start={start:.3f} ms",
                    f"end={end:.3f} ms",
                    f"duration={float(row['duration_ms']):.3f} ms",
                    f"tokens={row['total_tokens']}",
                ]
            )
        )
        parts.append(
            f"<g><rect class='{css_class}' x='{x:.2f}' y='{y}' width='{width:.2f}' "
            f"height='{bar_height}' rx='2' fill='{color}'><title>{tooltip}</title></rect>"
        )
        if width >= 24:
            parts.append(
                f"<text x='{x + 4:.2f}' y='{y + 22}' fill='white' font-size='10' "
                f"pointer-events='none'>B{row['batch_id']}</text>"
            )
        parts.append("</g>")

    parts.extend(["</svg></div></body></html>\n"])
    destination.write_text("".join(parts), encoding="utf-8")
