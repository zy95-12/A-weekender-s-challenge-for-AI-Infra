"""Recompute wall-clock C16 components from the packaged real trace."""

import json, statistics, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "demo/evidence"
with zipfile.ZipFile(BASE / "baseline-c16-raw.zip") as z:
    d = json.loads(z.read("c16.json"))
    rows = [json.loads(l) for l in z.read("trace.jsonl").splitlines()]
s = d["summary"]
ids = {
    r["request_id"]
    for r in d["all_requests"]
    if s["measurement_start"] <= r["end"] <= s["measurement_end"]
}
expected = json.loads((BASE / "baseline-c16.json").read_text())["breakdown"]
for phase, key in [("prefill", "ttft"), ("decode", "tpot")]:
    rr = [r for r in rows if r["client_request_id"] in ids and r["phase"] == phase]
    values = {
        "上行 RPC 区间": statistics.mean(r["upload_ms"] for r in rr),
        "下行 RPC 区间": statistics.mean(r["download_ms"] for r in rr),
        "云侧处理区间": statistics.mean(
            (r["cloud_send_ns"] - r["cloud_received_ns"]) / 1e6 for r in rr
        ),
    }
    wall = statistics.mean(r["step_wall_ms"] for r in rr)
    values["企业侧处理及本步其余开销"] = wall - sum(values.values())
    values["本步以外等待及交付残差"] = s[f"mean_{key}_ms"] - wall
    for name, v in values.items():
        assert abs(v - expected[key]["parts_ms"][name]) < 1e-9, (key, name)
    assert len(rr) == expected[key]["rows"]
print("C16 TTFT/TPOT evidence recomputes exactly from packaged raw data.")
