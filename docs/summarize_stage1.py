"""Reproduce stage-1 representative statistics from completed local anchors."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/optimization_stage1"
variants = {"off": ROOT / "results/optimization_stage0/anchor_tp22",
            "shm_wire": OUT / "anchor_both_tp22", "shm_wire_tcp16": OUT / "anchor_all_tp22"}
result = {"scope": "TP2+2 representative anchors, not the full matrix", "rows": []}
conditions = [(512,128,1),(512,128,4),(8192,256,1),(8192,256,4)]
stem = "representative"
matrix_mode = len(sys.argv) == 2
if len(sys.argv) > 2:
    raise SystemExit("Usage: summarize_stage1.py [completed-stage1-matrix-directory]")
if matrix_mode:
    OUT = Path(sys.argv[1]).resolve()
    assert json.loads((OUT/"completion.json").read_text())["result"] == "COMPLETED"
    config = json.loads((OUT/"matrix_config.json").read_text())
    assert config["repeats"] == 3
    variants = {"tp_"+p.replace(":","_"): OUT/("tp_"+p.replace(":","_")) for p in config["pairs"].split(",")}
    conditions = [(isl,osl,int(c)) for isl,osl in config["workloads"] for c in config["concurrency"].split(",")]
    result.update(scope="Completed stage-1 matrix: ordinary serial execution only",config=config)
    stem = "stage1_time_breakdown"
large = []
for name, root in variants.items():
    if not matrix_mode:
        assert json.loads((root / "anchor.json").read_text())["result"] == "PASS"
    for isl, osl, c in conditions:
        points, requests, per_request = [], [], []
        for r in range(3):
            folder = root / f"isl_{isl}_osl_{osl}_c_{c}_r_{r}"
            point = folder / "qps_inf"
            assert json.loads((point / "measurement_validation.json").read_text())["result"] == "PASS"
            points.append(json.loads((folder / "summary.json").read_text())["points"][0])
            req = [json.loads(line) for line in (point / "requests.jsonl").read_text().splitlines()]
            assert len(req) == max(8,4*c) and all(not x["error"] and x["output_tokens"] == osl for x in req)
            requests += req
            traces = {x["client_request_id"]: [] for x in req}
            trace_path = point / "split_trace.jsonl"
            large.append({"path": str(trace_path.relative_to(ROOT)), "bytes": trace_path.stat().st_size,
                          "sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(), "storage": "server only"})
            for line in trace_path.read_text().splitlines():
                row = json.loads(line)
                if row["client_request_id"] in traces:
                    traces[row["client_request_id"]].append(row)
            for request in req:
                rows = traces[request["client_request_id"]]
                assert len(rows) == osl and sum(x["phase"] == "prefill" for x in rows) == 1
                values = {}
                for phase, target in [("prefill", "ttft_ms"),("decode", "tpot_ms")]:
                    selected = [x for x in rows if x["phase"] == phase]
                    q,u,d,rpc,step = [statistics.mean(x[k] for x in selected) for k in
                                     ("queue_ms","upload_ms","download_ms","rpc_wall_ms","step_wall_ms")]
                    values[phase] = dict(queue_ms=q,upload_ms=u,download_ms=d,cloud_service_ms=rpc-u-d,
                                         enterprise_other_ms=step-rpc,step_ms=step,boundary_gap_ms=request[target]-q-step)
                per_request.append(values)
        entry = dict(variant=name,isl=isl,osl=osl,concurrency=c,repeats=3,requests=len(requests),metrics={},trace={})
        for key, repeat_key in [("ttft_ms","mean_ttft_ms"),("tpot_ms","mean_tpot_ms")]:
            vals = [x[key] for x in requests]
            entry["metrics"][key] = {"mean":statistics.mean(vals),"repeat_mean_sample_std":statistics.stdev(x[repeat_key] for x in points),
                                     **{f"p{p}":float(np.percentile(vals,p)) for p in (50,95,99)}}
        qps = [x["achieved_qps"] for x in points]
        entry["metrics"]["qps"] = dict(mean=statistics.mean(qps),repeat_sample_std=statistics.stdev(qps),min=min(qps),max=max(qps))
        for phase in ("prefill","decode"):
            entry["trace"][phase] = {k:statistics.mean(x[phase][k] for x in per_request) for k in per_request[0][phase]}
        result["rows"].append(entry)
(OUT / (stem+"_summary.json")).write_text(json.dumps(result,indent=2))
(OUT / (stem+"_large_artifacts.json")).write_text(json.dumps({"files":large},indent=2))
print(json.dumps({"rows":len(result["rows"]),"trace_files":len(large),"bytes":sum(x["bytes"] for x in large)}))
