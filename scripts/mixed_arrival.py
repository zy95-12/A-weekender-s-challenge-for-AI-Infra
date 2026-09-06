"""Controlled long-prefill arrival during decode; separate from official baseline.

SSE content-event gaps include empty text events: the current POC emits one such
event per target token. Verify event count against usage; do not silently reuse
this metric for a future protocol that groups speculative tokens into one event.
"""
import argparse
import asyncio
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import httpx
import numpy as np
from network_state import network_lock, expected, check, snapshot, verify
from benchmark_artifacts import export_trace

ROOT = Path(__file__).resolve().parents[1]


async def pair(client, unit, args, prefix):
    trigger = asyncio.Event()

    async def request(index, isl, osl):
        if index:
            await asyncio.wait_for(trigger.wait(), timeout=60)
        started = time.perf_counter_ns()
        stamps, usage, done = [], None, False
        async with client.stream("POST", "/v1/completions", headers={"X-Request-Id": prefix + str(index)},
                json={"prompt": unit * isl, "max_tokens": osl, "temperature": 0, "ignore_eos": True,
                      "stream": True, "stream_options": {"include_usage": True}}) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    done = True
                    break
                data = json.loads(line[6:])
                if "error" in data:
                    raise RuntimeError(data["error"])
                if data.get("choices"):
                    stamps.append(time.perf_counter_ns())
                    if index == 0 and len(stamps) == args.trigger_after:
                        trigger.set()
                if data.get("usage"):
                    usage = data["usage"]
        assert done and len(stamps) == osl, (len(stamps), osl, done)
        assert usage["prompt_tokens"] == isl and usage["completion_tokens"] == osl, usage
        gaps = np.diff(stamps) / 1e6
        return {"client_request_id": prefix + str(index), "isl": isl, "osl": osl,
                "started_monotonic_ns": started, "event_monotonic_ns": stamps, "usage": usage,
                "ttft_ms": (stamps[0]-started)/1e6, "tpot_ms": float(np.mean(gaps)),
                "event_gap_max_ms": float(np.max(gaps)),
                **{f"event_gap_p{p}_ms": float(np.percentile(gaps,p)) for p in (50,95,99)}}

    tasks = [asyncio.create_task(request(0,512,args.decode_osl)),
             asyncio.create_task(request(1,args.long_isl,args.long_osl))]
    try:
        rows = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    rows[0]["long_arrival_after_start_ms"] = (rows[1]["started_monotonic_ns"]-rows[0]["started_monotonic_ns"])/1e6
    assert rows[0]["event_monotonic_ns"][0] <= rows[1]["started_monotonic_ns"] < rows[0]["event_monotonic_ns"][-1], \
        "Long prefill did not arrive during decode; not a valid overlap workload"
    return rows


async def measure(args, output, unit, report):
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000",timeout=300,trust_env=False) as client:
        health = await client.get("/health")
        health.raise_for_status()
        current = health.json()
        validated = json.loads(Path(args.correctness_report).read_text())
        assert validated["result"] == "PASS"
        for key in ("split","layer_split","tp","cloud_tp","model_id","revision","protocol","optimizations"):
            assert current.get(key) == validated["server"].get(key), key
        assert current["active"] == 0 and current["waiting"] == 0
        report["server"] = current
        # Excluded warmup, not part of the controlled arrival workload.
        warmup = await client.post("/v1/completions",json={"prompt":unit*512,"max_tokens":8,"ignore_eos":True})
        warmup.raise_for_status()
        live = Path((ROOT/"run/current_results").read_text().strip())
        for name in ("environment.json","enterprise_config.json","cloud_config.json"):
            shutil.copyfile(live/name,output/("launch_environment.json" if name=="environment.json" else name))
        for repeat in range(args.repeats):
            folder = output / f"repeat_{repeat}"
            folder.mkdir()
            trace = live / "split_trace.jsonl"
            offset = trace.stat().st_size
            prefix = f"mixed-{time.time_ns()}-"
            started = time.perf_counter()
            rows = await pair(client,unit,args,prefix)
            wall = time.perf_counter()-started
            (folder/"requests.json").write_text(json.dumps(rows,indent=2))
            export_trace(trace,offset,folder,prefix,2)
            report["repeats"].append({"repeat":repeat,"pair_wall_seconds":wall,"pair_completion_qps":2/wall,
                                      "requests":rows})
            print(f"DONE repeat {repeat}: decode max event gap {rows[0]['event_gap_max_ms']:.2f} ms",flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--correctness-report",required=True)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--long-isl",type=int,default=8192)
    parser.add_argument("--long-osl",type=int,default=128)
    parser.add_argument("--decode-osl",type=int,default=128)
    parser.add_argument("--trigger-after",type=int,default=8)
    args = parser.parse_args()
    if args.repeats < 1 or not 1 <= args.trigger_after < args.decode_osl:
        parser.error("Invalid repetitions or trigger")
    if not (2 <= args.decode_osl <= 1024 and 2 <= args.long_osl <= 1024 and
            1 <= args.long_isl <= 16384-args.long_osl):
        parser.error("Invalid input/output lengths")
    output = Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    report = {"result":"RUNNING","config":vars(args),"repeats":[],
              "classification":"controlled mixed arrival, not official baseline or a capacity claim",
              "metric_scope":"SSE content-event gaps, including empty text; event count must equal usage tokens"}
    summary = output/"summary.json"
    summary.write_text(json.dumps(report,indent=2))
    try:
        deployment = json.loads((ROOT/"run/launch.json").read_text())
        assert not deployment["profile"] and not deployment.get("phase_profile",False)
        report["deployment"] = deployment
        intent = expected(deployment)
        check(intent,output/"network_check.json")
        before = snapshot()
        verify(before,intent)
        (output/"network_before.json").write_text(json.dumps(before,indent=2))
        subprocess.run([sys.executable,str(ROOT/"scripts/environment.py"),str(output/"environment.json")],cwd=ROOT,check=True)
        from transformers import AutoTokenizer
        unit = AutoTokenizer.from_pretrained(str(ROOT/"models/qwen"),local_files_only=True).encode(" apple",add_special_tokens=False)
        assert len(unit) == 1
        asyncio.run(measure(args,output,unit,report))
        after = snapshot()
        (output/"network_after.json").write_text(json.dumps(after,indent=2))
        verify(after,intent)
        report["result"] = "PASS"
    except BaseException as error:
        report.update(result="FAIL",error=repr(error))
        raise
    finally:
        summary.write_text(json.dumps(report,indent=2))


if __name__ == "__main__":
    with network_lock():
        main()
