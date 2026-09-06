"""Cancel only this script's request after a non-final prefill chunk.

Requires an idle, chunk-enabled deployment. No process termination. Checks both
KV pools eventually return to zero and records trace proving cancellation time.
"""
import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import sys
import time

import httpx
from network_state import network_lock, expected, check, snapshot, verify

ROOT = Path(__file__).resolve().parents[1]


async def cancel(args, output, report):
    from transformers import AutoTokenizer
    unit = AutoTokenizer.from_pretrained(str(ROOT/"models/qwen"),local_files_only=True).encode(" apple",add_special_tokens=False)
    assert len(unit) == 1
    live = Path((ROOT/"run/current_results").read_text().strip())
    trace = live/"split_trace.jsonl"
    offset = trace.stat().st_size
    rid = f"cancel-chunk-{time.time_ns()}"
    report["client_request_id"] = rid
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000",timeout=30,trust_env=False) as client:
        before = await client.get("/health")
        before.raise_for_status()
        report["before"] = before.json()
        assert report["before"]["active"] == 0 and report["before"]["waiting"] == 0
        size = report["before"]["optimizations"]["prefill_chunk_size"]
        assert 0 < size < args.isl, "Needs at least two prefill chunks"
        async with client.stream("POST","/v1/completions",headers={"X-Request-Id":rid},
                json={"prompt":unit*args.isl,"max_tokens":128,"ignore_eos":True,"stream":True}) as response:
            response.raise_for_status()
            deadline = time.monotonic()+30
            found = None
            with trace.open() as file:
                file.seek(offset)
                while time.monotonic() < deadline and found is None:
                    position = file.tell()
                    line = file.readline()
                    if not line.endswith("\n"):
                        file.seek(position)
                        await asyncio.sleep(.005)
                        continue
                    row = json.loads(line)
                    if row.get("client_request_id") == rid:
                        assert row["phase"] == "prefill" and not row["emits_token"], row
                        found = row
            assert found is not None, "No intermediate chunk observed"
            report["first_chunk"] = found
            report["cancel_time_ns"] = time.time_ns()
        deadline = time.monotonic()+15
        while time.monotonic() < deadline:
            after = await client.get("/health")
            after.raise_for_status()
            state = after.json()
            if state["active"] == 0 and state["waiting"] == 0 and state["kv_used_blocks"] == 0:
                report["after_enterprise"] = state
                break
            await asyncio.sleep(.05)
        assert "after_enterprise" in report, "Enterprise did not release cancelled KV"
    cloud = subprocess.run(["ip","netns","exec","split-enterprise","curl","-fsS","--max-time","5",
                            "http://10.205.0.2:8001/health"],capture_output=True,text=True,check=True)
    report["after_cloud"] = json.loads(cloud.stdout)
    assert report["after_cloud"]["active"] == 0 and report["after_cloud"]["kv_used_blocks"] == 0
    with trace.open() as file:
        file.seek(offset)
        rows = [row for line in file if (row:=json.loads(line)).get("client_request_id") == rid]
    (output/"split_trace.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    assert rows and all(not row["emits_token"] for row in rows if row["time_ns"] <= report["cancel_time_ns"]), \
        "Request already emitted a token before cancellation"
    report["observed_chunks"] = len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--isl",type=int,default=8192)
    args = parser.parse_args()
    if not 2 <= args.isl <= 16384-128:
        parser.error("Invalid input length")
    output = Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    report = {"result":"RUNNING","config":vars(args)}
    try:
        deployment = json.loads((ROOT/"run/launch.json").read_text())
        report["deployment"] = deployment
        intent = expected(deployment)
        check(intent,output/"network_check.json")
        subprocess.run([sys.executable,str(ROOT/"scripts/environment.py"),str(output/"environment.json")],cwd=ROOT,check=True)
        asyncio.run(cancel(args,output,report))
        observed = snapshot()
        (output/"network_after.json").write_text(json.dumps(observed,indent=2))
        verify(observed,intent)
        report["result"] = "PASS"
    except BaseException as error:
        report.update(result="FAIL",error=repr(error))
        raise
    finally:
        (output/"summary.json").write_text(json.dumps(report,indent=2))


if __name__ == "__main__":
    with network_lock():
        main()
