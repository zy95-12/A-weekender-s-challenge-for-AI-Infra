"""Read/echo real FP16 activations across the POC WAN; no model inference.

Matches the current HTTP + JSON/raw tensors + local multiprocessing Pipe path.
All phase intervals are explicitly scoped; server phases nest inside HTTP wall.
Run only on an idle POC. Creates only its own port 8099 server/CPU worker.
"""
import argparse
import asyncio
import json
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from split_poc.wire import pack, unpack


def echo_worker(pipe):
    while True:
        arrays = pipe.recv()
        if arrays is None:
            return
        pipe.send(arrays)


def server():
    from fastapi import FastAPI, Request, Response
    import uvicorn
    app = FastAPI()
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=echo_worker, args=(child,), daemon=True)
    process.start()
    child.close()
    lock = asyncio.Lock()

    @app.get("/health")
    def health():
        return {"ready": process.is_alive()}

    @app.post("/echo")
    async def echo(request: Request):
        start = time.perf_counter()
        body = bytearray()
        async for part in request.stream():
            body.extend(part)
            if len(body) > 160 * 1024**2:
                return Response(status_code=413)
        received = time.perf_counter()
        meta, arrays = unpack(body)
        parsed = time.perf_counter()
        async with lock:
            entered = time.perf_counter()
            if meta["ipc"]:
                await asyncio.to_thread(parent.send, arrays)
                arrays = await asyncio.to_thread(parent.recv)
            echoed = time.perf_counter()
        encoded = pack({}, arrays)
        finished = time.perf_counter()
        phases = {"body_receive_ms": (received-start)*1000,
                  "unpack_ms": (parsed-received)*1000,
                  "queue_ms": (entered-parsed)*1000,
                  "ipc_roundtrip_ms": (echoed-entered)*1000,
                  "pack_ms": (finished-echoed)*1000}
        return Response(encoded, media_type="application/octet-stream",
                        headers={"x-phases": json.dumps(phases)})

    @app.on_event("shutdown")
    def shutdown():
        if process.is_alive():
            process.terminate()
        process.join(5)
        parent.close()

    uvicorn.run(app, host="10.205.0.2", port=8099, access_log=False)


def client(args):
    import httpx
    sizes = [8192, 32768, 4*1024**2, 16*1024**2, 64*1024**2]
    rows = []
    concurrent_rows = []
    output = Path(args.output)
    with httpx.Client(base_url="http://10.205.0.2:8099", timeout=60, trust_env=False) as http:
        for size in sizes:
            # Both tensors together have exactly `size` bytes in each direction.
            arrays = [np.full((size//8192, 2048), value, np.float16) for value in (.25, -.5)]
            for ipc in (False, True):
                for repeat in range(-1, args.repeats):
                    start = time.perf_counter()
                    body = pack({"ipc": ipc}, arrays)
                    packed = time.perf_counter()
                    response = http.post("/echo", content=body)
                    response.raise_for_status()
                    received = time.perf_counter()
                    _, returned = unpack(response.content)
                    end = time.perf_counter()
                    for expected, actual in zip(arrays, returned):
                        np.testing.assert_array_equal(actual, expected)
                    if repeat < 0:
                        continue
                    rows.append({"bytes_each_direction": size, "ipc": ipc, "repeat": repeat,
                        "pack_ms": (packed-start)*1000, "http_wall_ms": (received-packed)*1000,
                        "unpack_ms": (end-received)*1000, "total_ms": (end-start)*1000,
                        "server": json.loads(response.headers["x-phases"])})
                print(f"PASS bytes={size} ipc={ipc}", flush=True)
        # Separate experiments: multiple messages in flight, same real TCP path.
        for size in (8192, 64*1024**2):
            arrays=[np.full((size//8192,2048),v,np.float16) for v in (.25,-.5)]
            body=pack({"ipc":True},arrays)
            def send_one(index):
                start=time.perf_counter()
                response=http.post("/echo",content=body)
                response.raise_for_status()
                end=time.perf_counter()
                _,result=unpack(response.content)
                for expected,actual in zip(arrays,result):
                    np.testing.assert_array_equal(expected,actual)
                return {"http_wall_ms":(end-start)*1000,"server":json.loads(response.headers["x-phases"])}
            for inflight in (2,4):
                with ThreadPoolExecutor(max_workers=inflight) as pool:
                    list(pool.map(send_one,range(inflight)))  # unmeasured warmup
                    for repeat in range(3):
                        start=time.perf_counter()
                        measured=list(pool.map(send_one,range(inflight)))
                        elapsed=time.perf_counter()-start
                        concurrent_rows.append({"bytes_each_direction":size,"inflight":inflight,
                            "repeat":repeat,"requests":measured,"group_wall_ms":elapsed*1000,
                            "aggregate_payload_gbps":2*size*inflight*8/elapsed/1e9})
                print(f"PASS inflight={inflight} bytes={size}",flush=True)
        tcp=subprocess.run(["ss","-tinm"],capture_output=True,text=True).stdout
    summary=[]
    for size in sizes:
        for ipc in (False, True):
            selected=[r for r in rows if r["bytes_each_direction"]==size and r["ipc"]==ipc]
            summary.append({"bytes_each_direction":size,"ipc":ipc,"n":len(selected),
                **{k:statistics.mean(r[k] for r in selected) for k in ("pack_ms","http_wall_ms","unpack_ms","total_ms")},
                "server":{k:statistics.mean(r["server"][k] for r in selected) for k in selected[0]["server"]}})
    output.write_text(json.dumps({"result":"PASS","command":sys.argv,"rows":rows,"summary":summary,
        "scope":"CPU tensor echo; no GPU compute. Server phases nest inside client HTTP wall. One unmeasured warmup per size/mode.",
        "concurrent_rows":concurrent_rows,"tcp":tcp},indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["run","server","client"])
    parser.add_argument("--output",default="results/optimization_stage0/transport")
    parser.add_argument("--repeats",type=int,default=10)
    args=parser.parse_args()
    if args.action=="server":
        server()
    elif args.action=="client":
        client(args)
    else:
        from network_state import network_lock,expected,check,snapshot,verify
        import httpx
        output=Path(args.output).resolve()
        output.mkdir(parents=True,exist_ok=False)
        with network_lock():
            intent=expected(json.loads((ROOT/"run/launch.json").read_text()))
            check(intent,output/"network_check.json")
            subprocess.run([sys.executable,str(ROOT/"scripts/environment.py"),str(output/"environment.json")],check=True,cwd=ROOT)
            with (output/"server.log").open("w") as log:
                process=subprocess.Popen(["ip","netns","exec","split-cloud",sys.executable,__file__,"server"],stdout=log,stderr=subprocess.STDOUT)
                try:
                    for _ in range(60):
                        if process.poll() is not None:
                            raise RuntimeError("microbench server exited")
                        ready=subprocess.run(["ip","netns","exec","split-enterprise","curl","-fsS","http://10.205.0.2:8099/health"],capture_output=True)
                        if ready.returncode==0:
                            break
                        time.sleep(.5)
                    else:
                        raise RuntimeError("microbench server not ready")
                    subprocess.run(["ip","netns","exec","split-enterprise",sys.executable,__file__,"client",
                        "--repeats",str(args.repeats),"--output",str(output/"measurement.json")],check=True)
                    after=snapshot()
                    (output/"network_after.json").write_text(json.dumps(after,indent=2))
                    verify(after,intent)
                finally:
                    process.terminate()
                    process.wait(timeout=15)


if __name__=="__main__":
    main()
