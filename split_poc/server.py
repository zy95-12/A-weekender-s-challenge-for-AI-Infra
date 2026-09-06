"""Streaming demonstration API and naive round-robin batch scheduler."""
import argparse
import asyncio
import contextlib
import json
import os
from pathlib import Path
import queue
import threading
import time
import uuid

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
import uvicorn

from split_poc import MODEL_ID, REVISION, SPLITS
from split_poc.runtime import Executor
from split_poc.wire import MAX_BODY, pack, unpack


class Job:
    def __init__(self, ids, limit, ignore_eos, forced=None, capture=False):
        self.id = uuid.uuid4().hex
        self.client_id = self.id
        self.ids, self.limit, self.ignore_eos = ids, limit, ignore_eos
        self.forced, self.capture = forced, capture
        self.tokens, self.logits = [], []
        self.events = queue.Queue()
        self.cancelled = False
        self.created = time.perf_counter()
        self.last_step = self.created
        self.prefilled = False
        self.position = 0


class Scheduler:
    def __init__(self, executor, args, eos):
        self.executor, self.args, self.eos = executor, args, eos
        self.pending = queue.Queue(maxsize=256)
        self.active = []
        self.closed = False
        self.completed = self.failed = self.steps = self.kv_used = 0
        self.prompt_tokens = self.generation_tokens = 0
        self.last_trace = None
        self.decode_rounds = 0
        self.http = httpx.Client(base_url=args.cloud, timeout=120, trust_env=False)
        Path(args.results).mkdir(parents=True, exist_ok=True)
        self.trace = open(Path(args.results) / "split_trace.jsonl", "a", buffering=1)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, job):
        if not self.executor.healthy:
            raise HTTPException(503, "GPU executor unhealthy; restart required")
        try:
            self.pending.put_nowait(job)
        except queue.Full:
            raise HTTPException(429, "Request queue is full")

    def release(self, jobs):
        ids = [job.id for job in jobs]
        if not ids:
            return
        try:
            local = self.executor.call({"op": "release", "ids": ids})
            self.kv_used = local["kv_used_blocks"]
            response = self.http.post("/release", json={"ids": ids})
            response.raise_for_status()
        except Exception:
            self.executor.healthy = False

    def run(self):
        while not self.closed:
            try:
                if not self.active:
                    try:
                        self.active.append(self.pending.get(timeout=0.1))
                    except queue.Empty:
                        continue
                cancelled = [j for j in self.active if j.cancelled]
                self.release(cancelled)
                self.active = [j for j in self.active if not j.cancelled]
                # At most one new prefill between decode batches. This is a
                # bounded, naive scheduler; no overlap or pipeline optimisation.
                if len(self.active) < self.args.max_active:
                    with contextlib.suppress(queue.Empty):
                        self.active.append(self.pending.get_nowait())
                if not self.active:
                    continue
                from split_poc.scheduling import choose
                batch, phase, self.decode_rounds = choose(self.active, self.args.scheduler_policy,
                                                         self.decode_rounds, self.args.decode_quota)
                items, tokens = [], []
                for job in batch:
                    ids = job.ids[job.position:job.position+(self.args.prefill_chunk_size or len(job.ids))] if phase == "prefill" else [
                        job.forced[len(job.tokens) - 1] if job.forced is not None else job.tokens[-1]]
                    items.append({"request_id": job.id, "position": job.position, "query_len": len(ids)})
                    tokens.extend(ids)
                command = {"op": "forward", "phase": phase, "items": items,
                           "token_ids": tokens, "batch_id": uuid.uuid4().hex,
                           "capture": any(j.capture for j in batch)}
                command["emit"] = phase == "decode" or all(
                    j.position + item["query_len"] == len(j.ids) for j,item in zip(batch,items))
                t = time.perf_counter()
                result = self.executor.call(command)
                elapsed = (time.perf_counter() - t) * 1000
                self.steps += 1
                self.kv_used = result["kv_used_blocks"]
                finished = []
                for i, job in enumerate(batch):
                    now = time.perf_counter()
                    token = result["tokens"][i]
                    emits = phase == "decode" or job.position + items[i]["query_len"] == len(job.ids)
                    if emits:
                        job.tokens.append(token)
                    if job.capture and emits:
                        job.logits.append(result["logits"][i])
                    trace = {"request_id": job.id, "client_request_id": job.client_id, "time_ns": time.time_ns(),
                        "batch_id": command["batch_id"], "batch_size": len(batch),
                        "phase": phase, "token_idx": len(job.tokens) - 1,
                        "emits_token": emits, "query_len": items[i]["query_len"], "position_start": job.position,
                        "context_len": job.position + items[i]["query_len"],
                        "queue_ms": (t - (job.created if job.position == 0 else job.last_step)) * 1000,
                        "step_wall_ms": elapsed, "timings_scope": "batch",
                        **result["timings"]}
                    self.trace.write(json.dumps(trace) + "\n")
                    self.last_trace = trace
                    if phase == "prefill":
                        self.prompt_tokens += items[i]["query_len"]
                    self.generation_tokens += int(emits)
                    job.position += items[i]["query_len"]
                    job.last_step, job.prefilled = now, job.position >= len(job.ids)
                    if not emits:
                        continue
                    done = len(job.tokens) >= job.limit or (token in self.eos and not job.ignore_eos)
                    job.events.put({"token": token, "done": done,
                                    "finish_reason": "length" if len(job.tokens) >= job.limit else "stop"})
                    if done:
                        finished.append(job)
                        self.completed += 1
                self.release(finished)
                self.active = [j for j in self.active if j not in finished]
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self.executor.healthy = False
                for job in self.active:
                    job.events.put({"error": str(exc)})
                    self.failed += 1
                self.active = []
                while not self.pending.empty():
                    self.pending.get_nowait().events.put({"error": "Executor unavailable"})
                time.sleep(0.1)


def optimization_config(args):
    return {key: getattr(args,key) for key in ("ipc_mode","wire_fast","tcp_buffer_mib",
            "prefill_chunk_size","scheduler_policy","decode_quota","pipeline_window")}


def create_app(args):
    app = FastAPI(title=f"Split-vLLM {args.role}")
    cloud_tp = None
    if args.role == "enterprise":
        remote = httpx.get(args.cloud + "/health", timeout=5, trust_env=False)
        remote.raise_for_status()
        remote_config = remote.json()
        if (remote_config.get("split"), remote_config.get("model_id"), remote_config.get("revision"), remote_config.get("protocol")) != (args.split, MODEL_ID, REVISION, 1):
            raise RuntimeError("Cloud model/split/protocol handshake mismatch")
        if tuple(remote_config.get("layer_split", [])) != SPLITS[args.split]:
            raise RuntimeError("Cloud layer partition differs from Enterprise")
        cloud_tp = remote_config["tp"]
        if remote_config.get("optimizations") != optimization_config(args):
            raise RuntimeError("Cloud optimization flags differ from Enterprise")
    executor = Executor(vars(args))
    app.state.executor = executor
    lock = threading.Lock()
    from split_poc.pipeline_state import CausalGate
    gate = CausalGate(timeout=30) if args.pipeline_window and args.pd_role!='decode' else None
    last_seen = {}
    cloud_metrics = {"kv_used_blocks": 0, "prefill_tokens": 0, "decode_tokens": 0, "forward_steps": 0}
    pd_control=None
    if args.pd_role:
        from split_poc.pd_control import install
        pd_control=install(app,args,executor,cloud_metrics,last_seen)
    Path(args.results).mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "model_id": MODEL_ID, "revision": REVISION,
              "vllm": "0.10.2", "dtype": "float16", "attention_backend": "FLASH_ATTN",
              "runtime": "custom vLLM V0 partial runner", "split_layers": SPLITS[args.split]}
    config["clock_sample"] = {"wall_time_ns": time.time_ns(), "monotonic_ns": time.perf_counter_ns()}
    label=args.role+('_'+args.pd_role if args.pd_role else '')
    (Path(args.results) / f"{label}_config.json").write_text(json.dumps(config, indent=2))
    scheduler = None
    if args.role == "enterprise":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        eos = {tokenizer.eos_token_id, 151643, 151645}
        if args.pd:
            remote=httpx.get(args.cloud_decode+'/health',timeout=5,trust_env=False)
            remote.raise_for_status();decode_config=remote.json()
            if (remote_config.get('pd_role'),decode_config.get('pd_role'))!=('prefill','decode') or any(
                c.get('pd_epoch')!=args.pd_epoch or c.get('layer_split')!=list(SPLITS[args.split])
                or c.get('revision')!=REVISION for c in (remote_config,decode_config)):
                raise RuntimeError('PD role/epoch/model handshake mismatch')
            from split_poc.pd_scheduler import PDScheduler
            scheduler=PDScheduler(executor,args,eos)
        elif args.pipeline_window:
            from split_poc.pipeline import PipelineScheduler
            scheduler = PipelineScheduler(executor, args, eos)
        else:
            scheduler = Scheduler(executor, args, eos)

    @app.get("/health")
    def health():
        if not executor.healthy or not all(p.is_alive() for p in executor.processes):
            raise HTTPException(503, "GPU workers unhealthy")
        if scheduler:
            try:
                reply = httpx.get(args.cloud + "/health", timeout=2, trust_env=False)
                reply.raise_for_status()
                if args.pd:
                    reply=httpx.get(args.cloud_decode+'/health',timeout=2,trust_env=False)
                    reply.raise_for_status()
            except Exception:
                raise HTTPException(503, "Cloud unavailable")
        return {"status": "ready", "role": args.role, "split": args.split, "tp": args.tp,
                "pd":args.pd,"pd_role":args.pd_role,"pd_epoch":args.pd_epoch,
                "pd_chunk_transfer":args.pd_chunk_transfer,"pd_control_channel":bool(args.pd_role),
                "pd_reservations":len(pd_control.records) if pd_control else None,
                "pd_reserving":len(scheduler.reserving) if scheduler and args.pd else 0,
                "pd_releasing":len(scheduler.releasing) if scheduler and args.pd else 0,
                "cloud_tp": cloud_tp, "model_id": MODEL_ID, "revision": REVISION, "protocol": 1,
                "layer_split": SPLITS[args.split],
                "optimizations": optimization_config(args),
                "gpu_pids": [p.pid for p in executor.processes],
                "worker_audits": executor.worker_audits,
                "active": len(scheduler.active) if scheduler else len(last_seen),
                "waiting": scheduler.pending.qsize() if scheduler else 0,
                "completed": scheduler.completed if scheduler else None,
                "kv_used_blocks": scheduler.kv_used if scheduler else cloud_metrics["kv_used_blocks"]}

    @app.get("/metrics")
    def metrics():
        values = {"healthy": int(executor.healthy), "running_requests": len(scheduler.active) if scheduler else len(last_seen)}
        if scheduler:
            values.update(waiting_requests=scheduler.pending.qsize(), requests_completed=scheduler.completed,
                          requests_failed=scheduler.failed, forward_steps=scheduler.steps,
                          kv_used_blocks=scheduler.kv_used, kv_total_blocks=args.kv_blocks,
                          prompt_tokens=scheduler.prompt_tokens, generation_tokens=scheduler.generation_tokens)
        else:
            values.update(cloud_metrics)
            values["kv_total_blocks"] = args.kv_blocks
        front, middle, back = SPLITS[args.split]
        owned = front + back if scheduler else middle
        bytes_per_block_per_rank = owned * 2 * 16 * (2 // args.tp) * 128 * 2
        values["kv_reserved_bytes_per_rank"] = args.kv_blocks * bytes_per_block_per_rank
        values["kv_used_bytes_per_rank"] = values["kv_used_blocks"] * bytes_per_block_per_rank
        if scheduler and args.pipeline_window:
            values["front_kv_used_blocks"]=scheduler.front_used
            values["back_kv_used_blocks"]=scheduler.kv_used
            values["kv_used_bytes_per_rank"]=(front*scheduler.front_used+back*scheduler.kv_used)*2*16*(2//args.tp)*128*2
            values["pipeline_inflight"]=len(scheduler.inflight)
            values["pipeline_window"]=args.pipeline_window
        return Response("\n".join(f'split_{key}{{role="{args.role}"}} {v}' for key, v in values.items()) + "\n",
                        media_type="text/plain")

    @app.post("/start_profile")
    def start_profile():
        executor.call({"op": "profile_start"})
        if scheduler:
            response = scheduler.http.post("/start_profile")
            response.raise_for_status()
        return {"profiling": True}

    @app.post("/stop_profile")
    def stop_profile():
        executor.call({"op": "profile_stop"})
        if scheduler:
            response = scheduler.http.post("/stop_profile")
            response.raise_for_status()
        return {"profiling": False}

    if args.role == "cloud":
        @app.post("/forward")
        async def forward(request: Request):
            started = time.perf_counter()
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > MAX_BODY:
                    raise HTTPException(413, "Activation too large")
            try:
                command, arrays = unpack(body)
                allowed = {"op", "phase", "items", "batch_id"}
                if set(command) != allowed or command["op"] != "forward" or command["phase"] not in {"prefill", "decode"}:
                    raise ValueError("Unsupported execution metadata")
                if args.pd_role and command['phase']!=args.pd_role:
                    raise ValueError('Wrong phase for PD role')
                if not 1 <= len(command["items"]) <= args.max_active:
                    raise ValueError("Invalid batch size")
                seen = set()
                for item in command["items"]:
                    if set(item) != {"request_id", "position", "query_len"}:
                        raise ValueError("Unsupported request metadata")
                    if not isinstance(item["request_id"], str) or len(item["request_id"]) != 32 or item["request_id"] in seen:
                        raise ValueError("Invalid request ID")
                    seen.add(item["request_id"])
                    if any(type(item[k]) is not int for k in ("position", "query_len")):
                        raise ValueError("Invalid positions")
                    if not 0 <= item["position"] < 16384 or not 1 <= item["query_len"] <= 16384 - item["position"]:
                        raise ValueError("Sequence too long")
                    if (command["phase"] == "prefill" and item["position"] != 0 and not args.prefill_chunk_size) or (command["phase"] == "decode" and item["query_len"] != 1):
                        raise ValueError("Invalid phase")
                n = sum(x["query_len"] for x in command["items"])
                if len(arrays) != 2 or any(a.shape != (n, 2048) for a in arrays):
                    raise ValueError("Activation shape mismatch")
            except (ValueError, KeyError, TypeError) as exc:
                raise HTTPException(400, str(exc))
            received = time.perf_counter()
            received_ns = time.perf_counter_ns()
            def execute():
                with lock:
                    t = time.perf_counter()
                    result = executor.call(command, arrays)
                    result["meta"]["timings"]["cloud_queue_ms"] = (t - received) * 1000
                    result["meta"]["timings"]["cloud_body_receive_ms"] = (received - started) * 1000
                    result["meta"]["timings"]["cloud_received_ns"] = received_ns
                    for item in command["items"]:
                        last_seen[item["request_id"]] = time.monotonic()
                    cloud_metrics["kv_used_blocks"] = result["meta"]["timings"]["cloud_kv_used_blocks"]
                    cloud_metrics["forward_steps"] += 1
                    cloud_metrics[command["phase"] + "_tokens"] += n
                    result["meta"]["timings"]["cloud_send_ns"] = time.perf_counter_ns()
                    if pd_control:pd_control.after_forward(command)
                    return pack(result["meta"], result["arrays"], fast=args.wire_fast)
            try:
                result = await asyncio.to_thread(lambda: gate.run(command["items"], execute)) if gate else await asyncio.to_thread(execute)
            except Exception as exc:
                raise HTTPException(503, str(exc))
            return Response(result, media_type="application/octet-stream")

        @app.post("/release")
        def release(body: dict):
            ids = body.get("ids", [])
            if not isinstance(ids, list) or len(ids) > 256 or any(not isinstance(x, str) for x in ids):
                raise HTTPException(400, "Invalid release")
            def release_owned():
                if pd_control:pd_control.release(ids)
                with lock:
                    result = executor.call({"op": 'pd_release' if args.pd_role else "release", "ids": ids})
                    cloud_metrics["kv_used_blocks"] = result["kv_used_blocks"]
                    for rid in ids:
                        last_seen.pop(rid, None)
                    return result
            return gate.release(ids, release_owned) if gate else release_owned()


        @app.on_event("startup")
        async def cleanup_start():
            async def cleanup():
                while True:
                    await asyncio.sleep(30)
                    if args.pd_role:continue  # PD ownership drains through the coordinator.
                    def expire():
                        with lock:
                            expired = [rid for rid, t in last_seen.items() if time.monotonic() - t > 300]
                            if expired and executor.healthy:
                                result = executor.call({"op": "release", "ids": expired})
                                cloud_metrics["kv_used_blocks"] = result["kv_used_blocks"]
                                for rid in expired:
                                    last_seen.pop(rid, None)
                    if gate:
                        def expire_gated():
                            with gate.condition:
                                expire()
                                gate.positions = {rid:pos for rid,pos in gate.positions.items() if rid in last_seen}
                        await asyncio.to_thread(expire_gated)
                    else:
                        await asyncio.to_thread(expire)
            app.state.cleanup = asyncio.create_task(cleanup())
    else:
        @app.get("/", response_class=HTMLResponse)
        def index():
            return Path(__file__).with_name("demo.html").read_text()

        @app.get("/v1/models")
        def models():
            return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]}

        async def generate(request, chat):
            try:
                body = await request.json()
            except ValueError:
                raise HTTPException(400, "Expected a JSON object")
            if not isinstance(body, dict):
                raise HTTPException(400, "Expected a JSON object")
            if body.get("model", MODEL_ID) not in {MODEL_ID, "split-qwen"}:
                raise HTTPException(404, "Unknown model")
            if body.get("temperature", 0) != 0 or body.get("n", 1) != 1:
                raise HTTPException(400, "POC supports greedy temperature=0, n=1")
            if body.get("stop") or body.get("logprobs"):
                raise HTTPException(400, "stop/logprobs are not supported by this POC")
            if (body.get("presence_penalty", 0) != 0 or body.get("frequency_penalty", 0) != 0
                    or body.get("repetition_penalty", 1) != 1 or body.get("logit_bias")
                    or body.get("tools") or body.get("response_format")):
                raise HTTPException(400, "Penalties, logit_bias, tools and response_format are not supported")
            if not isinstance(body.get("stream_options", {}), dict):
                raise HTTPException(400, "stream_options must be an object")
            try:
                if chat:
                    messages = body["messages"]
                    if not isinstance(messages, list) or not messages or any(
                            not isinstance(m, dict) or m.get("role") not in {"user", "system", "assistant"}
                            or not isinstance(m.get("content"), str) for m in messages):
                        raise ValueError("Expected text messages with system/user/assistant roles")
                    ids = tokenizer.apply_chat_template(body["messages"], tokenize=True, add_generation_prompt=True)
                else:
                    prompt = body["prompt"]
                    ids = tokenizer.encode(prompt, add_special_tokens=False) if isinstance(prompt, str) else prompt
                limit = body.get("max_tokens", body.get("max_completion_tokens", 256))
                if type(limit) is not int or not 1 <= limit <= 1024:
                    raise ValueError("max_tokens must be 1..1024")
                if not isinstance(ids, list) or not ids or any(type(x) is not int or not 0 <= x < 151936 for x in ids):
                    raise ValueError("Expected text or one token-ID list")
                if len(ids) + limit > 16384:
                    raise ValueError("Context including output must be <=16384 tokens")
            except (KeyError, ValueError, TypeError) as exc:
                raise HTTPException(400, str(exc))
            job = Job(ids, limit, bool(body.get("ignore_eos", False)))
            job.client_id = request.headers.get("x-request-id", job.id)[:128]
            scheduler.submit(job)
            async def events():
                text = ""
                emitted_tokens = []
                try:
                    while True:
                        if await request.is_disconnected():
                            return
                        try:
                            event = await asyncio.to_thread(job.events.get, True, 1)
                        except queue.Empty:
                            continue
                        if "error" in event:
                            yield "data: " + json.dumps({"error": {"message": event["error"], "type": "server_error"}}) + "\n\n"
                            raise RuntimeError("Split execution failed during streaming")
                        emitted_tokens.append(event["token"])
                        decoded = tokenizer.decode(emitted_tokens, skip_special_tokens=True)
                        # Do not emit an incomplete UTF-8 replacement at a token boundary.
                        if not event["done"]:
                            decoded = decoded.rstrip("\ufffd")
                        delta = decoded[len(text):]
                        text = decoded
                        choice = {"index": 0, "finish_reason": event["finish_reason"] if event["done"] else None}
                        choice["delta" if chat else "text"] = {"content": delta} if chat else delta
                        chunk = {"id": "chatcmpl-" + job.id, "object": "chat.completion.chunk" if chat else "text_completion",
                                 "created": int(time.time()), "model": MODEL_ID, "choices": [choice]}
                        yield "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"
                        if event["done"]:
                            if body.get("stream_options", {}).get("include_usage"):
                                usage = {**chunk, "choices": [], "usage": {
                                    "prompt_tokens": len(ids), "completion_tokens": len(emitted_tokens),
                                    "total_tokens": len(ids) + len(emitted_tokens)}}
                                yield "data: " + json.dumps(usage) + "\n\n"
                            yield "data: [DONE]\n\n"
                            return
                finally:
                    job.cancelled = True
            if body.get("stream", False):
                return StreamingResponse(events(), media_type="text/event-stream",
                                         headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            try:
                while True:
                    try:
                        event = await asyncio.to_thread(job.events.get, True, 1)
                    except queue.Empty:
                        if await request.is_disconnected():
                            raise HTTPException(499, "Client disconnected")
                        continue
                    if "error" in event:
                        raise HTTPException(503, event["error"])
                    if event["done"]:
                        text = tokenizer.decode(job.tokens, skip_special_tokens=True)
                        choice = {"index": 0, "finish_reason": event["finish_reason"]}
                        choice["message" if chat else "text"] = {"role": "assistant", "content": text} if chat else text
                        return {"id": "chatcmpl-" + job.id, "object": "chat.completion" if chat else "text_completion",
                                "created": int(time.time()), "model": MODEL_ID, "choices": [choice],
                                "usage": {"prompt_tokens": len(ids), "completion_tokens": len(job.tokens),
                                          "total_tokens": len(ids) + len(job.tokens)}}
            finally:
                job.cancelled = True

        @app.post("/v1/chat/completions")
        async def chat(request: Request):
            return await generate(request, True)

        @app.post("/v1/completions")
        async def completions(request: Request):
            return await generate(request, False)

        if args.diagnostics:
            @app.post("/debug/greedy")
            async def greedy(body: dict):
                ids, steps = body["prompt_ids"], body.get("steps", 32)
                if (not isinstance(ids, list) or not ids or type(steps) is not int or not 1 <= steps <= 257
                        or len(ids) + steps > 16384 or any(type(x) is not int or not 0 <= x < 151936 for x in ids)):
                    raise HTTPException(400, "Invalid diagnostic lengths")
                job = Job(ids, steps, True)
                scheduler.submit(job)
                while True:
                    event = await asyncio.to_thread(job.events.get)
                    if "error" in event:
                        raise HTTPException(503, event["error"])
                    if event["done"]:
                        return {"tokens": job.tokens}

            @app.post("/debug/teacher_force")
            async def teacher_force(body: dict):
                ids, forced = body["prompt_ids"], body["forced_tokens"]
                if (not isinstance(ids, list) or not isinstance(forced, list) or not ids or not 1 <= len(forced) <= 257
                        or len(ids) + len(forced) > 16384 or any(type(x) is not int or not 0 <= x < 151936 for x in ids + forced)):
                    raise HTTPException(400, "Invalid diagnostic lengths")
                job = Job(ids, len(forced), True, forced=forced, capture=True)
                scheduler.submit(job)
                while True:
                    event = await asyncio.to_thread(job.events.get)
                    if "error" in event:
                        raise HTTPException(503, event["error"])
                    if event["done"]:
                        import io
                        buffer = io.BytesIO()
                        np.savez(buffer, logits=np.stack(job.logits), tokens=np.array(job.tokens))
                        return Response(buffer.getvalue(), media_type="application/octet-stream")

    @app.on_event("shutdown")
    async def shutdown():
        if hasattr(app.state, "cleanup"):
            app.state.cleanup.cancel()
            await asyncio.gather(app.state.cleanup, return_exceptions=True)
        if scheduler:
            scheduler.closed = True
            await asyncio.to_thread(scheduler.thread.join, 180)
            if scheduler.thread.is_alive():
                raise RuntimeError("Scheduler did not drain; refusing concurrent executor close")
            scheduler.http.close()
            scheduler.trace.close()
        if pd_control:
            await asyncio.to_thread(pd_control.tasks.shutdown,True)
            pd_control.http.close()
            pd_control.trace.close()
        await asyncio.to_thread(executor.close)
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["enterprise", "cloud"], required=True)
    parser.add_argument("--model", default="models/qwen")
    parser.add_argument("--split", choices=SPLITS, default="1:3")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--kv-blocks", type=int, default=8192)
    parser.add_argument("--dist-port", type=int, default=29501)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--cloud", default="http://127.0.0.1:8001")
    parser.add_argument("--max-active", type=int, default=8)
    parser.add_argument("--results", default="results/live")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--ipc-mode", choices=["pipe", "shm"], default="pipe",
                        help="Cloud-local CPU IPC only; never bypasses WAN")
    parser.add_argument("--wire-fast", action="store_true", help="Single-join lossless FP16 wire encoding")
    parser.add_argument("--prefill-chunk-size", type=int, default=0, help="0 preserves full prefill")
    parser.add_argument("--scheduler-policy", choices=["legacy","decode-first"], default="legacy")
    parser.add_argument("--decode-quota", type=int, default=1, help="Maximum decode rounds before one waiting prefill chunk")
    parser.add_argument("--tcp-buffer-mib", type=int, default=0, help="0 preserves default sockets; nonzero requires Linux CAP_NET_ADMIN")
    parser.add_argument('--pd-prefill-window',type=int,default=0,help='PD prefill in-flight limit; 0 inherits pipeline-window; decode keeps pipeline-window')
    parser.add_argument("--pipeline-window", type=int, default=0, help="0 disables async front/RPC/back pipeline")
    parser.add_argument("--phase-profile", action="store_true",
                        help="Detailed synchronous GPU stage timings; disable for baseline throughput")
    parser.add_argument('--pd',action='store_true')
    parser.add_argument('--pd-role',choices=['','prefill','decode'],default='')
    parser.add_argument('--prefill-tp',type=int,choices=[1,2],default=2)
    parser.add_argument('--decode-tp',type=int,choices=[1,2],default=1)
    parser.add_argument('--cloud-decode',default='http://10.205.0.2:8002')
    parser.add_argument('--pd-kv-port',type=int,default=29611)
    parser.add_argument('--pd-epoch',default='')
    parser.add_argument('--pd-verify-kv',action='store_true')
    parser.add_argument('--pd-chunk-transfer',action='store_true',help='Migrate completed KV pages after each prefill chunk')
    args = parser.parse_args()
    if args.pd and (not args.pipeline_window or not args.pd_epoch):
        parser.error('PD requires pipelining and an explicit launch epoch')
    if not 0 <= args.prefill_chunk_size <= 16384 or args.decode_quota < 1:
        parser.error("Invalid prefill chunk or decode quota")
    if not 0 <= args.pd_prefill_window <= 8 or (args.pd_prefill_window and not args.pd):
        parser.error('PD prefill window requires PD and must be 0..8')
    if not 0 <= args.pipeline_window <= 8:
        parser.error("Pipeline window must be 0..8")
    if not 0 <= args.tcp_buffer_mib <= 64:
        parser.error("TCP buffer must be 0..64 MiB")
    if args.tp not in {1, 2}:
        parser.error("TP must be 1 or 2")
    from split_poc.transport import serve
    serve(create_app(args), args.host, args.port, args.tcp_buffer_mib if args.role == "cloud" else 0)


if __name__ == "__main__":
    main()
