"""Bounded front/RPC/back scheduler; enterprise GPU commands remain serialized.

The window includes responses waiting for back execution. Every task owns its
host arrays until back completes; cancellation drains submitted work before KV
release. No target GPU worker waits for HTTP in this mode.
"""
import concurrent.futures
import contextlib
import json
import queue
import time
import uuid

import httpx

from split_poc.pipeline_state import KVAdmission
from split_poc.server import Scheduler
from split_poc.scheduling import choose
from split_poc.transport import http_client
from split_poc.wire import pack, unpack


class PipelineScheduler(Scheduler):
    def __init__(self, executor, args, eos):
        self.window = args.pipeline_window
        self.transfers = concurrent.futures.ThreadPoolExecutor(max_workers=self.window)
        self.inflight = []
        self.admission = KVAdmission(args.kv_blocks)
        self.waiting_admission = None
        self.front_used = 0
        super().__init__(executor, args, eos)

    def submit(self, job):
        if self.admission.needed(job) > self.admission.capacity:
            from fastapi import HTTPException
            raise HTTPException(400, "Request exceeds KV capacity")
        job.front_position = 0
        job.outstanding = 0
        job.front_ready = job.created
        job.finished = False
        super().submit(job)

    def rpc(self, command, arrays):
        started = time.perf_counter_ns()
        metadata = {k:command[k] for k in ("phase","items","batch_id")}
        payload = pack({"op":"forward",**metadata},arrays,fast=self.args.wire_fast)
        sent = time.perf_counter_ns()
        def trace(event,info):
            import torch
            parts=event.split(".")
            names={"send_request_body":"pipeline_upload", "receive_response_headers":"pipeline_cloud_wait",
                   "receive_response_body":"pipeline_download"}
            if len(parts)==3 and parts[1] in names:
                if parts[2]=="started":
                    torch.cuda.nvtx.range_push(names[parts[1]]+" batch="+command["batch_id"])
                elif parts[2] in {"complete","failed"}:
                    torch.cuda.nvtx.range_pop()
        response = self.data_http.post("/forward",content=payload,
                                       extensions={"trace":trace} if self.args.phase_profile else {},
                                       headers={"content-type":"application/octet-stream"})
        response.raise_for_status()
        received = time.perf_counter_ns()
        meta, output = unpack(response.content)
        if meta["batch_id"] != command["batch_id"]:
            raise ValueError("Cloud batch mismatch")
        timings = meta["timings"]
        timings.update(enterprise_send_ns=sent,enterprise_received_ns=received,
                       rpc_wall_ms=(received-sent)/1e6,pack_ms=(sent-started)/1e6,
                       upload_bytes=len(payload),download_bytes=len(response.content),
                       upload_ms=(timings["cloud_received_ns"]-sent)/1e6,
                       download_ms=(received-timings["cloud_send_ns"])/1e6)
        return output,timings

    def release(self, jobs):
        if not jobs:
            return
        super().release(jobs)
        if not self.executor.healthy:
            raise RuntimeError("Pipeline KV release failed; restart required")
        for job in jobs:
            self.admission.release(job)
        if not self.active or all(j in jobs for j in self.active):
            self.front_used = 0

    def admit_pending(self):
        while len(self.active) < self.args.max_active:
            if self.waiting_admission is None:
                try:
                    self.waiting_admission = self.pending.get_nowait()
                except queue.Empty:
                    return
            job = self.waiting_admission
            if job.cancelled:
                self.waiting_admission = None
                continue
            if not self.admission.admit(job):
                return
            self.active.append(job)
            self.waiting_admission = None

    def submit_front(self, batch, phase):
        ids,items = [],[]
        for job in batch:
            pos = job.front_position
            tokens = (job.ids[pos:pos+(self.args.prefill_chunk_size or len(job.ids))]
                      if phase == "prefill" else
                      [job.forced[len(job.tokens)-1] if job.forced is not None else job.tokens[-1]])
            ids.extend(tokens)
            items.append({"request_id":job.id,"position":pos,"query_len":len(tokens)})
        command = {"op":"front","items":items,"phase":phase,"token_ids":ids,
                   "batch_id":uuid.uuid4().hex,"capture":any(j.capture for j in batch),
                   "emit":phase=="decode" or all(i["position"]+i["query_len"]==len(j.ids)
                                                   for i,j in zip(items,batch))}
        start = time.perf_counter_ns()
        waits = [(start/1e9-j.front_ready)*1000 for j in batch]
        front = self.executor.call(command)
        end = time.perf_counter_ns()
        self.front_used = front["front_kv_used_blocks"]
        future = self.transfers.submit(self.rpc,command,front["arrays"])
        self.inflight.append({"batch":batch,"command":command,"future":future,
                              "front_start_ns":start,"front_end_ns":end,
                              "front_timings":front["timings"],"queue_ms":waits})
        for job,item in zip(batch,items):
            job.front_position += item["query_len"]
            job.front_ready = end/1e9
            job.outstanding += 1

    def complete_back(self, task):
        arrays,timings = task["future"].result()
        batch,command = task["batch"],task["command"]
        start = time.perf_counter_ns()
        result = self.executor.call({**command,"op":"back"},arrays)
        end = time.perf_counter_ns()
        self.kv_used = result["kv_used_blocks"]
        self.steps += 1
        self.inflight.remove(task)
        for index,(job,item) in enumerate(zip(batch,command["items"])):
            job.position += item["query_len"]
            job.outstanding -= 1
            job.prefilled = job.position >= len(job.ids)
            emits = command["emit"] and not job.cancelled
            token = result["tokens"][index]
            if emits:
                job.tokens.append(token)
                if job.capture:
                    job.logits.append(result["logits"][index])
            row = {"request_id":job.id,"client_request_id":job.client_id,
                   "batch_id":command["batch_id"],"batch_size":len(batch),
                   "time_ns":time.time_ns(),"phase":command["phase"],
                   "token_idx":len(job.tokens)-1,"emits_token":emits,
                   "query_len":item["query_len"],"position_start":item["position"],
                   "context_len":job.position,"queue_ms":task["queue_ms"][index],
                   "step_wall_ms":(end-task["front_start_ns"])/1e6,
                   "timings_scope":"overlapping_pipeline_task_not_additive",
                   "front_start_ns":task["front_start_ns"],"front_end_ns":task["front_end_ns"],
                   "back_start_ns":start,"back_end_ns":end,
                   "window_occupancy":len(self.inflight)+1,"front_kv_used_blocks":self.front_used,
                   "back_kv_used_blocks":self.kv_used,
                   **task["front_timings"],**timings,**result["timings"]}
            self.trace.write(json.dumps(row)+"\n")
            self.last_trace = row
            self.prompt_tokens += item["query_len"] if command["phase"]=="prefill" else 0
            self.generation_tokens += int(emits)
            job.last_step = end/1e9
            if emits:
                job.finished = len(job.tokens)>=job.limit or (token in self.eos and not job.ignore_eos)
                job.events.put({"token":token,"done":job.finished,
                                "finish_reason":"length" if len(job.tokens)>=job.limit else "stop"})
                if job.finished:
                    self.completed += 1

    def run(self):
        self.data_http = http_client(self.args.tcp_buffer_mib,base_url=self.args.cloud,
                                    timeout=httpx.Timeout(30,connect=3),trust_env=False)
        try:
            while not self.closed or self.inflight:
                if self.closed:
                    for job in self.active:
                        job.cancelled = True
                else:
                    self.admit_pending()
                releasable = [j for j in self.active if (j.cancelled or j.finished) and j.outstanding==0]
                self.release(releasable)
                self.active = [j for j in self.active if j not in releasable]
                # Back follows committed KV order, independently of HTTP arrival order.
                ready = [t for t in self.inflight if t["future"].done() and
                         all(i["position"]==j.position for i,j in zip(t["command"]["items"],t["batch"]))]
                if ready:
                    self.complete_back(ready[0])
                    continue
                eligible = [j for j in self.active if not j.cancelled and not j.finished and
                            (j.front_position<len(j.ids) or (j.prefilled and j.outstanding==0))]
                if not self.closed and len(self.inflight)<self.window and eligible:
                    batch,phase,self.decode_rounds = choose(eligible,self.args.scheduler_policy,
                                                           self.decode_rounds,self.args.decode_quota)
                    self.submit_front(batch,phase)
                    continue
                if self.inflight:
                    concurrent.futures.wait([t["future"] for t in self.inflight],timeout=.001,
                                            return_when=concurrent.futures.FIRST_COMPLETED)
                    # Ready responses may still wait for a predecessor; do not spin.
                    if any(t["future"].done() for t in self.inflight):
                        time.sleep(.001)
                else:
                    time.sleep(.001)
            self.release(self.active)
            self.active=[]
        except BaseException as exc:
            self.executor.healthy=False
            for job in self.active:
                job.events.put({"error":str(exc)})
                self.failed+=1
            if self.waiting_admission:
                self.waiting_admission.events.put({"error":str(exc)})
            while not self.pending.empty():
                self.pending.get_nowait().events.put({"error":str(exc)})
            # RPCs have finite timeouts. No KV release can race their completion.
            for task in self.inflight:
                task["future"].cancel()
            self.transfers.shutdown(wait=True,cancel_futures=True)
            self.active=[]
            self.inflight=[]
        finally:
            self.transfers.shutdown(wait=True,cancel_futures=True)
            self.data_http.close()
