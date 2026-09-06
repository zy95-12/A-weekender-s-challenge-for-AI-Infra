"""PD routing and bounded asynchronous admission over the stage-3 pipeline."""
import concurrent.futures
import json
import queue
import time
from pathlib import Path

from split_poc.pipeline import PipelineScheduler
from split_poc.transport import http_client


class PDScheduler(PipelineScheduler):
    def __init__(self, executor, args, eos):
        self.control=concurrent.futures.ThreadPoolExecutor(max_workers=args.max_active+4)
        self.reserving={};self.releasing=[]
        self.decode_http=http_client(args.tcp_buffer_mib,base_url=args.cloud_decode,timeout=45,trust_env=False)
        self.pd_trace=open(Path(args.results)/'pd_trace.jsonl','a',buffering=1)
        self.back_decode_rounds=0
        super().__init__(executor,args,eos)

    def control_post(self,path,body):
        response=self.http.post(path,json={'epoch':self.args.pd_epoch,**body})
        response.raise_for_status();return response.json()

    def admit_pending(self):
        # Keep admission credits until remote release has completed.
        for jobs,future in self.releasing[:]:
            if future.done():
                future.result()
                for j in jobs:self.admission.release(j)
                self.releasing.remove((jobs,future))
        for rid,(job,future) in list(self.reserving.items()):
            if not future.done():continue
            future.result();del self.reserving[rid]
            self.active.append(job)
        while len(self.admission.reservations)<self.args.max_active:
            if self.waiting_admission is None:
                try:self.waiting_admission=self.pending.get_nowait()
                except queue.Empty:return
            job=self.waiting_admission
            if job.cancelled:self.waiting_admission=None;continue
            if not self.admission.admit(job):return
            self.waiting_admission=None
            future=self.control.submit(self.control_post,'/pd/reserve',{
                'request_id':job.id,'prompt_len':len(job.ids),'max_len':len(job.ids)+job.limit-1})
            self.reserving[job.id]=(job,future)

    def http_for(self, command):
        return self.decode_http if command['phase']=='decode' else self.data_http

    def remote_ready(self, task, timings):
        if task['command']['phase']=='prefill' and task['command']['emit']:
            for job in task['batch']:
                if not hasattr(job,'pd_ready'):
                    job.pd_ready=self.control.submit(self.control_post,'/pd/wait',{'request_id':job.id})

    @property
    def prefill_window(self):
        return getattr(self.args,'pd_prefill_window',0) or self.window

    def eligible(self):
        counts={p:sum(t['command']['phase']==p for t in self.inflight) for p in ['prefill','decode']}
        result=[]
        for job in super().eligible():
            phase='prefill' if job.front_position<len(job.ids) else 'decode'
            if counts[phase]>=(self.prefill_window if phase=='prefill' else self.window):continue
            if phase=='decode':
                if not hasattr(job,'pd_ready') or not job.pd_ready.done():continue
                ready=job.pd_ready.result()
                if not hasattr(job,'pd_logged'):
                    self.pd_trace.write(json.dumps({'request_id':job.id,'first_decode_ready_ns':time.perf_counter_ns(),**ready})+'\n')
                    job.pd_logged=True
            result.append(job)
        return result

    def can_submit(self):
        return len(self.inflight)<self.prefill_window+self.window

    def ready_task(self,tasks):
        decodes=[t for t in tasks if t['command']['phase']=='decode']
        prefills=[t for t in tasks if t['command']['phase']=='prefill']
        if decodes and (not prefills or self.back_decode_rounds<self.args.decode_quota):
            self.back_decode_rounds+=1
            return min(decodes,key=lambda t:min(j.last_step for j in t['batch']))
        self.back_decode_rounds=0
        return prefills[0]

    def release(self,jobs):
        if not jobs:return
        local=self.executor.call({'op':'release','ids':[j.id for j in jobs]})
        self.kv_used=local['kv_used_blocks']
        self.front_used=sum((j.front_position+15)//16 for j in self.active if j not in jobs)
        future=self.control.submit(self.control_post,'/release',{'ids':[j.id for j in jobs]})
        self.releasing.append((jobs[:],future))

    def close_clients(self):
        try:
            for job,future in self.reserving.values():
                future.result(timeout=45)
                self.control_post('/release',{'ids':[job.id]})
                self.admission.release(job)
            for jobs,future in self.releasing:
                future.result(timeout=45)
                for job in jobs:self.admission.release(job)
        finally:
            self.control.shutdown(wait=True,cancel_futures=True)
            self.decode_http.close();self.pd_trace.close()

    def fail_pending(self, exc):
        for job,future in self.reserving.values():
            job.events.put({'error':str(exc)})
            self.failed+=1
