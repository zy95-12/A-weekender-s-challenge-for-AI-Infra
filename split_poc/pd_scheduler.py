"""PD routing and bounded asynchronous admission over the stage-3 pipeline."""
import concurrent.futures
import json
import queue
import httpx
import time
from pathlib import Path

from split_poc.pipeline import PipelineScheduler
from split_poc.transport import http_client,post_control


class PDScheduler(PipelineScheduler):
    def __init__(self, executor, args, eos):
        self.control=concurrent.futures.ThreadPoolExecutor(max_workers=args.max_active+4)
        self.reserving={};self.releasing=[];self.routes={};self.next_replica=0
        urls=[args.cloud]+([args.cloud_prefill_secondary] if getattr(args,'prefill_replicas',1)==2 else [])
        self.prefill_urls=urls
        limits=httpx.Limits(keepalive_expiry=1)
        self.prefill_clients=[http_client(args.tcp_buffer_mib,base_url=url,timeout=45,trust_env=False,limits=limits) for url in urls]
        self.prefill_controls=[http_client(args.tcp_buffer_mib,base_url=url,timeout=45,trust_env=False,limits=limits) for url in urls]
        self.decode_http=http_client(args.tcp_buffer_mib,base_url=args.cloud_decode,timeout=45,trust_env=False,limits=limits)
        self.pd_trace=open(Path(args.results)/'pd_trace.jsonl','a',buffering=1)
        self.back_decode_rounds=0
        super().__init__(executor,args,eos)

    @property
    def replicas(self):
        return getattr(self.args,'prefill_replicas',1)

    def replica_for(self, rid):
        return getattr(self,'routes',{}).get(rid,0)

    def assign_replica(self, job):
        jobs=self.active+[j for j,f in self.reserving.values()]
        loads=[sum(max(0,len(j.ids)-j.position) for j in jobs if self.replica_for(j.id)==peer)
               for peer in range(self.replicas)]
        peer=min(range(self.replicas),key=lambda i:(loads[i],(i-self.next_replica)%self.replicas))
        self.next_replica=(peer+1)%self.replicas
        self.routes[job.id]=peer;job.prefill_replica=peer
        return peer

    def control_post(self,path,body):
        ids=[body['request_id']] if 'request_id' in body else body.get('ids',[])
        peers={self.replica_for(rid) for rid in ids}
        if len(peers)>1:raise ValueError('Control batch spans P replicas')
        peer=next(iter(peers),0)
        return post_control(self.prefill_controls[peer],self.prefill_urls[peer],path,
                            {'epoch':self.args.pd_epoch,**body},self.args.tcp_buffer_mib)

    def admit_pending(self):
        # Keep admission credits until remote release has completed.
        for jobs,future in self.releasing[:]:
            if future.done():
                future.result()
                for j in jobs:
                    self.admission.release(j);self.routes.pop(j.id,None)
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
            self.assign_replica(job)
            future=self.control.submit(self.control_post,'/pd/reserve',{
                'request_id':job.id,'prompt_len':len(job.ids),'max_len':len(job.ids)+job.limit-1})
            self.reserving[job.id]=(job,future)

    def http_for(self, command):
        if command['phase']=='decode':return self.decode_http
        peers={self.replica_for(i['request_id']) for i in command['items']}
        if len(peers)!=1:raise ValueError('Prefill batch spans P replicas')
        return self.prefill_clients[peers.pop()]

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
            if phase=='prefill':
                occupied=counts[phase] if self.replicas==1 else sum(
                    t['command']['phase']=='prefill' and self.replica_for(t['command']['items'][0]['request_id'])==self.replica_for(job.id)
                    for t in self.inflight)
                if occupied>=self.prefill_window:continue
            elif counts[phase]>=self.window:continue
            if phase=='decode':
                if not hasattr(job,'pd_ready') or not job.pd_ready.done():continue
                ready=job.pd_ready.result()
                if not hasattr(job,'pd_logged'):
                    self.pd_trace.write(json.dumps({'request_id':job.id,'first_decode_ready_ns':time.perf_counter_ns(),**ready})+'\n')
                    job.pd_logged=True
            result.append(job)
        return result

    def can_submit(self):
        return len(self.inflight)<self.prefill_window*self.replicas+self.window

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
        for peer in range(self.replicas):
            group=[j for j in jobs if self.replica_for(j.id)==peer]
            if not group:continue
            future=self.control.submit(self.control_post,'/release',{'ids':[j.id for j in group]})
            self.releasing.append((group,future))

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
            for client in self.prefill_clients+self.prefill_controls:client.close()

    def fail_pending(self, exc):
        for job,future in self.reserving.values():
            job.events.put({'error':str(exc)})
            self.failed+=1
