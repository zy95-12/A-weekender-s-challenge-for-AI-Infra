"""Cloud-local control plane for reservations and asynchronous KV handoff."""
import concurrent.futures
import threading
import time
import json
from pathlib import Path

import httpx
from fastapi import HTTPException


class PDControl:
    def __init__(self, args, executor, metrics):
        self.args=args;self.executor=executor;self.metrics=metrics
        self.records={};self.condition=threading.Condition()
        self.dispatch=threading.Lock()
        self.tasks=concurrent.futures.ThreadPoolExecutor(max_workers=2,thread_name_prefix='pd-handoff')
        self.http=httpx.Client(base_url=args.cloud_decode,timeout=45,trust_env=False)
        self.failed=None
        self.trace=open(Path(args.results)/'pd_kv_trace.jsonl','a',buffering=1)

    def remote(self, command):
        response=self.http.post('/pd/local',json={'epoch':self.args.pd_epoch,**command})
        response.raise_for_status();return response.json()

    def reserve(self, body):
        rid=body.get('request_id');length=body.get('prompt_len');maximum=body.get('max_len')
        if (not isinstance(rid,str) or len(rid)!=32 or any(c not in '0123456789abcdef' for c in rid)
            or type(length) is not int or type(maximum) is not int or not 1<=length<=maximum<=16384):
            raise HTTPException(400,'Invalid PD reservation')
        with self.condition:
            if self.failed:raise RuntimeError(self.failed)
            if rid in self.records:
                if self.records[rid]['signature']!=(length,maximum):raise HTTPException(409,'Conflicting reservation')
                return {'ok':True}
            if len(self.records)>=self.args.max_active:raise HTTPException(429,'PD admission full')
            command={'op':'pd_reserve','request_id':rid,'prompt_len':length,'max_len':maximum}
            self.remote(command)
            try:self.executor.call(command)
            except BaseException:
                self.remote({'op':'pd_release','ids':[rid]});raise
            self.records[rid]={'signature':(length,maximum),'state':'prefill','future':None,'enqueued_until':0,'chunks':[]}
        return {'ok':True}

    def after_forward(self, command):
        if command['phase']!='prefill':raise ValueError('Decode sent to prefill role')
        with self.condition:
            for item in command['items']:
                rid=item['request_id'];record=self.records[rid]
                length=record['signature'][0]
                end=item['position']+item['query_len']
                if end!=length:
                    if not getattr(self.args,'pd_chunk_transfer',False):continue
                    end=end//16*16
                start=record['enqueued_until']
                if end<=start:continue
                previous=record['future']
                record.update(state='copying',enqueued_until=end)
                record['future']=self.tasks.submit(self.migrate,rid,start,end,previous)

    def migrate(self, rid, start, end, previous):
        try:
            if previous is not None:previous.result()
            final=end==self.records[rid]['signature'][0]
            started=time.perf_counter_ns()
            # NCCL send/recv has no application tags. All workers must enqueue
            # the same request order even when two control tasks overlap.
            with self.dispatch:
                self.remote({'op':'pd_start','request_id':rid,'start':start,'end':end})
                self.executor.call({'op':'pd_start','request_id':rid,'start':start,'end':end})
            deadline=time.monotonic()+30
            send={'ranks':[]}
            while True:
                receive=self.remote({'op':'pd_status','ids':[rid]})
                if self.args.pd_verify_kv:send=self.executor.call({'op':'pd_status','ids':[rid]})
                if all(r[rid]['state']=='copied' for r in receive['ranks']+send['ranks']):break
                if time.monotonic()>deadline:raise TimeoutError('KV handoff timeout')
                time.sleep(.001)
            if self.args.pd_verify_kv:
                source={k:v for r in send['ranks'] for k,v in r[rid]['checksums'].items()}
                target={k:v for r in receive['ranks'] for k,v in r[rid]['checksums'].items()}
                if source!=target or set(source)!={'0','1'}:raise RuntimeError('KV payload mismatch')
            if final:self.remote({'op':'pd_commit','request_id':rid})
            with self.condition:
                if final:
                    self.records[rid].update(state='ready',transfer_start_ns=started,
                        kv_ready_ns=time.perf_counter_ns(),source=send['ranks'],destination=receive['ranks'])
                    self.condition.notify_all()
            # Destination commit is sufficient for decode readiness. Source
            # cleanup may wait behind another P batch without delaying D.
            while not send['ranks'] or not all(r[rid]['state']=='copied' for r in send['ranks']):
                send=self.executor.call({'op':'pd_status','ids':[rid]})
                if time.monotonic()>deadline:raise TimeoutError('Source completion timeout')
                if not all(r[rid]['state']=='copied' for r in send['ranks']):time.sleep(.001)
            chunk=dict(range_start=start,range_end=end,transfer_start_ns=started,
                       source=send['ranks'],destination=receive['ranks'])
            with self.condition:self.records[rid]['chunks'].append(chunk)
            if not final:return
            released=self.executor.call({'op':'pd_release','ids':[rid]})
            self.metrics['kv_used_blocks']=released['kv_used_blocks']
            with self.condition:
                self.records[rid].update(source=send['ranks'],source_released_ns=time.perf_counter_ns())
                self.trace.write(json.dumps({'request_id':rid,**{k:v for k,v in self.records[rid].items()
                    if k not in ('future','signature')}})+'\n')
        except BaseException as exc:
            with self.condition:
                self.failed=str(exc)
                self.records[rid].update(state='failed',error=str(exc))
                self.condition.notify_all()
            self.executor.healthy=False
            raise

    def wait(self, rid):
        deadline=time.monotonic()+35
        with self.condition:
            while self.records[rid]['state'] not in ('ready','failed'):
                if self.failed:raise RuntimeError(self.failed)
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('PD readiness timeout')
                self.condition.wait(remaining)
            r=self.records[rid]
            if r['state']=='failed':raise RuntimeError(r['error'])
            return {k:v for k,v in r.items() if k not in ('future','signature')}

    def release(self, ids):
        for rid in ids:
            with self.condition:record=self.records.get(rid)
            if record is None:continue
            if record['future'] is not None:record['future'].result(timeout=35)
            self.remote({'op':'pd_release','ids':[rid]})
            self.executor.call({'op':'pd_release','ids':[rid]})
            with self.condition:self.records.pop(rid,None)
        return {'ok':True}


def install(app,args,executor,metrics,last_seen):
    control=PDControl(args,executor,metrics) if args.pd_role=='prefill' else None

    def validate(body):
        if body.get('epoch')!=args.pd_epoch:raise HTTPException(409,'PD epoch mismatch')

    @app.post('/pd/local')
    def local(body:dict):
        validate(body)
        command={k:v for k,v in body.items() if k!='epoch'}
        if command.get('op') not in ('pd_reserve','pd_start','pd_status','pd_commit','pd_release'):
            raise HTTPException(400,'Invalid PD operation')
        result=executor.call(command)
        if command['op']=='pd_release':
            for rid in command['ids']:last_seen.pop(rid,None)
        if 'kv_used_blocks' in result:metrics['kv_used_blocks']=result['kv_used_blocks']
        return result

    if control:
        @app.post('/pd/reserve')
        def reserve(body:dict):
            validate(body);return control.reserve(body)

        @app.post('/pd/wait')
        def wait(body:dict):
            validate(body);return control.wait(body['request_id'])

    return control
