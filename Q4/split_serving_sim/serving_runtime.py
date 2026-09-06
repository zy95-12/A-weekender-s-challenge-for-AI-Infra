"""Virtual execution ports for the unmodified serving PDScheduler.run loop.

Only execution is simulated here: serial workers, asynchronous RPC/futures,
network costs and client arrivals. Admission/routing/batching/state transitions
are executed by the version-pinned serving code in serving_source.py.
"""
import heapq
import io
import itertools
import json
import logging
import queue
from collections import defaultdict, deque
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from .core import WorkItem, Phase, Stage
from .metrics import RequestRuntime, build_summary
from .simulator import Simulator, SimulationResult
from .serving_source import load_serving


class VirtualClock:
    def __init__(self, limit):
        self.now = 0.0
        self.limit = limit
        self.events = []
        self.ids = itertools.count()

    def at(self, timestamp, callback):
        heapq.heappush(self.events, (timestamp, next(self.ids), callback))

    def advance(self, until, predicate=lambda: False):
        if until > self.limit:
            raise RuntimeError('virtual serving exceeded max_time_s')
        while self.events and self.events[0][0] <= until and not predicate():
            timestamp, _, callback = heapq.heappop(self.events)
            self.now = timestamp
            callback()
        if not predicate():
            self.now = until

    def resolve(self, future):
        while not future.done():
            if not self.events:
                raise RuntimeError('virtual serving deadlock: unresolved future without events')
            self.advance(self.events[0][0], future.done)
        return future.result()

    def sleep(self, seconds):
        self.advance(self.now+seconds)

    def wait(self, futures, timeout, **kwargs):
        self.advance(self.now+timeout, lambda: any(f.done() for f in futures))
        return {f for f in futures if f.done()}, {f for f in futures if not f.done()}


class SerialLane:
    def __init__(self, runtime, name):
        self.runtime, self.name = runtime, name
        self.busy = False
        self.pending = deque()

    def submit(self, duration, done, metadata=None):
        self.pending.append((duration, done, metadata or {}, self.runtime.clock.now))
        self.pump()

    def pump(self):
        if self.busy or not self.pending:
            return
        duration, done, metadata, ready = self.pending.popleft()
        self.busy = True
        clock = self.runtime.clock
        start = clock.now
        self.runtime.execution.append(dict(resource=self.name,start_time_ms=start*1000,
            end_time_ms=(start+duration)*1000,queue_ms=(start-ready)*1000,**metadata))
        def finish():
            self.busy = False
            done()
            self.pump()
        clock.at(start+duration, finish)


class AsyncPort:
    def __init__(self, runtime):
        self.runtime = runtime

    def submit(self, function, *args):
        if function.__name__ == 'rpc':
            return self.runtime.rpc(*args)
        if function.__name__ == 'control_post':
            return self.runtime.control(*args)
        raise ValueError(f'unsupported asynchronous operation: {function.__name__}')

    def shutdown(self, **kwargs):
        pass


class Sink:
    def write(self, text):
        pass

    def close(self):
        pass


class VirtualServingSimulator:
    def __init__(self, config, gpu_cost_model=None, host_work=None):
        pd = config.scheduler.pd_disaggregation
        if (not pd.enabled or config.scheduler.policy != 'split_poc_pd'
            or config.workload.mode != 'closed_loop' or any(s.pp_degree != 1 for s in config.stages)
            or config.scheduler.allow_mixed_batch or (config.scheduler.kv_cache.enabled and config.scheduler.kv_cache.enable_preemption)):
            raise ValueError('serving backend requires closed-loop PD, PP1, no mixed batches or preemption')
        w = config.workload
        if w.warmup_requests not in (0, w.concurrency):
            raise ValueError('serving backend currently requires warmup=0 or concurrency')
        if config.stage('cloud_middle').prefill_replicas not in (1, 2):
            raise ValueError('serving source supports one or two P replicas')
        self.config = config
        # Reuse cost models, not the old simulator scheduler or lifecycle.
        costs = Simulator(config, gpu_cost_model)
        self.gpu, self.network = costs.roofline, costs.network
        self.clock = VirtualClock(config.simulation.max_time_s)
        self.execution = []
        self.decisions = []
        self.lanes = defaultdict(lambda: None)
        self.wire_free = defaultdict(float)
        self.http_queues = defaultdict(deque)
        self.http_busy = set()
        self.reserve_queues = defaultdict(deque)
        self.reserve_busy = set()
        self.connections = defaultdict(list)
        self.kv_ready = {}
        self.kv_end = defaultdict(int)
        self.kv_futures = {}
        self.jobs = {}
        self.metrics = {}
        self.warmup_done = 0
        self.measurement_start = 0.0 if not config.workload.warmup_requests else None
        self.measurement_end = (config.workload.measurement_duration_s or None) if self.measurement_start == 0 else None
        self.host_work = host_work
        if host_work is None:
            self.trace = io.StringIO()
        else:
            from .serving_host import TimedServingTrace
            self.trace = TimedServingTrace(self, host_work)
        self.closed_launches = False
        self.serial = itertools.count()
        clock = self.clock
        bindings = dict(queue=queue,json=json,Path=Path,logging=logging,
            time=SimpleNamespace(perf_counter=lambda:clock.now, perf_counter_ns=lambda:round(clock.now*1e9),
                monotonic=lambda:clock.now,time_ns=lambda:round(clock.now*1e9),sleep=clock.sleep),
            uuid=SimpleNamespace(uuid4=lambda:SimpleNamespace(hex=f'{next(self.serial):032x}')),
            concurrent=SimpleNamespace(futures=SimpleNamespace(wait=clock.wait,FIRST_COMPLETED='first')),
            httpx=SimpleNamespace(Timeout=lambda *a,**k:None),http_client=lambda *a,**k:Sink())
        source, self.manifest = load_serving(bindings)
        self.source = source
        scheduler = source['PDScheduler'].__new__(source['PDScheduler'])
        self.scheduler = scheduler
        args = SimpleNamespace(pipeline_window=pd.decode_window,pd_prefill_window=pd.prefill_window,
            prefill_replicas=config.stage('cloud_middle').prefill_replicas,
            max_active=config.scheduler.max_num_seqs,kv_blocks=config.scheduler.kv_cache.num_blocks,
            prefill_chunk_size=config.static_policy.prefill_chunk_size,
            scheduler_policy='decode-first' if config.scheduler.decode_first else 'naive',
            decode_quota=config.scheduler.max_consecutive_decode_batches,
            tcp_buffer_mib=config.data_path.tcp_buffer_mib,cloud='virtual-P',results='.',pd_epoch='virtual')
        scheduler.args=args; scheduler.executor=SimpleNamespace(call=self.enterprise,healthy=True)
        scheduler.eos=set();scheduler.window=args.pipeline_window
        scheduler.transfers=AsyncPort(self);scheduler.control=AsyncPort(self)
        scheduler.inflight=[];scheduler.admission=source['KVAdmission'](args.kv_blocks)
        scheduler.waiting_admission=None;scheduler.pending=queue.Queue();scheduler.active=[]
        scheduler.closed=False;scheduler.completed=scheduler.failed=scheduler.steps=scheduler.kv_used=0
        scheduler.prompt_tokens=scheduler.generation_tokens=scheduler.front_used=0
        scheduler.decode_rounds=scheduler.back_decode_rounds=0
        scheduler.reserving={};scheduler.releasing=[];scheduler.routes={};scheduler.next_replica=0
        scheduler.trace=self.trace;scheduler.pd_trace=Sink();scheduler.last_trace=None
        scheduler.prefill_clients=[];scheduler.prefill_controls=[];scheduler.decode_http=Sink()
        scheduler.control_post=lambda path,body:self.clock.resolve(self.control(path,body))
        scheduler.control_post.__name__='control_post'

    def lane(self, name):
        if self.lanes[name] is None:
            self.lanes[name] = SerialLane(self, name)
        return self.lanes[name]

    def work(self, command, stage):
        return [WorkItem(index, int(item['request_id'],16),Phase(command['phase']),stage,
            item['position'],item['query_len'],item['position']+item['query_len'],
            produces_logits=stage==Stage.EDGE_TAIL and command.get('emit',False))
            for index,item in enumerate(command['items'])]

    def duration(self, stage, command):
        items = self.work(command, stage)
        estimate = self.gpu.estimate(stage.value, items)
        if not getattr(self.gpu,'includes_host_staging',False):
            from .host_submission import schedule_submissions
            return schedule_submissions(estimate.sub_operations,0,0,self.config.execution.host_submission).end_time
        return estimate.total_time_s

    def enterprise(self, command, arrays=None):
        op = command['op']
        self.decisions.append(dict(time_ms=self.clock.now*1000,op=op,phase=command.get('phase'),
            ids=command.get('ids',[i['request_id'] for i in command.get('items',[])]),
            positions=[i['position'] for i in command.get('items',[])],
            routes=[self.scheduler.routes.get(i['request_id']) for i in command.get('items',[])]))
        if op=='release':
            duration=self.config.scheduler.pd_disaggregation.release_e_ms/1000 if self.config.scheduler.pd_disaggregation.admission_enabled else 0
            result={'kv_used_blocks':0}
        else:
            stage=Stage.EDGE_FRONT if op=='front' else Stage.EDGE_TAIL
            duration=self.duration(stage,command)
            result=dict(arrays=[],timings={},front_kv_used_blocks=0,kv_used_blocks=0,
                        tokens=[0]*len(command['items']))
        future=Future()
        self.lane('enterprise').submit(duration,lambda:future.set_result(result),dict(op=op,phase=command.get('phase'),ids=[i['request_id'] for i in command.get('items',[])],batch_size=len(command.get('items',[]))))
        return self.clock.resolve(future)

    def network_path(self, stage, command, done):
        estimate=self.network.estimate(stage,self.work(command,stage))
        cursor=self.clock.now
        for operation in estimate.sub_operations:
            if operation.name=='wan_serialization':cursor=max(cursor,self.wire_free[stage.value])
            cursor+=operation.duration_s
            if operation.name=='wan_serialization':self.wire_free[stage.value]=cursor
        self.clock.at(cursor,done)

    def rpc(self, command, arrays):
        future=Future()
        phase=command['phase']
        peer=self.scheduler.routes[command['items'][0]['request_id']]
        resource=f'P{peer}' if phase=='prefill' else 'D'
        sent=self.clock.now
        def arrived():
            received=self.clock.now
            def begin():
                start=self.clock.now
                def computed():
                    end=self.clock.now
                    if phase=='prefill':self.migrate(command,peer)
                    timings=dict(cloud_received_ns=round(received*1e9),cloud_send_ns=round(end*1e9),
                        cloud_queue_ms=(start-received)*1000,enterprise_send_ns=round(sent*1e9),
                        upload_ms=(received-sent)*1000)
                    def returned():
                        timings.update(enterprise_received_ns=round(self.clock.now*1e9),download_ms=(self.clock.now-end)*1000)
                        future.set_result(([],timings))
                    self.network_path(Stage.WAN_DOWN,command,returned)
                    self.http_busy.remove(resource)
                    self.pump_http(resource)
                self.lane(resource).submit(self.duration(Stage.CLOUD_MIDDLE,command),computed,
                    dict(op='forward',phase=phase,batch_size=len(command['items']),ids=[i['request_id'] for i in command['items']]))
            self.http_queues[resource].append(begin)
            self.pump_http(resource)
        self.network_path(Stage.WAN_UP,command,arrived)
        return future

    def pump_http(self, resource):
        if resource not in self.http_busy and self.http_queues[resource]:
            self.http_busy.add(resource)
            self.http_queues[resource].popleft()()

    def control(self, path, body):
        future=Future(); pd=self.config.scheduler.pd_disaggregation
        rid=body.get('request_id') or body['ids'][0]
        peer=self.scheduler.routes[rid]
        self.decisions.append(dict(time_ms=self.clock.now*1000,op=path,ids=body.get('ids',[rid]),routes=[peer]))
        rtt=self.config.network.rtt_ms/1000 if pd.admission_enabled or path=='/pd/wait' else 0
        pool=self.connections[peer]
        pool[:]=[t for t in pool if self.clock.now-t<=pd.rpc_keepalive_s] if pd.rpc_keepalive_s else []
        reused=bool(pool)
        if reused:pool.pop()
        def reply(value=None):
            def finish():
                pool.append(self.clock.now)
                future.set_result(value or {'ok':True})
            self.clock.at(self.clock.now+rtt/2,finish)
        def commands(ids,index=0):
            if index==len(ids):reply();return
            current=ids[index]
            def launch():
                self.control_pair(peer,'release',lambda:commands(ids,index+1))
            ready=self.kv_futures.get(current)
            if ready and not ready.done():ready.add_done_callback(lambda _:launch())
            else:launch()
        def arrived():
            if path=='/pd/reserve':
                def start():
                    self.reserve_busy.add(peer)
                    def done():
                        self.kv_ready[rid]=Future()
                        reply()
                        self.reserve_busy.remove(peer)
                        if self.reserve_queues[peer]:self.reserve_queues[peer].popleft()()
                    self.control_pair(peer,'reserve',done)
                if peer in self.reserve_busy:self.reserve_queues[peer].append(start)
                else:start()
            elif path=='/release':commands(body['ids'])
            elif path=='/pd/wait':
                ready=self.kv_ready[rid]
                ready.add_done_callback(lambda _:reply({'state':'ready'}))
            else:raise ValueError(path)
        self.clock.at(self.clock.now+rtt/2+(0 if reused else rtt),arrived)
        return future

    def control_pair(self, peer, op, done):
        pd=self.config.scheduler.pd_disaggregation
        local=pd.local_rpc_ms/2000 if pd.admission_enabled else 0
        d=getattr(pd,op+'_d_ms')/1000 if pd.admission_enabled else 0
        p=getattr(pd,op+'_p_ms')/1000 if pd.admission_enabled else 0
        def d_done():
            self.clock.at(self.clock.now+local,lambda:self.lane(f'P{peer}').submit(p,done,dict(op=op)))
        self.clock.at(self.clock.now+local,lambda:self.lane('D').submit(d,d_done,dict(op=op)))

    def migrate(self, command, peer):
        pd=self.config.scheduler.pd_disaggregation
        for item in command['items']:
            rid=item['request_id'];end=item['position']+item['query_len'];final=end==len(self.jobs[rid].ids)
            if not final and not pd.chunk_transfer:continue
            if not final:end=end//16*16
            start=self.kv_end[rid]
            if end<=start:continue
            self.kv_end[rid]=end
            previous=self.kv_futures.get(rid);finished=Future();self.kv_futures[rid]=finished
            work=[WorkItem(0,int(rid,16),Phase.PREFILL,Stage.PD_KV_TRANSFER,start,end-start,end)]
            duration=self.network.estimate(Stage.PD_KV_TRANSFER,work).total_time_s
            control=pd.control_latency_ms/1000
            def launch(rid=rid,final=final,duration=duration,finished=finished):
                def committed():
                    def done():
                        if final:
                            self.kv_ready[rid].set_result(True)
                            cost=pd.release_p_ms/1000 if pd.admission_enabled else 0
                            self.lane(f'P{peer}').submit(cost,lambda:finished.set_result(True),dict(op='source_release'))
                        else:finished.set_result(True)
                    lane='kv_control_D' if pd.control_channel else 'D'
                    self.lane(lane).submit(control if final else 0,done,dict(op='pd_commit'))
                def started():self.lane('kv_transfer').submit(duration,committed,dict(op='kv_transfer'))
                lane=f'kv_control_P{peer}' if pd.control_channel else f'P{peer}'
                self.lane(lane).submit(control,started,dict(op='pd_start'))
            if previous and not previous.done():previous.add_done_callback(lambda _,launch=launch:launch())
            else:launch()

    def launch(self):
        count=len(self.jobs);w=self.config.workload
        if count>=w.num_requests or (self.measurement_end is not None and self.clock.now>=self.measurement_end):
            return False
        job=self.source['Job']([0]*w.input_tokens,w.output_tokens,True)
        self.jobs[job.id]=job
        metric=RequestRuntime(count,self.clock.now,w.input_tokens,w.output_tokens)
        self.metrics[job.id]=metric
        class Events:
            def put(_,event):
                if 'error' in event:raise RuntimeError(event['error'])
                metric.token_times.append(self.clock.now)
                if metric.first_token_time is None:metric.first_token_time=self.clock.now
                if event['done']:
                    metric.finish_time=self.clock.now
                    if count<w.warmup_requests:
                        self.warmup_done+=1
                        if self.warmup_done==w.warmup_requests:
                            self.measurement_start=self.clock.now
                            self.measurement_end=self.clock.now+w.measurement_duration_s if w.measurement_duration_s else None
                    self.launch()
        job.events=Events()
        self.scheduler.submit(job)
        return True

    def run(self):
        w=self.config.workload
        for _ in range(min(w.concurrency,w.num_requests)):self.launch()
        # The real run loop checks closed only at its top. Shut down only after
        # clients and remote release futures drain, never cancel active jobs.
        original=self.scheduler.admit_pending
        def admit():
            original()
            if (not self.scheduler.active and not self.scheduler.reserving and not self.scheduler.releasing
                and self.scheduler.pending.empty() and self.scheduler.waiting_admission is None):
                self.scheduler.closed=True
        self.scheduler.admit_pending=admit
        self.scheduler.run()
        if not self.scheduler.executor.healthy:raise RuntimeError('serving scheduler failed')
        if self.scheduler.admission.reservations:raise RuntimeError('serving admission leaked')
        rows=[]
        lo=self.measurement_start
        if lo is None:raise ValueError('warmup did not finish')
        hi=self.measurement_end or max(m.finish_time for m in self.metrics.values())
        for serving_id,metric in self.metrics.items():
            row=metric.to_metrics()
            row['serving_request_id']=serving_id
            row['measured']=metric.request_id>=w.warmup_requests and lo<=metric.finish_time<=hi
            rows.append(row)
        selected=[r for r in rows if r['measured']]
        if not selected:raise ValueError('no measured requests')
        summary=build_summary(selected,[],{},lo,hi,self.config.slo)
        summary.update(scheduler_backend='serving_source',serving_source=self.manifest,
            command_cost_coverage=getattr(self.gpu,'coverage',{}),
            command_sampling=getattr(self.gpu,'sampling','mean'),cost_seed=getattr(self.gpu,'seed',None),
            residual_coverage=getattr(self.gpu,'residual_coverage',{}),
            measurement_window=dict(start_ms=lo*1000,end_ms=hi*1000),
            scheduling_decisions=self.decisions,
            lifecycle_drained=not self.scheduler.admission.reservations,
            serving_host_work=dict(enabled=self.host_work is not None,
                coverage=self.host_work.coverage if self.host_work else {},
                total_ms=self.host_work.total_ms if self.host_work else 0))
        self.serving_trace=[json.loads(line) for line in self.trace.getvalue().splitlines()]
        # Standard stage trace for existing CLI artifacts; serving trace remains
        # available for exact front/back/batch inspection.
        trace=[dict(batch_id=i,stage=r.get('op',''),resource=r['resource'],
                    start_time_ms=r['start_time_ms'],end_time_ms=r['end_time_ms'],
                    duration_ms=r['end_time_ms']-r['start_time_ms'],request_ids=r.get('ids',[]),batch_size=r.get('batch_size',1),
                    phases=[r['phase']] if r.get('phase') else []) for i,r in enumerate(self.execution)]
        return SimulationResult(summary,rows,trace)
