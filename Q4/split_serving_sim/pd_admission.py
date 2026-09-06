"""PD lifecycle RPCs with explicit worker queueing and admission ownership.

Service durations exclude queueing. Forward and lifecycle commands compete for
one worker command lane; RPC propagation runs asynchronously. No measured wait
or end-to-end TTFT is injected as a cost.
"""
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ControlCommand:
    request_id: int
    op: str
    resource: str
    ready_time: float
    duration: float
    done: Callable[[], None]
    sequence: int


class PDAdmission:
    def __init__(self, sim):
        self.sim = sim
        self.config = sim.config.scheduler.pd_disaggregation
        self.queues = defaultdict(deque)
        self.reserving = set()
        self.releasing = set()
        self.ready_admissions = deque()
        self.ready_releases = deque()
        self.gateway_busy = set()
        self.gateway_waiters = defaultdict(deque)
        self.connections = defaultdict(list)
        self.events = []
        self.sequence = 0
        self.forward_gate_available = {}

    def later(self, delay_ms, callback):
        self.sim._push_event(self.sim.clock + delay_ms / 1000, self.sim.lifecycle_event_type, callback)

    def record(self, request_id, event, **fields):
        self.events.append(dict(request_id=request_id, event=event, time_ms=self.sim.clock*1000, **fields))

    def resource(self, rid, phase):
        stage = self.sim.config.stage('cloud_middle').for_phase(phase)
        route = self.sim.pd_routes[rid] if phase == 'prefill' else 0
        return self.sim._parallel_resource_id(stage.resource, route, stage.replicas, 0, stage.pp_degree)

    def command(self, rid, op, resource, ms, done):
        self.sequence += 1
        self.queues[resource].append(ControlCommand(rid, op, resource, self.sim.clock, ms/1000, done, self.sequence))
        self.record(rid, op+'_queued', resource=resource)

    def try_start(self, resource, candidates):
        queue = self.queues[resource.resource_id]
        if not queue:
            return False
        task = queue[0]
        # FIFO at the shared worker command lane, including forward arrivals.
        if candidates and task.op != 'release_e':
            # HTTP forwards serialize before reaching Executor.call. Requests
            # behind an executing forward have not yet joined its worker lock.
            gate = self.forward_gate_available.get(resource.resource_id, 0)
            forward_arrival = max(min(i.ready_time for i in candidates), gate)
            if forward_arrival < task.ready_time:
                return False
        queue.popleft()
        marker = -task.sequence
        resource.running_batch_ids.add(marker)
        start = self.sim.clock
        self.record(task.request_id, task.op+'_start', resource=task.resource,
                    queue_ms=(start-task.ready_time)*1000, service_ms=task.duration*1000)
        def finish():
            resource.running_batch_ids.remove(marker)
            resource.busy_time += task.duration
            self.record(task.request_id, task.op+'_done', resource=task.resource)
            task.done()
        self.later(task.duration*1000, finish)
        return True

    def rpc(self, rid, op, body, done):
        peer = self.sim.pd_routes[rid]
        pool = self.connections[peer]
        cutoff = self.sim.clock-self.config.rpc_keepalive_s
        pool[:] = [last for last in pool if last >= cutoff] if self.config.rpc_keepalive_s else []
        reused = bool(pool)
        if reused:
            pool.pop()
        rtt = self.sim.config.network.rtt_ms
        handshake = 0 if reused else rtt
        self.record(rid, op+'_rpc_start', reused_connection=reused)
        def returned():
            pool.append(self.sim.clock)
            self.record(rid, op+'_rpc_done')
            done()
        self.later(handshake+rtt/2, lambda: body(lambda: self.later(rtt/2, returned)))

    def reserve(self, rid):
        self.reserving.add(rid)
        self.record(rid, 'reserve_submit')
        def arrived(reply):
            peer = self.sim.pd_routes[rid]
            def begin():
                self.gateway_busy.add(peer)
                self.record(rid, 'reserve_gateway_start')
                def p_done():
                    reply()
                    self.gateway_busy.remove(peer)
                    if self.gateway_waiters[peer]:
                        self.gateway_waiters[peer].popleft()()
                def d_done():
                    self.later(self.config.local_rpc_ms/2, lambda: self.command(
                        rid, 'reserve_p', self.resource(rid,'prefill'), self.config.reserve_p_ms, p_done))
                self.later(self.config.local_rpc_ms/2, lambda: self.command(
                    rid, 'reserve_d', self.resource(rid,'decode'), self.config.reserve_d_ms, d_done))
            self.record(rid, 'reserve_gateway_arrival')
            if peer in self.gateway_busy:
                self.gateway_waiters[peer].append(begin)
            else:
                begin()
        def active():
            self.ready_admissions.append(rid)
        self.rpc(rid, 'reserve', arrived, active)

    def poll(self):
        """The enterprise scheduler observes futures between blocking GPU calls."""
        while self.ready_releases:
            rid = self.ready_releases.popleft()
            self.releasing.remove(rid)
            self.sim.active_continuous_requests.discard(rid)
            self.record(rid, 'admission_credit_returned')
        while self.ready_admissions:
            rid = self.ready_admissions.popleft()
            self.reserving.remove(rid)
            self.sim.active_continuous_requests.add(rid)
            self.record(rid, 'active')
            self.sim._start_prefill(rid)

    def source_release(self, rid):
        self.command(rid, 'source_release_p', self.resource(rid,'prefill'), self.config.release_p_ms, lambda: None)

    def release(self, rid):
        if rid in self.releasing:
            return
        self.releasing.add(rid)
        self.record(rid, 'release_submit')
        enterprise = self.sim.config.stage('edge_tail').resource
        def local_done():
            self.sim._free_local_kv(rid)
            def arrived(reply):
                def d_done():
                    self.later(self.config.local_rpc_ms/2, lambda: self.command(
                        rid, 'release_p', self.resource(rid,'prefill'), self.config.release_p_ms, reply))
                self.later(self.config.local_rpc_ms/2, lambda: self.command(
                    rid, 'release_d', self.resource(rid,'decode'), self.config.release_d_ms, d_done))
            def finished():
                self.ready_releases.append(rid)
            self.rpc(rid, 'release', arrived, finished)
        self.command(rid, 'release_e', enterprise, self.config.release_e_ms, local_done)
