"""Asynchronous paged KV handoff using vLLM's NCCL communicator.

The executor owns pool mutations; the copy thread only reads reserved source
pages/writes unpublished destination pages. A completed CUDA event is required
before commit/release. TP groups and the KV transport group are independent.
"""
import concurrent.futures
import hashlib
import threading
import time

import torch


def head_routes(source_tp, destination_tp):
    if source_tp not in (1,2) or destination_tp not in (1,2):
        raise ValueError('Only Qwen two-KV-head TP1/TP2 are supported')
    return [(h, h//(2//source_tp), h%(2//source_tp),
             h//(2//destination_tp), h%(2//destination_tp)) for h in range(2)]


class KVTransfer:
    def __init__(self, runner):
        from vllm.distributed.utils import StatelessProcessGroup
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        self.runner=runner;self.args=runner.args
        self.role=self.args['pd_role'];self.rank=runner.rank
        self.p=self.args['prefill_tp'];self.d=self.args['decode_tp']
        global_rank=self.rank if self.role=='prefill' else self.p+self.rank
        self.group=StatelessProcessGroup.create(host='127.0.0.1',port=self.args['pd_kv_port'],
            rank=global_rank,world_size=self.p+self.d,store_timeout=120)
        self.comm=PyNcclCommunicator(self.group,device=self.rank)
        if not self.comm.available or self.comm.disabled:
            raise RuntimeError('vLLM NCCL KV communicator unavailable')
        self.stream=torch.cuda.Stream(device=self.rank)
        self.thread=concurrent.futures.ThreadPoolExecutor(max_workers=1,thread_name_prefix='kv-copy')
        self.records={};self.released=set();self.lock=threading.Lock()

    def reserve(self, request_id, prompt_len, max_len):
        signature=(prompt_len,max_len)
        if request_id in self.released:
            raise ValueError('Released request cannot be resurrected')
        if request_id in self.records:
            if self.records[request_id]['signature']!=signature:
                raise ValueError('Conflicting KV reservation')
            return
        self.runner.pool.reserve(request_id,prompt_len if self.role=='prefill' else max_len)
        self.records[request_id]=dict(signature=signature,state='reserved',future=None)

    def start(self, request_id):
        record=self.records[request_id]
        if record['future'] is not None:return
        if self.role=='prefill' and self.runner.pool.requests[request_id]['length']!=record['signature'][0]:
            raise ValueError('Export before complete prompt KV')
        # The source compute stream has committed all KV writes before copying.
        ready=torch.cuda.Event();ready.record()
        record['state']='copying'
        record['future']=self.thread.submit(self.copy,request_id,ready)

    @torch.inference_mode()
    def copy(self, request_id, ready):
        torch.cuda.set_device(self.rank)
        record=self.records[request_id];length=record['signature'][0]
        blocks=self.runner.pool.requests[request_id]['blocks'][:(length+15)//16]
        layers=list(self.runner.model.model.layers.values())
        start=time.perf_counter_ns();checksums={}
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(ready)
            ids=torch.tensor(blocks,device='cuda',dtype=torch.long)
            shape=(len(layers),2,len(blocks),16,128)
            if self.role=='prefill':
                pages=torch.stack([layer.self_attn.attn.kv_cache[0].index_select(1,ids) for layer in layers])
                if length%16:pages[:,:,-1,length%16:,:,:]=0
                sends=[]
                for head,src,local_src,dst,local_dst in head_routes(self.p,self.d):
                    if src!=self.rank:continue
                    tensor=pages[:,:,:,:,local_src,:].contiguous();sends.append(tensor)
                    if self.args.get('pd_verify_kv'):
                        checksums[str(head)]=hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()
                    self.comm.send(tensor,self.p+dst,self.stream)
            else:
                receives=[]
                for head,src,local_src,dst,local_dst in head_routes(self.p,self.d):
                    if dst!=self.rank:continue
                    tensor=torch.empty(shape,device='cuda',dtype=torch.float16)
                    self.comm.recv(tensor,src,self.stream);receives.append((head,local_dst,tensor))
                for head,local_dst,tensor in receives:
                    for index,layer in enumerate(layers):
                        # Use an explicit single-head view; index_copy_ scatters
                        # logical pages into the independently allocated D pool.
                        layer.self_attn.attn.kv_cache[0][:,:,:,local_dst,:].index_copy_(1,ids,tensor[index])
                    if self.args.get('pd_verify_kv'):
                        actual=torch.stack([layer.self_attn.attn.kv_cache[0][:,:,:,local_dst,:].index_select(1,ids) for layer in layers])
                        checksums[str(head)]=hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest()
            done=torch.cuda.Event();done.record()
        done.synchronize()
        return dict(start_ns=start,end_ns=time.perf_counter_ns(),checksums=checksums,
                    logical_bytes=len(layers)*2*length*2*128*2)

    def status(self, ids):
        result={}
        for rid in ids:
            if rid in self.released:
                result[rid]={'state':'released'};continue
            record=self.records[rid];future=record['future']
            value={'state':record['state']}
            if future is not None and future.done():
                value.update(future.result())
                if value['state']=='copying':value['state']='copied'
            result[rid]=value
        return result

    def commit(self, rid):
        record=self.records[rid]
        if record['state']=='ready':return
        future=record['future']
        if self.role!='decode' or future is None or not future.done():
            raise ValueError('Cannot commit incomplete D KV')
        future.result()
        self.runner.pool.requests[rid]['length']=record['signature'][0]
        record['state']='ready'

    def release(self, ids):
        for rid in ids:
            record=self.records.get(rid)
            if record:
                future=record['future']
                if future is not None:
                    if not future.done():raise ValueError('Cannot free in-flight KV')
                    future.result()
        self.runner.pool.release(ids)
        for rid in ids:
            self.records.pop(rid,None);self.released.add(rid)
        # IDs are unguessable per-request UUIDs; old epochs never share a worker.
        if len(self.released)>8192:self.released=set(list(self.released)[-4096:])

    def execute(self, command):
        op=command['op'];ids=command.get('ids',[])
        if op=='pd_reserve':self.reserve(command['request_id'],command['prompt_len'],command['max_len'])
        elif op=='pd_start':self.start(command['request_id'])
        elif op=='pd_status':return self.status(ids)
        elif op=='pd_commit':self.commit(command['request_id'])
        elif op=='pd_release':self.release(ids)
        else:raise ValueError('Unknown PD operation')
        return {'ok':True,'kv_used_blocks':self.args['kv_blocks']-len(self.runner.pool.free)}
