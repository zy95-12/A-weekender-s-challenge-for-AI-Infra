"""Validate the actual vLLM NCCL transport on three local GPUs."""
import json
from pathlib import Path
import torch
import torch.multiprocessing as mp


def worker(rank):
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    torch.cuda.set_device(rank)
    group=StatelessProcessGroup.create(host='127.0.0.1',port=29987,rank=rank,world_size=3,store_timeout=60)
    comm=PyNcclCommunicator(group,device=rank);comm.disabled=False
    stream=torch.cuda.Stream()
    # Each source rank sends one global KV head, representative of TP2 -> TP1.
    shape=(27,2,256,16,128)
    with torch.cuda.stream(stream):
        if rank<2:
            tensor=torch.full(shape,rank+.25,device='cuda',dtype=torch.float16)
            comm.send(tensor,2,stream)
        else:
            tensors=[]
            for src in range(2):
                tensor=torch.empty(shape,device='cuda',dtype=torch.float16)
                comm.recv(tensor,src,stream);tensors.append(tensor)
    stream.synchronize()
    if rank==2:
        assert all(torch.all(t==src+.25).item() for src,t in enumerate(tensors))
        print(json.dumps(dict(valid=True,bytes=sum(t.numel()*2 for t in tensors),backend='vLLM PyNcclCommunicator')),flush=True)
    group.barrier()


if __name__=='__main__':mp.spawn(worker,nprocs=3,join=True)
