"""Opt-in activation round-trip calibration using the serving transport/IPC."""
import os
import time
import numpy as np
import torch
from split_poc.wire import pack, unpack, MAX_BODY


def execute(runner, command, arrays):
    if os.environ.get('SPLIT_WAN_PROBE') != '1':
        raise ValueError('WAN probe is disabled')
    n = command['rows']
    if type(n) is not int or not 1 <= n <= 16384:
        raise ValueError('Rows outside serving activation range')
    if command['op'] == 'wan_prepare':
        runner.probe_tensors = [torch.full((n,2048),v,device='cuda',dtype=torch.float16) for v in (.5,-.25)]
        torch.cuda.synchronize()
        return {'ok':True}
    if command['op'] == 'wan_echo':
        start = time.perf_counter_ns()
        t = start
        hidden = runner.broadcast_hidden(arrays,n)
        torch.cuda.synchronize()
        received = time.perf_counter_ns()
        if runner.rank == 0:
            output = [x.cpu().numpy() for x in hidden]
            end = time.perf_counter_ns()
            return {'meta':{'timings':{'cloud_h2d_tp_ms':(received-t)/1e6,
                     'cloud_d2h_ms':(end-received)/1e6,'cloud_worker_ms':(end-start)/1e6}},'arrays':output}
        return None
    if command['op'] != 'wan_roundtrip':
        raise ValueError('Invalid probe operation')
    if runner.rank == 0:
        t0 = time.perf_counter_ns()
        host = [x.cpu().numpy() for x in runner.probe_tensors]
        t1 = time.perf_counter_ns()
        payload = pack({'op':'wan_echo','rows':n},host,fast=runner.args.get('wire_fast',False))
        t2 = time.perf_counter_ns()
        response = runner.http.post('/wan_echo',content=payload,headers={'content-type':'application/octet-stream'})
        response.raise_for_status()
        t3 = time.perf_counter_ns()
        meta, remote = unpack(response.content)
        t4 = time.perf_counter_ns()
    else:
        remote = None
    receive_start = time.perf_counter_ns()
    received_gpu = runner.broadcast_hidden(remote,n)
    torch.cuda.synchronize()
    end = time.perf_counter_ns()
    if runner.rank != 0:
        return None
    stats = meta['timings']
    header = lambda name: float(response.headers['x-probe-'+name])
    stats.update(rows=n,activation_bytes=n*8192,upload_wire_bytes=len(payload),download_wire_bytes=len(response.content),
        enterprise_d2h_ms=(t1-t0)/1e6,enterprise_pack_ms=(t2-t1)/1e6,rpc_ms=(t3-t2)/1e6,
        enterprise_unpack_ms=(t4-t3)/1e6,enterprise_h2d_tp_ms=(end-receive_start)/1e6,
        gpu_ready_rtt_ms=(end-t0)/1e6,cloud_pack_ms=header('pack-ms'),
        upload_path_ms=(header('body-end-ns')-t2)/1e6,
        download_path_ms=(t3-header('response-ready-ns'))/1e6)
    stats['http_transport_ms']=stats['upload_path_ms']+stats['download_path_ms']
    stats['cloud_ipc_residual_ms']=stats['cloud_executor_ms']-stats['cloud_worker_ms']
    stats['validation_ok']=bool(np.all(remote[0]==.5) and np.all(remote[1]==-.25))
    return stats


def install(app,args,executor):
    from fastapi import Request,HTTPException
    from fastapi.responses import Response
    import asyncio
    if args.role == 'cloud':
        @app.post('/wan_echo')
        async def echo(request:Request):
            start=time.perf_counter_ns();body=bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body)>MAX_BODY:raise HTTPException(413,'Body too large')
            body_end=time.perf_counter_ns()
            command,arrays=unpack(body)
            if command.get('op')!='wan_echo' or type(command.get('rows')) is not int or not 1<=command['rows']<=16384:
                raise HTTPException(400,'Invalid probe')
            if len(arrays)!=2 or any(a.shape!=(command['rows'],2048) for a in arrays):
                raise HTTPException(400,'Invalid activation shape')
            unpack_end=time.perf_counter_ns()
            def call():
                begin=time.perf_counter_ns()
                result=executor.call(command,arrays)
                done=time.perf_counter_ns()
                result['meta']['timings'].update(cloud_body_read_ms=(body_end-start)/1e6,
                    cloud_unpack_ms=(unpack_end-body_end)/1e6,cloud_executor_ms=(done-begin)/1e6,
                    cloud_dispatch_ms=(begin-unpack_end)/1e6)
                before=time.perf_counter_ns()
                output=pack(result['meta'],result['arrays'],fast=args.wire_fast)
                ready=time.perf_counter_ns()
                return Response(output,media_type='application/octet-stream',headers={
                    'x-probe-pack-ms':str((ready-before)/1e6),'x-probe-body-end-ns':str(body_end),
                    'x-probe-response-ready-ns':str(ready)})
            return await asyncio.to_thread(call)
    else:
        @app.post('/debug/wan_probe')
        def probe(body:dict):
            n=body.get('rows');op=body.get('op')
            if type(n) is not int or not 1<=n<=16384 or op not in ('wan_prepare','wan_roundtrip'):
                raise HTTPException(400,'Invalid probe')
            return executor.call({'op':op,'rows':n})
