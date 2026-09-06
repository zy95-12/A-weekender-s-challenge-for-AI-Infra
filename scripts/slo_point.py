"""Sustained concurrency point with start/end cohorts and occupancy samples."""
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


async def one(client,ids,rid,reference,max_tokens=128):
    start=time.perf_counter();stamps=[];pieces=[];usage=None;done=False;error=None
    try:
        async with client.stream('POST','/v1/completions',headers={'X-Request-Id':rid},
            json={'prompt':ids,'max_tokens':max_tokens,'stream':True,'stream_options':{'include_usage':True}}) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line=='data: [DONE]':done=True;break
                if line.startswith('data: {'):
                    event=json.loads(line[6:]);assert 'error' not in event,event
                    if event.get('choices'):
                        stamps.append(time.perf_counter());pieces.append(event['choices'][0]['text'])
                    if event.get('usage'):usage=event['usage']
        assert done and usage and len(stamps)==usage['completion_tokens']
        assert usage['prompt_tokens']==len(ids)
        assert reference is None or ''.join(pieces)==reference,'Output differs from reference'
    except Exception as exc:error=repr(exc)
    end=time.perf_counter();ttft=(stamps[0]-start)*1000 if stamps else None
    tpot=(stamps[-1]-stamps[0])*1000/(len(stamps)-1) if len(stamps)>1 else None
    itls=(np.diff(stamps)*1000).tolist()
    return dict(request_id=rid,start=start,end=end,ttft_ms=ttft,tpot_ms=tpot,e2e_ms=(end-start)*1000,
        first_second_ms=itls[0] if itls else None,max_itl_ms=max(itls) if itls else None,itls_ms=itls,
        tokens=len(stamps),text=''.join(pieces),error=error,
        slo_pass=error is None and ttft is not None and tpot is not None and ttft<=3000 and tpot<=100)


async def drained(client):
    for _ in range(200):
        r=await client.get('/health');r.raise_for_status();h=r.json()
        if all(h[k]==0 for k in ['active','waiting','kv_used_blocks']) and not h.get('pd_reserving') and not h.get('pd_releasing'):
            return h
        await asyncio.sleep(.05)
    raise RuntimeError(f'Service did not drain: {h}')


async def measure(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    prompt=json.loads(Path(args.prompt).read_text())['prompt_ids']
    reference=json.loads(Path(args.reference).read_text())
    samples=[];rows=[];counts=[0]*args.concurrency;stop=False
    async with httpx.AsyncClient(base_url='http://127.0.0.1:8000',timeout=180,trust_env=False,
        limits=httpx.Limits(max_connections=max(128,args.concurrency+8),max_keepalive_connections=max(128,args.concurrency+8))) as client:
        initial=await drained(client);(out/'initial_health.json').write_text(json.dumps(initial,indent=2))
        r=await client.post('/debug/greedy',json={'prompt_ids':prompt,'steps':79});r.raise_for_status()
        assert r.json()['tokens']==reference['greedy_ids'][:79]
        (out/'greedy.json').write_text(json.dumps(r.json()))
        await drained(client)
        async def worker(index):
            nonlocal stop
            while not stop:
                row=await one(client,prompt,f'pdbench-{args.variant}-{index}-{counts[index]}',reference['text'])
                row['worker']=index;rows.append(row);counts[index]+=1
                if row['error'] or row['tokens']!=79:
                    stop=True;raise RuntimeError(json.dumps(row))
        async def sample_health():
            while not stop:
                try:
                    response=await client.get('/health');response.raise_for_status()
                    h=response.json()
                    samples.append({'time':time.perf_counter(),**{k:h.get(k) for k in
                        ['active','waiting','kv_used_blocks','kv_total_blocks','pd_reserving','pd_releasing']}})
                except Exception as exc:samples.append({'time':time.perf_counter(),'error':repr(exc)})
                await asyncio.sleep(.5)
        sampler=asyncio.create_task(sample_health())
        tasks=[asyncio.create_task(worker(i)) for i in range(args.concurrency)]
        began=time.perf_counter()
        try:
            while min(counts)<1:
                await asyncio.sleep(.05)
                if stop:await asyncio.gather(*tasks)
                if time.perf_counter()-began>180:raise TimeoutError('Warmup timeout')
            start=time.perf_counter();start_counts=counts[:];last_completed=0
            while True:
                await asyncio.sleep(.1)
                if stop:await asyncio.gather(*tasks)
                completed=[r for r in rows if r['end']>=start]
                new=len(completed)>last_completed;last_completed=len(completed)
                elapsed=time.perf_counter()-start
                if elapsed>=args.seconds and new and min(n-b for n,b in zip(counts,start_counts))>=args.cycles:break
                if elapsed>240:raise TimeoutError('Steady completion timeout')
            end=time.perf_counter();stop=True;await asyncio.gather(*tasks)
        finally:
            stop=True;await asyncio.gather(*tasks,return_exceptions=True)
            await sampler
            (out/'health_samples.json').write_text(json.dumps(samples,indent=2))
            (out/'requests.json').write_text(json.dumps(rows,indent=2))
        measured=[r for r in rows if start<=r['end']<end]
        arrivals=[r for r in rows if start<=r['start']<end]
        report=dict(variant=args.variant,concurrency=args.concurrency,measurement_start=start,measurement_end=end,
            warmup_s=start-began,duration_s=end-start,requests=len(measured),completed_qps=len(measured)/(end-start),
            slo_attainment=sum(r['slo_pass'] for r in measured)/len(measured),
            start_cohort_slo_attainment=sum(r['slo_pass'] for r in arrivals)/len(arrivals),
            goodput_qps=sum(r['slo_pass'] for r in measured)/(end-start),all_outputs_match=True)
        for key in ['ttft_ms','tpot_ms','e2e_ms','first_second_ms','max_itl_ms']:
            values=[r[key] for r in measured]
            report['mean_'+key]=float(np.mean(values))
            report['p99_'+key]=float(np.percentile(values,99))
        report['p99_token_itl_ms']=float(np.percentile([v for r in measured for v in r['itls_ms']],99))
        report['slo_pass']=min(report['slo_attainment'],report['start_cohort_slo_attainment'])>=.99
        report['little_law_concurrency']=report['completed_qps']*report['mean_e2e_ms']/1000
        report['minimum_seconds']=args.seconds;report['minimum_worker_cycles']=args.cycles
        report['server_max_active']=96;report['server_kv_blocks']=32768
        (out/'summary.json').write_text(json.dumps(report,indent=2))
        final=await drained(client);(out/'final_health.json').write_text(json.dumps(final,indent=2))
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',required=True);p.add_argument('--out',required=True)
    p.add_argument('--prompt',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--concurrency',type=int,default=16);p.add_argument('--seconds',type=int,default=60)
    p.add_argument('--cycles',type=int,default=6)
    asyncio.run(measure(p.parse_args()))
