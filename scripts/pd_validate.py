"""Representative real-service PD correctness, cancellation and cleanup checks."""
import asyncio
import argparse
import json
import shutil
import subprocess
from pathlib import Path
import httpx
from transformers import AutoTokenizer
from scripts.pd_benchmark import one,drained

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/pd_validation'


async def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(OUT))
    parser.add_argument('--concurrency',type=int,choices=[1,16],default=16)
    args=parser.parse_args();OUT=Path(args.out)
    OUT.mkdir(exist_ok=False)
    prompt=json.loads(Path('/root/issue6-stage4-worktree/results/concurrency_4k_tp22/prompt.json').read_text())['prompt_ids']
    ref=json.loads(Path('/root/issue6-stage4-worktree/results/concurrency_4k_tp22/reference.json').read_text())
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/qwen'),local_files_only=True)
    unit=tok.encode('Explain why a computer needs memory and how it differs from storage. ',add_special_tokens=False)
    other=(unit*40)[:257]
    async with httpx.AsyncClient(base_url='http://127.0.0.1:8000',timeout=180,trust_env=False) as client:
        r=await client.post('/debug/greedy',json={'prompt_ids':prompt,'steps':79});r.raise_for_status()
        assert r.json()['tokens']==ref['greedy_ids'][:79]
        (OUT/'greedy.json').write_text(json.dumps(r.json()))
        rows=await asyncio.gather(*(one(client,prompt,f'pd-validate-c{args.concurrency}-{i}',ref['text']) for i in range(args.concurrency)))
        (OUT/f'c{args.concurrency}.json').write_text(json.dumps(rows,indent=2))
        assert all(r['error'] is None and r['tokens']==79 for r in rows),rows
        for name,ids in [('current4k',prompt),('nonaligned257',other)]:
            r=await client.post('/debug/teacher_force',json={'prompt_ids':ids,'forced_tokens':ref['greedy_ids'][:8]})
            r.raise_for_status();(OUT/(name+'.npz')).write_bytes(r.content)
        (OUT/'teacher_inputs.json').write_text(json.dumps({'current4k':prompt,'nonaligned257':other,'forced_tokens':ref['greedy_ids'][:8]}))
        single=await one(client,prompt,'pd-validate-one-token',None,max_tokens=1)
        assert single['error'] is None and single['tokens']==1,single
        (OUT/'one_token.json').write_text(json.dumps(single))
        # Disconnect before a full prompt can finish; submitted GPU work must
        # drain before P/D reservations and source pages are released.
        for delay in [.02,.15]:
            async with client.stream('POST','/v1/completions',json={'prompt':prompt,'stream':True,'max_tokens':128},
                headers={'X-Request-Id':f'pd-cancel-{delay}'}) as response:
                response.raise_for_status();await asyncio.sleep(delay)
            await drained(client)
        final=await drained(client)
        (OUT/'final_health.json').write_text(json.dumps(final,indent=2))
    for role,port in [('prefill',8001),('decode',8002)]:
        text=subprocess.check_output(['ip','netns','exec','split-enterprise','curl','-fsS',f'http://10.205.0.2:{port}/health'])
        h=json.loads(text);assert h['active']==h['kv_used_blocks']==0,h
        if role=='prefill':assert h['pd_reservations']==0,h
        (OUT/(role+'_health.json')).write_text(json.dumps(h,indent=2))
    live=Path((ROOT/'run/current_results').read_text())
    for name in ['split_trace.jsonl','pd_trace.jsonl','pd_kv_trace.jsonl','enterprise_config.json','cloud_prefill_config.json','cloud_decode_config.json']:
        shutil.copyfile(live/name,OUT/name)
    transfers=[json.loads(line) for line in (OUT/'pd_kv_trace.jsonl').read_text().splitlines()]
    chunk_count=0
    for r in transfers:
        chunks=r.get('chunks',[r]);end=0
        for chunk in chunks:
            src={k:v for rank in chunk['source'] for k,v in rank[r['request_id']]['checksums'].items()}
            dst={k:v for rank in chunk['destination'] for k,v in rank[r['request_id']]['checksums'].items()}
            assert src==dst and set(src)=={'0','1'}
            if 'range_start' in chunk:
                assert chunk['range_start']==end
                end=chunk['range_end']
                for rank in chunk['source']+chunk['destination']:
                    assert rank[r['request_id']]['range_end']==end
            chunk_count+=1
        if 'chunks' in r:
            assert all(rank[r['request_id']]['enqueued_until']==end for rank in r['destination'])
    result={'passed':True,'concurrency':args.concurrency,'requests':len(rows),'exact_kv_transfers':len(transfers),'exact_kv_chunks':chunk_count,'cancel_cases':2,
            'nonaligned_prompt_tokens':257,'max_tokens_one':True}
    (OUT/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':asyncio.run(main())
