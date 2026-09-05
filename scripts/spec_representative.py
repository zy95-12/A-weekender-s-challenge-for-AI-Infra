"""Fixed real-prompt greedy/output and burst-latency checks, with raw traces.

Compare mode requires exact full output IDs against the saved OFF reference.
No synthetic acceptance-rate extrapolation or maximum-capacity claims.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
import httpx
import numpy as np
from transformers import AutoTokenizer
from network_state import network_lock,expected,check,snapshot,verify

ROOT=Path(__file__).resolve().parents[1]
PASSAGE=('The library opens at nine in the morning and closes at six in the evening. '
         'Visitors can borrow books for three weeks. To renew a loan, use the online catalog '
         'or speak to the librarian. Quiet study rooms are on the second floor. '
         'The ground floor has a help desk, newspapers, and a small collection of maps. '
         'Please return books through the blue return slot beside the main entrance.')
QUESTIONS=[('explanation','Explain why a key-value cache makes language model decoding faster. '
            'Describe one limitation and one practical tradeoff.'),
           ('copy_passage','Copy the following passage exactly, without a preface or commentary:\n'+PASSAGE),
           ('code','Write a Python function named merge_sorted(a, b) that merges two sorted lists '
            'without calling sorted(). Include a short example and explain its time complexity.')]

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--reference')
 p.add_argument('--repeats',type=int,default=3);p.add_argument('--tokens',type=int,default=128)
 args=p.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
 ref=json.loads(Path(args.reference).read_text()) if args.reference else None
 launch=json.loads((ROOT/'run/launch.json').read_text());intent=expected(launch)
 check(intent,out/'network_check.json')
 subprocess.run([sys.executable,str(ROOT/'scripts/environment.py'),str(out/'environment.json')],check=True)
 tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'models/qwen'),local_files_only=True)
 live=Path((ROOT/'run/current_results').read_text().strip());trace=live/'split_trace.jsonl'
 report={'result':'RUNNING','configuration':vars(args),'cases':[],
         'scope':'real prompts, fixed output length, C1; descriptive tails, not capacity/SLO'}
 try:
  with httpx.Client(base_url='http://127.0.0.1:8000',timeout=300,trust_env=False) as client:
   health=client.get('/health');health.raise_for_status();report['server']=health.json()
   client.post('/v1/completions',json={'prompt':'Hello','max_tokens':8,'ignore_eos':True}).raise_for_status()
   for index,(name,question) in enumerate(QUESTIONS):
    ids=tokenizer.apply_chat_template([{'role':'user','content':question}],tokenize=True,add_generation_prompt=True)
    greedy=client.post('/debug/greedy',json={'prompt_ids':ids,'steps':args.tokens});greedy.raise_for_status()
    tokens=greedy.json()['tokens'];assert len(tokens)==args.tokens
    same=ref is None or (ids==ref['cases'][index]['prompt_ids'] and tokens==ref['cases'][index]['tokens'])
    case={'name':name,'prompt':question,'prompt_ids':ids,'tokens':tokens,'exact_match':same,'repeats':[]}
    report['cases'].append(case)
    assert same,f'{name}: speculative output differs from ordinary greedy'
    for repeat in range(args.repeats):
     rid=f'spec-{name}-{repeat}-{time.time_ns()}';offset=trace.stat().st_size
     stamps=[];usage=None;done=False;t=time.perf_counter_ns()
     with client.stream('POST','/v1/completions',headers={'X-Request-Id':rid},json={'prompt':ids,
          'max_tokens':args.tokens,'ignore_eos':True,'stream':True,'stream_options':{'include_usage':True}}) as response:
      response.raise_for_status()
      for line in response.iter_lines():
       if line=='data: [DONE]':done=True;break
       if line.startswith('data: {'):
        event=json.loads(line[6:]);assert 'error' not in event,event
        if event.get('choices'):stamps.append(time.perf_counter_ns())
        if event.get('usage'):usage=event['usage']
     assert done and len(stamps)==args.tokens and usage['completion_tokens']==args.tokens
     gaps=np.diff(stamps)/1e6
     with trace.open() as file:
      file.seek(offset);rows=[x for line in file if (x:=json.loads(line)).get('client_request_id')==rid]
     assert sum(r.get("emitted_count",int(r["emits_token"])) for r in rows)==args.tokens
     (out/f'{name}_{repeat}_trace.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
     case['repeats'].append({'client_request_id':rid,'ttft_ms':(stamps[0]-t)/1e6,
         'tpot_ms':float(np.mean(gaps)),'wall_ms':(stamps[-1]-t)/1e6,'event_gap_max_ms':float(max(gaps)),
         **{f'event_gap_p{q}_ms':float(np.percentile(gaps,q)) for q in [50,95,99]},
         'event_monotonic_ns':stamps,'output_tokens':args.tokens,
         'forward_roundtrips':len(rows),'rollback_roundtrips':sum(r.get('rollback_roundtrips',0) for r in rows),
         'draft_tokens':sum(r.get('draft_tokens',0) for r in rows),
         'accepted_draft_tokens':sum(r.get('accepted_draft_tokens',0) for r in rows),
         'draft_ms':sum(r.get('draft_ms',0) for r in rows),
         'rollback_ms':sum(r.get('rollback_ms',0) for r in rows)})
    print(name,'PASS',flush=True)
   # Normal EOS handling and a final one-token output budget must match OFF too.
   report['boundaries']=[]
   for prompt,limit in [('Answer with only the digit 2: what is 1+1?',32),('Hello',1)]:
    ids=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True)
    r=client.post('/v1/completions',json={'prompt':ids,'max_tokens':limit});r.raise_for_status();v=r.json()
    item={'text':v['choices'][0]['text'],'usage':v['usage'],'finish_reason':v['choices'][0]['finish_reason']}
    report['boundaries'].append(item)
   if ref:assert report['boundaries']==ref['boundaries']
  after=snapshot();(out/'network_after.json').write_text(json.dumps(after,indent=2));verify(after,intent)
  report['result']='PASS'
 except BaseException as error:
  report.update(result='FAIL',error=repr(error));raise
 finally:(out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
 with network_lock():main()
