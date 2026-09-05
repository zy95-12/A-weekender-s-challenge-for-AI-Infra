"""Aggregate the representative lookup draft evidence without hiding regressions."""
import argparse,json,statistics
from pathlib import Path
import numpy as np


def stats(values):
 return {'mean':float(np.mean(values)),'repeat_std':float(np.std(values,ddof=1)),
         **{f'p{q}':float(np.percentile(values,q)) for q in [50,95,99]},'count':len(values)}


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args();root=Path(args.root)
 report={'scope':'C1 real prompts, k0/2/4; 3 repeats; descriptive tails, no capacity claim','variants':{}}
 for k in [0,2,4]:
  s=json.loads((root/f'k{k}/real_prompts/summary.json').read_text());assert s['result']=='PASS'
  cases={}
  for c in s['cases']:
   assert c['exact_match']
   values={key:stats([r[key] for r in c['repeats']]) for key in
           ['ttft_ms','tpot_ms','wall_ms','event_gap_max_ms','forward_roundtrips','rollback_roundtrips','draft_ms','rollback_ms']}
   for key in ['draft_tokens','accepted_draft_tokens','output_tokens']:values[key]=sum(r[key] for r in c['repeats'])
   values['acceptance_rate']=(values['accepted_draft_tokens']/values['draft_tokens'] if values['draft_tokens'] else None)
   values['mean_confirmed_per_forward']=values['output_tokens']/sum(r['forward_roundtrips'] for r in c['repeats'])
   gaps=np.concatenate([np.diff(r['event_monotonic_ns'])/1e6 for r in c['repeats']])
   values['pooled_event_gaps']={**{f'p{q}':float(np.percentile(gaps,q)) for q in [50,95,99]},'count':len(gaps)}
   cases[c['name']]=values
  eos=json.loads((root/f'k{k}/eos_copy.json').read_text());assert eos['result']=='PASS'
  cases['normal_eos_copy']={key:stats([r[key] for r in eos['repeats']]) for key in ['ttft_ms','tpot_ms','wall_ms','event_gap_max_ms']}
  cases['normal_eos_copy']['output_tokens_per_request']=eos['expected_output_tokens']
  report['variants'][str(k)]=cases
 profile=Path((root/'k2/profile_source.txt').read_text().strip())
 rows=[r for line in (profile/'split_trace.jsonl').read_text().splitlines() if (r:=json.loads(line)).get('client_request_id')=='spec-profile-copy']
 assert sum(r['emitted_count'] for r in rows)==79
 (root/'k2/profile_trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 grouped=[r for r in rows if r['draft_tokens']==2]
 ordinary=[r for path in (root/'k2/real_prompts').glob('copy_passage_*_trace.jsonl')
           for line in path.read_text().splitlines() if (r:=json.loads(line))['draft_tokens']==2]
 report['ordinary_k2_copy_verification']={
     'groups':len(ordinary),'mean_D_ms':statistics.mean(r['draft_ms'] for r in ordinary),
     'mean_V_service_ms':statistics.mean((r['back_end_ns']-r['front_start_ns'])/1e6-r['upload_ms']-r['download_ms'] for r in ordinary),
     'V_definition':'front-to-back task wall minus upload and return PATH intervals; includes local/remote CPU, IPC and GPU; not pure GPU',
     'mean_upload_return_path_ms':statistics.mean(r['upload_ms']+r['download_ms'] for r in ordinary),
     'mean_confirmed_tokens':statistics.mean(r['emitted_count'] for r in ordinary),
     'mean_rollback_ms':statistics.mean(r['rollback_ms'] for r in ordinary)}
 report['independent_profile']={'scope':'separate synchronized/Nsight copy request, not ordinary TPOT decomposition',
       'verification_groups':len(grouped),'draft_length':2,
       'mean_D_ms':statistics.mean(r['draft_ms'] for r in grouped),
       'mean_target_stage_GPU_ms':statistics.mean(sum(r.get(key,0) for key in
          ['enterprise_front_decode_ms','cloud_middle_decode_ms','enterprise_back_decode_ms']) for r in grouped),
       'GPU_scope':'one rank per sequential partition; includes collectives, excludes WAN and CPU/IPC',
       'mean_confirmed_tokens':statistics.mean(r['emitted_count'] for r in grouped),
       'mean_rollback_ms':statistics.mean(r['rollback_ms'] for r in grouped),
       'accepted':sum(r['accepted_draft_tokens'] for r in grouped),'proposed':sum(r['draft_tokens'] for r in grouped)}
 (root/'representative_summary.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report['independent_profile'],indent=2))

if __name__=='__main__':main()
