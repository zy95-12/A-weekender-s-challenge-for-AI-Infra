"""Bounded C24 window/chunk ablation, with a separate diagnostic capture."""
import argparse,json,shutil,subprocess,hashlib
from pathlib import Path
import httpx
from scripts.network_state import network_lock,expected,snapshot,verify,check
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/root/A-weekender-s-challenge-for-AI-Infra')
OUT=ROOT/'results/pd_window'
PYTHON=str(ROOT/'.venv/bin/python')
def run(path,cmd,cwd=ROOT):
 with path.open('w') as f:subprocess.run(cmd,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=True)
def case(label,window,chunk,profile=False,seconds=90):
 folder=OUT/label;folder.mkdir(exist_ok=False)
 print('START '+label,flush=True)
 flags=['--wan','--pd','--max-active','96','--kv-blocks','32768','--ipc-mode','shm','--wire-fast','--tcp-buffer-mib','16',
 '--prefill-chunk-size',str(chunk),'--scheduler-policy','decode-first','--decode-quota','4','--pipeline-window','2','--pd-prefill-window',str(window)]
 if profile:flags+=['--profile']
 live=None
 try:
  run(folder/'startup.log',['./poc','up',*flags])
  live=Path((ROOT/'run/current_results').read_text())
  launch=json.loads((ROOT/'run/launch.json').read_text());(folder/'launch.json').write_text(json.dumps(launch,indent=2))
  intent=expected(launch)
  with network_lock():
   check(intent,folder/'network_check.json')
   with httpx.Client(base_url='http://127.0.0.1:8000',timeout=180,trust_env=False) as c:
    if profile:c.post('/start_profile').raise_for_status()
    try:
     run(folder/'point.log',[PYTHON,'-m','scripts.slo_point','--variant',label,'--concurrency','24','--seconds',str(seconds),
       '--cycles','2' if profile else '4','--out',str(folder/'point'),'--prompt',str(OUT/'prompt.json'),'--reference',str(OUT/'reference.json')])
    finally:
     if profile:c.post('/stop_profile').raise_for_status()
   observed=snapshot();verify(observed,intent);(folder/'network_after.json').write_text(json.dumps(observed,indent=2))
  s=json.loads((folder/'point/summary.json').read_text());s.update(prefill_window=window,decode_window=2,chunk=chunk,profile_only=profile)
  (folder/'summary.json').write_text(json.dumps(s,indent=2));print('RESULT '+json.dumps(s),flush=True)
  return s
 finally:
  run(folder/'shutdown.log',['./poc','down'])
  if live:
   for p in live.iterdir():
    if p.is_file():shutil.copyfile(p,folder/p.name)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--only',choices=['profile','performance','confirm'],default='profile');args=parser.parse_args()
 OUT.mkdir(parents=True,exist_ok=True)
 for n in ['prompt.json','reference.json']:shutil.copyfile(Path('/root/issue6-pd-serving/results/pd_c16')/n,OUT/n)
 hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list((ROOT/'split_poc').glob('*.py'))+list((ROOT/'scripts').glob('*.py'))}
 (OUT/f'{args.only}_source_hashes.json').write_text(json.dumps(hashes,indent=2))
 run(OUT/f'{args.only}_demo_down.log',['./poc','down'],BASE)
 try:
  if args.only=='profile':case('profile-p2-ch1024',2,1024,True,20)
  elif args.only=='performance':
   results=[case(f'p{w}-ch1024',w,1024) for w in [2,3,4]]
   # A modest window effect is expected; measure chunk 2048 at the best SLO-passing window.
   eligible=[r for r in results if r['slo_pass']] or results
   best=max(eligible,key=lambda r:r['completed_qps'])
   results.append(case(f'p{best["prefill_window"]}-ch2048',best['prefill_window'],2048))
   (OUT/'performance.json').write_text(json.dumps(results,indent=2))
  else:
   results=json.loads((OUT/'performance.json').read_text())
   best=max([r for r in results if r['slo_pass']] or results,key=lambda r:r['completed_qps'])
   values=[case('confirm-p2-ch1024',2,1024,seconds=180)]
   if best['prefill_window']!=2 or best['chunk']!=1024:
    values.append(case(f'confirm-p{best["prefill_window"]}-ch{best["chunk"]}',best['prefill_window'],best['chunk'],seconds=180))
   (OUT/'confirm.json').write_text(json.dumps(values,indent=2))
 finally:
  run(OUT/f'{args.only}_restore.log',['./poc','up','--wan','--ipc-mode','shm','--wire-fast','--tcp-buffer-mib','16'],BASE)
  assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
  print('RESTORED DEMO',flush=True)
if __name__=='__main__':main()
