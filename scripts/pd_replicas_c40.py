"""Single higher-load C40 comparison after C24 left both P replicas underutilized."""
import argparse,json,shutil,hashlib,subprocess
from pathlib import Path
from scripts.network_state import network_lock,expected,snapshot,verify,check
ROOT=Path(__file__).resolve().parents[1];BASE=Path('/root/A-weekender-s-challenge-for-AI-Infra')
OUT=ROOT/'results/pd_replicas';PYTHON=str(ROOT/'.venv/bin/python')
def run(log,cmd,cwd=ROOT):
 with log.open('w') as f:subprocess.run(cmd,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=True)
def case(label,replicas,validation=False,seconds=90):
 folder=OUT/label;folder.mkdir(exist_ok=False);live=None
 print('START '+label,flush=True)
 flags=['--pd','--wan','--max-active','96','--kv-blocks','32768','--ipc-mode','shm','--wire-fast','--tcp-buffer-mib','16',
 '--prefill-chunk-size','2048','--scheduler-policy','decode-first','--decode-quota','4','--pipeline-window','2','--pd-prefill-window','3',
 '--prefill-replicas',str(replicas),'--prefill-tp','1' if replicas==2 else '2','--decode-tp','1']
 if validation:flags+=['--pd-verify-kv']
 try:
  run(folder/'startup.log',['./poc','up',*flags]);live=Path((ROOT/'run/current_results').read_text())
  launch=json.loads((ROOT/'run/launch.json').read_text());(folder/'launch.json').write_text(json.dumps(launch,indent=2))
  intent=expected(launch)
  with network_lock():
   check(intent,folder/'network_check.json')
   if validation:
    run(folder/'validation.log',[PYTHON,'-m','scripts.pd_replicas_validate','--out',str(folder/'validation'),'--concurrency','40'])
    value=json.loads((folder/'validation/audit.json').read_text())
   else:
    run(folder/'point.log',[PYTHON,'-m','scripts.slo_point','--variant',label,'--concurrency','40','--seconds',str(seconds),'--cycles','4',
      '--out',str(folder/'point'),'--prompt',str(OUT/'prompt.json'),'--reference',str(OUT/'reference.json')])
    value=json.loads((folder/'point/summary.json').read_text());value.update(prefill_replicas=replicas,prefill_tp=1 if replicas==2 else 2,
      prefill_window_per_replica=3,decode_window=2,chunk=2048)
    (folder/'summary.json').write_text(json.dumps(value,indent=2))
   observed=snapshot();verify(observed,intent);(folder/'network_after.json').write_text(json.dumps(observed,indent=2))
  print('RESULT '+json.dumps(value),flush=True);return value
 finally:
  run(folder/'shutdown.log',['./poc','down'])
  if live:
   for p in live.iterdir():
    if p.is_file():shutil.copyfile(p,folder/p.name)
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['performance'],default='performance');a=p.parse_args()
 OUT.mkdir(parents=True,exist_ok=True)
 for n in ['prompt.json','reference.json']:shutil.copyfile(Path('/root/issue6-pd-serving/results/pd_c16')/n,OUT/n)
 hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list((ROOT/'split_poc').glob('*.py'))+list((ROOT/'scripts').glob('*.py'))}
 (OUT/f'c40_{a.mode}_source_hashes.json').write_text(json.dumps(hashes,indent=2))
 run(OUT/f'c40_{a.mode}_demo_down.log',['./poc','down'],BASE)
 try:
  if a.mode=='validation':case('validation',2,True)
  else:
   seconds=180 if a.mode=='confirm' else 90
   values=[case(f'c40-{a.mode}-tp2',1,seconds=seconds),case(f'c40-{a.mode}-replicas',2,seconds=seconds)]
   (OUT/f'c40_{a.mode}.json').write_text(json.dumps(values,indent=2))
 finally:
  run(OUT/f'c40_{a.mode}_restore.log',['./poc','up','--wan','--ipc-mode','shm','--wire-fast','--tcp-buffer-mib','16'],BASE)
  assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
  print('RESTORED DEMO',flush=True)
if __name__=='__main__':main()
