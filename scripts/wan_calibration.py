"""Activation-size sweep, no model compute; actual serving transport and IPC."""
import json,os,subprocess,time,shutil,random
from pathlib import Path
import httpx
from network_state import check,expected,network_lock

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/root/A-weekender-s-challenge-for-AI-Infra')
OUT=ROOT/'results/wan_calibration'
ROWS=[1,4,16,64,128,256,512,1024,2048,4096,8192,16384]

def run(name,args,cwd=ROOT,env=None):
    with (OUT/(name+'.log')).open('w') as f:
        subprocess.run(args,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    OUT.mkdir(exist_ok=False)
    run('demo_down',['./poc','down'],BASE)
    try:
        for variant in ['baseline','stage1']:
            print('Starting '+variant,flush=True)
            flags=[] if variant=='baseline' else ['--ipc-mode','shm','--wire-fast','--tcp-buffer-mib','16']
            run(variant+'_up',['./poc','up','--wan','--tp','2',*flags],env={**os.environ,'SPLIT_WAN_PROBE':'1'})
            dest=OUT/variant;dest.mkdir()
            live=Path((ROOT/'run/current_results').read_text().strip())
            for name in ['environment.json','enterprise_config.json','cloud_config.json']:
                shutil.copyfile(live/name,dest/name)
            launch=json.loads((ROOT/'run/launch.json').read_text());(dest/'launch.json').write_text(json.dumps(launch,indent=2))
            check(expected(launch),dest/'network_check.json')
            samples=[]
            # Two passes with different deterministic orders expose size/order variation.
            for pass_id in [0,1]:
                sizes=ROWS[:] if pass_id==0 else random.Random(5).sample(ROWS,len(ROWS))
                with network_lock(),httpx.Client(base_url='http://127.0.0.1:8000',timeout=300,trust_env=False) as client:
                    for n in sizes:
                        r=client.post('/debug/wan_probe',json={'op':'wan_prepare','rows':n});r.raise_for_status()
                        values=[]
                        for i in range(11):
                            r=client.post('/debug/wan_probe',json={'op':'wan_roundtrip','rows':n});r.raise_for_status()
                            v=r.json();assert v['validation_ok'],v
                            v.update(variant=variant,pass_id=pass_id,sample=i,warmup=i<3)
                            samples.append(v)
                            if i>=3:values.append(v['gpu_ready_rtt_ms'])
                        print(json.dumps({'variant':variant,'pass':pass_id,'rows':n,'mib':n/128,'rtt_mean_ms':sum(values)/len(values)}),flush=True)
                        (dest/'samples.json').write_text(json.dumps(samples,indent=2))
            with httpx.Client(trust_env=False) as client:
                h=client.get('http://127.0.0.1:8000/health').json();assert h['active']==h['waiting']==h['kv_used_blocks']==0
                (dest/'final_health.json').write_text(json.dumps(h,indent=2))
            run(variant+'_down',['./poc','down'])
    finally:
        run('cleanup',['./poc','down'])
        run('restore',['./poc','up','--wan','--ipc-mode','shm','--wire-fast','--tcp-buffer-mib','16'],BASE)

if __name__=='__main__':main()
