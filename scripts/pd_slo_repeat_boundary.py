"""Repeat a contradictory baseline boundary using the same fixed configuration."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from scripts.network_state import check,expected,network_lock,snapshot,verify

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/root/A-weekender-s-challenge-for-AI-Infra')
OUT=ROOT/'results/pd_slo_sweep';FOLDER=OUT/'baseline'


def main():
    initial=json.loads((FOLDER/'selection.json').read_text())
    assert initial['largest_confirmed_passing_concurrency']==4 and initial['next_concurrency_confirmed_failure']==5
    runtime=FOLDER/'repeat_runtime';runtime.mkdir(exist_ok=False)
    def run(name,cmd,cwd=ROOT):
        with (runtime/(name+'.log')).open('w') as f:subprocess.run(cmd,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=True)
    source=json.loads((FOLDER/'source_hashes.json').read_text())
    run('demo_down',['./poc','down'],BASE)
    try:
        run('startup',['./poc','up','--wan','--max-active','96','--kv-blocks','32768'])
        launch=json.loads((ROOT/'run/launch.json').read_text());intent=expected(launch)
        (runtime/'launch.json').write_text(json.dumps(launch,indent=2))
        with network_lock():
            check(intent,runtime/'network_check.json')
            run('measure',[str(ROOT/'.venv/bin/python'),'-m','scripts.slo_point','--variant','baseline-repeat-c4',
                '--concurrency','4','--seconds','180','--cycles','6','--out',str(FOLDER/'baseline-repeat-c4'),
                '--prompt',str(OUT/'prompt.json'),'--reference',str(OUT/'reference.json')])
            state=snapshot();verify(state,intent)
            (FOLDER/'baseline-repeat-c4/network_after.json').write_text(json.dumps(state,indent=2))
        r=json.loads((FOLDER/'baseline-repeat-c4/summary.json').read_text());r['stage']='confirm'
        points=json.loads((FOLDER/'summaries.json').read_text());points.append(r)
        (FOLDER/'summaries.json').write_text(json.dumps(points,indent=2))
        selection=json.loads((FOLDER/'selection.json').read_text())
        selection['repeated_boundary']={'concurrency':4,'latest':r,'reason':'Coarse C4 failed; first 180-second C4 passed'}
        if r['slo_pass']:
            selection['highest_confirmed_slo_qps']=r
            selection['largest_confirmed_passing_concurrency']=4
            selection['next_concurrency_confirmed_failure']=5
        else:
            selection['highest_confirmed_slo_qps']=next(p for p in points if p['stage']=='confirm' and p['concurrency']==3)
            selection['largest_confirmed_passing_concurrency']=3
            selection['next_concurrency_confirmed_failure']=4
        (FOLDER/'selection.json').write_text(json.dumps(selection,indent=2))
        print(json.dumps(selection,indent=2),flush=True)
    finally:
        live=Path((ROOT/'run/current_results').read_text())
        for p in list(live.glob('*config.json'))+list(live.glob('*trace.jsonl'))+[live/'environment.json']:
            if p.exists():shutil.copyfile(p,runtime/p.name)
        if (runtime/'split_trace.jsonl').exists():
            with (FOLDER/'split_trace.jsonl').open('ab') as dest:dest.write((runtime/'split_trace.jsonl').read_bytes())
        run('shutdown',['./poc','down'])
        run('restore',['./poc','up','--wan','--ipc-mode','shm','--wire-fast','--tcp-buffer-mib','16'],BASE)
        assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in source.items())
        print('RESTORED DEMO',flush=True)


if __name__=='__main__':main()
