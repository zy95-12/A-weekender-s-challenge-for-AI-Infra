"""Adaptive fixed-SLO sweep: coarse curve, integer boundary, longer confirmation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from scripts.network_state import check,expected,network_lock,snapshot,verify

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/root/A-weekender-s-challenge-for-AI-Infra')
OUT=ROOT/'results/pd_slo_sweep'
OPT=['--pd','--ipc-mode','shm','--wire-fast','--tcp-buffer-mib','16','--prefill-chunk-size','1024',
     '--scheduler-policy','decode-first','--decode-quota','4','--pipeline-window','2']


def run(log,cmd,cwd=ROOT):
    with log.open('w') as f:subprocess.run(cmd,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['baseline','optimized'],required=True)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    for name in ['prompt.json','reference.json']:
        shutil.copyfile(Path('/root/issue6-pd-serving/results/pd_c16')/name,OUT/name)
    folder=OUT/args.variant;folder.mkdir(exist_ok=False)
    source_files=list((ROOT/'split_poc').glob('*.py'))+[ROOT/'scripts/manage.py',ROOT/'scripts/slo_point.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    (folder/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    points=[];screen={};confirmed={}
    def point(c,stage,seconds,cycles):
        label=f'{args.variant}-{stage}-c{c}';dest=folder/label
        print('START '+label,flush=True)
        with network_lock():
            verify(snapshot(),intent)
            run(folder/(label+'.log'),[str(ROOT/'.venv/bin/python'),'-m','scripts.slo_point','--variant',label,
                '--concurrency',str(c),'--seconds',str(seconds),'--cycles',str(cycles),
                '--out',str(dest),'--prompt',str(OUT/'prompt.json'),'--reference',str(OUT/'reference.json')])
            observed=snapshot();verify(observed,intent)
            (dest/'network_after.json').write_text(json.dumps(observed,indent=2))
        value=json.loads((dest/'summary.json').read_text());value['stage']=stage
        points.append(value);(folder/'summaries.json').write_text(json.dumps(points,indent=2))
        print('RESULT '+json.dumps({k:value[k] for k in ['variant','concurrency','requests','duration_s','completed_qps',
            'mean_ttft_ms','p99_ttft_ms','mean_tpot_ms','p99_tpot_ms','slo_attainment','start_cohort_slo_attainment','slo_pass']}),flush=True)
        return value
    run(folder/'demo_down.log',['./poc','down'],BASE)
    try:
        flags=OPT if args.variant=='optimized' else []
        run(folder/'startup.log',['./poc','up','--wan','--max-active','96','--kv-blocks','32768',*flags])
        launch=json.loads((ROOT/'run/launch.json').read_text());intent=expected(launch)
        (folder/'launch.json').write_text(json.dumps(launch,indent=2))
        with network_lock():check(intent,folder/'network_check.json')
        coarse=[1,2,3,4,8,16] if args.variant=='baseline' else [1,2,4,8,16,24,32,48,64,96]
        failures=0
        for c in coarse:
            value=point(c,'coarse',45,3);screen[c]=value
            failures=0 if value['slo_pass'] else failures+1
            if args.variant=='optimized' and c>=32 and failures>=2:break
        passed=[c for c,v in screen.items() if v['slo_pass']]
        if not passed:raise RuntimeError('No SLO-passing concurrency')
        lo=max(passed);fails=[c for c,v in screen.items() if c>lo and not v['slo_pass']]
        if fails:
            hi=min(fails)
            while hi-lo>1:
                c=(lo+hi)//2;value=point(c,'refine',60,4);screen[c]=value
                if value['slo_pass']:lo=c
                else:hi=c
        # Longer trials at both the largest passing C and best observed QPS.
        candidates={lo,max((c for c,v in screen.items() if v['slo_pass']),key=lambda c:screen[c]['completed_qps'])}
        for c in sorted(candidates):confirmed[c]=point(c,'confirm',180,6)
        while not any(v['slo_pass'] for v in confirmed.values()):
            lo=min(confirmed)-1
            if lo<1:raise RuntimeError('No confirmed passing concurrency')
            confirmed[lo]=point(lo,'confirm',180,6)
        boundary=max(c for c,v in confirmed.items() if v['slo_pass'])
        if boundary<96:
            c=boundary+1
            while c<=96:
                if c not in confirmed:confirmed[c]=point(c,'confirm',180,6)
                if not confirmed[c]['slo_pass']:break
                boundary=c;c+=1
        winner=max((v for v in confirmed.values() if v['slo_pass']),key=lambda v:v['completed_qps'])
        (folder/'selection.json').write_text(json.dumps({'highest_confirmed_slo_qps':winner,
            'largest_confirmed_passing_concurrency':boundary,'next_concurrency_confirmed_failure':
            boundary+1 if boundary+1 in confirmed and not confirmed[boundary+1]['slo_pass'] else None},indent=2))
        print('SELECT '+json.dumps(json.loads((folder/'selection.json').read_text())),flush=True)
    finally:
        if (ROOT/'run/current_results').exists():
            live=Path((ROOT/'run/current_results').read_text())
            for p in list(live.glob('*config.json'))+list(live.glob('*trace.jsonl'))+[live/'environment.json']:
                if p.exists():shutil.copyfile(p,folder/p.name)
        run(folder/'shutdown.log',['./poc','down'])
        run(folder/'restore.log',['./poc','up','--wan','--ipc-mode','shm','--wire-fast','--tcp-buffer-mib','16'],BASE)
        print('RESTORED DEMO',flush=True)
        assert hashes=={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}


if __name__=='__main__':main()
