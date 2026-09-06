"""Small controlled C16 allocation/transfer ablations; always restore demo."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from network_state import check,expected

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/root/issue6-pd-serving')
BASE=Path('/root/A-weekender-s-challenge-for-AI-Infra')


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['allocations','chunks','reverse_perf'],required=True)
    args=p.parse_args();out=ROOT/'results'/('pd_'+args.phase);out.mkdir(parents=True,exist_ok=False)
    for name in ['prompt.json','reference.json']:shutil.copyfile(OLD/'results/pd_c16'/name,out/name)
    def run(name,cmd,cwd):
        with (out/(name+'.log')).open('w') as f:subprocess.run(cmd,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=True)
    cases=[('old_p1d2',OLD,['--prefill-tp','1','--decode-tp','2']),('old_p2d1',OLD,[])] if args.phase=='allocations' else [
        ('notify_only',ROOT,[]),('chunk_transfer',ROOT,['--pd-chunk-transfer'])]
    if args.phase=='reverse_perf':cases=[('chunk_p1d2',ROOT,['--prefill-tp','1','--decode-tp','2','--pd-chunk-transfer'])]
    flags=['--wan','--pd','--max-active','16','--ipc-mode','shm','--wire-fast','--tcp-buffer-mib','16',
           '--prefill-chunk-size','1024','--scheduler-policy','decode-first','--decode-quota','4','--pipeline-window','2']
    run('demo_down',['./poc','down'],BASE)
    try:
        for name,workspace,extra in cases:
            print('Starting '+name,flush=True)
            files=list((workspace/'split_poc').glob('*.py'))+[workspace/'scripts/manage.py']
            hashes={str(f.relative_to(workspace)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
            run(name+'_up',['./poc','up',*flags,*extra],workspace)
            launch=json.loads((workspace/'run/launch.json').read_text())
            check(expected(launch),out/(name+'_network.json'))
            run(name+'_measure',[str(workspace/'.venv/bin/python'),'-m','scripts.pd_benchmark',
                '--variant',name,'--out',str(out/name),'--prompt',str(out/'prompt.json'),'--reference',str(out/'reference.json')],workspace)
            dest=out/name;(dest/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
            (dest/'launch.json').write_text(json.dumps(launch,indent=2))
            live=Path((workspace/'run/current_results').read_text())
            for f in list(live.glob('*config.json'))+list(live.glob('*trace.jsonl'))+[live/'environment.json']:
                shutil.copyfile(f,dest/f.name)
            assert hashes=={str(f.relative_to(workspace)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
            print((dest/'summary.json').read_text(),flush=True)
            run(name+'_down',['./poc','down'],workspace)
    finally:
        for workspace in {w for _,w,_ in cases}:run(workspace.name+'_cleanup',['./poc','down'],workspace)
        run('restore',['./poc','up','--wan','--ipc-mode','shm','--wire-fast','--tcp-buffer-mib','16'],BASE)
        print('Restored demo',flush=True)


if __name__=='__main__':main()
