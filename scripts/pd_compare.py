"""Bounded stage123-vs-PD C16 comparison; restore the previous demo on exit."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from network_state import check,expected

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/root/A-weekender-s-challenge-for-AI-Infra')
OUT=ROOT/'results/pd_c16'


def run(name,cmd,cwd=ROOT):
    with (OUT/(name+'.log')).open('w') as f:
        subprocess.run(cmd,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=True)


def main():
    OUT.mkdir(exist_ok=False)
    source=Path('/root/issue6-stage4-worktree/results/concurrency_4k_tp22')
    for name in ['prompt.json','reference.json']:shutil.copyfile(source/name,OUT/name)
    files=list((ROOT/'split_poc').glob('*.py'))+[ROOT/'scripts/manage.py',ROOT/'scripts/pd_benchmark.py',ROOT/'scripts/pd_compare.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (OUT/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    flags=['--wan','--max-active','16','--ipc-mode','shm','--wire-fast','--tcp-buffer-mib','16',
           '--prefill-chunk-size','1024','--scheduler-policy','decode-first','--decode-quota','4','--pipeline-window','2']
    run('initial_down',['./poc','down'])
    try:
        for variant,extra in [('stage123_q4',[]),('pd_stage123_p2d1',['--pd'])]:
            print('Starting '+variant,flush=True)
            run(variant+'_up',['./poc','up',*flags,*extra])
            launch=json.loads((ROOT/'run/launch.json').read_text())
            check(expected(launch),OUT/(variant+'_network.json'))
            run(variant+'_measure',[str(ROOT/'.venv/bin/python'),'-m','scripts.pd_benchmark',
                '--variant',variant,'--out',str(OUT/variant),'--prompt',str(OUT/'prompt.json'),
                '--reference',str(OUT/'reference.json')])
            dest=OUT/variant
            shutil.copyfile(ROOT/'run/launch.json',dest/'launch.json')
            live=Path((ROOT/'run/current_results').read_text())
            for p in list(live.glob('*config.json'))+list(live.glob('*trace.jsonl'))+[live/'environment.json']:
                shutil.copyfile(p,dest/p.name)
            print((dest/'summary.json').read_text(),flush=True)
            run(variant+'_down',['./poc','down'])
        assert hashes=={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'Runtime changed during measurements'
    finally:
        run('cleanup',['./poc','down'])
        run('restore',['./poc','up','--wan','--ipc-mode','shm','--wire-fast','--tcp-buffer-mib','16'],BASE)
        print('Restored original demo',flush=True)


if __name__=='__main__':main()
