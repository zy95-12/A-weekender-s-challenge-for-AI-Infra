"""Reverse TP handoff validation and a fixed-position TP2 decode control."""
import json
import subprocess
from pathlib import Path
import httpx

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/root/A-weekender-s-challenge-for-AI-Infra')
OUT=ROOT/'results/pd_followup'


def run(name,cmd,cwd=ROOT):
    with (OUT/(name+'.log')).open('w') as f:
        subprocess.run(cmd,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=True)


def main():
    OUT.mkdir(exist_ok=False)
    flags=['--wan','--max-active','16','--ipc-mode','shm','--wire-fast','--tcp-buffer-mib','16',
        '--prefill-chunk-size','1024','--scheduler-policy','decode-first','--decode-quota','4','--pipeline-window','2']
    run('demo_down',['./poc','down'],BASE)
    try:
        run('reverse_up',['./poc','up',*flags,'--pd','--prefill-tp','1','--decode-tp','2','--pd-verify-kv'])
        run('reverse_validate',[str(ROOT/'.venv/bin/python'),'-m','scripts.pd_validate','--out',str(OUT/'reverse'),'--concurrency','1'])
        print('Reverse TP1 -> TP2 validated',flush=True)
        run('reverse_down',['./poc','down'])
        run('reference_up',['./poc','up',*flags,'--enterprise-tp','1','--cloud-tp','2'])
        inputs=json.loads((ROOT/'results/pd_validation/teacher_inputs.json').read_text())
        with httpx.Client(base_url='http://127.0.0.1:8000',timeout=180,trust_env=False) as client:
            for name in ['current4k','nonaligned257']:
                r=client.post('/debug/teacher_force',json={'prompt_ids':inputs[name],'forced_tokens':inputs['forced_tokens']})
                r.raise_for_status();(OUT/(name+'_reference.npz')).write_bytes(r.content)
        live=Path((ROOT/'run/current_results').read_text())
        (OUT/'reference_config.json').write_text((live/'enterprise_config.json').read_text())
        print('Fixed-position reference captured',flush=True)
    finally:
        run('cleanup',['./poc','down'])
        run('restore',['./poc','up','--wan','--ipc-mode','shm','--wire-fast','--tcp-buffer-mib','16'],BASE)
        print('Original demo restored',flush=True)


if __name__=='__main__':main()
