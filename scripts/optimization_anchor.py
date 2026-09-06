"""Small repeated representative anchor; does not replace the full matrix."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    launch=json.loads((ROOT/'run/launch.json').read_text())
    manifest={'result':'RUNNING','launch':launch,'started_unix':time.time(),'command':sys.argv}
    (out/'anchor.json').write_text(json.dumps(manifest,indent=2))
    for rep in range(args.repeats):
        for isl,osl in ((512,128),(8192,256)):
            for c in (1,4):
                label=f'isl_{isl}_osl_{osl}_c_{c}_r_{rep}'
                command=[sys.executable,str(ROOT/'scripts/benchmark.py'),'--baseline','--rates','inf',
                    '--input',str(isl),'--output-tokens',str(osl),'--max-concurrency',str(c),
                    '--requests',str(max(8,4*c)),'--seed',str(rep),'--output',str(out/label)]
                print('START',label,flush=True)
                with (out/(label+'.log')).open('w') as log:
                    result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:
                    manifest.update(result='FAIL',failed=label)
                    (out/'anchor.json').write_text(json.dumps(manifest,indent=2))
                    raise RuntimeError((out/(label+'.log')).read_text()[-4000:])
                print('DONE',label,flush=True)
    manifest.update(result='PASS',finished_unix=time.time())
    (out/'anchor.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
