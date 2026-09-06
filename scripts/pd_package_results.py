"""Package the bounded experiment evidence without weights or Nsight captures."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'docs/evidence/issue-6/pd-c16'


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    for src,name in [(ROOT/'results/pd_c16/comparison.csv','comparison.csv'),
                     (ROOT/'results/pd_c16/audit.json','audit.json'),
                     (ROOT/'results/pd_followup/logits_comparison.json','logits_comparison.json')]:
        shutil.copyfile(src,DEST/name)
    with zipfile.ZipFile(DEST/'raw_results.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for folder in ['pd_c16','pd_validation','pd_followup']:
            for path in sorted((ROOT/'results'/folder).rglob('*')):
                if path.is_file():z.write(path,str(path.relative_to(ROOT)))
        for path in sorted((ROOT/'scripts').glob('pd_*.py')):z.write(path,str(path.relative_to(ROOT)))
        for path in sorted((ROOT/'split_poc').glob('*.py')):z.write(path,str(path.relative_to(ROOT)))
        z.write(ROOT/'scripts/manage.py','scripts/manage.py')
        for path in sorted((ROOT/'configs').glob('*.json')):z.write(path,str(path.relative_to(ROOT)))
    hashes={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in DEST.iterdir() if p.is_file() and p.name!='SHA256SUMS.json'}
    (DEST/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(hashes,indent=2))


if __name__=='__main__':main()
