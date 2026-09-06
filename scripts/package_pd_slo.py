"""Bundle per-system raw evidence and exact benchmark source snapshots."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'results/pd_slo_sweep'
OUT=ROOT/'docs/evidence/issue-6/pd-slo'


def main():
    for variant in ['baseline','optimized']:
        with zipfile.ZipFile(OUT/f'{variant}_raw.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted((RAW/variant).rglob('*')):
                if p.is_file():z.write(p,str(p.relative_to(ROOT)))
            for name in ['prompt.json','reference.json']:z.write(RAW/name,str((RAW/name).relative_to(ROOT)))
    with zipfile.ZipFile(OUT/'reproduce.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for folder,pattern in [('split_poc','*.py'),('scripts','*.py'),('configs','*.json'),('tests','*.py')]:
            for p in sorted((ROOT/folder).glob(pattern)):z.write(p,str(p.relative_to(ROOT)))
        z.write(ROOT/'poc','poc')
    hashes={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for p in OUT.iterdir() if p.is_file() and p.name!='SHA256SUMS.json'}
    (OUT/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(hashes,indent=2))


if __name__=='__main__':main()
