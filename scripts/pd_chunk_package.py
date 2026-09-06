"""Publish compact, reproducible chunk-transfer evidence without model weights."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'docs/evidence/issue-6/pd-chunks'


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    for path in (ROOT/'results/pd_chunk_report').iterdir():
        if path.is_file():shutil.copyfile(path,DEST/path.name)
    with zipfile.ZipFile(DEST/'raw_results.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in ['pd_allocations','pd_chunks','pd_reverse_perf','pd_chunk_validation','pd_chunk_reverse','pd_chunk_report']:
            for path in sorted((ROOT/'results'/name).rglob('*')):
                if path.is_file():z.write(path,str(path.relative_to(ROOT)))
        for folder,pattern in [('scripts','pd_*.py'),('split_poc','*.py'),('configs','*.json'),('tests','test_pd.py')]:
            for path in sorted((ROOT/folder).glob(pattern)):z.write(path,str(path.relative_to(ROOT)))
        z.write(ROOT/'scripts/manage.py','scripts/manage.py')
    hashes={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for p in DEST.iterdir() if p.is_file() and p.name!='SHA256SUMS.json'}
    (DEST/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(hashes,indent=2))


if __name__=='__main__':main()
