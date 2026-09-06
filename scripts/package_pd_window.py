"""Bundle raw measurements and reproducible source; SQLite is re-exportable."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'results/pd_window';DOC=ROOT/'docs/evidence/issue-6/pd-window'
archive=RAW/'evidence.zip'
files=[(p,'raw/'+str(p.relative_to(RAW))) for p in RAW.rglob('*') if p.is_file() and p.suffix not in {'.sqlite','.zip'}]
files += [(p,'source/'+str(p.relative_to(ROOT))) for folder in ['split_poc','scripts','tests'] for p in (ROOT/folder).glob('*.py')]
files += [(p,'source/'+str(p.relative_to(ROOT))) for p in (ROOT/'configs').glob('*') if p.is_file()]
files += [(ROOT/p,'source/'+p) for p in ['poc','docs/pd-window.md']]
files += [(p,'source/'+str(p.relative_to(ROOT))) for p in DOC.glob('*') if p.is_file() and p.name!='archive.json']
manifest={name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p,name in files}
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p,name in sorted(files,key=lambda item:item[1]):z.write(p,name)
 z.writestr('SHA256SUMS.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
DOC.mkdir(exist_ok=True,parents=True)
(DOC/'archive.json').write_text(json.dumps({'local_path':str(archive),'bytes':archive.stat().st_size,
 'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(files),'zip_crc_verified':True},indent=2))
print((DOC/'archive.json').read_text())
