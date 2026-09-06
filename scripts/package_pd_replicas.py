"""Archive complete windows, excluded failures, source snapshots and final code."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RAW=ROOT/'results/pd_replicas';DOC=ROOT/'docs/evidence/issue-6/pd-replicas'
archive=RAW/'evidence.zip';files=[]
for name in ['pd_replicas_initial','pd_replicas_v2','pd_replicas']:
    base=ROOT/'results'/name
    files.extend((p,'results/'+name+'/'+str(p.relative_to(base))) for p in base.rglob('*') if p.is_file() and p!=archive)
for folder in ['split_poc','scripts','tests']:
    files.extend((p,'source/'+str(p.relative_to(ROOT))) for p in (ROOT/folder).glob('*.py'))
files.extend((p,'source/'+str(p.relative_to(ROOT))) for p in (ROOT/'configs').glob('*') if p.is_file())
files.extend((ROOT/p,'source/'+p) for p in ['poc','docs/pd-replicas.md'])
files.extend((p,'source/'+str(p.relative_to(ROOT))) for p in DOC.glob('*') if p.is_file() and p.name!='archive.json')
manifest={n:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p,n in files}
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p,n in sorted(files,key=lambda item:item[1]):z.write(p,n)
    z.writestr('SHA256SUMS.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for n,m in manifest.items():
        b=z.read(n);assert len(b)==m['bytes'] and hashlib.sha256(b).hexdigest()==m['sha256']
(DOC/'archive.json').write_text(json.dumps({'local_path':str(archive),'bytes':archive.stat().st_size,
    'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(files),'zip_crc_and_member_hashes_verified':True},indent=2))
print((DOC/'archive.json').read_text())
