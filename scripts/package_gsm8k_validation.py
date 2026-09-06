"""Package the fixed GSM8K integration run, preserving arrays and per-position audit."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'results/gsm8k-logits'
DEST = ROOT/'docs/evidence/issue-6/gsm8k-integration'


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    summary = json.loads((RAW/'summary.json').read_text())
    assert summary.get('comparison_completed') if summary.get('metric_mode')=='ranking' else summary['passed']
    for name in ['summary.json','manifest.json','baseline-positions.json','optimized-positions.json','curl.sse']:
        shutil.copy2(RAW/name,DEST/name)
    members = {}
    def archive(name, files):
        manifest = {}
        with zipfile.ZipFile(DEST/name,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for path,label in files:
                z.write(path,label)
                manifest[label] = hashlib.sha256(path.read_bytes()).hexdigest()
        with zipfile.ZipFile(DEST/name) as z:
            assert z.testzip() is None
            for label,digest in manifest.items():
                assert hashlib.sha256(z.read(label)).hexdigest() == digest
        members[name] = manifest
    for variant in ['baseline','optimized']:
        files=[]
        for prefix in ['native-','split-']:
            files += [(p,str(p.relative_to(RAW))) for p in sorted((RAW/(prefix+variant)).rglob('*')) if p.is_file()]
        archive(variant+'_logits.zip',files)
    files = [(p,'evaluation/'+p.name) for p in sorted(RAW.iterdir()) if p.is_file()]
    for folder in ['scripts','split_poc','tests','configs']:
        files += [(p,'source/'+str(p.relative_to(ROOT))) for p in sorted((ROOT/folder).rglob('*'))
                  if p.is_file() and p.suffix in ['.py','.sh','.json'] and '__pycache__' not in p.parts]
    files += [(ROOT/name,'source/'+name) for name in ['poc','requirements.txt','requirements.lock.txt',
        'docs/integrated-serving.md','docs/gsm8k-validation.md']]
    control = ROOT/'results/gsm8k-logits-control-off'
    files += [(p,'control-off/'+str(p.relative_to(control))) for p in sorted(control.rglob('*'))
              if p.is_file() and ('native-' not in str(p.relative_to(control)) and 'split-baseline' not in str(p.relative_to(control)))]
    archive('reproduce.zip',files)
    (DEST/'members.sha256.json').write_text(json.dumps(members,indent=2))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DEST.iterdir())
            if p.is_file() and p.name!='SHA256SUMS.json'}
    (DEST/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps({p.name:p.stat().st_size for p in DEST.glob('*.zip')},indent=2))


if __name__=='__main__':main()
