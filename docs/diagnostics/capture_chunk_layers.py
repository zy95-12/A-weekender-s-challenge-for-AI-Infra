import json,os,subprocess
from pathlib import Path
import httpx
from transformers import AutoTokenizer
root=Path.cwd();unit=AutoTokenizer.from_pretrained(str(root/'models/qwen'),local_files_only=True).encode(' apple',add_special_tokens=False)
base=root/'results/chunk_layer_diagnostic';base.mkdir(parents=True,exist_ok=True)
for size in [0,1024]:
 folder=base/f'chunk{size}';folder.mkdir(exist_ok=True)
 env={**os.environ,'SPLIT_NUMERICS_CAPTURE':str(folder/'layers')}
 with (folder/'start.log').open('w') as log:
  subprocess.run(['bash','poc','up','--wan','--tp','2','--ipc-mode','shm','--wire-fast','--tcp-buffer-mib','16','--prefill-chunk-size',str(size),'--scheduler-policy','decode-first'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 with httpx.Client(base_url='http://127.0.0.1:8000',timeout=180,trust_env=False) as client:
  r=client.post('/debug/greedy',json={'prompt_ids':unit*8192,'steps':6});r.raise_for_status()
  (folder/'tokens.json').write_text(json.dumps(r.json()))
 subprocess.run(['bash','poc','down'],check=True)
(base/'complete.json').write_text(json.dumps({'result':'CAPTURED','scope':'diagnostic only, timing perturbed; not performance'}))
