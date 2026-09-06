from check_model import check_model
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from metrics import score
from retrieval import recover


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--root',type=Path,default=Path('Q1'));a=p.parse_args()
    check_model(a.model)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    mapping=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
    with safe_open(a.model/mapping['model.embed_tokens.weight'],framework='pt') as f:embedding=f.get_tensor('model.embed_tokens.weight').to('cuda',dtype=torch.float16)
    prompt=json.loads((a.root/'data/current_prompt.json').read_text())['prompt_ids']
    output=json.loads((a.root/'data/current_reference.json').read_text())['greedy_ids'][:79]
    result={'method':'Full 151936-row cosine search; FP16 exposed embeddings, FP32 normalized dot products; TF32 off','depth':0,'model_revision':(a.model/'revision.txt').read_text().strip(),'evaluations':{}}
    cases=[('current_prompt',prompt),('current_output_tokens',output)]
    if (a.root/'cache/test/labels.npy').exists():cases.append(('public_test',np.load(a.root/'cache/test/labels.npy').tolist()))
    for name,ids in cases:
        start=time.perf_counter();predicted=recover(embedding[torch.tensor(ids,device='cuda')],embedding)
        result['evaluations'][name]={**score(ids,predicted,tokenizer),'seconds':time.perf_counter()-start}
        (a.root/f'results/depth0_{name}_predictions.json').write_text(json.dumps({'target':ids,'prediction':predicted},indent=2))
        (a.root/f'results/depth0_{name}_recovered.txt').write_text(tokenizer.decode(predicted))
    (a.root/'results/embedding_attack.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
