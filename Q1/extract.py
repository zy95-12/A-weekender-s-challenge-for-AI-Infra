from check_model import check_model
"""Frozen Qwen boundary states BEFORE final RMSNorm, preserving absolute positions."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
DEPTHS=[4,9,18,27,36]


def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args();root=Path(__file__).resolve().parent
 check_model(a.model)
 torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
 model=AutoModel.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
 seen={};hooks=[]
 for depth in DEPTHS:
  def capture(module,inputs,output,depth=depth):seen[depth]=output.detach().cpu().numpy()
  hooks.append(model.layers[depth-1].register_forward_hook(capture))
 data=json.loads((root/'cache/public_tokens.json').read_text())
 prompt=json.loads((root/'data/current_prompt.json').read_text())['prompt_ids']
 output=json.loads((root/'data/current_reference.json').read_text())['greedy_ids'][:78]
 # Decode input tokens: final EOS is never sent back after the request terminates.
 data['current']=[prompt+output]
 shapes={}
 with torch.inference_mode():
  embedding=model.embed_tokens.weight.detach().cpu().numpy();np.save(root/'cache/embedding.npy',embedding)
  for name,sequences in data.items():
   count=sum(map(len,sequences));dest=root/'cache'/name;dest.mkdir(exist_ok=True)
   arrays={d:np.lib.format.open_memmap(dest/f'depth{d}.npy',mode='w+',dtype=np.float16,shape=(count,2048)) for d in DEPTHS}
   labels=[];position=0;start=time.perf_counter()
   for i in range(0,len(sequences),4):
    batch=sequences[i:i+4];ids=torch.tensor(batch,device='cuda')
    model(input_ids=ids,use_cache=False)
    n=ids.numel()
    for d in DEPTHS:arrays[d][position:position+n]=seen[d].reshape(-1,2048)
    labels.extend(sum(batch,[]));position+=n
   for array in arrays.values():array.flush()
   np.save(dest/'labels.npy',np.array(labels,dtype=np.int64))
   shapes[name]={'tokens':count,'sequences':len(sequences),'context_tokens':len(sequences[0]),'seconds':time.perf_counter()-start}
   print(name,shapes[name],flush=True)
 for hook in hooks:hook.remove()
 (root/'results/extraction.json').write_text(json.dumps({'backend':'Transformers4.55.2 Qwen2Model FP16 SDPA, single GPU; same pinned weights as serving, not a bitwise TP2 wire capture','representation':'decoder block residual-stream output; before final RMSNorm at depth36','depths':DEPTHS,'splits':shapes,'current_prompt_tokens':4096,'current_output_input_tokens':78},indent=2))

if __name__=='__main__':main()
