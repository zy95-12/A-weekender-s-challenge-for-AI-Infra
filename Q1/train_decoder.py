from check_model import check_model
"""Equal-budget two-layer recovery probes; select checkpoints ONLY on public validation."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from metrics import score
from retrieval import recover


def normalize(x):
 return torch.nn.functional.normalize(x.float(),dim=-1)*2048**.5


class Decoder(torch.nn.Module):
 def __init__(self,vocab):
  super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(2048,512),torch.nn.GELU(),torch.nn.Linear(512,vocab))
 def forward(self,x):return self.net(normalize(x))


@torch.inference_mode()
def predict(model,features):
 result=[]
 for start in range(0,len(features),256):
  x=torch.tensor(np.array(features[start:start+256]),device='cuda')
  with torch.autocast('cuda',dtype=torch.float16):result.extend(model(x).argmax(-1).tolist())
 return result


def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--depths',default='4,9,18,27,36');p.add_argument('--epochs',type=int,default=4);p.add_argument('--seed',type=int,default=42);a=p.parse_args();root=Path(__file__).resolve().parent
 check_model(a.model)
 torch.set_num_threads(4);tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
 result_dir=root/'results' if a.seed==42 else root/'results'/f'seed{a.seed}';result_dir.mkdir(exist_ok=True)
 labels={s:np.load(root/'cache'/s/'labels.npy') for s in ['train','validation','test','current']}
 embedding=torch.tensor(np.load(root/'cache/embedding.npy'),device='cuda')
 for depth in map(int,a.depths.split(',')):
  torch.manual_seed(a.seed);np.random.seed(a.seed)
  features={s:np.load(root/'cache'/s/f'depth{depth}.npy',mmap_mode='r') for s in labels}
  model=Decoder(len(embedding)).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01,fused=True)
  scaler=torch.amp.GradScaler('cuda');best=-1;history=[];checkpoint=root/'checkpoints'/f'depth{depth}_seed{a.seed}.pt';checkpoint.parent.mkdir(exist_ok=True)
  start=time.perf_counter()
  for epoch in range(a.epochs):
   model.train();order=np.random.permutation(len(labels['train']));losses=[]
   for step in range(0,len(order),256):
    idx=order[step:step+256];x=torch.tensor(np.array(features['train'][idx]),device='cuda');y=torch.tensor(labels['train'][idx],device='cuda')
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):loss=torch.nn.functional.cross_entropy(model(x),y)
    scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();losses.append(loss.item())
   model.eval();pred=predict(model,features['validation']);acc=float(np.mean(np.array(pred)==labels['validation']))
   r={'epoch':epoch+1,'training_loss':float(np.mean(losses)),'public_validation_accuracy':acc,'elapsed_s':time.perf_counter()-start};history.append(r)
   print(json.dumps({'depth':depth,**r}),flush=True)
   if acc>best:best=acc;torch.save(model.state_dict(),checkpoint)
  model.load_state_dict(torch.load(checkpoint,weights_only=True));model.eval()
  result={'depth':depth,'seed':a.seed,'epochs':a.epochs,'train_tokens':len(labels['train']),'validation_tokens':len(labels['validation']),'hidden_width':512,'input_width':2048,'vocab_size':len(embedding),'optimizer':'AdamW lr0.001 weight_decay0.01','batch_size':256,'checkpoint_selection':'highest public-validation top1; target never used','history':history,'evaluations':{}}
  for name in ['test','current']:
   target=labels[name].tolist();pred=predict(model,features[name]);vec=recover(torch.tensor(np.array(features[name]),device='cuda'),embedding)
   selections=[('public_test',0,len(target))] if name=='test' else [('current_prompt',0,4096),('current_output_inputs',4096,len(target))]
   for tag,l,r in selections:
    result['evaluations'][tag]={'mlp':score(target[l:r],pred[l:r],tokenizer,set(labels['train'].tolist())),'cosine':score(target[l:r],vec[l:r],tokenizer,set(labels['train'].tolist()))}
   (result_dir/f'depth{depth}_{name}_predictions.json').write_text(json.dumps({'target':target,'mlp':pred,'cosine':vec}))
   if name=='current':
    (result_dir/f'depth{depth}_mlp_prompt.txt').write_text(tokenizer.decode(pred[:4096]))
    (result_dir/f'depth{depth}_cosine_prompt.txt').write_text(tokenizer.decode(vec[:4096]))
  (result_dir/f'depth{depth}.json').write_text(json.dumps(result,indent=2));print('COMPLETE',depth,flush=True)
  del model,optimizer;torch.cuda.empty_cache()

if __name__=='__main__':main()
