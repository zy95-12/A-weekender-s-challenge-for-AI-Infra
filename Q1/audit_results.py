"""Recompute saved recovery metrics and data-separation checks before publication."""
import json,hashlib
from pathlib import Path
import numpy as np
from metrics import score


def main():
 root=Path(__file__).resolve().parent
 data=json.loads((root/'cache/public_tokens.json').read_text())
 train=sum(data['train'],[]);prompt=json.loads((root/'data/current_prompt.json').read_text())['prompt_ids']
 overlaps=len(set(map(tuple,data['train']))&set(map(tuple,data['validation'])))
 ngrams={tuple(train[i:i+16]) for i in range(len(train)-15)}
 audit={'train_validation_exact_4096token_window_overlap':overlaps,'target_16grams_present_in_training':sum(tuple(prompt[i:i+16]) in ngrams for i in range(len(prompt)-15)),
        'target_positions_with_token_seen_in_training':sum(t in set(train) for t in prompt),'target_distinct_tokens_seen_in_training':len(set(prompt)&set(train)),
        'target_total_positions':4096,'target_distinct_tokens':len(set(prompt))}
 assert overlaps==audit['target_16grams_present_in_training']==0
 for seed in [42,43]:
  folder=root/'results'/('' if seed==42 else 'seed43')
  for depth in [4,9,18,27,36]:
   result=json.loads((folder/f'depth{depth}.json').read_text())
   for name in ['test','current']:
    pred=json.loads((folder/f'depth{depth}_{name}_predictions.json').read_text())
    assert pred['target']==np.load(root/'cache'/name/'labels.npy').tolist()
    slices=[('public_test',0,len(pred['target']))] if name=='test' else [('current_prompt',0,4096),('current_output_inputs',4096,4174)]
    for tag,l,r in slices:
     for method in ['cosine','mlp']:
      computed=score(pred['target'][l:r],pred[method][l:r],seen_token_ids=set(train))
      for key,value in computed.items():assert result['evaluations'][tag][method][key]==value,(depth,seed,tag,key)
 audit['recomputed_depth_seed_results']=10
 (root/'results/leakage_audit.json').write_text(json.dumps(audit,indent=2))
 hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'checkpoints').glob('*_seed*.pt'))}
 (root/'results/checkpoint_hashes.json').write_text(json.dumps(hashes,indent=2))
 print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
