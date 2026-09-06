import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
DEPTHS=[4,9,18,27,36]


def main():
 rows=[];summary=[]
 for depth in DEPTHS:
  records=[]
  for seed in [42,43]:
   path=ROOT/'results'/('' if seed==42 else 'seed43')/f'depth{depth}.json'
   r=json.loads(path.read_text());records.append(r)
   for dataset,attacks in r['evaluations'].items():
    for attack,scores in attacks.items():rows.append({'depth':depth,'seed':seed,'dataset':dataset,'attack':attack,**scores})
  out={'depth':depth}
  for dataset in ['current_prompt','current_output_inputs','public_test']:
   for attack in ['mlp','cosine']:
    metrics=records[0]['evaluations'][dataset][attack]
    for metric in ['token_accuracy','macro_token_type_accuracy','training_seen_token_accuracy']:
     v=[r['evaluations'][dataset][attack][metric] for r in records]
     out[f'{dataset}_{attack}_{metric}_mean']=float(np.mean(v));out[f'{dataset}_{attack}_{metric}_min']=min(v);out[f'{dataset}_{attack}_{metric}_max']=max(v)
  summary.append(out)
 with (ROOT/'results/recovery_by_seed.csv').open('w') as f:
  w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
 with (ROOT/'results/recovery_summary.csv').open('w') as f:
  w=csv.DictWriter(f,list(summary[0]));w.writeheader();w.writerows(summary)
 (ROOT/'results/recovery_summary.json').write_text(json.dumps(summary,indent=2))
 fig,axes=plt.subplots(1,2,figsize=(12,4.7))
 for ax,dataset,title in zip(axes,['current_prompt','public_test'],['Current 4K prompt (92 distinct token types)','Independent public test (8192 tokens)']):
  for attack,color in [('cosine','#2864b7'),('mlp','#d97706')]:
   ys=[100*r[f'{dataset}_{attack}_token_accuracy_mean'] for r in summary]
   low=[100*r[f'{dataset}_{attack}_token_accuracy_min'] for r in summary];high=[100*r[f'{dataset}_{attack}_token_accuracy_max'] for r in summary]
   ax.plot(DEPTHS,ys,'-o',color=color,label=attack);ax.fill_between(DEPTHS,low,high,color=color,alpha=.2)
  ax.scatter([0],[100],marker='*',s=110,color='#2864b7',label='Embedding-only retrieval')
  ax.set_ylim(0,105);ax.set_xticks([0,*DEPTHS]);ax.set_xlabel('Decoder layers executed locally');ax.set_ylabel('Position-aligned token recovery (%)');ax.set_title(title);ax.grid(alpha=.2);ax.legend()
 fig.suptitle('Qwen2.5-3B-Instruct FP16 | equal-budget MLP | matched 4K training context')
 fig.tight_layout()
 for ext in ['png','svg','pdf']:fig.savefig(ROOT/'results'/f'recovery_depth.{ext}',dpi=180,bbox_inches='tight')
 print(json.dumps([{k:r[k] for k in ['depth','current_prompt_mlp_token_accuracy_mean','public_test_mlp_token_accuracy_mean','current_prompt_cosine_token_accuracy_mean']} for r in summary],indent=2))

if __name__=='__main__':main()
