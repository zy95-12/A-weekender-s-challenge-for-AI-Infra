import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parents[1]
rows=json.loads((root/'results/pd_window/performance.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
x=np.arange(len(rows));labels=[f'P{r["prefill_window"]} / {r["chunk"]}' for r in rows]
axes[0].bar(x,[r['completed_qps'] for r in rows],color=['#247eaa' if r['slo_pass'] else '#bd5943' for r in rows])
for i,r in enumerate(rows):axes[0].text(i,r['completed_qps']+.025,f'{r["completed_qps"]:.3f}',ha='center',fontsize=9)
axes[0].set(title='Completed QPS',ylabel='requests/s');axes[0].set_ylim(0,max(r['completed_qps'] for r in rows)*1.17)
for ax,key,title,limit in [(axes[1],'ttft','TTFT',3000),(axes[2],'tpot','Request-average TPOT',100)]:
 ax.bar(x-.18,[r[f'mean_{key}_ms'] for r in rows],.36,label='Mean',color='#247eaa')
 ax.bar(x+.18,[r[f'p99_{key}_ms'] for r in rows],.36,label='P99',color='#94bdd0')
 ax.axhline(limit,color='#bd5943',ls='--',label='SLO limit');ax.set(title=title,ylabel='ms');ax.legend(fontsize=8)
for ax in axes:ax.set_xticks(x,labels,rotation=20);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('C24, 4K prompt / 79 output tokens, E1/P2/D1; D window = 2\n90s+ trials; profiling OFF; red QPS bar = joint SLO failure',fontsize=11)
out=root/'docs/evidence/issue-6/pd-window';out.mkdir(exist_ok=True,parents=True)
fig.savefig(out/'comparison.png',dpi=180)
