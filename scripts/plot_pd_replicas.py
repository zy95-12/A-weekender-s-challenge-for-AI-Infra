"""Plot only complete audited C24/C40 windows."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/evidence/issue-6/pd-replicas'
rows=sorted(json.loads((OUT/'audit_summary.json').read_text()),key=lambda r:(r['concurrency'],r['prefill_replicas']))
x=np.arange(len(rows));labels=[f'{"TP2" if r["prefill_replicas"]==1 else "2 x TP1"}\nC{r["concurrency"]}' for r in rows]
fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
axes[0].bar(x,[r['completed_qps'] for r in rows],color=['#247eaa' if r['slo_pass'] else '#bd5943' for r in rows])
for i,r in enumerate(rows):axes[0].text(i,r['completed_qps']+.04,f'{r["completed_qps"]:.3f}',ha='center',fontsize=9)
axes[0].set(title='Completed QPS',ylabel='requests/s',ylim=(0,6))
for ax,key,title,limit in [(axes[1],'ttft','TTFT',3000),(axes[2],'tpot','Request-average TPOT',100)]:
    ax.bar(x-.18,[r[f'mean_{key}_ms'] for r in rows],.36,label='Mean',color='#247eaa')
    ax.bar(x+.18,[r[f'p99_{key}_ms'] for r in rows],.36,label='P99',color='#94bdd0')
    ax.axhline(limit,color='#bd5943',ls='--',label='SLO limit');ax.set(title=title,ylabel='ms');ax.legend(fontsize=8)
for ax in axes:ax.set_xticks(x,labels);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('4K prompt / 79 output tokens; same four GPUs; stage 1+2+3\n90s+ windows, profiling OFF; red QPS bar fails joint SLO',fontsize=11)
fig.savefig(OUT/'comparison.png',dpi=180)
