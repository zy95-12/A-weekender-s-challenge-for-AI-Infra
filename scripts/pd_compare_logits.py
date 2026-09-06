"""Absolute-position logits/softmax comparison against a TP2 decode control."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def softmax(x):
    x=x.astype(np.float64);e=np.exp(x-x.max(axis=-1,keepdims=True));return e/e.sum(axis=-1,keepdims=True)


def main():
    result={'qualification':'E1/P2/D1 PD versus E1/cloudTP2 shared serving; decode TP differs, not bitwise equivalence acceptance', 'prompts':{}}
    for name,length in [('current4k',4096),('nonaligned257',257)]:
        pd=np.load(ROOT/'results/pd_validation'/f'{name}.npz')
        ref=np.load(ROOT/'results/pd_followup'/f'{name}_reference.npz')
        a,b=pd['logits'],ref['logits'];assert a.shape==b.shape==(8,151936)
        p,q=softmax(a),softmax(b);diff=np.abs(a.astype(np.float64)-b)
        rows=[]
        for i in range(8):
            rows.append(dict(logits_row=i,absolute_position=length-1+i,phase='prefill' if i==0 else 'decode',
                pd_top1=int(a[i].argmax()),reference_top1=int(b[i].argmax()),max_logit_error=float(diff[i].max()),
                max_probability_error=float(np.abs(p[i]-q[i]).max()),total_variation=float(np.abs(p[i]-q[i]).sum()/2)))
        result['prompts'][name]=dict(rows=rows,prefill_logits_exact=bool(np.array_equal(a[0],b[0])),
            top1_matches=int(np.sum(a.argmax(-1)==b.argmax(-1))),max_logit_error=float(diff.max()),
            mean_logit_error=float(diff.mean()),p99_logit_error=float(np.percentile(diff,99)),
            max_probability_error=float(np.abs(p-q).max()),mean_total_variation=float(np.abs(p-q).sum(axis=-1).mean()/2))
    (ROOT/'results/pd_followup/logits_comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
