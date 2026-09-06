"""Publication files for TTFT-QPS, TPOT-QPS and the concurrency knee."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/evidence/issue-6/pd-slo'
COLORS={'baseline':'#2864b7','optimized':'#db7420'}
LABELS={'baseline':'Baseline (Stages 1/2/3 OFF): E2 + shared cloud TP2','optimized':'Optimized: E1/P2/D1 + Stages 1/2/3 + control channel'}


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.input
    rows=list(csv.DictReader((OUT/'curve.csv').open()))
    all_rows=list(csv.DictReader((OUT/'all_points.csv').open()))
    selection=json.loads((OUT/'selection.json').read_text())
    names=[name for name in COLORS if name in selection]
    data={name:sorted([r for r in rows if r['system']==name],key=lambda r:int(r['concurrency'])) for name in names}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    handles=[Line2D([0],[0],color=c,label=LABELS[n]) for n,c in COLORS.items() if n in names]+[
        Line2D([0],[0],color='#555',marker='o',linestyle='',label='Joint SLO passes (both cohorts)'),
        Line2D([0],[0],color='#555',marker='x',linestyle='',label='Joint SLO fails'),
        Line2D([0],[0],color='#555',marker='*',markersize=12,linestyle='',label='Selected confirmed SLO point'),
        Line2D([0],[0],color='#555',marker='o',alpha=.3,linestyle='',label='Earlier trials (faint markers')]
    best={n:selection[n]['highest_confirmed_slo_qps'] for n in names}
    for metric,title,limit,scale,unit in [('ttft','TTFT',3,1000,'s'),('tpot','TPOT',100,1,'ms')]:
        fig,axes=plt.subplots(1,2,figsize=(13,5.6),layout='constrained')
        for ax,stat in zip(axes,['mean','p99']):
            for name,points in data.items():
                color=COLORS[name];x=[float(p['completed_qps']) for p in points]
                y=[float(p[f'{stat}_{metric}_ms'])/scale for p in points]
                ax.plot(x,y,color=color,lw=1.5,alpha=.85)
                chosen={p['variant'] for p in points}
                for trial in all_rows:
                    if trial['system']==name and trial['variant'] not in chosen:
                        ax.scatter(float(trial['completed_qps']),float(trial[f'{stat}_{metric}_ms'])/scale,color=color,
                            marker='o' if trial['slo_pass']=='True' else 'x',s=24,alpha=.3,zorder=2)
                crowded=0
                for i,(p,xx,yy) in enumerate(zip(points,x,y)):
                    passed=p['slo_pass']=='True';c=int(p['concurrency'])
                    ax.scatter(xx,yy,color=color,marker='o' if passed else 'x',s=34,zorder=4)
                    if p['stage']=='coarse' or c in (best[name]['concurrency'],selection[name]['largest_confirmed_passing_concurrency']+1):
                        offset=(5,8 if i%2==0 else -13);arrow=None
                        if metric=='tpot' and name=='optimized' and c>=16:
                            offset=[(-48,28),(-50,-20),(16,-32),(20,16),(18,42)][crowded%5];crowded+=1
                            arrow=dict(arrowstyle='-',color=color,lw=.5)
                        ax.annotate(f'C{c}',(xx,yy),xytext=offset,textcoords='offset points',fontsize=8,color=color,arrowprops=arrow)
                b=best[name]
                ax.scatter(b['completed_qps'],b[f'{stat}_{metric}_ms']/scale,marker='*',s=180,color=color,edgecolor='black',lw=.6,zorder=5)
            ax.axhline(limit,color='#b33333',ls='--',lw=1,label='Per-request limit')
            ax.text(.5,limit,f' {limit:g} {unit} request limit',transform=ax.get_yaxis_transform(),ha='center',va='bottom',color='#b33333',fontsize=8)
            ax.set(xlabel='Completed requests / second (QPS)',ylabel=f'{stat.upper() if stat=="p99" else "Mean"} {title} ({unit})',ylim=(0,None),xlim=(0,max(float(p['completed_qps']) for p in rows)*1.18))
            ax.grid(alpha=.2)
        fig.suptitle(f'{title} vs QPS | Qwen2.5-3B | 4096 / 79 tokens | 4 x A10 | WAN: 10 Gbps, 5 ms one-way\nSLO: >=99% of requests meet TTFT <=3 s AND average TPOT <=100 ms',fontsize=12)
        fig.legend(handles=handles,loc='outside lower center',ncol=2,fontsize=9)
        fig.savefig(OUT/f'{metric}_qps.png');fig.savefig(OUT/f'{metric}_qps.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5.4),layout='constrained')
    for name,points in data.items():
        color=COLORS[name]
        chosen={p['variant'] for p in points}
        for trial in all_rows:
            if trial['system']==name and trial['variant'] not in chosen:
                ax.scatter(int(trial['concurrency']),float(trial['completed_qps']),color=color,
                    marker='o' if trial['slo_pass']=='True' else 'x',s=24,alpha=.3)
        ax.plot([int(p['concurrency']) for p in points],[float(p['completed_qps']) for p in points],color=color,label=LABELS[name])
        for p in points:ax.scatter(int(p['concurrency']),float(p['completed_qps']),marker='o' if p['slo_pass']=='True' else 'x',color=color,s=34)
        b=best[name];ax.scatter(b['concurrency'],b['completed_qps'],marker='*',s=180,color=color,edgecolor='black',lw=.6)
        ax.annotate(f"C{b['concurrency']}: {b['completed_qps']:.3f} QPS",(b['concurrency'],b['completed_qps']),xytext=(6,10),textcoords='offset points',color=color)
        if name=='optimized':
            knee=min((p for p in points if p['slo_pass']=='True' and float(p['completed_qps'])>=.95*b['completed_qps']),key=lambda p:int(p['concurrency']))
            if int(knee['concurrency'])!=b['concurrency']:
                ax.annotate(f"C{knee['concurrency']}: >=95% of best SLO QPS",(int(knee['concurrency']),float(knee['completed_qps'])),
                    xytext=(-80,-35),textcoords='offset points',color=color,arrowprops=dict(arrowstyle='-',color=color,lw=.7))
    ax.set(xlabel='Closed-loop client concurrency',ylabel='Completed QPS',title='Throughput knee and fixed-SLO boundary',xlim=(0,None),ylim=(0,None));ax.grid(alpha=.2)
    fig.legend(handles=handles,loc='outside lower center',ncol=2,fontsize=9)
    fig.savefig(OUT/'concurrency_qps.png');fig.savefig(OUT/'concurrency_qps.svg')
    for path in OUT.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    print('Saved PNG and SVG charts to',OUT)


if __name__=='__main__':main()
