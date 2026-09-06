import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]/'results/wan_calibration'
METRICS=['gpu_ready_rtt_ms','rpc_ms','http_transport_ms','upload_path_ms','download_path_ms','enterprise_d2h_ms','enterprise_pack_ms','enterprise_unpack_ms','enterprise_h2d_tp_ms','cloud_body_read_ms','cloud_unpack_ms','cloud_dispatch_ms','cloud_executor_ms','cloud_worker_ms','cloud_h2d_tp_ms','cloud_d2h_ms','cloud_pack_ms','cloud_ipc_residual_ms']

def main():
 allrows=[];points=[];fits={}
 for variant in ['baseline','stage1']:
  samples=json.loads((ROOT/variant/'samples.json').read_text());rows=[r for r in samples if not r['warmup']];allrows+=rows
  ps=[]
  for size in sorted({r['activation_bytes'] for r in rows}):
   selected=[r for r in rows if r['activation_bytes']==size]
   p={'variant':variant,'activation_bytes_each_direction':size,'activation_mib_each_direction':size/2**20,'samples':len(selected),'rows':selected[0]['rows']}
   for metric in METRICS:
    vs=[r[metric] for r in selected]
    for tag,fn in [('mean',np.mean),('p50',np.median),('p95',lambda v:np.percentile(v,95)),('std',np.std)]:p[metric+'_'+tag]=float(fn(vs))
   p['upload_wire_bytes_mean']=float(np.mean([r['upload_wire_bytes'] for r in selected]));p['download_wire_bytes_mean']=float(np.mean([r['download_wire_bytes'] for r in selected]))
   ps.append(p)
  points+=ps
  models={}
  for metric in ['gpu_ready_rtt_ms','http_transport_ms','upload_path_ms','download_path_ms']:
   # Fit first pass, evaluate second pass at the same measured sizes.
   training=[r for r in rows if r['pass_id']==0];testing=[r for r in rows if r['pass_id']==1]
   x=np.array([r['activation_bytes']/2**20 for r in training]);y=np.array([r[metric] for r in training]);b,a=np.polyfit(x,y,1)
   residual=np.array([r[metric]-(a+b*r['activation_bytes']/2**20) for r in testing])
   xf=np.array([p['activation_mib_each_direction'] for p in ps]);yf=np.array([p[metric+'_mean'] for p in ps]);bf,af=np.polyfit(xf,yf,1)
   models[metric]={'affine_formula':'latency_ms = intercept_ms + slope_ms_per_MiB * activation_MiB_each_direction','intercept_ms':float(af),'slope_ms_per_MiB':float(bf),'fit_mean_point_rmse_ms':float(np.sqrt(np.mean((yf-af-bf*xf)**2))),'pass0_affine_test_pass1_rmse_ms':float(np.sqrt(np.mean(residual**2))),'affine_recommended':False,'recommended':'Interpolate piecewise_linear_knots in linear byte space; inspect per-size variation and do not extrapolate.','piecewise_linear_knots':[[float(x),float(y)] for x,y in zip(xf,yf)]}
  fits[variant]=models
 with (ROOT/'curves.csv').open('w') as f:
  w=csv.DictWriter(f,list(points[0]));w.writeheader();w.writerows(points)
 with (ROOT/'samples.csv').open('w') as f:
  w=csv.DictWriter(f,list(allrows[0]));w.writeheader();w.writerows(allrows)
 (ROOT/'curves.json').write_text(json.dumps(points,indent=2));(ROOT/'fit_models.json').write_text(json.dumps({'scope':'C1, warm persistent HTTP, TP2+2, 10Gbps/10ms RTT. X is activation bytes EACH direction, two FP16 tensors. No model compute. No extrapolation beyond measured range. HTTP path includes stack overhead, not pure wire latency.','models':fits},indent=2))
 fig,axes=plt.subplots(1,2,figsize=(13,5))
 for variant,color in [('baseline','#2864b7'),('stage1','#d97706')]:
  ps=[p for p in points if p['variant']==variant];x=[p['activation_mib_each_direction'] for p in ps]
  for ax,metric,title in zip(axes,['gpu_ready_rtt_ms','http_transport_ms'],['GPU-ready round trip','HTTP upload + download path']):
   ax.plot(x,[p[metric+'_mean'] for p in ps],'-o',color=color,label=variant+' mean')
   ax.plot(x,[p[metric+'_p95'] for p in ps],'--',color=color,alpha=.7,label=variant+' P95')
   ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks([1/128,1/8,1,8,32,128],['8 KiB','128 KiB','1 MiB','8 MiB','32 MiB','128 MiB']);ax.set_xlabel('Activation bytes each direction (hidden + residual)');ax.set_ylabel('Latency (ms)');ax.set_title(title);ax.grid(alpha=.2);ax.legend()
 fig.suptitle('WAN size sweep | TP2+2 | 10 Gbps / 10 ms RTT | C1 | no model compute')
 fig.tight_layout()
 for ext in ['png','svg','pdf']:fig.savefig(ROOT/('wan_latency.'+ext),dpi=180,bbox_inches='tight')
 print(json.dumps([{k:p[k] for k in ['variant','activation_mib_each_direction','gpu_ready_rtt_ms_mean','http_transport_ms_mean']} for p in points],indent=2))

if __name__=='__main__':main()
