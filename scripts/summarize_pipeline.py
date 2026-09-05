"""Offline representative statistics and Nsight proof on ONE enterprise GPU.

Kernel/transport interval unions avoid double-counting TP ranks or concurrent
messages. Transport ranges are socket I/O, not pure propagation delay.
"""
import argparse,json,sqlite3
from pathlib import Path
import numpy as np


def union(intervals):
 result=[]
 for a,b in sorted(intervals):
  if b<=a:continue
  if result and a<=result[-1][1]:result[-1]=(result[-1][0],max(b,result[-1][1]))
  else:result.append((a,b))
 return result


def intersections(a,b):
 a,b=union(a),union(b);out=[];i=j=0
 while i<len(a) and j<len(b):
  lo,hi=max(a[i][0],b[j][0]),min(a[i][1],b[j][1])
  if lo<hi:out.append((lo,hi))
  if a[i][1]<b[j][1]:i+=1
  else:j+=1
 return out


def measure(intervals):return sum(b-a for a,b in union(intervals))/1e6


def profile(sqlite,trace,cloud_sqlite=None):
 ids={r['batch_id'] for l in Path(trace).read_text().splitlines() if (r:=json.loads(l)).get('client_request_id','').startswith('pipeline-profile-')}
 c=sqlite3.connect(sqlite);strings=dict(c.execute('select id,value from StringIds'))
 nvtx=[(a,b,text or strings.get(tid,''),thread) for a,b,text,tid,thread in c.execute('select start,end,text,textId,globalTid from NVTX_EVENTS where end is not null')]
 transfers=[(a,b,name.split('batch=')[1]) for a,b,name,tid in nvtx
            if name.startswith(('pipeline_upload ','pipeline_download ')) and name.split('batch=')[1] in ids]
 outer=[(a,b,name.split('batch=')[1].split()[0],tid) for a,b,name,tid in nvtx if name.startswith('batch=')]
 model=[]
 for a,b,name,tid in nvtx:
  if name.startswith(('enterprise_front_','enterprise_back_')):
   owner=next((bid for x,y,bid,t in outer if t==tid and x<=a and b<=y and bid in ids),None)
   if owner:model.append((a,b,tid,owner))
 runtime={(tid//16777216*16777216,corr):(a,tid) for a,tid,corr in c.execute('select start,globalTid,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME')}
 kernels=[]
 for a,b,pid,corr,name in c.execute('select start,end,globalPid,correlationId,demangledName from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=0'):
  name=strings[name]
  if 'nccl' in name.lower():continue
  launch=runtime.get((pid,corr))
  if launch is None:continue
  start,tid=launch
  owner=next((bid for x,y,t,bid in model if t==tid and x<=start<=y),None)
  if owner:kernels.append((a,b,owner,name))
 overlap=[];examples=[]
 for a,b,bid,name in kernels:
  for x,y,other in transfers:
   if bid!=other and max(a,x)<min(b,y):
    overlap.append((max(a,x),min(b,y)))
    if len(examples)<8:examples.append({'gpu_kernel':name,'compute_batch':bid,'transport_batch':other,
       'kernel_start_ns':a,'kernel_end_ns':b,'transport_start_ns':x,'transport_end_ns':y,
       'overlap_ms':(min(b,y)-max(a,x))/1e6})
 gpu=[(a,b) for a,b,_,_ in kernels];io=[(a,b) for a,b,_ in transfers]
 assert overlap,'No non-NCCL model GPU kernel overlaps another batch socket transfer'
 result = {'result':'PASS','classification':'independent Nsight capture, not ordinary TTFT/TPOT',
         'gpu_scope':'enterprise device 0 only; non-NCCL kernels launched inside front/back NVTX ranges',
         'transport_scope':'upload body / download body socket I/O; excludes headers/cloud wait',
         'profile_batches':len(ids),'gpu_kernel_count':len(kernels),'transfer_ranges':len(transfers),
         'gpu_compute_union_ms':measure(gpu),'transport_union_ms':measure(io),
         'cross_batch_compute_transport_overlap_union_ms':measure(overlap),
         'transport_without_model_compute_ms':measure(io)-measure(intersections(gpu,io)),
         'examples':examples}
 if cloud_sqlite:
  cloud=sqlite3.connect(cloud_sqlite)
  offset=cloud.execute('select * from TARGET_INFO_SESSION_START_TIME').fetchone()[0]-c.execute('select * from TARGET_INFO_SESSION_START_TIME').fetchone()[0]
  names=dict(cloud.execute('select id,value from StringIds'))
  ranges=[(a,b,text or names.get(textid,''),tid) for a,b,text,textid,tid in cloud.execute('select start,end,text,textId,globalTid from NVTX_EVENTS where end is not null')]
  parents=[(a,b,text.split('batch=')[1].split()[0],tid) for a,b,text,tid in ranges if text.startswith('batch=')]
  middle=[]
  for a,b,text,tid in ranges:
   if text.startswith('cloud_middle_'):
    owner=next((bid for x,y,bid,t in parents if t==tid and x<=a and b<=y and bid in ids),None)
    if owner:middle.append((a,b,tid,owner))
  runtime={(tid//16777216*16777216,corr):(a,tid) for a,tid,corr in cloud.execute('select start,globalTid,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME')}
  cloudkernels=[]
  for a,b,pid,corr,name in cloud.execute('select start,end,globalPid,correlationId,demangledName from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=0'):
   if 'nccl' in names[name].lower() or (pid,corr) not in runtime:continue
   start,tid=runtime[(pid,corr)]
   owner=next((bid for x,y,t,bid in middle if t==tid and x<=start<=y),None)
   if owner:cloudkernels.append((a+offset,b+offset,owner))
  cloud_overlap=[(max(a,x),min(b,y)) for a,b,bid in cloudkernels for x,y,other in transfers
                 if bid!=other and max(a,x)<min(b,y)]
  result['cloud_gpu0_cross_batch_overlap_union_ms']=measure(cloud_overlap)
  result['either_side_cross_batch_overlap_union_ms']=measure(overlap+cloud_overlap)
  result['cloud_clock_alignment']='same-host Nsight session epoch start + activity timestamps; one GPU per side'
  result['cloud_session_offset_ns']=offset
  # Aligning reports must preserve upload-before-middle causality for every batch.
  uploads={name.split('batch=')[1]:b for a,b,name,tid in nvtx if name.startswith('pipeline_upload ')}
  first={bid:min(a for a,b,owner in cloudkernels if owner==bid) for bid in ids}
  result['aligned_upload_before_cloud_kernel']=all(uploads[bid]<=first[bid] for bid in ids)
  assert result['aligned_upload_before_cloud_kernel']
 return result


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--sqlite');p.add_argument('--trace');p.add_argument('--cloud-sqlite');args=p.parse_args()
 root=Path(args.root);report={'scope':'TP2+2 8192/256 C1/C4, three repeats of four requests; descriptive tails','variants':{}}
 for variant in ['window0','window2','window2_shm']:
  folder=root/variant;cells={}
  for concurrency in [1,4]:
   summaries=sorted(folder.glob(f'isl8192_osl256_c{concurrency}_r*/summary.json'))
   assert len(summaries)==3
   points=[json.loads(s.read_text())['points'][0] for s in summaries]
   assert all(x['success_rate']==1 and x['fixed_output_length_valid'] for x in points)
   rows=[json.loads(l) for s in summaries for l in (s.parent/'qps_inf/requests.jsonl').read_text().splitlines()]
   cell={'repeats':3,'requests':len(rows)}
   for key in ['mean_ttft_ms','mean_tpot_ms','achieved_qps']:
    values=[x[key] for x in points];cell[key]={'mean':float(np.mean(values)),'repeat_std':float(np.std(values,ddof=1))}
   for key in ['ttft_ms','tpot_ms']:
    cell[key+'_pooled']={f'p{q}':float(np.percentile([r[key] for r in rows],q)) for q in [50,95,99]}
   cells[str(concurrency)]=cell
  mixed=json.loads((folder/'mixed/summary.json').read_text());assert mixed['result']=='PASS'
  cells['mixed']={'repeats':3,'decode_max_gap_ms':[r['requests'][0]['event_gap_max_ms'] for r in mixed['repeats']],
                  'long_ttft_ms':[r['requests'][1]['ttft_ms'] for r in mixed['repeats']]}
  report['variants'][variant]=cells
 if args.sqlite:report['overlap']=profile(args.sqlite,args.trace,args.cloud_sqlite)
 (root/'representative_summary.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2))

if __name__=='__main__':main()
