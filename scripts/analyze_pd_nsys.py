import sqlite3,json,bisect,collections,csv
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'results/pd_window/profile-p2-ch1024'
s=json.loads((root/'summary.json').read_text());ids=set()
for line in (root/'split_trace.jsonl').open():
 r=json.loads(line)
 if r['phase']=='prefill' and s['measurement_start']*1e9<=r['front_start_ns']<s['measurement_end']*1e9:ids.add(r['batch_id'])
c=sqlite3.connect(f'file:{root}/cloud_prefill.sqlite?mode=ro',uri=True)
strings=dict(c.execute('select id,value from StringIds'));ranges=collections.defaultdict(list);thread_ranges=collections.defaultdict(list)
for start,end,tid,text,textid in c.execute('select start,end,globalTid,text,textId from NVTX_EVENTS where end is not null'):
 label=text or strings.get(textid,'')
 if label.startswith('batch=') and label.split()[0][6:] in ids:
  value=(start,end,label.split()[0][6:]);ranges[(tid>>24)&0xffffff].append(value);thread_ranges[tid].append(value)
assert len(ranges)==2
for rs in ranges.values():
 assert {r[2] for r in rs}==ids,'Incomplete rank capture'
 rs.sort()
for rs in thread_ranges.values():rs.sort()
starts={tid:[r[0] for r in rs] for tid,rs in thread_ranges.items()}
launches={}
# Associate by the launching CPU thread and CUDA correlation ID, excluding
# independent KV-transfer stream work that merely overlaps the forward in time.
for a,b,tid,corr in c.execute('select start,end,globalTid,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME'):
 rs=thread_ranges.get(tid,[]);i=bisect.bisect_right(starts.get(tid,[]),a)-1
 if i>=0 and b<=rs[i][1]:launches[((tid>>24)&0xffffff,corr)]=rs[i][2]
agg=collections.defaultdict(lambda:[0,0]);copies=collections.defaultdict(lambda:[0,0,0]);events=collections.defaultdict(list)
for a,b,pid,name,corr in c.execute('select start,end,globalPid,demangledName,correlationId from CUPTI_ACTIVITY_KIND_KERNEL'):
 pid=(pid>>24)&0xffffff
 bid=launches.get((pid,corr))
 if bid:
  name=strings[name];agg[pid,name][0]+=1;agg[pid,name][1]+=(b-a);events[pid].append((a,b))
for a,b,pid,kind,n,corr in c.execute('select start,end,globalPid,copyKind,bytes,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY'):
 pid=(pid>>24)&0xffffff
 if launches.get((pid,corr)):v=copies[pid,kind];v[0]+=1;v[1]+=b-a;v[2]+=n
result=[]
for pid,rs in ranges.items():
 total=len(rs);cats=collections.defaultdict(float)
 for (p,name),(count,dur) in agg.items():
  if p!=pid:continue
  lower=name.lower();kind='NCCL' if 'nccl' in lower else 'GEMM' if any(x in lower for x in ['gemm','xmma','cutlass']) else 'attention' if 'flash' in lower else 'other'
  cats[kind]+=dur/1e6/total
 end=0;union=0
 for a,b in sorted(events[pid]):union+=max(0,b-max(a,end));end=max(end,b)
 result.append(dict(pid=pid,batches=total,kernel_sum_mean_ms=dict(cats),kernel_union_mean_ms=union/1e6/total,
  copies=[dict(kind=k,count=v[0],mean_ms_per_batch=v[1]/1e6/total,bytes_per_batch=v[2]/total) for (p,k),v in copies.items() if p==pid]))
(root/'nsys_breakdown.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
with (root/'kernel_summary.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['pid','kernel','calls','total_ms','mean_ms_per_batch'])
 for (pid,name),(count,dur) in sorted(agg.items(),key=lambda kv:-kv[1][1]):w.writerow([pid,name,count,dur/1e6,dur/1e6/len(ranges[pid])])
print('copy kinds',c.execute('select * from ENUM_CUDA_MEMCPY_OPER').fetchall())
