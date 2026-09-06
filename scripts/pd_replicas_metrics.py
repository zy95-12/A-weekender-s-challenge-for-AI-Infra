"""Per-role service coverage from deduplicated pipeline tasks, not GPU utilization."""
import json,statistics as st
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RAW=ROOT/'results/pd_replicas'
def union(intervals,lo,hi):
 end=lo;total=0
 for a,b in sorted(intervals):
  a=max(lo,a);b=min(hi,b)
  if b>a:total+=max(0,b-max(a,end));end=max(end,b)
 return total/(hi-lo)
results=[]
for folder in sorted(RAW.iterdir()):
 if not (folder/'summary.json').is_file():continue
 s=json.loads((folder/'summary.json').read_text());lo=s['measurement_start']*1e9;hi=s['measurement_end']*1e9
 batches={}
 for line in (folder/'split_trace.jsonl').open():
  r=json.loads(line);batches[r['batch_id']]=r
 allrows=list(batches.values());out={'variant':s['variant']}
 out['enterprise_command_coverage']=union([(r[a],r[b]) for r in allrows for a,b in [('front_start_ns','front_end_ns'),('back_start_ns','back_end_ns')]],lo,hi)
 out['prefill']=[]
 for peer in range(s['prefill_replicas']):
  rows=[r for r in allrows if r['phase']=='prefill' and r['prefill_replica']==peer]
  measured=[r for r in rows if lo<=r['front_start_ns']<hi]
  out['prefill'].append({'replica':peer,'chunks':len(measured),'chunks_per_s':len(measured)/s['duration_s'],
    'service_ms':st.mean((r['cloud_send_ns']-r['cloud_received_ns'])/1e6-r['cloud_queue_ms'] for r in measured),
    'service_coverage':union([(r['cloud_received_ns']+r['cloud_queue_ms']*1e6,r['cloud_send_ns']) for r in rows],lo,hi)})
 d=[r for r in allrows if r['phase']=='decode'];m=[r for r in d if lo<=r['front_start_ns']<hi]
 out['decode']={'batch_mean':st.mean(r['batch_size'] for r in m),'batches_per_s':len(m)/s['duration_s'],
 'service_coverage':union([(r['cloud_received_ns']+r['cloud_queue_ms']*1e6,r['cloud_send_ns']) for r in d],lo,hi)}
 results.append(out)
(RAW/'service_metrics.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
