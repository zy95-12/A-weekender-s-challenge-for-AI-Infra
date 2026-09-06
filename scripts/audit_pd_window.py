import json,statistics as st,hashlib,csv
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'results/pd_window'
out=[]
for snap in root.glob('*_source_hashes.json'):
 for name,h in json.loads(snap.read_text()).items():
  assert hashlib.sha256((root.parents[1]/name).read_bytes()).hexdigest()==h,(snap,name)
for folder in sorted(root.iterdir()):
 if not folder.is_dir() or not (folder/'summary.json').exists() or not (folder/'split_trace.jsonl').exists():continue
 s=json.loads((folder/'summary.json').read_text());req=json.loads((folder/'point/requests.json').read_text())
 traces={};batches={}
 for line in (folder/'split_trace.jsonl').open():
  r=json.loads(line);traces.setdefault(r['client_request_id'],[]).append(r);batches[r['batch_id']]=r
 refs=json.loads((root/'reference.json').read_text())
 kv={}
 for line in (folder/'pd_kv_trace.jsonl').open():
  k=json.loads(line);kv[k['request_id']]=k
 waits=[];spans=[]
 for r in req:
  assert r['error'] is None and r['tokens']==79 and r['text']==refs['text']
  steps=traces[r['request_id']];expected=list(range(0,4096,s['chunk']))+list(range(4096,4174))
  assert [t['position_start'] for t in steps]==expected
  assert [t['query_len'] for t in steps]==[s['chunk']]*(4096//s['chunk'])+[1]*78
  assert sum(t['emits_token'] for t in steps)==79
  k=kv[steps[0]['request_id']]
  assert steps[4096//s['chunk']]['front_start_ns']>=k['kv_ready_ns']
  assert [(v['range_start'],v['range_end']) for v in k['chunks']]==[(0,4096)]
  assert k['source_released_ns']>=k['kv_ready_ns']
  assert r['slo_pass']==(r['ttft_ms']<=3000 and r['tpot_ms']<=100)
  assert abs(st.mean(r['itls_ms'])-r['tpot_ms'])<1e-8
  if s['measurement_start']<=r['end']<s['measurement_end']:
   waits.append(steps[0]['front_start_ns']/1e6-r['start']*1000)
   spans.append((steps[4096//s['chunk']-1]['back_end_ns']-steps[0]['front_start_ns'])/1e6)
 samples=json.loads((folder/'point/health_samples.json').read_text())
 assert not any('error' in h for h in samples)
 for key in ['ttft_ms','tpot_ms','e2e_ms','first_second_ms','max_itl_ms']:
  measured=[r[key] for r in req if s['measurement_start']<=r['end']<s['measurement_end']]
  assert abs(st.mean(measured)-s['mean_'+key])<1e-7
 h=json.loads((folder/'point/final_health.json').read_text());assert all(h[k]==0 for k in ['active','waiting','kv_used_blocks','pd_reserving','pd_releasing'])
 cohort=[r for r in req if s['measurement_start']<=r['end']<s['measurement_end']]
 starts=[r for r in req if s['measurement_start']<=r['start']<s['measurement_end']]
 assert len(cohort)==s['requests']
 assert abs(len(cohort)/s['duration_s']-s['completed_qps'])<1e-10
 assert s['slo_attainment']==st.mean(r['slo_pass'] for r in cohort)
 assert s['start_cohort_slo_attainment']==st.mean(r['slo_pass'] for r in starts)
 lo=s['measurement_start']*1e9;hi=s['measurement_end']*1e9
 rows=[r for r in batches.values() if lo<=r['front_start_ns']<hi and r['phase']=='prefill']
 cloud=st.mean((r['cloud_send_ns']-r['cloud_received_ns'])/1e6-r['cloud_queue_ms'] for r in rows)
 s.update(all_requests_audited=len(req),absolute_positions_valid=True,drained=True,mean_first_front_wait_ms=st.mean(waits),mean_prefill_span_ms=st.mean(spans),mean_cloud_service_ms=cloud,prefill_chunks_per_s=len(rows)/s['duration_s'])
 for key in ['cloud_middle_prefill_ms','cloud_receive_ms','cloud_d2h_ms']:
  if rows and key in rows[0]:s['mean_'+key]=st.mean(r[key] for r in rows)
 out.append(s)
(root/'audit_summary.json').write_text(json.dumps(out,indent=2))
keys=['variant','profile_only','prefill_window','decode_window','chunk','duration_s','completed_qps','mean_ttft_ms','p99_ttft_ms','mean_tpot_ms','p99_tpot_ms','slo_attainment','start_cohort_slo_attainment','slo_pass','mean_first_front_wait_ms','mean_prefill_span_ms','mean_cloud_service_ms','prefill_chunks_per_s','all_requests_audited']
with (root/'comparison.csv').open('w') as f:
 w=csv.DictWriter(f,keys,extrasaction='ignore');w.writeheader();w.writerows(out)
print(json.dumps(out,indent=2))
