"""Audit completed windows across preserved transport revisions; never trim failures."""
import json,hashlib,csv,zipfile,statistics as st
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/evidence/issue-6/pd-replicas'
OUT.mkdir(parents=True,exist_ok=True);results=[]
for raw in [ROOT/'results/pd_replicas_v2',ROOT/'results/pd_replicas']:
    archive=zipfile.ZipFile(raw/'source_snapshot.zip') if (raw/'source_snapshot.zip').exists() else None
    for snap in raw.glob('*_source_hashes.json'):
        for name,h in json.loads(snap.read_text()).items():
            data=archive.read(name) if archive else (ROOT/name).read_bytes()
            assert hashlib.sha256(data).hexdigest()==h,(snap,name)
    if archive:archive.close()
    ref=json.loads((raw/'reference.json').read_text())
    for folder in sorted(raw.iterdir()):
        if not (folder/'summary.json').is_file():continue
        s=json.loads((folder/'summary.json').read_text());traces={};kv={}
        for line in (folder/'split_trace.jsonl').open():
            t=json.loads(line);traces.setdefault(t['client_request_id'],[]).append(t)
        for path in folder.glob('pd_kv_trace*.jsonl'):
            for line in path.open():
                k=json.loads(line);assert k['request_id'] not in kv;kv[k['request_id']]=k
        rows=json.loads((folder/'point/requests.json').read_text());counts={p:0 for p in range(s['prefill_replicas'])}
        for r in rows:
            assert r['error'] is None and r['tokens']==79 and r['text']==ref['text']
            ts=traces[r['request_id']];peer=ts[0]['prefill_replica'];counts[peer]+=1
            assert {t['prefill_replica'] for t in ts}=={peer}
            assert [t['position_start'] for t in ts]==[0,2048]+list(range(4096,4174))
            assert [t['query_len'] for t in ts]==[2048,2048]+[1]*78
            k=kv[ts[0]['request_id']];assert k['prefill_replica']==peer
            assert ts[2]['front_start_ns']>=k['kv_ready_ns']
            assert k['source_released_ns']>=k['kv_ready_ns']
            assert [(c['range_start'],c['range_end']) for c in k['chunks']]==[(0,4096)]
            assert len(r['itls_ms'])==78 and abs(st.mean(r['itls_ms'])-r['tpot_ms'])<1e-7
            assert r['slo_pass']==(r['ttft_ms']<=3000 and r['tpot_ms']<=100)
        cohort=[r for r in rows if s['measurement_start']<=r['end']<s['measurement_end']]
        starts=[r for r in rows if s['measurement_start']<=r['start']<s['measurement_end']]
        assert len(cohort)==s['requests'] and abs(len(cohort)/s['duration_s']-s['completed_qps'])<1e-9
        assert s['slo_attainment']==st.mean(r['slo_pass'] for r in cohort)
        assert s['start_cohort_slo_attainment']==st.mean(r['slo_pass'] for r in starts)
        assert s['slo_pass']==(min(s['slo_attainment'],s['start_cohort_slo_attainment'])>=.99)
        for key in ['ttft_ms','tpot_ms','e2e_ms','first_second_ms','max_itl_ms']:
            values=[r[key] for r in cohort]
            assert abs(st.mean(values)-s['mean_'+key])<1e-7
            assert abs(float(np.percentile(values,99))-s['p99_'+key])<1e-7
        h=json.loads((folder/'point/final_health.json').read_text())
        assert all(h[k]==0 for k in ['active','waiting','kv_used_blocks','pd_reserving','pd_releasing'])
        health=json.loads((folder/'point/health_samples.json').read_text());assert not any('error' in h for h in health)
        launch=json.loads((folder/'launch.json').read_text())
        assert not launch['profile'] and not launch['phase_profile'] and not launch['pd_verify_kv']
        assert launch['prefill_replicas']==s['prefill_replicas'] and launch['prefill_tp']==s['prefill_tp']
        assert launch['decode_tp']==1 and launch['pd_prefill_window']==3 and launch['pipeline_window']==2
        assert launch['max_active']==96 and launch['kv_blocks']==32768 and launch['prefill_chunk_size']==2048
        s.update(raw_directory=str(folder),all_requests_audited=len(rows),routes=counts,source_hashes_match=True,correctness_passed=True)
        results.append(s)
(OUT/'audit_summary.json').write_text(json.dumps(results,indent=2))
keys=['variant','concurrency','prefill_replicas','prefill_tp','requests','duration_s','completed_qps',
      'mean_ttft_ms','p99_ttft_ms','mean_tpot_ms','p99_tpot_ms','slo_attainment','start_cohort_slo_attainment','slo_pass',
      'p99_token_itl_ms','mean_first_second_ms','p99_first_second_ms','all_requests_audited','raw_directory']
with (OUT/'comparison.csv').open('w') as f:
    w=csv.DictWriter(f,keys,extrasaction='ignore');w.writeheader();w.writerows(results)
print(json.dumps({'windows':len(results),'all_requests_audited':sum(s['all_requests_audited'] for s in results),'passed':True}))
