"""Recompute the SLO sweep, verify traces, and export report/plot input CSVs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'results/pd_slo_sweep'
OUT=ROOT/'docs/evidence/issue-6/pd-slo'


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--systems',nargs='+',choices=['baseline','optimized'],default=['baseline','optimized'])
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.out
    OUT.mkdir(parents=True,exist_ok=True)
    exported=[];curves=[];audits=[];selections={}
    reference=json.loads((RAW/'reference.json').read_text())['text']
    for variant in args.systems:
        folder=RAW/variant
        source=json.loads((folder/'source_hashes.json').read_text())
        assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in source.items())
        points=json.loads((folder/'summaries.json').read_text())
        selection=json.loads((folder/'selection.json').read_text());selections[variant]=selection
        traces={}
        for line in (folder/'split_trace.jsonl').open():
            r=json.loads(line)
            if r['client_request_id'].startswith('pdbench-'):
                traces.setdefault(r['client_request_id'],[]).append(r)
        kv={}
        if variant=='optimized':
            for line in (folder/'pd_kv_trace.jsonl').open():
                r=json.loads(line);kv[r['request_id']]=r
        total=0;chosen={}
        for p in points:
            d=folder/p['variant'];rows=json.loads((d/'requests.json').read_text());total+=len(rows)
            cohort=[r for r in rows if p['measurement_start']<=r['end']<p['measurement_end']]
            starts=[r for r in rows if p['measurement_start']<=r['start']<p['measurement_end']]
            assert len(cohort)==p['requests']
            assert abs(len(cohort)/p['duration_s']-p['completed_qps'])<1e-12
            first_wait=[];prefill_span=[];decode_batches=[]
            for r in rows:
                assert r['error'] is None and r['tokens']==79 and r['text']==reference
                assert r['slo_pass']==(r['ttft_ms']<=3000 and r['tpot_ms']<=100)
                assert len(r['itls_ms'])==78
                assert abs(np.mean(r['itls_ms'])-r['tpot_ms'])<1e-9
                steps=traces[r['request_id']]
                positions=([0] if variant=='baseline' else [0,1024,2048,3072])+list(range(4096,4174))
                lengths=([4096] if variant=='baseline' else [1024]*4)+[1]*78
                assert [s['position_start'] for s in steps]==positions
                assert [s['query_len'] for s in steps]==lengths
                assert sum(s['emits_token'] for s in steps)==79
                if p['measurement_start']<=r['end']<p['measurement_end']:
                    last_prefill=0 if variant=='baseline' else 3
                    first_wait.append(max(0,steps[0]['front_start_ns']/1e6-r['start']*1000) if variant=='optimized' else steps[0]['queue_ms'])
                    if variant=='optimized':prefill_span.append((steps[last_prefill]['back_end_ns']-steps[0]['front_start_ns'])/1e6)
                    decode_batches.extend(s['batch_size'] for s in steps if s['phase']=='decode')
                if variant=='optimized':
                    rid=steps[0]['request_id'];k=kv[rid]
                    assert steps[4]['front_start_ns']>=k['kv_ready_ns']
                    assert [(c['range_start'],c['range_end']) for c in k['chunks']]==[(0,4096)]
                    assert k['kv_ready_ns']>=max(rank[rid]['end_ns'] for rank in k['destination'])
                    assert k['source_released_ns']>=max(k['kv_ready_ns'],max(rank[rid]['end_ns'] for rank in k['source']))
            for key in ['ttft_ms','tpot_ms','e2e_ms','first_second_ms','max_itl_ms']:
                assert abs(np.mean([r[key] for r in cohort])-p['mean_'+key])<1e-9
                assert abs(np.percentile([r[key] for r in cohort],99)-p['p99_'+key])<1e-9
            assert p['slo_attainment']==np.mean([r['slo_pass'] for r in cohort])
            assert p['start_cohort_slo_attainment']==np.mean([r['slo_pass'] for r in starts])
            assert p['slo_pass']==(min(p['slo_attainment'],p['start_cohort_slo_attainment'])>=.99)
            samples=json.loads((d/'health_samples.json').read_text())
            steady=[s for s in samples if p['measurement_start']<=s['time']<p['measurement_end']]
            assert steady and not any('error' in s for s in steady)
            final=json.loads((d/'final_health.json').read_text())
            assert all(final[k]==0 for k in ['active','waiting','kv_used_blocks'])
            p={**p,'system':variant,'ttft_violation_fraction':float(np.mean([r['ttft_ms']>3000 for r in cohort])),
                'tpot_violation_fraction':float(np.mean([r['tpot_ms']>100 for r in cohort])),
                'mean_active':float(np.mean([s['active'] for s in steady])),
                'mean_pd_reserving':float(np.mean([s.get('pd_reserving') or 0 for s in steady])),
                'peak_active':max(s['active'] for s in steady),'peak_waiting':max(s['waiting'] for s in steady),
                'peak_kv_blocks':max(s['kv_used_blocks'] for s in steady),
                'mean_first_prefill_wait_ms':float(np.mean(first_wait)),
                'mean_prefill_pipeline_span_ms':float(np.mean(prefill_span)) if prefill_span else None,
                'request_weighted_decode_batch_size':float(np.mean(decode_batches))}
            exported.append(p)
            c=p['concurrency']
            priority={'coarse':0,'refine':1,'confirm':2}
            if c not in chosen or priority[p['stage']]>=priority[chosen[c]['stage']]:chosen[c]=p
        curves.extend(chosen[c] for c in sorted(chosen))
        winner=selection['highest_confirmed_slo_qps'];assert winner['stage']=='confirm' and winner['slo_pass']
        audits.append(dict(system=variant,points=len(points),all_requests=total,outputs_match=True,
            absolute_positions_valid=True,kv_handoff_order_valid=True if variant=='optimized' else None,both_slo_cohorts_verified=True,
            steady_health_verified=True,drain_verified=True,source_hashes_match=True))
    fields=['system','stage','concurrency','server_max_active','server_kv_blocks','requests','duration_s',
        'completed_qps','goodput_qps','slo_attainment','start_cohort_slo_attainment','slo_pass',
        'mean_ttft_ms','p99_ttft_ms','mean_tpot_ms','p99_tpot_ms','mean_e2e_ms','p99_e2e_ms',
        'mean_first_second_ms','p99_first_second_ms','p99_token_itl_ms','p99_max_itl_ms',
        'ttft_violation_fraction','tpot_violation_fraction','little_law_concurrency','mean_active',
        'mean_pd_reserving','peak_active','peak_waiting','peak_kv_blocks','mean_first_prefill_wait_ms','mean_prefill_pipeline_span_ms',
        'request_weighted_decode_batch_size','minimum_seconds','minimum_worker_cycles','variant']
    for name,rows in [('all_points.csv',exported),('curve.csv',curves)]:
        with (OUT/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
    (OUT/'audit.json').write_text(json.dumps(dict(passed=True,variants=audits),indent=2))
    (OUT/'selection.json').write_text(json.dumps(selections,indent=2))
    print(json.dumps(selections,indent=2))


if __name__=='__main__':main()
