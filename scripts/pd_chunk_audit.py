"""Audit the bounded chunk-migration ablation from raw request/KV traces."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/pd_chunk_report'


def stats(values):
    return dict(mean=float(np.mean(values)),p99=float(np.percentile(values,99)))


def main():
    OUT.mkdir(exist_ok=True)
    summaries=[];audits=[]
    for group,names in [('pd_allocations',['old_p1d2','old_p2d1']),
                        ('pd_chunks',['notify_only','chunk_transfer']),('pd_reverse_perf',['chunk_p1d2'])]:
        for name in names:
            folder=ROOT/'results'/group/name
            summary=json.loads((folder/'summary.json').read_text());summaries.append(summary)
            rows=json.loads((folder/'requests.json').read_text())
            measured=[r for r in rows if summary['measurement_start']<=r['end']<summary['measurement_end']]
            assert len(measured)==summary['requests']
            assert all(r['error'] is None and r['tokens']==79 for r in rows)
            reference=json.loads((ROOT/'results'/group/'reference.json').read_text())['text']
            assert all(r['text']==reference for r in rows)
            assert all(r['slo_pass']==(r['ttft_ms']<=3000 and r['tpot_ms']<=100) for r in rows)
            starts=[r for r in rows if summary['measurement_start']<=r['start']<summary['measurement_end']]
            assert float(np.mean([r['slo_pass'] for r in measured]))==summary['slo_attainment']
            assert float(np.mean([r['slo_pass'] for r in starts]))==summary['start_cohort_slo_attainment']
            assert abs(len(measured)/summary['duration_s']-summary['completed_qps'])<1e-12
            for key in ['ttft_ms','tpot_ms','first_second_ms']:
                value=stats([r[key] for r in measured])
                for metric in ['mean','p99']:assert abs(value[metric]-summary[metric+'_'+key])<1e-9
            traces={}
            for row in map(json.loads,(folder/'split_trace.jsonl').read_text().splitlines()):
                if row['client_request_id'].startswith('pdbench-'):
                    traces.setdefault(row['client_request_id'],[]).append(row)
            kv={r['request_id']:r for r in map(json.loads,(folder/'pd_kv_trace.jsonl').read_text().splitlines())}
            for r in rows:
                steps=traces[r['request_id']]
                assert [s['position_start'] for s in steps]==[0,1024,2048,3072]+list(range(4096,4174))
                assert [s['query_len'] for s in steps]==[1024]*4+[1]*78
                assert sum(s['emits_token'] for s in steps)==79
                rid=steps[0]['request_id'];k=kv[rid]
                assert steps[4]['front_start_ns']>=k['kv_ready_ns']
                assert k['kv_ready_ns']>=max(rank[rid]['end_ns'] for rank in k['destination'])
                assert k['source_released_ns']>=max(k['kv_ready_ns'],max(rank[rid]['end_ns'] for rank in k['source']))
                if 'chunks' in k:
                    chunks=k['chunks']
                    expected=[(0,1024),(1024,2048),(2048,3072),(3072,4096)] if name.startswith('chunk_') else [(0,4096)]
                    assert [(c['range_start'],c['range_end']) for c in chunks]==expected
                    assert sum(c['source'][0][rid]['logical_bytes'] for c in chunks)==108*1024**2
                    for c in chunks:
                        for rank in c['source']+c['destination']:
                            assert (rank[rid]['range_start'],rank[rid]['range_end'])==(c['range_start'],c['range_end'])
            exposed=[];receive_time=[];commit=[];early_bytes=[];ready_to_front=[]
            for r in measured:
                steps=traces[r['request_id']];rid=steps[0]['request_id'];k=kv[rid]
                exposed.append(max(0,k['kv_ready_ns']-steps[3]['back_end_ns'])/1e6)
                end=max(rank[rid]['end_ns'] for rank in k['destination'])
                receive_time.append((end-min(rank[rid]['start_ns'] for rank in k['destination']))/1e6)
                commit.append((k['kv_ready_ns']-end)/1e6)
                ready_to_front.append((steps[4]['front_start_ns']-max(k['kv_ready_ns'],steps[3]['back_end_ns']))/1e6)
                early_bytes.append(sum(c['destination'][0][rid]['logical_bytes'] for c in k.get('chunks',[])
                    if c['range_end']<4096 and max(rank[rid]['end_ns'] for rank in c['destination'])<steps[3]['cloud_send_ns']))
            audits.append(dict(variant=name,all_requests=len(rows),formal_requests=len(measured),positions_valid=True,
                handoff_order_valid=True,exposed_kv_ms=stats(exposed),final_receive_thread_ms=stats(receive_time),
                import_to_ready_ms=stats(commit),eligible_to_first_decode_front_ms=stats(ready_to_front),
                fraction_kv_ready_before_first_token=float(np.mean(np.array(exposed)==0)),
                mean_bytes_migrated_before_final_prefill_response=float(np.mean(early_bytes))))
    (OUT/'audit.json').write_text(json.dumps(dict(passed=True,variants=audits),indent=2))
    fields=['variant','requests','completed_qps','slo_attainment','mean_ttft_ms','p99_ttft_ms','mean_tpot_ms',
            'p99_tpot_ms','mean_first_second_ms','p99_first_second_ms','p99_token_itl_ms','p99_max_itl_ms']
    with (OUT/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(summaries)
    logits={}
    for name,length in [('current4k',4096),('nonaligned257',257)]:
        a=np.load(ROOT/'results/pd_chunk_validation'/f'{name}.npz')['logits']
        b=np.load(ROOT/'results/pd_chunk_validation'/f'{name}_full_transfer_reference.npz')['logits']
        assert a.shape==b.shape==(8,151936)
        assert np.array_equal(a,b)
        logits[name]=dict(absolute_positions=list(range(length-1,length+7)),exact=True,
                         max_logit_error=0,mean_logit_error=0,max_softmax_probability_error=0,top1_matches=8)
    (OUT/'logits_comparison.json').write_text(json.dumps(logits,indent=2))
    print(json.dumps(audits,indent=2))


if __name__=='__main__':main()
