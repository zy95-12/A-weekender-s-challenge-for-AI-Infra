"""Recompute C16 metrics and audit absolute positions and handoff timestamps."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/pd_c16'


def main():
    audits=[];summaries=[]
    for variant in ['stage123_q4','pd_stage123_p2d1']:
        folder=OUT/variant
        summary=json.loads((folder/'summary.json').read_text());summaries.append(summary)
        rows=json.loads((folder/'requests.json').read_text())
        measured=[r for r in rows if summary['measurement_start']<=r['end']<summary['measurement_end']]
        assert len(measured)==summary['requests']
        assert all(r['error'] is None and r['tokens']==79 for r in rows)
        assert abs(len(measured)/summary['duration_s']-summary['completed_qps'])<1e-12
        for key in ['ttft_ms','tpot_ms','first_second_ms']:
            assert abs(float(np.mean([r[key] for r in measured]))-summary['mean_'+key])<1e-9
            assert abs(float(np.percentile([r[key] for r in measured],99))-summary['p99_'+key])<1e-9
        traces={}
        for line in (folder/'split_trace.jsonl').read_text().splitlines():
            row=json.loads(line)
            if row['client_request_id'].startswith('pdbench-'):
                traces.setdefault(row['client_request_id'],[]).append(row)
        for r in rows:
            steps=traces[r['request_id']]
            assert [s['position_start'] for s in steps]==[0,1024,2048,3072]+list(range(4096,4174))
            assert [s['query_len'] for s in steps]==[1024]*4+[1]*78
            assert sum(s['emits_token'] for s in steps)==79
        item=dict(variant=variant,formal_requests=len(measured),all_requests=len(rows),positions_valid=True)
        if variant.startswith('pd_'):
            kv={r['request_id']:r for r in map(json.loads,(folder/'pd_kv_trace.jsonl').read_text().splitlines())}
            exposed=[];transfer=[];commit=[]
            for r in measured:
                steps=traces[r['request_id']];rid=steps[0]['request_id'];k=kv[rid]
                assert steps[4]['front_start_ns']>=k['kv_ready_ns']
                src_end=max(rank[rid]['end_ns'] for rank in k['source'])
                dst_end=max(rank[rid]['end_ns'] for rank in k['destination'])
                assert k['kv_ready_ns']>=dst_end
                assert k['source_released_ns']>=max(src_end,k['kv_ready_ns'])
                exposed.append(max(0,k['kv_ready_ns']-steps[3]['back_end_ns'])/1e6)
                transfer.append((dst_end-min(rank[rid]['start_ns'] for rank in k['destination']))/1e6)
                commit.append((k['kv_ready_ns']-dst_end)/1e6)
            item.update(handoff_order_valid=True,mean_exposed_kv_after_first_token_ms=float(np.mean(exposed)),
                p99_exposed_kv_after_first_token_ms=float(np.percentile(exposed,99)),
                fraction_kv_ready_before_first_token=float(np.mean(np.array(exposed)==0)),
                mean_receive_thread_duration_ms=float(np.mean(transfer)),mean_import_to_ready_ms=float(np.mean(commit)))
        audits.append(item)
    result=dict(passed=True,variants=audits)
    (OUT/'audit.json').write_text(json.dumps(result,indent=2))
    fields=['variant','requests','completed_qps','slo_attainment','mean_ttft_ms','p99_ttft_ms','mean_tpot_ms',
            'p99_tpot_ms','mean_first_second_ms','p99_first_second_ms','p99_token_itl_ms','p99_max_itl_ms']
    with (OUT/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(summaries)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
