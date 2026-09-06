"""Calibrate only measured post-model CPU body; excludes following admission/wait."""
import argparse,csv,gzip,json,statistics,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from split_serving_sim.config import load_config
from split_serving_sim.presets import configure_serving,ServingFeatures
from split_serving_sim.command_cost import command_scope

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--point',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);a=p.parse_args()
    summary=json.loads((a.point/'summary.json').read_text())
    batches={}
    for line in (a.run/'split_trace.jsonl').open():
        r=json.loads(line);batches[r['batch_id']]=r
    groups=defaultdict(list);rows=[];raw=[]
    for path in a.run.glob('precompute-*.jsonl'):
        for line in path.open():
            r=json.loads(line)
            if r['kind']!='scheduler_back_post' or not summary['measurement_start']<=r['model_return_ns']/1e9<=summary['measurement_end']:continue
            raw.append(r)
            b=batches[r['batch_id']]
            total=(r['post_end_ns']-r['model_return_ns'])/1e6
            trace=r['trace_wall_ns']/1e6;events=r['events_wall_ns']/1e6;other=total-trace-events
            assert other>=0
            row=dict(phase=r['phase'],batch_size=r['batch_size'],emits_token=b['emits_token'],total_ms=total,
                     trace_ms=trace,token_events_ms=events,other_ms=other)
            rows.append(row);groups[r['phase'],r['batch_size'],b['emits_token']].append(row)
    samples=[]
    for (phase,b,emits),values in sorted(groups.items()):
        samples.append(dict(phase=phase,batch_size=b,emits_token=emits,sample_count=len(values),
            latency_ms=statistics.mean(v['total_ms'] for v in values),
            trace_ms=statistics.mean(v['trace_ms'] for v in values),
            token_events_ms=statistics.mean(v['token_events_ms'] for v in values),
            other_ms=statistics.mean(v['other_ms'] for v in values)))
    cfg=configure_serving(load_config(ROOT/'configs/issue6_baseline_host.json'),ServingFeatures.optimized())
    profile=dict(scope=command_scope(cfg),operation='scheduler_back_post',samples=samples,
        provenance=dict(run=str(a.run),point=str(a.point),measurement_start=summary['measurement_start'],measurement_end=summary['measurement_end'],
            boundary='enterprise back executor return to complete_back body end; trace/event/state work only',
            limitations='CPU wall time includes GIL/I/O scheduling; spread across rows; same hardware and logging implementation required'))
    a.output.write_text(json.dumps(profile,indent=2)+'\n')
    a.evidence.mkdir(exist_ok=True)
    with (a.evidence/'post_back_samples.csv').open('w') as f:
        w=csv.DictWriter(f,rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
    (a.evidence/'post_back_probe.jsonl.gz').write_bytes(gzip.compress(''.join(json.dumps(r)+'\n' for r in raw).encode(),mtime=0))
    (a.evidence/'probe_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for phase in ('prefill','decode'):
        selected=[r for r in rows if r['phase']==phase]
        print(phase,len(selected),{k:statistics.mean(r[k] for r in selected) for k in ('total_ms','trace_ms','token_events_ms','other_ms')})

if __name__=='__main__':main()
