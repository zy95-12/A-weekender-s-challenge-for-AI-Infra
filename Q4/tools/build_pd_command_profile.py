"""Extract command service times from a normal, uninstrumented PD run.

Never use TTFT, TPOT, QPS or queue time as calibration targets. Deduplicate
per-request trace rows into RPC batches and restrict to the measurement window.
"""
import argparse
import collections
import json
import statistics
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from split_serving_sim.config import load_config
from split_serving_sim.command_cost import command_scope
from split_serving_sim.presets import ServingFeatures,configure_serving


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--include-distribution',action='store_true',help='preserve per-command service samples for empirical sampling')
    args=parser.parse_args()
    summary=json.loads((args.run/'point/summary.json').read_text())
    groups=collections.defaultdict(list);seen=set()
    for line in (args.run/'split_trace.jsonl').open():
        row=json.loads(line)
        if row['batch_id'] in seen or not summary['measurement_start']<=row['back_end_ns']/1e9<=summary['measurement_end']:
            continue
        seen.add(row['batch_id'])
        if row['phase']=='prefill':
            low=high=row['context_len']
        else:
            # This profile is explicitly restricted to the 4K/79-token workload.
            low,high=4096,4175
        durations={'edge_front':(row['front_end_ns']-row['front_start_ns'])/1e6,
                   'edge_tail':(row['back_end_ns']-row['back_start_ns'])/1e6,
                   'cloud_middle':(row['cloud_send_ns']-row['cloud_received_ns'])/1e6-row['cloud_queue_ms']}
        for stage,ms in durations.items():
            key=(stage,row['phase'],row['batch_size'],row['query_len'],
                 row['emits_token'] if stage=='edge_tail' else False,low,high)
            groups[key].append(ms)
    samples=[]
    for (stage,phase,batch,query,emits,low,high),values in sorted(groups.items()):
        samples.append(dict(stage=stage,phase=phase,tp_degree=1,batch_size=batch,
                            query_len=query,emits_logits=emits,context_min=low,context_max=high,
                            latency_ms=statistics.mean(values),median_ms=statistics.median(values),
                            sample_count=len(values),**({'service_samples_ms':values} if args.include_distribution else {})))
    cfg=configure_serving(load_config(ROOT/'configs/issue6_baseline_host.json'),ServingFeatures.optimized())
    profile=dict(scope=command_scope(cfg),
                 provenance=dict(run=str(args.run),concurrency=summary['concurrency'],
                     measurement_start=summary['measurement_start'],measurement_end=summary['measurement_end'],
                     method='mean service wall time per command; cloud queue subtracted; RPC batch deduplicated',
                     limitations='includes CPU/IPC/device staging; not GPU kernel profiling; decode context distribution pooled within 4K/79'),
                 samples=samples)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(profile,indent=2)+'\n')
    print(f'{len(samples)} cost samples from {len(seen)} batches')

if __name__=='__main__':main()
