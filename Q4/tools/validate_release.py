"""CPU-only baseline and current PD regression; no external trace checkout."""
import argparse,json,sys
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from split_serving_sim.config import load_config
from split_serving_sim.presets import configure_serving,ServingFeatures
from split_serving_sim.simulator import Simulator
from split_serving_sim.command_cost import CommandCostModel
from split_serving_sim.serving_host import ServingHostWork
from split_serving_sim.serving_runtime import VirtualServingSimulator

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/release-validation')
    parser.add_argument('--baseline-only',action='store_true');args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    baseline=[]
    for anchor in json.loads((ROOT/'docs/validation/baseline_regression.json').read_text()):
        c=anchor['concurrency'];cap=8 if c==1 else 16
        cfg=load_config(ROOT/'configs/issue6_baseline_host.json')
        cfg=replace(cfg,workload=replace(cfg.workload,num_requests=512,concurrency=c,warmup_requests=c,measurement_duration_s=60),
            static_policy=replace(cfg.static_policy,max_batch_size=cap),scheduler=replace(cfg.scheduler,max_num_seqs=cap),
            simulation=replace(cfg.simulation,trace_enabled=False,max_trace_records=0,max_detailed_trace_records=0))
        s=Simulator(configure_serving(cfg,ServingFeatures())).run().summary
        values=dict(qps=s['observed_request_throughput_qps'],ttft=s['ttft_ms']['mean'],tpot=s['tpot_ms']['mean'])
        delta={k:values[k]-anchor['values'][k] for k in values};assert all(abs(v)<1e-8 for v in delta.values()),delta
        row=dict(concurrency=c,values=values,delta=delta);baseline.append(row);print(json.dumps(row),flush=True)
    (args.output_dir/'baseline.json').write_text(json.dumps(baseline,indent=2)+'\n')
    if args.baseline_only:return
    pd=[]
    for seed in (17,29,43):
        cfg=configure_serving(load_config(ROOT/'configs/issue6_baseline_host.json'),ServingFeatures.optimized())
        cfg=replace(cfg,workload=replace(cfg.workload,num_requests=1024,concurrency=40,warmup_requests=40,measurement_duration_s=90))
        model=CommandCostModel.from_file(cfg,ROOT/'profiles/issue6_pd_tp1_c40_empirical_commands.json',sampling='empirical',seed=seed)
        host=ServingHostWork.from_file(cfg,ROOT/'profiles/issue6_c40_serving_host.json')
        s=VirtualServingSimulator(cfg,model,host).run().summary
        old=json.loads((ROOT/f'docs/validation/serving_cost_distribution/host-empirical-{seed}.json').read_text())
        for key in ('ttft_ms','tpot_ms'):assert abs(s[key]['mean']-old[key]['mean'])<1e-8
        assert abs(s['observed_request_throughput_qps']-old['observed_request_throughput_qps'])<1e-8
        assert s['lifecycle_drained'];s.pop('scheduling_decisions');pd.append(dict(seed=seed,summary=s))
        print(json.dumps(dict(seed=seed,qps=s['observed_request_throughput_qps'],ttft_ms=s['ttft_ms']['mean'],tpot_ms=s['tpot_ms']['mean'])),flush=True)
    (args.output_dir/'pd.json').write_text(json.dumps(pd,indent=2)+'\n')
if __name__=='__main__':main()
