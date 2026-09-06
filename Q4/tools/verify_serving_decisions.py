"""Feed the same admission/completion state sequence to live and vendored code.

Run --live from the real serving checkout with its Python environment. No
scheduler thread, HTTP client, GPU executor or server is started.
"""
import argparse
from concurrent.futures import Future
import json
from pathlib import Path
import queue
import sys
from types import SimpleNamespace


def replay(PDScheduler, KVAdmission, choose):
    events=[];calls=[]
    s=PDScheduler.__new__(PDScheduler)
    s.args=SimpleNamespace(max_active=2,prefill_replicas=2,pd_prefill_window=3,decode_quota=4)
    s.window=2;s.active=[];s.reserving={};s.releasing=[];s.routes={};s.next_replica=0
    s.admission=KVAdmission(100);s.pending=queue.Queue();s.waiting_admission=None;s.inflight=[]
    s.back_decode_rounds=0;s.control_post=lambda *a:None
    def submit(fn,path,body):
        f=Future();calls.append((path,body,f));return f
    s.control=SimpleNamespace(submit=submit)
    s.executor=SimpleNamespace(call=lambda command:{'kv_used_blocks':0})
    jobs=[SimpleNamespace(id=f'{i:032x}',ids=[0]*32,limit=3,cancelled=False,finished=False,
        front_position=0,position=0,prefilled=False,outstanding=0,last_step=i) for i in range(3)]
    for j in jobs:s.pending.put(j)
    def record(label):
        events.append(dict(event=label,active=[j.id for j in s.active],reserving=list(s.reserving),
            reservations=dict(s.admission.reservations),routes=dict(s.routes),
            calls=[dict(path=p,body=b,done=f.done()) for p,b,f in calls]))
    s.admit_pending();record('submit_initial_reserves')
    calls[1][2].set_result({'ok':True});s.admit_pending();record('second_reserve_finishes_first')
    batch,phase,rounds=choose(s.eligible(),'decode-first',0,4)
    events.append(dict(event='front_choice',ids=[j.id for j in batch],phase=phase,rounds=rounds))
    calls[0][2].set_result({'ok':True});s.admit_pending();record('first_reserve_finishes')
    for j in jobs[:2]:
        j.prefilled=True;j.position=j.front_position=32;j.pd_ready=Future();j.pd_logged=True
    jobs[1].pd_ready.set_result({'state':'ready'})
    events.append(dict(event='kv_gated_eligible',ids=[j.id for j in s.eligible()]))
    jobs[0].pd_ready.set_result({'state':'ready'})
    dec={'batch':[jobs[0]],'command':{'phase':'decode'}}
    pre={'batch':[jobs[1]],'command':{'phase':'prefill'}}
    events.append(dict(event='back_quota',phases=[s.ready_task([pre,dec])['command']['phase'] for _ in range(6)]))
    jobs[0].finished=True;s.release([jobs[0]]);s.active.remove(jobs[0])
    s.admit_pending();record('release_pending_holds_credit')
    calls[-1][2].set_result({'ok':True});s.admit_pending();record('release_done_admits_third')
    return events


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--live',action='store_true');parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.live:
        sys.path.insert(0,str(Path.cwd()))
        from split_poc.pd_scheduler import PDScheduler
        from split_poc.pipeline_state import KVAdmission
        from split_poc.scheduling import choose
    else:
        root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
        from split_serving_sim.serving_runtime import VirtualServingSimulator
        from split_serving_sim.config import load_config
        from split_serving_sim.presets import configure_serving,ServingFeatures
        sim=VirtualServingSimulator(configure_serving(load_config(root/'configs/issue6_baseline_host.json'),ServingFeatures.optimized()))
        PDScheduler=sim.source['PDScheduler'];KVAdmission=sim.source['KVAdmission'];choose=sim.source['choose']
    Path(args.output).write_text(json.dumps(replay(PDScheduler,KVAdmission,choose),indent=2)+'\n')

if __name__=='__main__':main()
