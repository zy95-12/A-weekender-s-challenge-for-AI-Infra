from concurrent.futures import Future
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from split_serving_sim.core import Stage
from split_serving_sim.serving_runtime import VirtualClock, VirtualServingSimulator
from split_serving_sim.serving_source import SOURCE, load_serving
from tests.test_pd_serving import toy_pd


def config(**workload):
    cfg=toy_pd()
    defaults=dict(mode='closed_loop',num_requests=4,concurrency=2,warmup_requests=0,measurement_duration_s=0)
    defaults.update(workload)
    return replace(cfg,workload=replace(cfg.workload,**defaults))


class ServingRuntimeTests(unittest.TestCase):
    def test_pinned_sources_and_method_bodies(self):
        sim=VirtualServingSimulator(config())
        manifest=sim.manifest
        for name,digest in manifest['files'].items():
            self.assertEqual(hashlib.sha256((SOURCE/name).read_bytes()).hexdigest(),digest)
        # Both instances execute the same original run/choose/admission code,
        # while having separate virtual clocks/globals.
        other=VirtualServingSimulator(config())
        for name,method in [('PDScheduler','admit_pending'),('PDScheduler','eligible'),('PipelineScheduler','run')]:
            a=getattr(sim.source[name],method);b=getattr(other.source[name],method)
            self.assertEqual(a.__code__.co_code,b.__code__.co_code)
            self.assertNotEqual(id(a.__globals__),id(b.__globals__))
            self.assertEqual(Path(a.__code__.co_filename).parent,SOURCE)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for p in SOURCE.iterdir():
                if p.is_file(): (root/p.name).write_bytes(p.read_bytes())
            (root/'scheduling.py').write_text('# drift\n')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                load_serving({},root)

    def test_same_event_script_same_decisions_and_drained(self):
        a=VirtualServingSimulator(config());b=VirtualServingSimulator(config())
        ar=a.run();br=b.run()
        self.assertEqual(a.decisions,b.decisions)
        self.assertEqual(ar.requests,br.requests)
        self.assertTrue(ar.summary['lifecycle_drained'])
        self.assertFalse(a.clock.events)
        for row in ar.requests:self.assertEqual(len(row['token_timestamps_ms']),config().workload.output_tokens)

    def test_decisions_match_live_serving_replay(self):
        from tools.verify_serving_decisions import replay
        sim=VirtualServingSimulator(config())
        expected=json.loads((Path(__file__).parents[1]/'tests/fixtures/serving_decisions.json').read_text())
        self.assertEqual(replay(sim.source['PDScheduler'],sim.source['KVAdmission'],sim.source['choose']),expected)

    def test_out_of_order_responses_preserve_back_positions(self):
        class Reordered(VirtualServingSimulator):
            def network_path(self,stage,command,done):
                if stage==Stage.WAN_DOWN and command['phase']=='prefill':
                    delay=.050 if command['items'][0]['position']==0 else .001
                    self.clock.at(self.clock.now+delay,done)
                else:super().network_path(stage,command,done)
        sim=Reordered(config(num_requests=1,concurrency=1));sim.run()
        prefills=[r for r in sim.serving_trace if r['phase']=='prefill']
        self.assertEqual([r['position_start'] for r in prefills],[0,16])
        self.assertGreater(prefills[0]['enterprise_received_ns'],prefills[1]['enterprise_received_ns'])
        self.assertLess(prefills[0]['back_end_ns'],prefills[1]['back_end_ns'])

    def test_release_holds_credit_and_next_request_reuses_capacity(self):
        cfg=config(output_tokens=1)
        cfg=replace(cfg,scheduler=replace(cfg.scheduler,max_num_seqs=1,pd_disaggregation=replace(
            cfg.scheduler.pd_disaggregation,release_d_ms=50,release_p_ms=50)))
        sim=VirtualServingSimulator(cfg);sim.run()
        reserves=[e for e in sim.decisions if e['op']=='/pd/reserve']
        releases=[e for e in sim.execution if e.get('op')=='release' and e['resource'].startswith('P')]
        self.assertEqual(len(reserves),4)
        self.assertGreaterEqual(reserves[1]['time_ms'],releases[0]['end_time_ms'])
        self.assertFalse(sim.scheduler.admission.reservations)

    def test_original_back_priority_and_quota(self):
        sim=VirtualServingSimulator(config());s=sim.scheduler
        job=type('J',(),{'last_step':0})()
        pre={'command':{'phase':'prefill'},'batch':[job]}
        dec={'command':{'phase':'decode'},'batch':[job]}
        for _ in range(s.args.decode_quota):self.assertIs(s.ready_task([pre,dec]),dec)
        self.assertIs(s.ready_task([pre,dec]),pre)
        self.assertIs(s.ready_task([pre,dec]),dec)

    def test_future_completes_during_blocking_call_but_observed_after(self):
        clock=VirtualClock(1)
        future=Future();seen=[]
        clock.at(.002,lambda:future.set_result('ready'))
        # A synchronous worker call keeps the scheduler blocked until .010.
        worker=Future();clock.at(.010,lambda:worker.set_result(None))
        clock.resolve(worker)
        seen.append((clock.now,future.result()))
        self.assertEqual(seen,[(.010,'ready')])

if __name__=='__main__':unittest.main()
