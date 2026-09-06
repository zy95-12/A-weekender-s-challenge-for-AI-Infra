from dataclasses import replace
import unittest

from split_serving_sim.config import RequestSpec,validate_config,ConfigError
from split_serving_sim.core import Phase
from split_serving_sim.performance import RooflineModel
from split_serving_sim.simulator import Simulator
from split_serving_sim.presets import ServingFeatures,configure_serving
from tests.test_pd_serving import toy_pd


class PDAdmissionTest(unittest.TestCase):
    def test_reserve_orders_d_then_p_and_gates_front(self):
        sim=Simulator(toy_pd());result=sim.run()
        for rid in sim.requests:
            events={e['event']:e for e in sim.pd_admission.events if e['request_id']==rid}
            self.assertLessEqual(events['reserve_d_done']['time_ms'],events['reserve_p_start']['time_ms'])
            self.assertLessEqual(events['reserve_p_done']['time_ms'],events['reserve_rpc_done']['time_ms'])
            first=min(r['start_time_ms'] for r in result.trace if r['stage']=='edge_front' and rid in r['request_ids'])
            self.assertGreaterEqual(first,events['active']['time_ms'])
            self.assertTrue(events['reserve_p_start']['resource'].startswith('cloud_prefill'))
            self.assertEqual(events['reserve_d_start']['resource'],'cloud_decode')
        self.assertFalse(sim.pd_admission.reserving)
        self.assertFalse(sim.pd_admission.releasing)

    def test_running_prefill_delays_reserve_on_the_same_worker(self):
        cfg=toy_pd(prefill_replicas=1)
        cfg=replace(cfg,workload=replace(cfg.workload,mode='trace',requests=(
            RequestSpec(request_id=0,arrival_time_ms=0,input_tokens=32,output_tokens=3),
            RequestSpec(request_id=1,arrival_time_ms=30,input_tokens=32,output_tokens=3))))
        class SlowPrefill(RooflineModel):
            def estimate(self,name,items):
                e=super().estimate(name,items)
                if name=='cloud_middle' and items[0].phase==Phase.PREFILL:
                    return replace(e,total_time_s=0.05,sub_operations=())
                return e
        sim=Simulator(cfg,SlowPrefill(cfg));result=sim.run()
        event=next(e for e in sim.pd_admission.events if e['event']=='reserve_p_start' and e['request_id']==1)
        self.assertGreater(event['queue_ms'],10)
        queued=event['time_ms']-event['queue_ms']
        forwards=[r for r in result.trace if r['resource']==event['resource'] and r['stage']=='cloud_middle']
        self.assertTrue(any(r['start_time_ms']<=queued<r['end_time_ms']<=event['time_ms'] for r in forwards))

    def test_release_holds_admission_credit_after_client_finishes(self):
        cfg=toy_pd(prefill_replicas=1)
        cfg=replace(cfg,workload=replace(cfg.workload,output_tokens=1),scheduler=replace(
            cfg.scheduler,max_num_seqs=1,kv_cache=replace(cfg.scheduler.kv_cache,enabled=True,enable_preemption=False),
            pd_disaggregation=replace(cfg.scheduler.pd_disaggregation,release_e_ms=50,release_d_ms=200,release_p_ms=200)))
        sim=Simulator(cfg);sim.run()
        a={e['event']:e for e in sim.pd_admission.events if e['request_id']==0}
        b={e['event']:e for e in sim.pd_admission.events if e['request_id']==1}
        self.assertLess(sim.requests[0].finish_time*1000,a['admission_credit_returned']['time_ms'])
        self.assertGreaterEqual(b['reserve_submit']['time_ms'],a['admission_credit_returned']['time_ms'])
        freed=next(e for e in sim.kv_events if e['event']=='free' and e['request_id']==0)
        self.assertEqual(freed['time_ms'],a['release_e_done']['time_ms'])
        self.assertLess(freed['time_ms'],a['admission_credit_returned']['time_ms'])
        self.assertFalse(sim.active_continuous_requests)
        self.assertTrue(all(not q for q in sim.pd_admission.queues.values()))

    def test_release_waits_for_kv_commit_for_single_token(self):
        cfg=toy_pd()
        cfg=replace(cfg,workload=replace(cfg.workload,output_tokens=1),scheduler=replace(cfg.scheduler,
                    pd_disaggregation=replace(cfg.scheduler.pd_disaggregation,kv_transfer_latency_ms=500)))
        sim=Simulator(cfg);result=sim.run()
        for rid in sim.requests:
            commit=next(r for r in result.trace if r['stage']=='pd_kv_commit' and rid in r['request_ids'])
            release=next(e for e in sim.pd_admission.events if e['event']=='release_submit' and e['request_id']==rid)
            self.assertGreaterEqual(release['time_ms'],commit['end_time_ms'])

    def test_idle_reserve_uses_service_and_rpc_costs_not_a_wait_constant(self):
        cfg=toy_pd(prefill_replicas=1)
        cfg=replace(cfg,workload=replace(cfg.workload,num_requests=1))
        sim=Simulator(cfg);sim.run()
        active=next(e for e in sim.pd_admission.events if e['event']=='active')
        self.assertAlmostEqual(active['time_ms'],21.45)
        waits=[e['queue_ms'] for e in sim.pd_admission.events
               if e['event'] in ('reserve_p_start','reserve_d_start')]
        self.assertEqual(waits,[0,0])

    def test_preset_preserves_calibrated_lifecycle_costs(self):
        from split_serving_sim.config import load_config
        from pathlib import Path
        base=load_config(Path(__file__).parents[1]/'configs/issue6_baseline_host.json')
        base=replace(base,scheduler=replace(base.scheduler,pd_disaggregation=replace(
            base.scheduler.pd_disaggregation,reserve_p_ms=0.7)))
        cfg=configure_serving(base,ServingFeatures.optimized())
        self.assertEqual(cfg.scheduler.pd_disaggregation.reserve_p_ms,0.7)

    def test_switch_and_validation(self):
        cfg=toy_pd(pd_admission=False)
        sim=Simulator(cfg);result=sim.run()
        self.assertIsNone(sim.pd_admission)
        self.assertFalse(result.summary['pd_admission']['enabled'])
        invalid=replace(cfg,scheduler=replace(cfg.scheduler,pd_disaggregation=replace(
            cfg.scheduler.pd_disaggregation,admission_enabled=True,reserve_p_ms=-1)))
        with self.assertRaises(ConfigError):validate_config(invalid)

if __name__=='__main__':unittest.main()
