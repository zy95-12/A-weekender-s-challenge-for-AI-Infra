from dataclasses import replace
from copy import deepcopy
from pathlib import Path
import unittest

from split_serving_sim.config import load_config
from split_serving_sim.presets import configure_serving,ServingFeatures
from split_serving_sim.command_cost import CommandCostModel,CommandNetworkModel
from split_serving_sim.performance import NetworkModel
from split_serving_sim.core import WorkItem,Phase,Stage
from split_serving_sim.simulator import Simulator

ROOT=Path(__file__).parents[1]


class CommandCostTest(unittest.TestCase):
    def setUp(self):
        self.config=configure_serving(load_config(ROOT/'configs/issue6_baseline_host.json'),ServingFeatures.optimized())
        self.model=CommandCostModel.from_file(self.config,ROOT/'profiles/issue6_pd_tp1_c24_commands.json')

    def empirical_profile(self):
        profile=deepcopy(self.model.profile)
        sample=next(s for s in profile['samples'] if s['stage']=='cloud_middle' and s['phase']=='decode')
        profile['samples']=[dict(sample,batch_size=1,latency_ms=10,sample_count=2,service_samples_ms=[5,15]),
                            dict(sample,batch_size=3,latency_ms=20,sample_count=2,service_samples_ms=[10,30])]
        return profile

    def test_empirical_seed_reproducible_without_changing_mean_mode(self):
        profile=self.empirical_profile()
        a=CommandCostModel(self.config,profile,sampling='empirical',seed=17)
        b=CommandCostModel(self.config,profile,sampling='empirical',seed=17)
        mean=CommandCostModel(self.config,profile)
        item=WorkItem(0,0,Phase.DECODE,Stage.CLOUD_MIDDLE,4100,1,4101)
        av=[a.estimate('cloud_middle',[item]).total_time_s*1000 for _ in range(20)]
        bv=[b.estimate('cloud_middle',[item]).total_time_s*1000 for _ in range(20)]
        self.assertEqual(av,bv)
        self.assertEqual(set(av),{5,15})
        self.assertEqual(mean.estimate('cloud_middle',[item]).total_time_s*1000,10)

    def test_interpolated_batch_preserves_target_mean_with_relative_residuals(self):
        model=CommandCostModel(self.config,self.empirical_profile(),sampling='empirical',seed=17)
        items=[WorkItem(i,i,Phase.DECODE,Stage.CLOUD_MIDDLE,4100,1,4101) for i in range(2)]
        values=[model.estimate('cloud_middle',items).total_time_s*1000 for _ in range(20)]
        self.assertEqual(set(values),{7.5,22.5})
        self.assertEqual(model.residual_coverage,{'nearest_batch_relative_residual':20})

    def test_empirical_rejects_missing_and_inconsistent_distributions(self):
        with self.assertRaisesRegex(ValueError,'service_samples_ms'):
            CommandCostModel(self.config,self.model.profile,sampling='empirical')
        profile=self.empirical_profile()
        profile['samples'][0]['service_samples_ms']=[5,16]
        with self.assertRaisesRegex(ValueError,'calibrated mean'):
            CommandCostModel(self.config,profile,sampling='empirical')
        profile['samples'][0]['service_samples_ms']=[-5,25]
        with self.assertRaisesRegex(ValueError,'invalid empirical'):
            CommandCostModel(self.config,profile,sampling='empirical')

    def test_invalid_service_time_rejected(self):
        profile=deepcopy(self.model.profile)
        profile['samples'][0]['latency_ms']=-1
        with self.assertRaises(ValueError):
            CommandCostModel(self.config,profile)

    def test_scope_rejects_different_transport(self):
        cfg=replace(self.config,data_path=replace(self.config.data_path,wire_fast=False))
        with self.assertRaises(ValueError):
            CommandCostModel(cfg,self.model.profile)

    def test_unknown_query_fails_instead_of_using_wrong_cost(self):
        item=WorkItem(0,0,Phase.PREFILL,Stage.EDGE_FRONT,0,1024,1024)
        with self.assertRaises(ValueError):
            self.model.estimate('edge_front',[item])

    def test_measured_cost_disables_host_submission_double_count(self):
        cfg=replace(self.config,workload=replace(self.config.workload,num_requests=2,
            concurrency=1,warmup_requests=0,measurement_duration_s=0,output_tokens=3))
        sim=Simulator(cfg,self.model);sim.run()
        self.assertFalse(sim.host_totals)
        self.assertTrue(all('command_' in key for counts in self.model.coverage.values() for key in counts))

    def test_staging_removed_but_wire_serialization_preserved(self):
        item=WorkItem(0,0,Phase.PREFILL,Stage.WAN_UP,0,2048,2048)
        full=NetworkModel(self.config).estimate(Stage.WAN_UP,[item])
        wire=CommandNetworkModel(self.config).estimate(Stage.WAN_UP,[item])
        names={op.name for op in wire.sub_operations}
        self.assertFalse(names & {'device_to_host','host_to_device','cloud_ipc'})
        self.assertIn('wan_serialization',names)
        self.assertIn('host_pack',names)
        removed=sum(op.duration_s for op in full.sub_operations if op.name not in names)
        self.assertAlmostEqual(full.total_time_s-wire.total_time_s,removed)

    def test_large_decode_batch_marked_extrapolation(self):
        items=[WorkItem(i,i,Phase.DECODE,Stage.CLOUD_MIDDLE,4100,1,4101) for i in range(40)]
        estimate=self.model.estimate('cloud_middle',items)
        self.assertEqual(estimate.sub_operations[0].profile_source,'command_extrapolated_batch')
        self.assertGreater(estimate.total_time_s,0)

if __name__=='__main__':unittest.main()
