import io,json,unittest
from types import SimpleNamespace
from split_serving_sim.command_cost import command_scope
from split_serving_sim.serving_host import ServingHostWork,TimedServingTrace
from split_serving_sim.serving_runtime import VirtualClock,VirtualServingSimulator
from tests.test_serving_runtime import config


def profile(cfg,ms=10):
    return dict(scope=command_scope(cfg),operation='scheduler_back_post',samples=[
        dict(phase=phase,emits_token=emits,batch_size=1,sample_count=2,latency_ms=ms)
        for phase,emits in [('prefill',False),('prefill',True),('decode',True)]])


class ServingHostTests(unittest.TestCase):
    def test_cpu_work_advances_clock_while_cloud_completes(self):
        cfg=config();work=ServingHostWork(cfg,profile(cfg))
        clock=VirtualClock(1);completed=[]
        clock.at(.003,lambda:completed.append(clock.now))
        runtime=SimpleNamespace(clock=clock,execution=[])
        trace=TimedServingTrace(runtime,work)
        row=dict(phase='decode',emits_token=True,batch_size=1,request_id='0')
        trace.write(json.dumps(row)+'\n')
        self.assertEqual(completed,[.003]);self.assertEqual(clock.now,.010)
        self.assertEqual(runtime.execution[0]['resource'],'enterprise_host')
        self.assertEqual(json.loads(trace.getvalue()),row)

    def test_token_delivery_includes_post_work_outside_model_interval(self):
        cfg=config(num_requests=1,concurrency=1,output_tokens=3)
        work=ServingHostWork(cfg,profile(cfg,ms=2))
        sim=VirtualServingSimulator(cfg,host_work=work);r=sim.run()
        emitted=[x for x in sim.serving_trace if x['emits_token']]
        self.assertEqual(len(emitted),3)
        for row,token in zip(emitted,r.requests[0]['token_timestamps_ms']):
            self.assertAlmostEqual(token-row['back_end_ns']/1e6,2,places=5)
        self.assertTrue(r.summary['lifecycle_drained'])

    def test_invalid_scope_and_negative_cost_rejected(self):
        cfg=config();p=profile(cfg);p['operation']='gpu_compute'
        with self.assertRaises(ValueError):ServingHostWork(cfg,p)
        with self.assertRaises(ValueError):ServingHostWork(cfg,profile(cfg,ms=-1))

if __name__=='__main__':unittest.main()
