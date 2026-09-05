from __future__ import annotations

import unittest

from split_serving_sim.core import Stage
from split_serving_sim.dag import ExecutionDAG


class DAGTest(unittest.TestCase):
    def test_prefill_chunks_are_causal_but_pipelineable(self) -> None:
        dag = ExecutionDAG()
        ready = dag.add_prefill(request_id=0, input_tokens=32, chunk_size=16, ready_time=0)
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].stage, Stage.EDGE_FRONT)
        first_front = ready[0]

        next_ready = dag.mark_complete(first_front.id, 1.0)
        self.assertEqual(
            {item.stage for item in next_ready}, {Stage.EDGE_FRONT, Stage.WAN_UP}
        )
        self.assertEqual({item.chunk_index for item in next_ready}, {0, 1})

    def test_decode_graph_is_created_one_iteration_at_a_time(self) -> None:
        dag = ExecutionDAG()
        ready = dag.add_decode(0, iteration=1, context_tokens=33, ready_time=2.0)
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].stage, Stage.EDGE_FRONT)
        self.assertEqual(len(dag.items), 5)


if __name__ == "__main__":
    unittest.main()
