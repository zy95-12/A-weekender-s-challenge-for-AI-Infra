import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from split_poc.runtime import KVPool, Runner


class ChunkMetadataTests(unittest.TestCase):
    def test_unchunked_prefill_keeps_original_attention_path(self):
        runner = Runner.__new__(Runner)
        runner.pool = KVPool(8)
        runner.args = {"prefill_chunk_size": 0}
        meta = self.metadata(runner, [{"request_id": "a", "position": 0, "query_len": 17}], "prefill")
        self.assertEqual(meta.block_tables.numel(), 0)

    def metadata(self, runner, items, phase):
        tensor, empty = torch.tensor, torch.empty
        def cpu_tensor(*args, **kwargs):
            return tensor(*args, **{**kwargs, "device": "cpu"})
        def cpu_empty(*args, **kwargs):
            return empty(*args, **{**kwargs, "device": "cpu"})
        module = SimpleNamespace(FlashAttentionMetadata=lambda **kwargs: SimpleNamespace(**kwargs))
        with patch.dict(sys.modules,{"vllm.attention.backends.flash_attn":module}), \
                patch("torch.tensor",side_effect=cpu_tensor), patch("torch.empty",side_effect=cpu_empty):
            return runner.metadata(items,phase)

    def test_nonzero_chunk_keeps_prefix_blocks_and_positions(self):
        runner = Runner.__new__(Runner)
        runner.pool = KVPool(8)
        runner.args = {"prefill_chunk_size":17}
        first = [{"request_id":"a","position":0,"query_len":17}]
        meta = self.metadata(runner,first,"prefill")
        self.assertEqual(meta.block_tables.tolist(),[runner.pool.requests["a"]["blocks"]])
        self.assertEqual(meta.context_lens_tensor.tolist(),[0])
        runner.pool.commit(first)
        blocks = runner.pool.requests["a"]["blocks"][:]
        second = [{"request_id":"a","position":17,"query_len":2}]
        meta = self.metadata(runner,second,"prefill")
        self.assertEqual(meta.block_tables.tolist(),[blocks])
        self.assertEqual(meta.slot_mapping.tolist(),[blocks[1]*16+1,blocks[1]*16+2])
        self.assertEqual(meta.seq_lens,[19])
        self.assertEqual(meta.context_lens_tensor.tolist(),[17])
        self.assertEqual(meta.query_start_loc.tolist(),[0,2])
        self.assertEqual(meta.num_prefill_tokens,2)
        self.assertEqual(meta.num_decode_tokens,0)
        runner.pool.commit(second)
        meta = self.metadata(runner,[{"request_id":"a","position":19,"query_len":1}],"decode")
        self.assertEqual(meta.num_prefills,0)
        self.assertEqual(meta.num_decode_tokens,1)
        self.assertEqual(meta.seq_lens,[20])


if __name__ == "__main__":
    unittest.main()
