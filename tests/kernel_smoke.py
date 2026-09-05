"""Initialization-only CUDA test with zero weights, NOT a model correctness test.

Useful while the real checkpoint is downloading. This test override is confined
to this test process; production startup always loads and verifies real weights.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"
os.environ["GLOO_SOCKET_IFNAME"] = "lo"
import numpy as np
import torch
import torch.multiprocessing as mp
from split_poc.runtime import Runner, PartialModel


def run(rank):
    def zero_weights(self, path):
        with torch.no_grad():
            for param in self.parameters():
                param.zero_()
    PartialModel.load_owned_weights = zero_weights
    args = {"model": "models/qwen", "role": "cloud", "split": "1:3", "tp": 2,
            "kv_blocks": 32, "dist_port": 29509}
    runner = Runner(rank, args)
    for step in range(3):
        length = 32 if step == 0 else 1
        command = {"op": "forward", "phase": "prefill" if step == 0 else "decode", "batch_id": str(step),
                   "items": [{"request_id": "a" * 32, "position": 0 if step == 0 else 31 + step, "query_len": length}]}
        arrays = [np.ones((length, 2048), dtype=np.float16)] * 2 if rank == 0 else None
        result = runner.execute(command, arrays)
        if rank == 0:
            assert result["arrays"][0].shape == (length, 2048)
            assert np.isfinite(result["arrays"][0]).all()
            print(f"CUDA initialization test step {step} PASS", flush=True)
    from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    mp.spawn(run, nprocs=2, join=True)
