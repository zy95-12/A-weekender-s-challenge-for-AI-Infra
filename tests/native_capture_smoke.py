"""Check the native logits hook with vLLM's explicit dummy loader.

Does not generate reference artifacts or claim real-model correctness.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from split_poc.reference import capture_logits
from vllm import LLM, SamplingParams
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np


if __name__ == "__main__":
    engine = LLM(model="models/qwen", load_format="dummy", dtype="half", tensor_parallel_size=2,
                 enforce_eager=True, max_model_len=512, max_num_seqs=1,
                 max_num_batched_tokens=512, gpu_memory_utilization=.35,
                 enable_prefix_caching=False, enable_chunked_prefill=False,
                 disable_custom_all_reduce=True, disable_async_output_proc=True)
    with capture_logits(engine) as captured:
        params = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)
        engine.generate([{"prompt_token_ids": [1234] * 32}], params)
        assert len(captured) == 3
        assert captured[0].shape == (1, 151936)
        print("Native capture hook initialization test PASS")
