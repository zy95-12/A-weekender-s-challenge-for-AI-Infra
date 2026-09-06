"""Native vLLM LLMEngine reference. Hook only copies unmodified logits."""
import argparse
import contextlib
import json
from pathlib import Path
import numpy as np

from split_poc import MODEL_ID, REVISION


@contextlib.contextmanager
def capture_logits(engine):
    """Read-only diagnostic hook around the native driver's compute_logits.

v0.10.2 V0 rejects request-level logits_processors, so observe the model
output directly. The exact original tensor is returned to the native sampler.
    """
    model = engine.llm_engine.model_executor.driver_worker.model_runner.model
    original = model.compute_logits
    captured = []
    def observe(*args, **kwargs):
        logits = original(*args, **kwargs)
        if logits is not None:
            captured.append(logits.detach().float().cpu().numpy().copy())
        return logits
    model.compute_logits = observe
    try:
        yield captured
    finally:
        model.compute_logits = original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/qwen")
    parser.add_argument("--lengths", default="128,1024,4096,8192")
    parser.add_argument("--steps", type=int, default=257)
    parser.add_argument("--output", default="results/reference")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import torch
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    engine = LLM(model=args.model, dtype="half", tensor_parallel_size=args.tp,
        enforce_eager=True, max_model_len=16384, max_num_seqs=8,
        max_num_batched_tokens=16384, gpu_memory_utilization=0.65,
        enable_prefix_caching=False, enable_chunked_prefill=False,
        disable_custom_all_reduce=True, disable_async_output_proc=True, seed=0)
    config = {**vars(args), "model_id": MODEL_ID, "revision": REVISION,
              "vllm": "0.10.2", "torch": torch.__version__, "cuda": torch.version.cuda,
              "dtype": "float16", "attention_backend": "FLASH_ATTN", "seed": 0,
              "sampling": {"temperature": 0, "ignore_eos": True},
              "logits_capture": "read-only compute_logits hook, native LLMEngine"}
    (root / "config.json").write_text(json.dumps(config, indent=2))
    base = tokenizer.encode("The enterprise keeps its documents locally. The cloud computes intermediate neural network layers. "
                            "Explain how attention and key value caches help language models generate text.\n", add_special_tokens=False)
    for length in map(int, args.lengths.split(",")):
        prompt = (base * (length // len(base) + 1))[:length]
        for repeat in range(args.repeat):
            dest = root / f"len_{length}" / f"repeat_{repeat}"
            dest.mkdir(parents=True, exist_ok=True)
            params = SamplingParams(temperature=0, max_tokens=args.steps, ignore_eos=True)
            with capture_logits(engine) as captured:
                output = engine.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)[0]
            tokens = list(output.outputs[0].token_ids)
            assert len(captured) == args.steps, (len(captured), args.steps)
            logits = np.concatenate(captured)
            np.savez(dest / "reference.npz", prompt_ids=np.array(prompt), tokens=np.array(tokens), logits=logits)
            print(json.dumps({"length": length, "repeat": repeat, "steps": len(tokens),
                              "text": output.outputs[0].text[:200]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
