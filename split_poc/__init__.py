"""Naive split inference on the vLLM 0.10.2 CUDA runtime."""
import os
import json
from pathlib import Path

os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
WEIGHT_SHA256 = {
    "model-00001-of-00002.safetensors": "67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2",
    "model-00002-of-00002.safetensors": "a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d",
}
SPLITS = {}
for path in sorted((Path(__file__).resolve().parents[1] / "configs").glob("split_*.json")):
    config = json.loads(path.read_text())
    layers = tuple(config[k] for k in ("front_layers", "cloud_layers", "back_layers"))
    if any(type(x) is not int or x < 1 for x in layers) or sum(layers) != 36:
        raise ValueError(f"Invalid layer partition: {path}")
    SPLITS[config["ratio"]] = layers
