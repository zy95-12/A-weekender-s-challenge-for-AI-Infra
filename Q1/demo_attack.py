"""Live embedding exposure/recovery adapter for PR #15's exact attack.

Encoding is not cryptographic encryption. Recovery receives the exposed tensor
and public embedding table; ground truth is read only afterwards for scoring.
"""

import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from check_model import check_model
from retrieval import recover
from metrics import score


def run(model, out, phase):
    check_model(model)
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    mapping = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    with safe_open(model / mapping["model.embed_tokens.weight"], framework="pt") as f:
        embedding = f.get_tensor("model.embed_tokens.weight").to(dtype=torch.float16)
    start = time.perf_counter()
    if phase == "encode":
        text = json.loads((out / "input.json").read_text())["text"]
        ids = tok.encode(text, add_special_tokens=False)
        if not 1 <= len(ids) <= 4096:
            raise ValueError("Input must contain 1–4096 tokens")
        hidden = embedding[torch.tensor(ids)]
        np.save(out / "hidden.npy", hidden.numpy())
        (out / "target.json").write_text(json.dumps(ids))
        result = {
            "phase": phase,
            "shape": list(hidden.shape),
            "dtype": "float16",
            "bytes": hidden.numel() * 2,
            "tensor_preview": hidden[0, :12].float().tolist(),
            "tokens": len(ids),
            "model_revision": (model / "revision.txt").read_text().strip(),
            "hidden_sha256": hashlib.sha256(
                (out / "hidden.npy").read_bytes()
            ).hexdigest(),
            "seconds": time.perf_counter() - start,
            "method": "PR #15 embedding-only boundary; not encryption",
        }
    else:
        hidden = torch.from_numpy(np.load(out / "hidden.npy", allow_pickle=False))
        predicted = recover(hidden, embedding)  # No target IDs enter the attacker.
        target = json.loads((out / "target.json").read_text())
        result = {
            "phase": phase,
            **score(target, predicted, tok),
            "recovered_text": tok.decode(predicted),
            "positions": [
                {"position": i, "target_id": a, "recovered_id": b, "match": a == b}
                for i, (a, b) in enumerate(zip(target, predicted))
            ],
            "seconds": time.perf_counter() - start,
            "method": "PR #15 full-vocabulary FP32 cosine nearest-neighbor search on FP16 exposed embeddings; CPU",
        }
    (out / f"{phase}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--phase", choices=["encode", "recover"], required=True)
    a = p.parse_args()
    run(a.model, a.output, a.phase)
