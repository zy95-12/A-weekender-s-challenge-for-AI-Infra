"""One real 4K/32 request inside an explicit Nsight capture window."""
import json
from pathlib import Path
import time
import httpx
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    launch = json.loads((ROOT / "run/launch.json").read_text())
    assert launch["profile"], "Start with ./poc up --profile --wan first"
    folder = Path((ROOT / "run/current_results").read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "models/qwen"), local_files_only=True)
    base = tokenizer.encode("Explain how key value caches help language models generate text. ")
    prompt = (base * (4096 // len(base) + 1))[:4096]
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=180, trust_env=False) as client:
        client.post("/start_profile").raise_for_status()
        try:
            start = time.perf_counter()
            response = client.post("/v1/completions", json={"prompt": prompt, "max_tokens": 32,
                "ignore_eos": True, "temperature": 0})
            response.raise_for_status()
            result = response.json()
            assert result["usage"]["completion_tokens"] == 32
            (folder / "profile_request.json").write_text(json.dumps({"response": result,
                "seconds": time.perf_counter() - start, "classification": "profile_only_not_benchmark"}, indent=2))
            print(json.dumps(result, ensure_ascii=False), flush=True)
        finally:
            client.post("/stop_profile").raise_for_status()


if __name__ == "__main__":
    main()
