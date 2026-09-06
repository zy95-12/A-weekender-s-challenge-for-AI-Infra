"""Real-model POC acceptance: long KV, streaming, concurrency, QA, cancellation."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import time
import httpx
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="results/acceptance.json")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "models/qwen"), local_files_only=True)
    rows = []
    with httpx.Client(base_url=args.url, timeout=300, trust_env=False) as client:
        health = client.get("/health").json()
        for length in (1024, 4096, 8192):
            ids = tokenizer.encode("Explain how computers store information. ", add_special_tokens=False)
            ids = (ids * (length // len(ids) + 1))[:length]
            start = time.perf_counter()
            response = client.post("/v1/completions", json={"prompt": ids, "max_tokens": 256,
                                                           "ignore_eos": True, "temperature": 0})
            response.raise_for_status()
            result = response.json()
            assert result["usage"]["prompt_tokens"] == length
            assert result["usage"]["completion_tokens"] == 256
            rows.append({"case": "kv_length", "input_tokens": length, "output_tokens": 256,
                         "wall_seconds": time.perf_counter() - start, "result": "PASS"})
            print(f"Real KV {length}/256 PASS", flush=True)

        prompts = [tokenizer.apply_chat_template([{"role": "user", "content": question}],
                    tokenize=True, add_generation_prompt=True) for question in [
            "Write the numbers from one to ten in English.", "Explain why the sky is blue in one sentence.",
            "列出四种常见水果。", "What is 17 plus 25? Give only the answer."]]
        def generate(ids):
            with httpx.Client(base_url=args.url, timeout=180, trust_env=False) as c:
                response = c.post("/debug/greedy", json={"prompt_ids": ids, "steps": 32})
                response.raise_for_status()
                return response.json()["tokens"]
        expected = [generate(ids) for ids in prompts]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(generate, prompts))
        isolation_path = Path(args.output).with_suffix('.isolation.json')
        isolation_path.parent.mkdir(parents=True, exist_ok=True)
        isolation_path.write_text(json.dumps({"prompts": prompts, "isolated": expected,
            "concurrent": actual, "isolated_text": tokenizer.batch_decode(expected),
            "concurrent_text": tokenizer.batch_decode(actual)}, indent=2, ensure_ascii=False))
        # The diagnostic endpoint forces 32 steps, even after a chat has ended.
        # Compare the actual response through its first EOS, not fictitious
        # subsequent turns. Preserve all forced tokens above for audit.
        eos = {tokenizer.eos_token_id, 151643, 151645}
        def response_tokens(tokens):
            end = next((i + 1 for i, token in enumerate(tokens) if token in eos), len(tokens))
            return tokens[:end]
        assert [response_tokens(x) for x in actual] == [response_tokens(x) for x in expected], \
            "Concurrent and isolated response token sequences differ"
        rows.append({"case": "concurrent_4_vs_isolated", "max_steps": 32,
                     "comparison": "exact IDs through first EOS inclusive, or 32 steps",
                     "forced_after_eos_identical": actual == expected, "result": "PASS"})
        print("Real concurrent=4 isolation PASS", flush=True)

        for target in (4096, 8192):
            needle = "The secret access code for Project Willow is WILLOW-7319."
            filler = "This record describes an ordinary office meeting and contains no access codes. "
            repeats = max(1, target // (2 * len(tokenizer.encode(filler))) - 8)
            record = filler * repeats + needle + " " + filler * repeats
            def make_messages(padding):
                question = "Read the records below.\n" + record + " apple" * padding + "\nWhat is the access code for Project Willow? Output only the code."
                return [{"role": "user", "content": question}]
            base_len = len(tokenizer.apply_chat_template(make_messages(0), tokenize=True, add_generation_prompt=True))
            padding = target - base_len
            for _ in range(4):
                messages = make_messages(padding)
                actual_len = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
                if actual_len == target:
                    break
                padding += target - actual_len
            assert len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)) == target
            response = client.post("/v1/chat/completions", json={"messages": messages, "max_tokens": 32, "temperature": 0})
            response.raise_for_status()
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            passed = "WILLOW-7319" in text
            rows.append({"case": "needle_qa", "target_tokens": target,
                         "input_tokens": result["usage"]["prompt_tokens"], "answer": text,
                         "result": "PASS" if passed else "FAIL"})
            print(f"Needle QA target={target}: {text}", flush=True)

        chunks = []
        with client.stream("POST", "/v1/completions", json={"prompt": "The benefits of exercise include", "max_tokens": 32,
                "temperature": 0, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: {"):
                    chunks.append(json.loads(line[6:]))
        assert sum(bool(x.get("choices")) for x in chunks) == 32
        assert chunks[-1]["usage"]["completion_tokens"] == 32
        rows.append({"case": "streaming_usage", "result": "PASS"})

        with client.stream("POST", "/v1/completions", json={"prompt": "Count from one to one thousand.",
                "max_tokens": 256, "ignore_eos": True, "stream": True, "temperature": 0}) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    break
        for _ in range(50):
            state = client.get("/health").json()
            if state["active"] == 0 and state["kv_used_blocks"] == 0:
                break
            time.sleep(.1)
        assert state["active"] == 0 and state["kv_used_blocks"] == 0
        rows.append({"case": "cancel_and_release", "result": "PASS"})
    report = {"server": health, "cases": rows,
              "result": "PASS" if all(x["result"] == "PASS" for x in rows) else "FAIL"}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if report["result"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
