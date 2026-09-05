"""Full vocabulary teacher-forcing comparison, plus free-running validation."""
import argparse
import io
import json
from pathlib import Path
import numpy as np
import httpx


def compare(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    diff = a - b
    top_a, top_b = np.argsort(a)[-5:], np.argsort(b)[-5:]
    result = {"mae": float(np.abs(diff).mean()), "rmse": float(np.sqrt(np.mean(diff ** 2))),
              "max_abs_error": float(np.abs(diff).max()),
              "cosine_similarity": float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))),
              "top1_agreement": bool(a.argmax() == b.argmax()),
              "top5_agreement": len(set(top_a) & set(top_b)) / 5}
    result["result"] = "PASS" if (result["mae"] <= 1e-2 and result["rmse"] <= 2e-2 and
        result["max_abs_error"] <= .1 and result["cosine_similarity"] >= .9999 and result["top1_agreement"]) else "FAIL"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="results/reference")
    parser.add_argument("--output", default="results/correctness")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--steps", type=int, default=33)
    parser.add_argument("--cross-tp-observation", action="store_true",
                        help="Record cross-TP differences, not same-TP precision acceptance")
    args = parser.parse_args()
    root, output = Path(args.reference), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    free_reports = []
    with httpx.Client(base_url=args.url, timeout=300, trust_env=False) as client:
        health = client.get("/health").json()
        reference_config = json.loads((root / "config.json").read_text())
        if args.cross_tp_observation and health["tp"] == health["cloud_tp"] == reference_config["tp"]:
            raise RuntimeError("Cross-TP mode requires genuinely different TP configurations")
        for path in sorted(root.glob("len_*/repeat_0/reference.npz")):
            ref = np.load(path)
            prompt, tokens = ref["prompt_ids"].tolist(), ref["tokens"][:args.steps].tolist()
            length = len(prompt)
            repeated_path = path.parent.parent / "repeat_1/reference.npz"
            if repeated_path.exists():
                repeated = np.load(repeated_path)
                stable = [compare(a, b) for a, b in zip(ref["logits"], repeated["logits"])]
                if any(r["result"] != "PASS" for r in stable) or np.mean([r["top5_agreement"] for r in stable]) < .99:
                    raise RuntimeError("Native repeatability FAIL; do not relax split thresholds")
            response = client.post("/debug/teacher_force", json={"prompt_ids": prompt, "forced_tokens": tokens})
            response.raise_for_status()
            split = np.load(io.BytesIO(response.content))
            if split["logits"].shape != (len(tokens), 151936) or not np.isfinite(split["logits"]).all():
                raise RuntimeError("Incomplete or non-finite split logits")
            (output / f"split_{health['split'].replace(':', '_')}_{length}.npz").write_bytes(response.content)
            for step, (native, actual) in enumerate(zip(ref["logits"][:len(tokens)], split["logits"])):
                report = {"split": health["split"], "prompt_tokens": length,
                          "phase": "prefill" if step == 0 else "decode", "decode_step": step,
                          **compare(native, actual)}
                reports.append(report)
            # Free-running is a separate request, using the normal public API.
            response = client.post("/v1/completions", json={"model": "split-qwen", "prompt": prompt,
                "temperature": 0, "max_tokens": 32, "ignore_eos": True})
            response.raise_for_status()
            # Exact token IDs are verified below through an additional diagnostic
            # run without teacher forcing; text alone is not used as evidence.
            response = client.post("/debug/greedy", json={"prompt_ids": prompt, "steps": 32})
            response.raise_for_status()
            free = response.json()["tokens"]
            expected_free = ref["tokens"][:32].tolist()
            free_reports.append({"prompt_tokens": length, "expected_tokens": expected_free,
                                 "actual_tokens": free, "exact_match": free == expected_free})
            print(f"Checked split={health['split']} length={length} steps={len(tokens)}", flush=True)
    if not reports:
        raise RuntimeError("No reference cases")
    top5 = float(np.mean([r["top5_agreement"] for r in reports]))
    passed = all(r["result"] == "PASS" for r in reports) and top5 >= .99 and all(r["exact_match"] for r in free_reports)
    with (output / "comparison.jsonl").open("w") as f:
        for row in reports:
            f.write(json.dumps(row) + "\n")
    summary = {"result": "CROSS_TP_OBSERVATION" if args.cross_tp_observation else "PASS" if passed else "FAIL",
               "same_tp_thresholds_met": passed, "reference_tp": reference_config["tp"],
               "free_running": free_reports, "comparisons": len(reports),
               "max_mae": max(r["mae"] for r in reports),
               "max_abs_error": max(r["max_abs_error"] for r in reports),
               "min_cosine": min(r["cosine_similarity"] for r in reports),
               "top5_agreement": top5, "top5_definition": "mean unordered top-5 intersection / 5",
               "configuration": vars(args), "server": health}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not passed and not args.cross_tp_observation:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
