# Q1: reproducible hidden-state token recovery

This standalone directory addresses issue#3. It uses the exact Qwen2.5-3B-Instruct revision and4K prompt from the serving experiments, without importing or modifying serving code. See [the experiment report](../docs/hidden-state-security.md).

Create an isolated Python environment and install `Q1/requirements.txt`. Download `Qwen/Qwen2.5-3B-Instruct` at revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`; the local snapshot must contain `revision.txt` with that SHA, as the existing serving snapshot does. Weights are not redistributed. Approximate needs: one CUDA GPU with20GB free for extraction/training and several GB disk for cached activations. Runs here used A10; timings are not inference performance measurements.

From repository root, set a model path and run:

```bash
MODEL_DIR=/path/to/pinned/qwen
python Q1/prepare_public.py --model "$MODEL_DIR"
python Q1/extract.py --model "$MODEL_DIR"
python Q1/embedding_attack.py --model "$MODEL_DIR"
python Q1/train_decoder.py --model "$MODEL_DIR" --seed 42
python Q1/train_decoder.py --model "$MODEL_DIR" --seed 43
python Q1/summarize.py
python -m unittest discover -s Q1/tests
```

`prepare_public.py` downloads pinned WikiText-2 raw parquet files and stores them under ignored `cache/`. Training uses the first65,536 token IDs from train, validation/test each8,192, packed into4096-token causal contexts; exact source rows and checksums are in `data/public_manifest.json`. The source model is frozen. Both seeds use a2048→512→151936 GELU MLP,4 epochs, batch256, AdamW lr0.001/weight_decay0.01. Select the checkpoint by public-validation accuracy only.

`extract.py` hooks decoder block outputs BEFORE final RMSNorm. It uses FP16 Transformers SDPA on one GPU with the same serving weights, not bitwise captures from the TP2 vLLM wire. Current target extraction is causal teacher forcing over the existing4096-token prompt plus first78 output tokens. The79th output is EOS and is not fed back by the real request. The separate embedding-only control tests all79 token embeddings and must not be read as an observed transmission of the final EOS.

`retrieval.py` searches the full151936-row public embedding matrix with FP32 cosine similarity. It does not restrict candidates to the input's token vocabulary. `metrics.py` measures exact absolute-position recovery, whole-sequence exact match, distinct-token macro average, non-special and training-seen subsets. `summarize.py` reports both seeds and the current/public held-out evaluations.

Results contain token predictions and recovered target text, aggregate CSV/JSON, learning curves, source/environment manifests and plots. `short_context_pilot/` retains the initial256-context/seed42 run; it is not pooled into the final result. It exposed a training/target length mismatch, so the final experiment controls that factor at4K. Current target data were never included in training or checkpoint selection. Two-seed ranges are descriptive, not confidence intervals over user documents.

Model weights, public parquet source copies, activation arrays and decoder checkpoints are ignored. Checkpoints can be regenerated; hashes of retained local checkpoints are provided for this run. Source-data attribution and redistribution terms are in `DATA_LICENSE.md`.
