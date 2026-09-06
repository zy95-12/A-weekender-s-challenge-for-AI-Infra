# Live attack adapter

`retrieval.py`, `metrics.py`, `check_model.py`, and the original `embedding_attack.py`
are taken unchanged from PR #15, commit d880c281e9a0ddfc997689867e40cea3fc97c638.
`demo_attack.py` exposes its embedding-only experiment as two real operations:
encode input to an FP16 boundary tensor, then recover tokens using the public
embedding table. This is not cryptographic encryption/decryption. The attack is
CPU-based to avoid competing with serving for GPU memory. Ground-truth IDs are
used only after recovery for scoring. It does not claim to attack the 4-layer
serving boundary or reproduce the depth-dependent trained MLP experiments.

The original fixed 4K prompt/reference are retained for the serving benchmark.
See DATA_LICENSE.md and PR #15 for their provenance. The original batch attack
also needs the PR #15 data/cache directories; the demo adapter needs only the
pinned model and its per-run input.json.
