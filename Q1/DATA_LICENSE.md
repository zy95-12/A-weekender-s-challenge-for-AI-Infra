# Data provenance

`data/current_prompt.json` and `data/current_reference.json` reproduce the constructed library-document extraction workload already used in this repository's serving measurements. They are synthetic test material, not customer records.

Auxiliary training/validation/test material comes from [Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext), WikiText-2 raw configuration, derived from Wikipedia by Stephen Merity and collaborators. The dataset card lists CC-BY-SA3.0 and GFDL. Exact pinned revision, source-row indices and file/token checksums are in `data/public_manifest.json`; the original text and article metadata can be retrieved from that snapshot. Wikipedia article authors and revision histories remain the underlying attribution source.

Any redistributed tokenized WikiText excerpts in prediction JSONs are derived dataset material and retain [CC-BY-SA3.0](https://creativecommons.org/licenses/by-sa/3.0/) attribution/share-alike terms; their modification here is tokenization and pairing with experimental model predictions. Public source parquet files are downloaded by the reproduction script and not committed.

The model is [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), whose model card specifies the Qwen research license. Model weights are not included in this PR.
