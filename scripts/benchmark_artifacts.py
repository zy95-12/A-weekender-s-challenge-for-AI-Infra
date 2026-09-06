"""Keep measured requests separate from warmup/unattributed concurrent traffic."""
import json


def export_trace(source_path, offset, folder, prefix, requests):
    measured_ids = {prefix + str(i) for i in range(requests)}
    counts = {"measured_rows": 0, "excluded_rows": 0}
    with source_path.open("rb") as source, (folder / "split_trace.jsonl").open("w") as measured, \
            (folder / "excluded_trace.jsonl").open("w") as excluded:
        source.seek(offset)
        for line in source:
            row = json.loads(line)
            included = row.get("client_request_id") in measured_ids
            row.update(experiment_id=prefix, is_measured=included,
                       is_warmup=False if included else None,
                       measurement_role="measured" if included else "warmup_or_unattributed")
            (measured if included else excluded).write(json.dumps(row) + "\n")
            counts["measured_rows" if included else "excluded_rows"] += 1
    (folder / "trace_scope.json").write_text(json.dumps({**counts, "experiment_id": prefix,
        "policy": "Exact measured request IDs only in split_trace.jsonl. Excluded rows may be client warmup or other traffic; is_warmup=null means unknown.",
        "telemetry_scope": "Entire vLLM client process, including initial warmup; not a measured-only window."}, indent=2))
