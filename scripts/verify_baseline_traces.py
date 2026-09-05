"""Offline evidence check: one prefill, incremental decode and activation bytes.

Only client IDs from the measured request list are checked; the separate warmup
is excluded. This does not substitute for logits correctness verification.
"""
import argparse
import collections
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root)
    reports = []
    for summary in sorted(root.glob("tp_*/isl_*_r_*/summary.json")):
        point = summary.parent / "qps_inf"
        config = json.loads((point / "config.json").read_text())
        deployment = config["deployment"]
        assert deployment["wan"] and not deployment["profile"] and not deployment.get("phase_profile", False), point
        _, enterprise_tp, cloud_tp = summary.parent.parent.name.split("_")
        assert (deployment["enterprise_tp"], deployment["cloud_tp"]) == (int(enterprise_tp), int(cloud_tp)), point
        network = json.loads((point / "network.json").read_text())
        for namespace in ("split-enterprise", "split-cloud"):
            saved = network[namespace]
            assert saved["exit_code"] == 0, (point, namespace)
            qdiscs = {entry["kind"]: entry for entry in json.loads(saved["stdout"])}
            assert qdiscs["tbf"]["options"]["rate"] == 1_250_000_000, (point, namespace)
            assert qdiscs["netem"]["options"]["delay"]["delay"] == .005, (point, namespace)
        isl, osl = config["input"], config["output_tokens"]
        requests = [json.loads(line) for line in (point / "requests.jsonl").read_text().splitlines()]
        states = {request["client_request_id"]: {"steps": 0, "prefills": 0} for request in requests}
        batch_sizes = {}
        with (point / "split_trace.jsonl").open() as file:
            for line in file:
                row = json.loads(line)
                state = states.get(row["client_request_id"])
                if state is None:
                    continue
                step = state["steps"]
                assert row["token_idx"] == step and row["context_len"] == isl + step, (point, row)
                assert row["phase"] == ("prefill" if step == 0 else "decode"), (point, row)
                state["prefills"] += row["phase"] == "prefill"
                state["steps"] += 1
                n = isl if step == 0 else row["batch_size"]
                raw_bytes = 2 * n * 2048 * 2  # hidden + residual, FP16
                assert raw_bytes < row["upload_bytes"] < raw_bytes + 1024 * 1024
                assert raw_bytes < row["download_bytes"] < raw_bytes + 1024 * 1024
                batch_sizes[row["batch_id"]] = row["batch_size"]
        assert len(states) == config["requests"]
        assert all(state == {"steps": osl, "prefills": 1} for state in states.values()), point
        reports.append({"point": str(summary.parent.relative_to(root)), "result": "PASS",
                        "requests": len(states), "isl": isl, "osl": osl,
                        "wan_10gbps_one_way_5ms_verified": True, "profiler_disabled_verified": True,
                        "unique_batch_size_histogram": dict(collections.Counter(batch_sizes.values()))})
    profiles = []
    for path in sorted(root.glob("tp_*/profiling/profile_cases.json")):
        cases = {case["client_request_id"]: case for case in json.loads(path.read_text())["cases"]}
        steps = collections.Counter()
        batches = set()
        with (path.parent / "split_trace.jsonl").open() as file:
            for line in file:
                row = json.loads(line)
                rid = row["client_request_id"]
                if rid not in cases:
                    continue
                case, step = cases[rid], steps[rid]
                assert row["token_idx"] == step and row["context_len"] == case["isl"] + step, path
                assert row["phase"] == ("prefill" if step == 0 else "decode"), path
                steps[rid] += 1
                batches.add(row["batch_id"])
        assert all(steps[rid] == case["osl"] for rid, case in cases.items()), path
        profiles.append({"topology": path.parent.parent.name, "result": "PASS", "requests": len(cases),
                         "token_steps": sum(steps.values()), "unique_batches": len(batches)})
    output = {"result": "PASS" if reports else "NO_DATA", "points": len(reports),
              "checked_requests": sum(row["requests"] for row in reports), "reports": reports,
              "profiles": profiles,
              "meaning": "Per-request incremental execution and RPC byte-size consistency, not a logits proof"}
    (root / "trace_verification.json").write_text(json.dumps(output, indent=2))
    print(json.dumps({k: output[k] for k in ("result", "points", "checked_requests")}, indent=2))


if __name__ == "__main__":
    main()
