"""Offline evidence check: complete/chunked prefill, decode and activation bytes.

Only client IDs from the measured request list are checked; the separate warmup
is excluded. This does not substitute for logits correctness verification.
"""
import argparse
import collections
import json
from pathlib import Path


def initial_state():
    return {"steps": 0, "prefills": 0, "prompt_tokens": 0, "emitted": 0, "position": 0}


def check_step(row, state, isl):
    """Support historical traces and explicit chunk traces without counting chunks as tokens."""
    prefill = row["phase"] == "prefill"
    assert row["phase"] in {"prefill", "decode"}, row
    query = row.get("query_len", isl if prefill else 1)
    assert type(query) is int and query > 0, row
    assert row.get("position_start", state["position"]) == state["position"], row
    if prefill:
        assert state["emitted"] == 0 and state["prompt_tokens"] < isl, row
        state["prompt_tokens"] += query
        assert state["prompt_tokens"] <= isl, row
        emits = state["prompt_tokens"] == isl
        state["prefills"] += 1
    else:
        assert state["prompt_tokens"] == isl and query == 1, row
        emits = True
    assert row.get("emits_token", True) == emits, row
    state["emitted"] += int(emits)
    state["position"] += query
    state["steps"] += 1
    assert row["token_idx"] == state["emitted"] - 1, row
    assert row["context_len"] == state["position"], row
    n = query if prefill else row["batch_size"]
    raw_bytes = 2 * n * 2048 * 2
    for key in ("upload_bytes", "download_bytes"):
        assert raw_bytes < row[key] < raw_bytes + 1024 * 1024, row


def check_complete(state, isl, osl):
    assert state["prompt_tokens"] == isl and state["emitted"] == osl, state
    assert state["position"] == isl + osl - 1 and state["prefills"] >= 1, state
    assert state["steps"] == state["prefills"] + osl - 1, state


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
        states = {request["client_request_id"]: initial_state() for request in requests}
        batch_sizes = {}
        with (point / "split_trace.jsonl").open() as file:
            for line in file:
                row = json.loads(line)
                state = states.get(row["client_request_id"])
                if state is None:
                    continue
                check_step(row, state, isl)
                batch_sizes[row["batch_id"]] = row["batch_size"]
        assert len(states) == config["requests"]
        for state in states.values():
            check_complete(state, isl, osl)
        reports.append({"point": str(summary.parent.relative_to(root)), "result": "PASS",
                        "requests": len(states), "isl": isl, "osl": osl,
                        "wan_10gbps_one_way_5ms_verified": True, "profiler_disabled_verified": True,
                        "unique_batch_size_histogram": dict(collections.Counter(batch_sizes.values()))})
    profiles = []
    for path in sorted(root.glob("tp_*/profiling/profile_cases.json")):
        cases = {case["client_request_id"]: case for case in json.loads(path.read_text())["cases"]}
        states = {rid: initial_state() for rid in cases}
        batches = set()
        with (path.parent / "split_trace.jsonl").open() as file:
            for line in file:
                row = json.loads(line)
                rid = row["client_request_id"]
                if rid not in cases:
                    continue
                check_step(row, states[rid], cases[rid]["isl"])
                batches.add(row["batch_id"])
        for rid, case in cases.items():
            check_complete(states[rid], case["isl"], case["osl"])
        profiles.append({"topology": path.parent.parent.name, "result": "PASS", "requests": len(cases),
                         "token_steps": sum(s["emitted"] for s in states.values()),
                         "forward_steps": sum(s["steps"] for s in states.values()), "unique_batches": len(batches)})
    output = {"result": "PASS" if reports else "NO_DATA", "points": len(reports),
              "checked_requests": sum(row["requests"] for row in reports), "reports": reports,
              "profiles": profiles,
              "meaning": "Per-request incremental execution and RPC byte-size consistency, not a logits proof"}
    (root / "trace_verification.json").write_text(json.dumps(output, indent=2))
    print(json.dumps({k: output[k] for k in ("result", "points", "checked_requests")}, indent=2))


if __name__ == "__main__":
    main()
