from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .config import load_config, validate_config
from .simulator import Simulator
from .command_cost import CommandCostModel
from .presets import ServingFeatures, configure_serving
from .visualization import render_gantt_html, render_gantt_svg


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a behavioral split-LLM serving simulation."
    )
    parser.add_argument("--config", required=True, help="experiment JSON file")
    parser.add_argument(
        "--output-dir", default="outputs/latest", help="directory for JSON results"
    )
    for name in ("concurrency", "num-requests", "warmup-requests"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--measurement-duration-s", type=float)
    parser.add_argument("--preset", choices=("baseline", "optimized"))
    parser.add_argument("--scheduler-backend", choices=("behavioral", "serving"), default="behavioral")
    parser.add_argument("--cost-model", choices=("profile", "roofline", "command"), default="profile")
    parser.add_argument("--serving-host-profile", help="measured CPU post-back profile for serving backend")
    parser.add_argument("--command-sampling", choices=("mean", "empirical"), default="mean")
    parser.add_argument("--cost-seed", type=int, default=0)
    parser.add_argument("--command-profile", help="measured forward-command cost JSON")
    for name in ("stage1", "chunked-prefill", "pipeline", "decode-first", "pd", "chunk-transfer", "control-channel", "pd-admission"):
        parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--control-latency-ms", type=float)
    for name in ("prefill-replicas", "prefill-tp", "decode-tp", "prefill-window", "decode-window", "decode-quota", "chunk-size"):
        parser.add_argument(f"--{name}", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    workload_overrides = {name: getattr(args, name) for name in (
        "concurrency", "num_requests", "warmup_requests", "measurement_duration_s")
        if getattr(args, name) is not None}
    if workload_overrides:
        config = replace(config, workload=replace(config.workload, **workload_overrides))
    features = None
    overrides = {name: getattr(args, name) for name in ServingFeatures.__dataclass_fields__
                 if getattr(args, name, None) is not None}
    if args.preset or overrides:
        features = ServingFeatures.optimized() if args.preset == "optimized" else ServingFeatures()
        features = replace(features, **overrides)
        config = configure_serving(config, features, "profile" if args.cost_model == "command" else args.cost_model)
    elif args.cost_model == "roofline":
        config = replace(config, performance_profile=replace(config.performance_profile, enabled=False))
    if args.command_sampling == "empirical" and (args.cost_model != "command" or args.scheduler_backend != "serving"):
        raise ValueError("empirical command sampling requires command costs and the serving scheduler backend")
    if args.serving_host_profile and args.scheduler_backend != "serving":
        raise ValueError("serving host profile requires serving scheduler backend")
    model = None
    if args.cost_model == "command":
        if not args.command_profile:
            raise ValueError("--cost-model command requires --command-profile")
        model = CommandCostModel.from_file(config, args.command_profile, sampling=args.command_sampling, seed=args.cost_seed)
    validate_config(config)
    if args.scheduler_backend == "serving":
        from .serving_runtime import VirtualServingSimulator
        from .serving_host import ServingHostWork
        host_work = ServingHostWork.from_file(config, args.serving_host_profile) if args.serving_host_profile else None
        simulator = VirtualServingSimulator(config, gpu_cost_model=model, host_work=host_work)
    else:
        simulator = Simulator(config, gpu_cost_model=model)
    result = simulator.run()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "resolved_config.json", asdict(config))
    _write_json(output_dir / "serving_features.json", {
        "preset": args.preset, "scheduler_backend": args.scheduler_backend, "cost_model": args.cost_model, "command_profile": args.command_profile,
        "command_sampling": args.command_sampling, "cost_seed": args.cost_seed, "serving_host_profile": args.serving_host_profile,
        "features": asdict(features) if features else None,
    })
    _write_json(output_dir / "summary.json", result.summary)
    _write_json(
        output_dir / "metrics.json",
        {
            "requests": result.summary["num_requests"],
            "qps": result.summary["observed_request_throughput_qps"],
            "ttft_ms": result.summary["ttft_ms"],
            "tpot_ms": result.summary["tpot_ms"],
        },
    )
    _write_jsonl(output_dir / "requests.jsonl", result.requests)
    _write_jsonl(output_dir / "trace.jsonl", result.trace)
    if args.scheduler_backend == "serving":
        _write_jsonl(output_dir / "serving_trace.jsonl", simulator.serving_trace)
        _write_jsonl(output_dir / "decisions.jsonl", result.summary["scheduling_decisions"])
    if config.simulation.trace_enabled:
        render_gantt_html(result.trace, output_dir / "gantt.html")
        render_gantt_svg(result.trace, output_dir / "gantt.svg")
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
