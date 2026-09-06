from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config
from .simulator import Simulator
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    result = Simulator(config).run()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    if config.simulation.trace_enabled:
        render_gantt_html(result.trace, output_dir / "gantt.html")
        render_gantt_svg(result.trace, output_dir / "gantt.svg")
    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
