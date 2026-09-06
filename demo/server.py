#!/usr/bin/env python3
"""Dependency-free web server for the security inference research demo."""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import threading
from dataclasses import replace
from functools import lru_cache, partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
Q4_DIR = REPO_ROOT / "Q4"
sys.path.insert(0, str(Q4_DIR))

from split_serving_sim.config import load_config  # noqa: E402
from split_serving_sim.simulator import Simulator  # noqa: E402

SUPPORTED = {"model": "Qwen3-32B", "hardware": "A10", "cards": 9, "tp_degree": 4}
SIMULATION_LOCK = threading.Lock()


def apply_memory_limit(limit_mib: int) -> None:
    """Leave most of a 4 GiB host available to SSH and other processes."""
    if limit_mib <= 0:
        return
    limit = limit_mib * 1024 * 1024
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new_hard = limit if hard in (-1, resource.RLIM_INFINITY) else min(hard, limit)
    resource.setrlimit(resource.RLIMIT_AS, (min(limit, new_hard), new_hard))


def validate_payload(payload: dict[str, Any]) -> tuple[int, int, int]:
    for key, expected in SUPPORTED.items():
        if payload.get(key) != expected:
            raise NotImplementedError(
                "该组合仅预留接口；当前真实后端支持 Qwen3-32B / A10 / 9 卡 / 云侧 TP4。"
            )
    concurrency = int(payload.get("concurrency", 0))
    input_tokens = int(payload.get("input_tokens", 0))
    output_tokens = int(payload.get("output_tokens", 0))
    if concurrency not in {1, 2, 4, 8}:
        raise ValueError("concurrency 必须是 1、2、4 或 8")
    if not 32 <= input_tokens <= 2048:
        raise ValueError("input_tokens 必须在 32–2048 之间")
    if not 2 <= output_tokens <= 64:
        raise ValueError("output_tokens 必须在 2–64 之间")
    return concurrency, input_tokens, output_tokens


@lru_cache(maxsize=32)
def simulate_point(concurrency: int, input_tokens: int, output_tokens: int) -> dict[str, Any]:
    base = load_config(Q4_DIR / "configs" / "example.json")
    # A10 FP16 nominal preset. This is intentionally marked uncalibrated in the UI.
    hardware = {
        name: replace(
            item,
            peak_flops_tflops=125.0,
            hbm_bandwidth_gb_s=600.0,
            memory_gb=24.0,
            compute_efficiency=0.55,
            memory_efficiency=0.65,
        )
        for name, item in base.hardware.items()
    }
    config = replace(
        base,
        hardware=hardware,
        workload=replace(
            base.workload,
            mode="closed_loop",
            num_requests=128,
            concurrency=concurrency,
            warmup_requests=concurrency,
            measurement_duration_s=5.0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
        static_policy=replace(base.static_policy, max_batch_size=8),
        scheduler=replace(base.scheduler, max_num_seqs=8),
        simulation=replace(
            base.simulation,
            trace_enabled=False,
            max_trace_records=0,
            max_detailed_trace_records=0,
            max_time_s=120.0,
        ),
    )
    with SIMULATION_LOCK:
        summary = Simulator(config).run().summary
    point = {
        "concurrency": concurrency,
        "qps": float(summary["observed_request_throughput_qps"]),
        "ttft_mean_ms": float(summary["ttft_ms"]["mean"]),
        "ttft_p99_ms": float(summary["ttft_ms"]["p99"]),
        "tpot_mean_ms": float(summary["tpot_ms"]["mean"]),
        "tpot_p99_ms": float(summary["tpot_ms"]["p99"]),
        "measured_requests": int(summary["num_requests"]),
    }
    gc.collect()
    return point


class DemoHandler(SimpleHTTPRequestHandler):
    server_version = "SplitShieldDemo/1.0"

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/simulate":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 16_384:
                raise ValueError("请求体大小不合法")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("请求体必须是 JSON object")
            concurrency, input_tokens, output_tokens = validate_payload(payload)
            point = simulate_point(concurrency, input_tokens, output_tokens)
            self.send_json(HTTPStatus.OK, {"status": "ok", "engine": "Q4", "point": point})
        except NotImplementedError as exc:
            self.send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"status": "not_implemented", "message": str(exc)})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"status": "invalid_request", "message": str(exc)})
        except Exception as exc:  # keep the demo response useful without leaking a traceback
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"status": "error", "message": f"仿真失败：{exc}"})

    def send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the SplitShield HTML demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--memory-limit-mib", type=int, default=768)
    args = parser.parse_args()
    apply_memory_limit(args.memory_limit_mib)
    handler = partial(DemoHandler, directory=str(DEMO_DIR))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"SplitShield demo: http://{args.host}:{args.port}", flush=True)
    print(f"PID {os.getpid()} · address-space limit {args.memory_limit_mib} MiB", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
