#!/usr/bin/env python3
"""Local research UI: real subprocess jobs, serving SSE, and Q4 simulation."""

from __future__ import annotations
import argparse, hashlib, json, math, subprocess, sys, threading, time, uuid
import urllib.request, urllib.error
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
RUNS = ROOT / "results/demo"
PYTHON = ROOT / ".venv/bin/python"
JOBS = {}
LOCK = threading.Lock()


def health():
    try:
        owner = json.loads((ROOT / "run/enterprise.pid.json").read_text())
        if (
            Path(f"/proc/{owner['pid']}/stat").read_text().split()[21]
            != owner["start_ticks"]
        ):
            return {"status": "stopped"}
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3) as r:
            return json.load(r)
    except (OSError, ValueError, KeyError, IndexError):
        return {"status": "stopped"}


def validate_sim(payload):
    allowed = {
        "variant",
        "concurrency",
        "workload_mode",
        "arrival_rate_qps",
        "model",
        "hardware",
    }
    if set(payload) - allowed:
        raise ValueError("Unsupported simulation parameter")
    if payload.get("variant") not in ("baseline", "optimized"):
        raise ValueError("Choose baseline or optimized")
    if payload.get("model", "qwen2.5-3b") not in (
        "qwen2.5-3b",
        "qwen3-32b",
        "deepseek-v4-flash",
    ):
        raise ValueError("Unknown model")
    if payload.get("hardware", "a10") not in ("a10", "l20", "h20", "ascend910b"):
        raise ValueError("Unknown hardware")
    if payload.get("workload_mode", "closed_loop") == "open_loop":
        rate = payload.get("arrival_rate_qps")
        if (
            type(rate) not in (int, float)
            or not math.isfinite(rate)
            or not 0.05 <= rate <= 8
        ):
            raise ValueError("Arrival rate must be finite and in 0.05–8 QPS")
        if "concurrency" in payload:
            raise ValueError("Open loop has no client concurrency cap")
    elif payload.get("workload_mode", "closed_loop") == "closed_loop":
        c = payload.get("concurrency")
        if type(c) is not int or not 1 <= c <= 96:
            raise ValueError("Concurrency must be an integer in 1–96")
        if "arrival_rate_qps" in payload:
            raise ValueError("Closed loop does not use arrival rate")
    else:
        raise ValueError("Unknown workload mode")
    return dict(payload)


def launch(kind, payload):
    if kind == "simulate":
        payload = validate_sim(payload)
    elif kind == "encode":
        if (
            not isinstance(payload.get("text"), str)
            or not 1 <= len(payload["text"]) <= 20000
        ):
            raise ValueError("Enter 1–20000 characters")
    elif kind == "recover":
        parent = JOBS.get(payload.get("encode_id", ""))
        if not parent or parent["kind"] != "encode" or parent["status"] != "completed":
            raise ValueError("First complete encoding")
    elif kind != "start":
        raise ValueError("Unknown job")
    if not LOCK.acquire(blocking=False):
        raise RuntimeError("Another experiment is running; wait for it to finish")
    jid = uuid.uuid4().hex
    out = RUNS / jid
    try:
        out.mkdir(parents=True)
        job = {
            "id": jid,
            "kind": kind,
            "status": "running",
            "started_at": time.time(),
            "result": None,
            "error": None,
        }
        JOBS[jid] = job
        (out / "input.json").write_text(json.dumps(payload, ensure_ascii=False))
    except Exception:
        LOCK.release()
        raise

    def worker():
        attempted_start = False
        try:
            if kind == "start":
                h = health()
                if any(
                    h.get(k, 0)
                    for k in ("active", "waiting", "pd_reserving", "pd_releasing")
                ):
                    raise RuntimeError(
                        "Serving has in-flight requests; retry after they drain"
                    )
                if h["status"] != "ready":
                    try:
                        with urllib.request.urlopen(
                            "http://127.0.0.1:8000/health", timeout=2
                        ) as r:
                            foreign = json.load(r)
                    except (OSError, ValueError):
                        foreign = None
                    if foreign:
                        raise RuntimeError(
                            "Port 8000 belongs to another checkout; stop that service from its own directory first"
                        )
                # Fixed command, no arbitrary shell interpolation from the browser.
                command = ["./poc", "up", "--wan"]
            elif kind in ("encode", "recover"):
                if kind == "recover":
                    import shutil

                    for name in ("hidden.npy", "target.json"):
                        shutil.copyfile(RUNS / payload["encode_id"] / name, out / name)
                command = [
                    str(PYTHON),
                    str(ROOT / "Q1/demo_attack.py"),
                    "--model",
                    str(ROOT / "models/qwen"),
                    "--output",
                    str(out),
                    "--phase",
                    kind,
                ]
            else:
                command = [
                    sys.executable,
                    str(DEMO / "simulate.py"),
                    str(out / "input.json"),
                    str(out / "result.json"),
                ]
            (out / "command.json").write_text(json.dumps(command))
            source_files = [
                "demo/server.py",
                "demo/simulate.py",
                "Q1/demo_attack.py",
                "Q1/retrieval.py",
                "scripts/manage.py",
            ]
            if kind == "simulate":
                source_files += [
                    str(p.relative_to(ROOT))
                    for folder in (
                        "split_serving_sim",
                        "models",
                        "hardware",
                        "profiles",
                        "configs",
                    )
                    for p in sorted((ROOT / "Q4" / folder).rglob("*"))
                    if p.is_file() and p.suffix in (".py", ".json")
                ]
            provenance = {
                "command": command,
                "requested_model": payload.get("model") if kind == "simulate" else None,
                "sources_sha256": {
                    name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in source_files
                },
                "model_revision": (
                    (ROOT / "models/qwen/revision.txt").read_text().strip()
                    if kind != "simulate"
                    and (ROOT / "models/qwen/revision.txt").is_file()
                    else None
                ),
            }
            (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
            attempted_start = kind == "start"
            with (out / "log.txt").open("w") as log:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=1800,
                )
            if kind == "start":
                result = health()
                if result["status"] != "ready":
                    raise RuntimeError("Startup exited without a healthy service")
                result = {"health": result, "command": command}
            else:
                result = json.loads(
                    (
                        out
                        / (
                            f"{kind}.json"
                            if kind in ("encode", "recover")
                            else "result.json"
                        )
                    ).read_text()
                )
            job.update(status="completed", result=result)
        except Exception as e:
            job.update(status="failed", error=str(e))
            if attempted_start:
                with (out / "log.txt").open("a") as log:
                    try:
                        subprocess.run(
                            ["./poc", "down"],
                            cwd=ROOT,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=120,
                        )
                    except (OSError, subprocess.SubprocessError) as cleanup:
                        log.write("Cleanup failed: " + str(cleanup))
        finally:
            job["finished_at"] = time.time()
            (out / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2))
            LOCK.release()

    threading.Thread(target=worker, daemon=True).start()
    return dict(job)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(DEMO), **kw)

    def send_json(self, status, payload):
        b = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            return self.send_json(200, health())
        if path == "/api/evidence":
            data = {
                p.stem: json.loads(p.read_text())
                for p in (DEMO / "evidence").glob("*.json")
            }
            data["accuracy"] = data["accuracy-tp1-ranking"]
            data["accuracy_manifest"] = data["accuracy-tp1-manifest"]
            return self.send_json(200, data)
        if path.startswith("/api/jobs/"):
            jid = path.rsplit("/", 1)[-1]
            j = JOBS.get(jid)
            if not j:
                return self.send_json(404, {"error": "Unknown job"})
            log = RUNS / jid / "log.txt"
            return self.send_json(
                200,
                {
                    **j,
                    "log": (
                        log.read_text(errors="replace")[-30000:] if log.exists() else ""
                    ),
                },
            )
        if path.startswith("/api/artifacts/"):
            pieces = path.split("/")
            if len(pieces) != 5:
                return self.send_json(404, {"error": "Not found"})
            jid, name = pieces[-2:]
            if jid not in JOBS or name not in (
                "hidden.npy",
                "encode.json",
                "recover.json",
                "result.json",
                "log.txt",
                "job.json",
                "provenance.json",
            ):
                return self.send_json(404, {"error": "Not found"})
            file = RUNS / jid / name
            if not file.is_file():
                return self.send_json(404, {"error": "Not ready"})
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.send_header("Content-Length", str(file.stat().st_size))
            self.end_headers()
            with file.open("rb") as f:
                while chunk := f.read(65536):
                    self.wfile.write(chunk)
            return
        return super().do_GET()

    def do_POST(self):
        try:
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                return self.send_json(403, {"error": "Same-origin requests only"})
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Content-Type must be application/json")
            n = int(self.headers.get("Content-Length", "0"))
            if not 0 < n <= 100000:
                raise ValueError("Invalid body size")
            p = json.loads(self.rfile.read(n))
            if not isinstance(p, dict):
                raise ValueError("Expected JSON object")
            routes = {
                "/api/service/start": "start",
                "/api/security/encode": "encode",
                "/api/security/recover": "recover",
                "/api/simulate": "simulate",
            }
            if self.path in routes:
                return self.send_json(202, launch(routes[self.path], p))
            if self.path == "/api/chat":
                return self.chat(p)
            return self.send_json(404, {"error": "Unknown endpoint"})
        except (ValueError, TypeError) as e:
            self.send_json(400, {"error": str(e)})
        except RuntimeError as e:
            self.send_json(409, {"error": str(e)})
        except (OSError, subprocess.SubprocessError) as e:
            self.send_json(503, {"error": str(e)})

    def chat(self, p):
        messages = p.get("messages")
        if (
            not isinstance(messages, list)
            or not 1 <= len(messages) <= 40
            or any(
                not isinstance(m, dict)
                or m.get("role") not in ("user", "assistant", "system")
                or not isinstance(m.get("content"), str)
                for m in messages
            )
        ):
            raise ValueError("Invalid messages")
        if health()["status"] != "ready":
            raise RuntimeError("Service is not ready")
        if not LOCK.acquire(blocking=False):
            raise RuntimeError("Experiment in progress; wait before chatting")
        try:
            return self.stream_chat(messages)
        finally:
            LOCK.release()

    def stream_chat(self, messages):
        body = json.dumps(
            {
                "model": "split-qwen",
                "messages": messages,
                "temperature": 0,
                "max_tokens": 128,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        ).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:8000/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as upstream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                for line in upstream:
                    self.wfile.write(line)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except OSError as e:
                self.wfile.write(
                    ("data: " + json.dumps({"error": str(e)}) + "\n\n").encode()
                )
                self.wfile.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8088)
    a = p.parse_args()
    print(f"Live research demo: http://{a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
