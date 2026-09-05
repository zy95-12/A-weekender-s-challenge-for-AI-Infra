"""Version-pinned vLLM model runner with local paged KV and independent TP.

The scheduler lives in server.py. Both sides run actual vLLM decoder layers,
FlashAttention and vocabulary/row/column parallel kernels. Only owned weights
are instantiated. Cloud has neither an embedding nor a language-model head.
"""
import contextlib
import glob
import json
import os
import time
import traceback
import threading
from datetime import timedelta

import numpy as np
import torch
import torch.multiprocessing as mp

from split_poc import SPLITS, WEIGHT_SHA256
from split_poc.wire import pack, unpack


class KVPool:
    def __init__(self, blocks, block_size=16):
        self.free = list(range(blocks))
        self.requests = {}
        self.block_size = block_size

    def prepare(self, items):
        # Validate the entire batch before changing any state.
        need = 0
        seen = set()
        for item in items:
            rid, pos, n = item["request_id"], item["position"], item["query_len"]
            if rid in seen:
                raise ValueError("Duplicate request in batch")
            seen.add(rid)
            old = self.requests.get(rid, {"length": 0, "blocks": []})
            if pos != old["length"] or n < 1:
                raise ValueError(f"KV position mismatch: {rid} expected={old['length']} got={pos}")
            need += max(0, (pos + n + 15) // 16 - len(old["blocks"]))
        if need > len(self.free):
            raise ValueError("KV capacity exhausted")
        slots, tables = [], []
        for item in items:
            rid, pos, n = item["request_id"], item["position"], item["query_len"]
            state = self.requests.setdefault(rid, {"length": 0, "blocks": []})
            while len(state["blocks"]) < (pos + n + 15) // 16:
                state["blocks"].append(self.free.pop())
            slots.extend(state["blocks"][p // 16] * 16 + p % 16 for p in range(pos, pos + n))
            tables.append(state["blocks"][:])
        return slots, tables

    def commit(self, items):
        for item in items:
            self.requests[item["request_id"]]["length"] += item["query_len"]

    def release(self, ids):
        for rid in ids:
            state = self.requests.pop(rid, None)
            if state:
                self.free.extend(state["blocks"])


class PartialModel(torch.nn.Module):
    def __init__(self, config, role, split):
        super().__init__()
        from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer
        from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        front, middle, back = SPLITS[split]
        self.front, self.end = front, front + middle
        self.role = role
        hf = config.model_config.hf_config
        if (hf.num_hidden_layers, hf.hidden_size, hf.num_attention_heads, hf.num_key_value_heads) != (36, 2048, 16, 2):
            raise ValueError("This runner supports only Qwen2.5-3B-Instruct")
        owned = (list(range(front)) + list(range(self.end, 36))
                 if role == "enterprise" else list(range(front, self.end)))
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleDict({str(i): Qwen2DecoderLayer(
            hf, cache_config=config.cache_config, prefix=f"model.layers.{i}") for i in owned})
        if role == "enterprise":
            self.model.embed_tokens = VocabParallelEmbedding(hf.vocab_size, hf.hidden_size,
                                                            prefix="model.embed_tokens")
            self.model.norm = RMSNorm(hf.hidden_size, eps=hf.rms_norm_eps)
            self.logits_processor = LogitsProcessor(hf.vocab_size)

    def load_owned_weights(self, model_dir):
        from safetensors import safe_open
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader
        params = dict(self.named_parameters())
        loaded = set()
        mapping = [("q_proj", "qkv_proj", "q"), ("k_proj", "qkv_proj", "k"),
                   ("v_proj", "qkv_proj", "v"), ("gate_proj", "gate_up_proj", 0),
                   ("up_proj", "gate_up_proj", 1)]
        for filename in sorted(WEIGHT_SHA256):
            path = os.path.join(model_dir, filename)
            with safe_open(path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    dest, shard = name, None
                    for source, target, sid in mapping:
                        if f".{source}." in name:
                            dest, shard = name.replace(f".{source}.", f".{target}."), sid
                            break
                    if dest not in params:
                        continue
                    param = params[dest]
                    weight = f.get_tensor(name)
                    loader = getattr(param, "weight_loader", default_weight_loader)
                    if shard is None:
                        loader(param, weight)
                    else:
                        loader(param, weight, shard)
                    loaded.add((dest, shard))
        for name in params:
            shards = [s for n, s in loaded if n == name]
            expected = 3 if ".qkv_proj." in name else 2 if ".gate_up_proj." in name else 1
            if len(shards) != expected:
                raise RuntimeError(f"Incomplete weight: {name}, shards={shards}")

    def layers(self, positions, hidden, residual, indices):
        for i in indices:
            hidden, residual = self.model.layers[str(i)](positions, hidden, residual)
        return hidden, residual


class Runner:
    def __init__(self, rank, args):
        from vllm.engine.arg_utils import EngineArgs
        from vllm.config import set_current_vllm_config
        from vllm.distributed import init_distributed_environment, initialize_model_parallel
        from vllm.distributed.parallel_state import set_custom_all_reduce
        self.rank, self.args = rank, args
        torch.set_num_threads(4)
        torch.cuda.set_device(rank)
        torch.set_default_dtype(torch.float16)
        self.config = EngineArgs(model=args["model"], dtype="half", tensor_parallel_size=args["tp"],
            enforce_eager=True, max_model_len=16384, enable_prefix_caching=False,
            disable_custom_all_reduce=True, block_size=16,
            max_num_batched_tokens=16384, max_num_seqs=16).create_engine_config()
        self.context = set_current_vllm_config(self.config)
        self.context.__enter__()
        set_custom_all_reduce(False)
        init_distributed_environment(world_size=args["tp"], rank=rank, local_rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{args['dist_port']}",
            backend="nccl", timeout=timedelta(seconds=120))
        initialize_model_parallel(tensor_model_parallel_size=args["tp"])
        with torch.device(f"cuda:{rank}"):
            self.model = PartialModel(self.config, args["role"], args["split"])
        self.model.load_owned_weights(args["model"])
        self.model.eval()
        self.pool = KVPool(args["kv_blocks"])
        for layer in self.model.model.layers.values():
            attn = layer.self_attn.attn
            shape = attn.attn_backend.get_kv_cache_shape(args["kv_blocks"], 16,
                                                        attn.num_kv_heads, attn.head_size)
            attn.kv_cache = [torch.zeros(shape, device="cuda", dtype=torch.float16)]
        if rank == 0 and args["role"] == "enterprise":
            import httpx
            self.http = httpx.Client(base_url=args["cloud"], timeout=httpx.Timeout(30, connect=3), trust_env=False)
        self.trace = []

    @contextlib.contextmanager
    def timed(self, name, timings):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.nvtx.range_push(name)
        start.record()
        try:
            yield
        finally:
            end.record()
            end.synchronize()
            timings[name + "_ms"] = start.elapsed_time(end)
            torch.cuda.nvtx.range_pop()

    def metadata(self, items, phase):
        from vllm.attention.backends.flash_attn import FlashAttentionMetadata
        slots, tables = self.pool.prepare(items)
        lengths = [x["position"] + x["query_len"] for x in items]
        queries = [x["query_len"] for x in items]
        max_blocks = max(map(len, tables))
        tensor = lambda x, dtype=torch.int32: torch.tensor(x, dtype=dtype, device="cuda")
        prefill = phase == "prefill"
        return FlashAttentionMetadata(
            num_prefills=len(items) if prefill else 0,
            num_prefill_tokens=sum(queries) if prefill else 0,
            num_decode_tokens=0 if prefill else sum(queries),
            slot_mapping=tensor(slots, torch.int64), multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False, seq_lens=lengths,
            seq_lens_tensor=tensor(lengths), max_prefill_seq_len=max(lengths) if prefill else 0,
            max_decode_seq_len=0 if prefill else max(lengths),
            context_lens_tensor=tensor([x["position"] for x in items]),
            block_tables=(torch.empty((len(items), 0), dtype=torch.int32, device="cuda")
                          if prefill else tensor([x + [0] * (max_blocks - len(x)) for x in tables])),
            use_cuda_graph=False, max_query_len=max(queries), max_decode_query_len=1,
            query_start_loc=tensor([0] + np.cumsum(queries).tolist()),
            seq_start_loc=tensor([0] + np.cumsum(lengths).tolist()))

    def broadcast_hidden(self, arrays, n):
        from vllm.distributed import get_tp_group
        outputs = []
        for i in range(2):
            value = (torch.from_numpy(arrays[i]).to(device="cuda") if self.rank == 0
                     else torch.empty((n, 2048), dtype=torch.float16, device="cuda"))
            get_tp_group().broadcast(value, src=0)
            outputs.append(value)
        return outputs

    @torch.inference_mode()
    def execute(self, command, arrays=None):
        from vllm.forward_context import set_forward_context
        from vllm.distributed import get_tp_group
        if command["op"] in {"profile_start", "profile_stop"}:
            if command["op"] == "profile_start":
                torch.cuda.profiler.start()
            else:
                torch.cuda.profiler.stop()
            return {"ok": True}
        if command["op"] == "release":
            self.pool.release(command["ids"])
            return {"ok": True, "kv_used_blocks": self.args["kv_blocks"] - len(self.pool.free)}
        items, phase = command["items"], command["phase"]
        if phase == "decode" and any(item["request_id"] not in self.pool.requests for item in items):
            raise ValueError("Decode without a live prefill cache")
        meta = self.metadata(items, phase)
        positions = torch.tensor([p for x in items for p in range(x["position"], x["position"] + x["query_len"])],
                                 device="cuda", dtype=torch.int64)
        n = len(positions)
        timings = {}
        with set_forward_context(meta, self.config), torch.cuda.nvtx.range(
                f"batch={command['batch_id']} phase={phase} requests=" + ",".join(x["request_id"] for x in items)):
            if self.args["role"] == "cloud":
                with self.timed("cloud_receive", timings):
                    hidden, residual = self.broadcast_hidden(arrays, n)
                with self.timed(f"cloud_middle_{phase}", timings):
                    hidden, residual = self.model.layers(positions, hidden, residual,
                                                         range(self.model.front, self.model.end))
                self.pool.commit(items)
                if self.rank == 0:
                    t = time.perf_counter()
                    output = [hidden.cpu().numpy(), residual.cpu().numpy()]
                    timings["cloud_d2h_ms"] = (time.perf_counter() - t) * 1000
                    timings["cloud_kv_used_blocks"] = self.args["kv_blocks"] - len(self.pool.free)
                    return {"meta": {"batch_id": command["batch_id"], "timings": timings}, "arrays": output}
            else:
                with self.timed(f"enterprise_front_{phase}", timings):
                    ids = torch.tensor(command["token_ids"], device="cuda", dtype=torch.int64)
                    hidden = self.model.model.embed_tokens(ids)
                    hidden, residual = self.model.layers(positions, hidden, None, range(self.model.front))
                remote, error = None, None
                if self.rank == 0:
                    try:
                        start = time.perf_counter()
                        payload = pack({k: command[k] for k in ("op", "items", "phase", "batch_id")},
                                       [hidden.cpu().numpy(), residual.cpu().numpy()])
                        timings["enterprise_stage_pack_ms"] = (time.perf_counter() - start) * 1000
                        start = time.perf_counter()
                        sent_ns = time.perf_counter_ns()
                        response = self.http.post("/forward", content=payload,
                            headers={"content-type": "application/octet-stream"})
                        response.raise_for_status()
                        received_ns = time.perf_counter_ns()
                        timings["rpc_wall_ms"] = (time.perf_counter() - start) * 1000
                        timings["upload_bytes"], timings["download_bytes"] = len(payload), len(response.content)
                        remote_meta, remote = unpack(response.content)
                        if remote_meta["batch_id"] != command["batch_id"]:
                            raise ValueError("Cloud batch mismatch")
                        timings.update(remote_meta["timings"])
                        # Both namespaces use the same host monotonic clock.
                        # These path intervals include HTTP/CPU overhead; they
                        # are not estimates of pure WAN propagation delay.
                        timings["upload_ms"] = (timings["cloud_received_ns"] - sent_ns) / 1e6
                        timings["download_ms"] = (received_ns - timings["cloud_send_ns"]) / 1e6
                        timings["enterprise_send_ns"] = sent_ns
                        timings["enterprise_received_ns"] = received_ns
                    except Exception as exc:
                        error = str(exc)
                # Propagate failure before the next collective to avoid rank-1 deadlock.
                status = get_tp_group().broadcast_object(error, src=0)
                if status:
                    raise RuntimeError(f"Cloud execution failed: {status}")
                with self.timed("enterprise_receive", timings):
                    hidden, residual = self.broadcast_hidden(remote, n)
                with self.timed(f"enterprise_back_{phase}", timings):
                    hidden, residual = self.model.layers(positions, hidden, residual, range(self.model.end, 36))
                    hidden, _ = self.model.model.norm(hidden, residual)
                    indices = torch.tensor(np.cumsum([x["query_len"] for x in items]) - 1, device="cuda")
                    logits = self.model.logits_processor(self.model.model.embed_tokens,
                                                        hidden.index_select(0, indices), None)
                self.pool.commit(items)
                if self.rank == 0:
                    logits = logits.float().cpu().numpy()
                    if not np.isfinite(logits).all():
                        raise RuntimeError("Non-finite logits; refusing to return a generated token")
                    return {"tokens": logits.argmax(-1).tolist(), "timings": timings,
                            "logits": logits if command.get("capture") else None,
                            "kv_used_blocks": self.args["kv_blocks"] - len(self.pool.free)}
        return None


def worker(rank, args, pipe):
    try:
        runner = Runner(rank, args)
        pipe.send({"ready": True, "rank": rank})
        while True:
            command, arrays = pipe.recv()
            if command["op"] == "stop":
                break
            try:
                result = runner.execute(command, arrays)
                pipe.send({"result": result})
            except Exception:
                pipe.send({"error": traceback.format_exc()})
                # A failed step may have written part of the cache. The server
                # marks the executor unhealthy and restart is required.
                break
    except BaseException:
        try:
            pipe.send({"error": traceback.format_exc()})
        except Exception:
            pass
    finally:
        with contextlib.suppress(Exception):
            from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
            destroy_model_parallel()
            destroy_distributed_environment()


class Executor:
    def __init__(self, args):
        ctx = mp.get_context("spawn")
        self.pipes, self.processes = [], []
        self.healthy = True
        self.lock = threading.Lock()
        for rank in range(args["tp"]):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=worker, args=(rank, args, child), daemon=True)
            process.start()
            child.close()
            self.pipes.append(parent)
            self.processes.append(process)
        for pipe in self.pipes:
            if not pipe.poll(600):
                raise RuntimeError("GPU worker startup timeout")
            reply = pipe.recv()
            if "error" in reply:
                raise RuntimeError(reply["error"])

    def call(self, command, arrays=None):
        with self.lock:
            return self._call(command, arrays)

    def _call(self, command, arrays=None):
        if not self.healthy:
            raise RuntimeError("Executor unhealthy; restart required")
        for rank, pipe in enumerate(self.pipes):
            pipe.send((command, arrays if rank == 0 else None))
        replies = []
        for pipe in self.pipes:
            if not pipe.poll(150):
                self.healthy = False
                raise RuntimeError("GPU execution timeout")
            reply = pipe.recv()
            if "error" in reply:
                self.healthy = False
                raise RuntimeError(reply["error"])
            replies.append(reply["result"])
        return replies[0]

    def close(self):
        if self.healthy:
            for pipe in self.pipes:
                try:
                    pipe.send(({"op": "stop"}, None))
                except (BrokenPipeError, EOFError):
                    pass
            for process in self.processes:
                process.join(timeout=3)
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=10)
