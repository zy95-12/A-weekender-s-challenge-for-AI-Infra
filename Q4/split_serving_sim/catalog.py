"""Public model/device presets; calibrated costs only for the original A10/Qwen2.5."""

from dataclasses import asdict, replace
import json
from pathlib import Path
from .config import HardwareConfig, ModelConfig, load_config, validate_config
from .presets import configure_serving, ServingFeatures

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "qwen2.5-3b": (
        "Qwen/Qwen2.5-3B-Instruct",
        "qwen2_5_3b_instruct_config.json",
        "aa8e72537993ba99e69dfaafa59ed015b17504d1",
    ),
    "qwen3-32b": (
        "Qwen/Qwen3-32B",
        "qwen3_32b_config.json",
        "9216db5781bf21249d130ec9da846c4624c16137",
    ),
    "deepseek-v4-flash": (
        "deepseek-ai/DeepSeek-V4-Flash",
        "deepseek_v4_flash_config.json",
        "60d8d70770c6776ff598c94bb586a859a38244f1",
    ),
}


def public_model(key):
    name, file, revision = MODELS[key]
    raw = json.loads((ROOT / "models" / file).read_text())
    return ModelConfig(
        name,
        raw["model_type"],
        raw["num_hidden_layers"],
        raw["hidden_size"],
        raw.get("intermediate_size", raw.get("moe_intermediate_size")),
        raw["num_attention_heads"],
        raw["num_key_value_heads"],
        raw.get("head_dim", raw["hidden_size"] // raw["num_attention_heads"]),
        vocab_size=raw["vocab_size"],
        dtype="bfloat16",
        dtype_bytes=2,
        attention_bias=raw.get("attention_bias", False),
        tie_word_embeddings=raw.get("tie_word_embeddings", False),
        architecture_config=raw if key == "deepseek-v4-flash" else {},
    )


def stage_storage(model, stages, tp, active=96):
    weights = (
        sum(s.num_layers for s in stages)
        * model.parameters_per_layer
        * model.dtype_bytes
        / tp
    )
    for s in stages:
        if s.layer_start == 0:
            weights += model.vocab_size * model.hidden_size * model.dtype_bytes / tp
        if s.layer_end == model.num_layers:
            weights += model.vocab_size * model.hidden_size * model.dtype_bytes / tp
    if model.architecture == "deepseek_v4":
        from .deepseek_v4 import cache_bytes

        kv = active * sum(
            cache_bytes(model, l, 4175)
            for s in stages
            for l in range(s.layer_start, s.layer_end)
        )
    else:
        local_heads = max(1, model.num_key_value_heads // tp)
        kv = (
            active
            * 4175
            * sum(s.num_layers for s in stages)
            * 2
            * local_heads
            * model.head_dim
            * model.dtype_bytes
        )
    return weights, kv


def build_config(model_key="qwen2.5-3b", hardware_key="a10", variant="optimized"):
    if variant not in ("baseline", "optimized"):
        raise ValueError("Unknown serving variant")
    if model_key not in MODELS:
        raise ValueError("Unknown model")
    catalog = json.loads((ROOT / "hardware/public_roofline.json").read_text())
    if hardware_key not in catalog["devices"]:
        raise ValueError("Unknown hardware")
    calibrated = model_key == "qwen2.5-3b" and hardware_key == "a10"
    cfg = configure_serving(
        load_config(ROOT / "configs/issue6_baseline_host.json"),
        ServingFeatures.optimized() if variant == "optimized" else ServingFeatures(),
    )
    cfg = replace(
        cfg,
        static_policy=replace(cfg.static_policy, max_batch_size=96),
        scheduler=replace(
            cfg.scheduler,
            max_num_seqs=96,
            kv_cache=replace(cfg.scheduler.kv_cache, num_blocks=32768),
        ),
    )
    if not calibrated:
        model = public_model(model_key)
        n = model.num_layers
        ranges = [(0, 4), (4, n - 5), (n - 5, n)]
        stages = tuple(
            replace(s, layer_start=lo, layer_end=hi)
            for s, (lo, hi) in zip(cfg.stages, ranges)
        )
        spec = catalog["devices"][hardware_key]
        hardware = {}
        planned = []
        for role, group in [
            ("enterprise", [stages[0], stages[2]]),
            ("cloud", [stages[1]]),
        ]:
            minimum = 1 if variant == "optimized" else 2
            tp = next(
                (
                    t
                    for t in [1, 2, 4, 8, 16, 32, 64]
                    if t >= minimum
                    and model.num_attention_heads % t == 0
                    and sum(stage_storage(model, group, t))
                    <= 0.8 * spec["memory_gb"] * 1e9
                ),
                None,
            )
            if tp is None:
                raise ValueError(
                    "Model weights + 96-request KV exceed supported TP capacity"
                )
            weight, kv = stage_storage(model, group, tp)
            planned.append(
                dict(
                    role=role,
                    tp=tp,
                    weight_gb_per_rank=weight / 1e9,
                    kv_gb_per_rank=kv / 1e9,
                )
            )
        etp, ctp = [r["tp"] for r in planned]
        for name, old in cfg.hardware.items():
            tp = etp if name == cfg.stages[0].resource else ctp
            count = tp * (2 if name == "cloud_prefill" else 1)
            hardware[name] = HardwareConfig(
                count,
                spec["peak_flops_tflops"],
                spec["hbm_bandwidth_gb_s"],
                spec["memory_gb"],
                compute_efficiency=0.7,
                memory_efficiency=0.7,
                kernel_overhead_us=10,
                interconnect_bandwidth_gb_s=(
                    25 if tp > 8 else spec["interconnect_bandwidth_gb_s"]
                ),
                interconnect_latency_us=8,
            )
        stages = tuple(
            (
                replace(
                    s,
                    tp_degree=ctp,
                    prefill_tp_degree=ctp if variant == "optimized" else None,
                    decode_tp_degree=ctp if variant == "optimized" else None,
                )
                if s.name == "cloud_middle"
                else replace(s, tp_degree=etp)
            )
            for s in stages
        )
        cfg = replace(
            cfg,
            model=model,
            stages=stages,
            hardware=hardware,
            operator_backend=replace(
                cfg.operator_backend, name="logical", version="HF architecture Roofline"
            ),
            performance_profile=replace(cfg.performance_profile, enabled=False),
            execution=replace(
                cfg.execution,
                host_submission=replace(cfg.execution.host_submission, enabled=False),
            ),
            data_path=replace(
                cfg.data_path,
                wan_calibration=replace(cfg.data_path.wan_calibration, enabled=False),
            ),
            network=replace(
                cfg.network,
                activation_tensor_count=1 if model.architecture == "deepseek_v4" else 2,
            ),
        )
    else:
        planned = []
    validate_config(cfg)
    info = dict(
        calibrated=calibrated,
        model=MODELS[model_key][0],
        model_revision=MODELS[model_key][2],
        hardware=catalog["devices"][hardware_key],
        split=[s.num_layers for s in cfg.stages],
        total_devices=sum(h.count for h in cfg.hardware.values()),
        stages=[asdict(s) for s in cfg.stages],
        capacity_plan=planned,
        assumptions=(
            catalog["assumptions"]
            if not calibrated
            else "Original A10/Qwen2.5 command/operator/host/WAN calibration; see validation table."
        ),
        dtype=cfg.model.dtype,
        quantization=(
            "BF16 dequantized architectural scenario; no FP4/FP8 speedup modeled"
            if model_key == "deepseek-v4-flash"
            else cfg.model.dtype
        ),
    )
    return cfg, info
