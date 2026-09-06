"""BF16 architectural Roofline approximation of HF DeepSeek-V4-Flash.

Models routed/shared MoE, low-rank attention, SWA/CSA/HCA and mHC.
Uniform expert routing, TP-sharded experts, no EP or quantized-kernel claim.
"""


def layer_parameters(m, layer):
    c = m.architecture_config
    h = m.hidden_size
    d = m.head_dim
    q = c["q_lora_rank"]
    o = c["o_lora_rank"]
    g = c["o_groups"]
    n = m.num_attention_heads
    attention = h * q + q * n * d + h * d + n * d * o + g * o * h
    ratio = c["compress_ratios"][layer]
    if ratio:
        width = (2 if ratio == 4 else 1) * d
        attention += 2 * h * width + ratio * width + d
        if ratio == 4:
            di = c["index_head_dim"]
            ni = c["index_n_heads"]
            attention += 4 * h * di + q * ni * di + h * ni + ratio * 2 * di + di
    experts = (
        3
        * h
        * c["moe_intermediate_size"]
        * (c["n_routed_experts"] + c["n_shared_experts"])
    )
    router = (
        m.vocab_size * c["num_experts_per_tok"]
        if layer < c["num_hash_layers"]
        else h * c["n_routed_experts"]
    )
    hc = c["hc_mult"]
    mix = (2 + hc) * hc
    return (
        attention + experts + router + 2 * (mix * hc * h + mix + 3) + 2 * h + q + d + n
    )


def cache_bytes(m, layer, length):
    """Conservative replicated per-rank cache incl. compression working buffers."""
    c = m.architecture_config
    ratio = c["compress_ratios"][layer]
    d = m.head_dim
    elements = min(length, c["sliding_window"]) * d
    if ratio:
        elements += (length // ratio + 2 * ratio) * d
        if ratio == 4:
            elements += (length // ratio + 2 * ratio) * c["index_head_dim"]
    return elements * m.dtype_bytes


class DeepSeekV4WorkloadModel:
    def __init__(self, config):
        self.config = config
        self.model = config.model

    def input_shape(self, items):
        return f"{self.model.dtype} [{sum(i.token_count for i in items)}, {self.model.hidden_size}]"

    def build_operators(self, stage, items):
        from .performance import ModeledOperator
        from .core import OperatorWorkload

        m = self.model
        c = m.architecture_config
        h = m.hidden_size
        d = m.head_dim
        tp = stage.tp_degree
        tokens = sum(i.token_count + i.recompute_tokens for i in items)
        b = m.dtype_bytes
        ops = []

        def add(name, flops, memory, category="compute", comm=0):
            ops.append(
                ModeledOperator(
                    name,
                    category,
                    OperatorWorkload(flops, memory, comm),
                    f"{m.dtype} [tokens={tokens}, hidden={h}, tp={tp}]",
                    dependencies=(ops[-1].name,) if ops else (),
                    communication_kind=(
                        "all_reduce" if category == "communication" else None
                    ),
                )
            )

        def mm(name, params, activation_width):
            add(
                name,
                2 * tokens * params / tp,
                (params / tp + tokens * activation_width) * b,
            )

        if stage.layer_start == 0:
            add("embed_tokens", 0, 2 * tokens * h * c["hc_mult"] * b)
        for layer in range(stage.layer_start, stage.layer_end):
            p = f"layer_{layer:02d}"
            hc = c["hc_mult"]
            mix = (2 + hc) * hc
            # pre/post stream mappings for attention and MoE; Sinkhorn vector work.
            add(
                p + ".mhc",
                4 * tokens * mix * hc * h
                + 4 * tokens * hc * hc * h
                + 8 * tokens * hc * hc * c["hc_sinkhorn_iters"],
                (2 * mix * hc * h + 12 * tokens * hc * h) * b,
            )
            add(
                p + ".rms_norm_rope",
                tokens * (16 * h + 8 * m.query_width),
                tokens * (8 * h + 4 * m.query_width) * b,
            )
            mm(p + ".q_a_proj", h * c["q_lora_rank"], h + c["q_lora_rank"])
            mm(
                p + ".q_b_proj",
                c["q_lora_rank"] * m.num_attention_heads * d,
                c["q_lora_rank"] + m.num_attention_heads * d / tp,
            )
            # Shared K=V projection and cache are replicated across TP ranks.
            add(p + ".kv_proj", 2 * tokens * h * d, (h * d + tokens * (h + d)) * b)
            ratio = c["compress_ratios"][layer]
            if ratio:
                width = (2 if ratio == 4 else 1) * d
                add(
                    p + ".compressor",
                    4 * tokens * h * width + 8 * tokens * width,
                    (2 * h * width + 6 * tokens * width) * b,
                )
                if ratio == 4:
                    di = c["index_head_dim"]
                    ni = c["index_n_heads"]
                    mm(
                        p + ".indexer_proj",
                        4 * h * di + c["q_lora_rank"] * ni * di + h * ni,
                        h + ni * di,
                    )
                    index_pairs = sum(
                        sum(
                            (pos + 1) // ratio
                            for pos in range(
                                i.token_start, i.token_start + i.token_count
                            )
                        )
                        for i in items
                    )
                    add(
                        p + ".indexer_scores_topk",
                        index_pairs * ni * (2 * di + 4) / tp,
                        index_pairs * 4 / tp + tokens * ni * di * b / tp,
                    )
            pairs = 0
            for i in items:
                for pos in range(i.token_start, i.token_start + i.token_count):
                    long = (pos + 1) // ratio if ratio else 0
                    if ratio == 4:
                        long = min(long, c["index_topk"])
                    pairs += min(pos + 1, c["sliding_window"]) + long
            add(
                p + ".compressed_attention",
                4 * pairs * m.num_attention_heads * d / tp,
                sum(cache_bytes(m, layer, i.context_tokens) for i in items)
                + 2 * tokens * m.query_width * b / tp,
            )
            mm(
                p + ".o_a_proj",
                m.num_attention_heads * d * c["o_lora_rank"],
                m.query_width / tp + c["o_groups"] * c["o_lora_rank"],
            )
            mm(
                p + ".o_b_proj",
                c["o_groups"] * c["o_lora_rank"] * h,
                c["o_groups"] * c["o_lora_rank"] + h,
            )
            if tp > 1:
                add(
                    p + ".tp_all_reduce_attention",
                    0,
                    0,
                    "communication",
                    tokens * h * b,
                )
            if layer < c["num_hash_layers"]:
                add(p + ".hash_router", 0, tokens * c["num_experts_per_tok"] * 8)
            else:
                mm(p + ".router", h * c["n_routed_experts"], h + c["n_routed_experts"])
            e = c["n_routed_experts"]
            k = c["num_experts_per_tok"]
            shared = c["n_shared_experts"]
            mid = c["moe_intermediate_size"]
            touched = e * (1 - (1 - k / e) ** tokens)
            # Only active experts compute, all experts must remain resident.
            add(
                p + ".routed_shared_experts",
                6 * tokens * h * mid * (k + shared) / tp,
                (
                    3 * h * mid * (touched + shared) / tp
                    + tokens * (k + shared) * (3 * h + 6 * mid / tp)
                )
                * b,
            )
            add(
                p + ".expert_scatter_gather",
                tokens * k * h * 2 / tp,
                tokens * k * h * 4 * b,
            )
            if tp > 1:
                add(
                    p + ".tp_all_reduce_attention",
                    0,
                    0,
                    "communication",
                    tokens * h * b,
                )
                add(p + ".tp_all_reduce_moe", 0, 0, "communication", tokens * h * b)
        if stage.layer_end == m.num_layers:
            logits = sum(1 for i in items if i.produces_logits)
            add(
                "hyper_head_final_norm",
                tokens * (2 * hc * hc * h + hc * h * 3),
                tokens * hc * h * 4 * b,
            )
            if logits:
                add(
                    "lm_head",
                    2 * logits * h * m.vocab_size / tp,
                    (h * m.vocab_size / tp + logits * (h + m.vocab_size / tp)) * b,
                )
        return ops
