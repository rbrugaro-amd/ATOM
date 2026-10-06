# SPDX-License-Identifier: MIT
"""Kimi-K3 DFlash draft: the SpecForge/SGLang DSpark speculator.

`RadixArk/Kimi-K3-DSpark`, trained by SpecForge on hidden states from a live
SGLang Kimi-K3 target. Same algorithm as the TorchSpec draft in
`kimi_k3_dspark.py` -- block-parallel draft, low-rank Markov head, confidence
head -- on a different backbone: 5 Qwen3-style GQA layers instead of MLA.

Checkpoint layout (62 tensors, BF16, no biases)::

    fc.weight                                   [7168, 35840]  (= 5 x 7168)
    hidden_norm.weight                          [7168]
    norm.weight                                 [7168]
    layers.{0..4}.self_attn.q_proj.weight       [4096, 7168]
    layers.{0..4}.self_attn.k_proj.weight       [1024, 7168]
    layers.{0..4}.self_attn.v_proj.weight       [1024, 7168]
    layers.{0..4}.self_attn.{q,k}_norm.weight   [64]           per-head QK-norm
    layers.{0..4}.self_attn.o_proj.weight       [7168, 4096]
    layers.{0..4}.mlp.{gate,up}_proj.weight     [14336, 7168]
    layers.{0..4}.mlp.down_proj.weight          [7168, 14336]
    layers.{0..4}.{input,post_attention}_layernorm.weight  [7168]
    markov_head.markov_w{1,2}.weight            [163840, 256]
    confidence_head.proj.{weight,bias}          [1, 7424] / [1]  TRAINING ONLY

No `embed_tokens`, no `lm_head` -- both shared from the target, as the MLA draft
does. The fusion width is ``len(target_layer_ids)`` (5), NOT the config's
``num_target_layers`` (93, which is the TARGET's depth); `fc.weight`'s 35840
confirms it.

DUAL-SOURCE ATTENTION
---------------------
Per layer, the reference (`dflash.py` in the checkpoint) computes::

    q = q_norm(q_proj(noise))                   # q-norm BEFORE rope
    k = k_norm(cat([k_proj(ctx), k_proj(noise)]))   # SAME weights both sources
    v =        cat([v_proj(ctx), v_proj(noise)])    # v is NOT normed
    q, k = rope(q, k)                           # rope AFTER norm
    cache.update(k, v)                          # cache holds post-rope K
    attention is NON-CAUSAL over the whole concatenation

`k_norm` is an RMSNorm over `head_dim`, i.e. row-wise, so
``norm(cat(a, b)) == cat(norm(a), norm(b))`` exactly. That identity is what lets
the context half be normed and cached at `write_context_kv` time and the noise
half be normed in the block pass, and still equal the reference's single concat.

The norms run HERE, in the model, not inside `Attention`: the op's non-fused
path applies rope first and the norms second (`attention_mha.py`), which is the
wrong order for this draft.

NON-CAUSALITY, AND WHY THIS BYPASSES `PagedAttentionImpl`
--------------------------------------------------------
`PagedAttentionImpl.dispatch_backend` is causal at EVERY arm: `causal=True` is a
literal in all four runners and `attn_metadata.causal` is read only by the MLA
impl, so the proposer setting it False is inert here. Routing this block through
the op would silently apply a causal mask to an attention the reference defines
as bidirectional -- no error, no fault, just lost acceptance length.

So the block pass does its own write and its own read:

  write  `aiter.reshape_and_cache_flash` -- the block WRITES the KV it attends,
         joining the context rows already in the pool, so one read then sees
         `[0, anchor] ++ T`, i.e. the reference's single attention over the cat.
  read   `aiter.ops.triton.attention.mha_v3.flash_attn_with_kvcache(causal=False)`
         -- the only aiter entry point that is non-causal at `qlen > 1` over a
         paged GQA cache. `unified_attention` asserts causal; `pa_decode_gluon`
         derives `is_causal = query_length > 1` and caps qlen at 4; gfx950 ships
         no `Msk=0` persistent ASM kernel (verified against the kernel manifest).

That kernel hard-unpacks FOUR cache axes, so this draft's pool is bound in FLASH
layout (`MhaKvPool.fa_views`) -- SHUFFLE's x-pack splits `head_dim` across two
non-adjacent axes and cannot be offered to it by any permute. The pool is also
pinned to bf16 (`needs_non_causal_kv`): mha_v3 takes fp8 only when q is fp8 too,
and its descales are per-(batch, kv_head), not the per-token scales this pool
would carry.
"""

from typing import ClassVar

import torch
from aiter.dist.parallel_state import get_tensor_model_parallel_world_size
from aiter.rotary_embedding import get_rope
from torch import nn

from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from atom.model_ops.paged_attention import Attention
from atom.models.dspark_draft import DSparkDraftModel
from atom.models.kimi_k3_dspark import DSparkMarkovHead, _linear_out


class KimiK3DFlashAttention(nn.Module):
    """One DFlash GQA layer's attention, with a second entry for context rows.

    `forward` is the block pass over the noise stream. `write_context_kv` is the
    target-derived half: it runs the same k/v projections over the projected aux
    context and scatters the result into this layer's rows of the draft's own KV
    pool, so the block pass then attends context and noise in one call.
    """

    def __init__(self, config, layer_num: int, quant_config=None, prefix: str = ""):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        tp_size = get_tensor_model_parallel_world_size()
        if self.total_num_heads % tp_size != 0:
            raise ValueError(
                f"DFlash draft has {self.total_num_heads} query heads, which is "
                f"not divisible by TP size {tp_size}."
            )
        self.num_heads = self.total_num_heads // tp_size
        # The GQA-tolerant form, not a bare divisibility assert: below TP the kv
        # heads replicate rather than shard, and QKVParallelLinear already does
        # that. With 16 kv heads this only bites past TP16.
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self._unit_scale = None
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Per-head QK-norm, Qwen3 style: RMSNorm over head_dim is row-wise, so a
        # flat [N, q_size] tensor is normalized in head_dim-wide groups without
        # an explicit reshape.
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # The DRAFT'S OWN rope, never the target's. This checkpoint is YaRN x16
        # from 65536 at theta 10000; Kimi-K3 is x32 from 32768 at theta 50000.
        # Inheriting the target's loads clean and only costs acceptance length.
        rope_params = dict(getattr(config, "rope_parameters", None) or {})
        rope_theta = rope_params.get("rope_theta") or getattr(
            config, "rope_theta", 10000.0
        )
        rope_scaling = None
        if rope_params.get("rope_type") == "yarn" or rope_params.get(
            "factor", 1.0
        ) not in (
            1.0,
            None,
        ):
            rope_scaling = dict(rope_params)
            rope_scaling.setdefault(
                "original_max_position_embeddings", config.max_position_embeddings
            )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=True,
        )

        self.attn = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=getattr(config, "_atom_kv_cache_dtype", "bf16"),
            layer_num=layer_num,
            rotary_emb=self.rotary_emb,
            prefix=f"{prefix}.attn",
            # Deliberately None: see the module docstring. The op would rope
            # before norming, which is the opposite of the reference.
            q_norm=None,
            k_norm=None,
        )

    def _qkv(self, hidden_states: torch.Tensor):
        qkv = _linear_out(self.qkv_proj(hidden_states))
        return qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        """One block pass: write this block's own K/V, then attend non-causally.

        Deliberately NOT `self.attn(...)`. `PagedAttentionImpl.dispatch_backend`
        is causal at every arm -- `causal=True` is a literal in all four runners
        and `attn_metadata.causal` is read only by the MLA impl -- so routing
        the block through it would silently apply a causal mask to an attention
        the reference defines as bidirectional. That is a numerical bug with no
        error and no fault, visible only as lost acceptance length.
        """
        from atom.utils.forward_context import get_forward_context

        fwd_ctx = get_forward_context()
        attn_metadata = fwd_ctx.attn_metadata

        q, k, v = self._qkv(hidden_states)
        q = self.q_norm(q)
        k = self.k_norm(k)
        # Norm, THEN rope -- the reference's order, and the reason the norms are
        # not handed to the op (its non-fused path ropes first).
        q, k = self.rotary_emb(positions, q, k)

        n = q.shape[0]
        k_cache, v_cache = self._caches(fwd_ctx)
        if k_cache is None:
            # warmup_model() runs BEFORE allocate_kv_cache(), so there is no
            # paged state to write or read. The block forward still runs -- that
            # pass doubles as memory profiling, and skipping it would leave the
            # draft's activations out of the KV budget -- but its attention has
            # nothing to attend, so the output is shaped and left at zero.
            o = q.new_zeros((n, self.num_heads * self.head_dim))
            return _linear_out(self.o_proj(o))

        self._write_block_kv(k, v, attn_metadata.slot_mapping[:n], k_cache, v_cache)
        o = self._attend_non_causal(q, attn_metadata, k_cache, v_cache)
        return _linear_out(self.o_proj(o))

    def _write_block_kv(self, k, v, slot_mapping, k_cache, v_cache) -> None:
        """Scatter the block's own K/V into this layer's rows of the pool.

        Unlike V4's rolling window, this block WRITES the KV it attends: the T
        draft rows join the context rows already there, and the single read
        below then sees `[0, anchor] ++ T` -- the reference's one attention over
        `cat([k_ctx, k_noise])`.
        """
        import aiter

        # Unit scales, not None: the binding types arg6/arg7 as tensors, so a
        # bf16 ("auto") write still has to hand it something. One element each,
        # allocated once -- this runs per layer per block pass.
        if self._unit_scale is None or self._unit_scale.device != k.device:
            self._unit_scale = torch.ones(1, dtype=torch.float32, device=k.device)
        aiter.reshape_and_cache_flash(
            k.view(-1, self.num_kv_heads, self.head_dim),
            v.view(-1, self.num_kv_heads, self.head_dim),
            k_cache,
            v_cache,
            slot_mapping,
            "auto",
            self._unit_scale,
            self._unit_scale,
        )

    def _attend_non_causal(self, q, attn_metadata, k_cache, v_cache):
        """The block's one non-causal paged read.

        `mha_v3.flash_attn_with_kvcache` is the only aiter entry point that is
        non-causal at `qlen > 1` over a paged GQA cache: `unified_attention`
        asserts causal, `pa_decode_gluon` derives `is_causal = query_length > 1`
        and caps qlen at 4, and gfx950 ships no `Msk=0` persistent ASM kernel.
        """
        from aiter.ops.triton.attention.mha_v3 import flash_attn_with_kvcache

        bs = attn_metadata.block_tables.shape[0]
        t = q.shape[0] // bs
        out = flash_attn_with_kvcache(
            q=q.view(bs, t, self.num_heads, self.head_dim),
            k_cache=k_cache,
            v_cache=v_cache,
            # A Tensor, and mandatory: `page_table` alone does not route to the
            # decode kernel, and the call falls into the prefill kernel and
            # asserts on a batch mismatch.
            cache_seqlens=attn_metadata.context_lens[:bs],
            softmax_scale=self.scaling,
            causal=False,
            page_table=attn_metadata.block_tables,
        )
        return out.reshape(-1, self.num_heads * self.head_dim)

    def _caches(self, fwd_ctx):
        """This layer's pool rows in FLASH layout, or `(None, None)` if unbound.

        `DraftKvBuilder` bound them that way (`MhaKvPool.fa_views`) because
        mha_v3 hard-unpacks four axes and SHUFFLE's x-pack splits `head_dim`
        across two of them.

        Unbound is a real state, not an error: `warmup_model()` runs before
        `allocate_kv_cache()`, so `kv_cache_data` is None on that pass. Keyed on
        the POOL rather than on `is_dummy_run`, because the DP-alignment dummy
        carries that flag with a fully bound pool.
        """
        cache_data = fwd_ctx.kv_cache_data or {}
        entry = cache_data.get(f"layer_{self.attn.layer_num}")
        k_cache = getattr(entry, "k_cache", None) if entry is not None else None
        if k_cache is None or k_cache.numel() == 0:
            return None, None
        return k_cache, entry.v_cache

    def write_context_kv(
        self,
        ctx_hidden: torch.Tensor,  # [N, hidden] projected aux context
        positions: torch.Tensor,  # [N] absolute positions
        slot_mapping: torch.Tensor,  # [N] destination slots in the draft pool
        fwd_ctx,
    ) -> None:
        """Project the target-derived context and cache it as this layer's K/V.

        The whole fused QKV GEMM runs and the q columns are dropped. Not
        bookkeeping that could be avoided by slicing the weight: with
        per-output-channel quant and preshuffled rows, a row slice of the merged
        weight is not that shard's weight. The MLA draft does the same.
        """
        _, k, v = self._qkv(ctx_hidden)
        # Only k is normed -- v is not, matching the reference. Row-wise RMSNorm
        # makes norming this half separately identical to norming the concat,
        # which is what lets the context live in the pool at all.
        k = self.k_norm(k)
        # Rope at the token's ABSOLUTE position, before the write: that is what
        # makes a cached row valid on every later step without being rewritten.
        _, k = self.rotary_emb(positions, torch.empty_like(k), k)
        k_cache, v_cache = self._caches(fwd_ctx)
        if k_cache is None:
            return
        self._write_block_kv(k, v, slot_mapping, k_cache, v_cache)


class KimiK3DFlashMLP(nn.Module):
    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = _linear_out(self.gate_up_proj(x))
        gate, up = gate_up.chunk(2, dim=-1)
        return _linear_out(self.down_proj(self.act_fn(gate) * up))


class KimiK3DFlashDecoderLayer(nn.Module):
    def __init__(self, config, layer_num: int, quant_config=None, prefix: str = ""):
        super().__init__()
        self.self_attn = KimiK3DFlashAttention(
            config, layer_num, quant_config, prefix=f"{prefix}.self_attn"
        )
        self.mlp = KimiK3DFlashMLP(config, quant_config, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(positions, hidden_states)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)

    def write_context_kv(self, ctx_hidden: torch.Tensor, positions: torch.Tensor):
        """The context rows bypass `input_layernorm`.

        The reference feeds `hidden_norm(fc(aux))` straight into k_proj/v_proj;
        only the noise stream goes through the layer's input norm.
        """
        from atom.utils.forward_context import get_forward_context

        fwd_ctx = get_forward_context()
        # The TARGET's slot mapping: `compute_draft_kv` runs while attn_metadata
        # still describes the target forward, and the draft's pool rides the
        # target's block ids (one entry per scheduler block), so the slot a
        # verified token occupies is the same number in both pools.
        slot_mapping = fwd_ctx.attn_metadata.slot_mapping[: ctx_hidden.shape[0]]
        self.self_attn.write_context_kv(ctx_hidden, positions, slot_mapping, fwd_ctx)


class KimiK3DFlash(DSparkDraftModel):
    """The SpecForge DFlash draft for Kimi-K3.

    Registered as ``K3DFlashModel``: the checkpoint ships
    ``architectures: ["DSparkDraftModel"]``, which SpecForge stamps on every
    DFlash draft and which collides with this module's own base class, so
    `SpeculativeConfig.hf_config_override` renames it.
    """

    packed_modules_mapping: ClassVar[dict[str, tuple]] = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    # confidence_head predicts a per-position accept probability and is used at
    # training time (and by SGLang for adaptive block length); ATOM runs a fixed
    # block, so it is never read. embed_tokens / lm_head are the target's.
    skip_weight_prefixes: ClassVar[list[str]] = [
        "confidence_head.",
        "embed_tokens.",
        "lm_head.",
    ]

    def __init__(self, atom_config, layer_offset: int = 0):
        super().__init__()
        config = atom_config.hf_config
        quant_config = getattr(atom_config, "quant_config", None)

        self.config = config
        self.vocab_size = int(config.vocab_size)
        self.hidden_size = config.hidden_size
        self.noise_token_id = int(config.dspark_noise_token_id)
        self.target_layer_ids = list(config.dspark_target_layer_ids)

        # Aux fusion: [N, hidden * len(target_layer_ids)] -> [N, hidden].
        # Replicated on purpose: the aux hidden states arrive replicated (they
        # are the target's residual stream), so a row-parallel split would buy
        # memory at the cost of an all-reduce every drafting step.
        self.fc = ReplicatedLinear(
            self.hidden_size * len(self.target_layer_ids),
            self.hidden_size,
            bias=False,
            prefix="fc",
        )
        self.hidden_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        self.layers = nn.ModuleList(
            [
                KimiK3DFlashDecoderLayer(
                    config,
                    layer_offset + i,
                    quant_config,
                    prefix=f"layers.{i}",
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.markov_head = DSparkMarkovHead(
            self.vocab_size, int(config.dspark_markov_rank)
        )

        # Bound by `share_with_target`.
        self.embed_tokens = None
        self.lm_head = None

    # -- DSparkDraftModel contract ----------------------------------------

    def project_context(self, aux_concat: torch.Tensor) -> torch.Tensor:
        return self.hidden_norm(_linear_out(self.fc(aux_concat)))

    @property
    def context_layers(self):
        """Every draft layer holds context rows; `layers` is already in order."""
        return self.layers

    def block_backbone(self, input_ids, positions, num_draft: int):
        """One block: [anchor, MASK, ...] of width T in, T hidden rows out.

        `positions` is [B*T] absolute block positions -- the standalone-draft
        shape `DSparkProposer._block_positions` produces.
        """
        bs = input_ids.shape[0]
        draft_ids = input_ids.new_full((bs, num_draft), self.noise_token_id)
        draft_ids[:, 0] = input_ids
        hidden = self.embed_tokens(draft_ids.view(-1))
        for layer in self.layers:
            hidden = layer(positions, hidden)
        return self.norm(hidden)

    def head_and_sample(self, out, anchor_ids: torch.Tensor, num_draft: int):
        base_logits = self.lm_head(out).view(anchor_ids.shape[0], num_draft, -1)
        return self._sample_block(base_logits, anchor_ids), None

    def forward_spec(self, input_ids, positions, num_draft=None):
        out = self.block_backbone(input_ids, positions, num_draft)
        return self.head_and_sample(out, input_ids, num_draft)

    def _sample_block(self, base_logits: torch.Tensor, anchor_ids: torch.Tensor):
        """Markov-biased left-to-right sampling over the computed block.

            logits_k <- U_k + B(x_{k-1}, .);  x_k <- argmax(logits_k)

        Only this loop is sequential, and it is T rank-256 matmuls rather than T
        backbone passes -- which is the point of DSpark.
        """
        bs, num_draft, _ = base_logits.shape
        out_ids = anchor_ids.new_empty(bs, num_draft + 1)
        out_ids[:, 0] = anchor_ids
        for k in range(num_draft):
            # Greedy: temperature/sampling is applied by the target's verify.
            # The column IS the op's destination -- `sample_next` writes ids
            # into `out`, and returns the markov embedding, not the ids -- so
            # no id is written twice and a strided view costs no copy.
            self.markov_head.sample_next(
                out_ids[:, k], base_logits[:, k], out_ids[:, k + 1]
            )
        return out_ids[:, 1:]

    def share_with_target(self, target_base, loaded) -> None:
        """Bind the target's embedding and unembedding.

        The checkpoint ships neither; the draft drafts in the target's token
        space, so a vocab disagreement is a load-time error, not a runtime one.
        """
        # `target_base.model.embed_tokens`, not `target_base.embed_tokens`:
        # ModelRunner unwraps the multimodal `language_model` wrapper but
        # hands over the KimiLinearForCausalLM, whose embedding lives one
        # level down on `.model`.
        target_vocab = int(target_base.model.embed_tokens.num_embeddings)
        if target_vocab != self.vocab_size:
            raise ValueError(
                f"DFlash draft vocab {self.vocab_size} != target vocab "
                f"{target_vocab}; the draft samples in the target's token space."
            )
        self.embed_tokens = target_base.model.embed_tokens
        self.lm_head = target_base.lm_head
