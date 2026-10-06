#!/usr/bin/env python3
"""Phase-0 reference for the SpecForge "DFlash" speculative draft (Kimi-K3-DSpark).

Runs the checkpoint's OWN modeling code (``$CKPT/dflash.py``) with the real BF16
weights on one GPU, for ONE forward, on a FIXED SEEDED input, and dumps the
intermediate tensors listed in ``KIMI_K3_DFLASH_DESIGN.md`` section 7.2 so that
the ATOM implementation can be compared against them.

!!!! LOUD CAVEAT — SUBSTITUTED INPUT !!!!
The draft checkpoint has NO ``embed_tokens`` and NO ``lm_head``; both live on the
1T target, which we deliberately do not load.  ``DFlashDraftModel.forward`` takes
``noise_embedding`` (already-embedded tokens), so we SUBSTITUTE a FIXED, SEEDED
pseudo-embedding: for token id ``t``, ``embed(t) = 0.02 * randn(hidden,
generator=manual_seed(EMBED_SEED_BASE + t))``.  See ``seeded_embed()``.  This is
NOT the target's real embedding table.  Consequences:
  * every dump below is only meaningful when the ATOM side is fed the SAME
    ``noise_embedding`` tensor (saved as ``inputs.noise_embedding``), not when it
    is fed the real target embedding of the same ids;
  * the ``target_hidden`` aux states are likewise seeded gaussian noise, not real
    1T hidden states, so magnitudes/condition numbers are not representative —
    this validates the OPERATOR GRAPH, not the numerical regime;
  * section 7.2 checkpoint 5 asks for "markov head logits"; those need the
    target's ``lm_head`` to produce ``base_logits``, so they are NOT dumped.
    We dump the final ``norm`` output (the draft's true last op) instead, plus
    the markov bias term, which only needs the draft's own weights.

Usage (host has no python/torch; run under docker):

  docker run --rm --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined --group-add video --ipc=host \
    --shm-size 16G \
    -v /home/rbrugaro/code/ATOM-k3-dflash:/work -w /work \
    -v /data/workloads-inference/hf_hub_cache:/data/workloads-inference/hf_hub_cache:ro \
    -e AITER_LOG_LEVEL=WARNING -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
    -e HIP_VISIBLE_DEVICES=7 rocm/atom-dev:nightly_202610051453 \
    python3 tools/ref_dflash.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import torch

CKPT_DEFAULT = (
    "/data/workloads-inference/hf_hub_cache/models--RadixArk--Kimi-K3-DSpark/"
    "snapshots/3c5bac301d9cf392706189d82ed947feca6c2f0f"
)
OUT_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_dump")

# ---- fixed seeds (the whole point: the ATOM side must reproduce these) -------
AUX_SEED = 1234  # seeded aux (target) hidden states
EMBED_SEED_BASE = 777000  # per-token-id pseudo-embedding seed base
EMBED_SCALE = 0.02
AUX_SCALE = 1.0


def load_dflash_module(ckpt: str):
    """Import ``$CKPT/dflash.py`` by path (trust_remote_code style, manually)."""
    spec = importlib.util.spec_from_file_location(
        "ckpt_dflash", os.path.join(ckpt, "dflash.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ckpt_dflash"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_config(ckpt: str, attn_impl: str):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

    with open(os.path.join(ckpt, "config.json")) as f:
        raw = json.load(f)
    raw.pop("architectures", None)
    raw.pop("auto_map", None)
    cfg = Qwen3Config(**raw)
    cfg._attn_implementation = attn_impl
    # Qwen3Config may not round-trip every extra key; force the DFlash ones.
    for k in ("block_size", "num_target_layers", "dflash_config", "head_dim"):
        setattr(cfg, k, raw[k])
    return cfg


def seeded_embed(token_ids: torch.Tensor, hidden: int) -> torch.Tensor:
    """FIXED, SEEDED stand-in for the target's embed_tokens (see module docstring).

    Deterministic per token id, so any other implementation can regenerate it
    exactly without sharing a file: row(t) = EMBED_SCALE * randn(hidden,
    generator=torch.Generator('cpu').manual_seed(EMBED_SEED_BASE + int(t))).
    """
    flat = token_ids.reshape(-1).tolist()
    rows = []
    for t in flat:
        g = torch.Generator(device="cpu").manual_seed(EMBED_SEED_BASE + int(t))
        rows.append(EMBED_SCALE * torch.randn(hidden, generator=g, dtype=torch.float32))
    out = torch.stack(rows).reshape(*token_ids.shape, hidden)
    return out


def _maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


@torch.inference_mode()
def verify(model, dflash, D, kproj_calls, vproj_calls, rc, ctx_len, T, cfg):
    """Numerically check the three Appendix-A claims the design relies on."""
    hd = cfg.head_dim
    attn = model.layers[0].self_attn
    print("\n[verify] --- Appendix A claims ---")

    # (iii) norm(cat) == cat(norm): k_norm is row-wise over head_dim, so the
    # paged-context term and the dense block term may be normed separately.
    k_ctx, k_noise = kproj_calls
    B = k_ctx.shape[0]
    k_cat = torch.cat([k_ctx, k_noise], dim=1).view(B, ctx_len + T, -1, hd)
    joint = attn.k_norm(k_cat)
    split = torch.cat(
        [
            attn.k_norm(k_ctx.view(B, ctx_len, -1, hd)),
            attn.k_norm(k_noise.view(B, T, -1, hd)),
        ],
        dim=1,
    )
    eq = torch.equal(joint, split)
    print(
        f"[verify] norm(cat)==cat(norm): bitwise={eq} maxdiff={_maxdiff(joint, split)}"
    )
    print(
        f"[verify] joint matches dumped c.l0_k_after_knorm: "
        f"{torch.equal(joint, D['l0_k_after_knorm'])}"
    )

    # (v) apply_rotary_pos_emb slicing: q takes cos[..., -q_len:, :], k full cos.
    from transformers.models.qwen3.modeling_qwen3 import rotate_half

    cos, sin = rc["cos"].unsqueeze(1), rc["sin"].unsqueeze(1)
    q, k = rc["q_pre_rope"], rc["k_pre_rope"]
    q_ref = q * cos[..., -T:, :] + rotate_half(q) * sin[..., -T:, :]
    k_ref = k * cos + rotate_half(k) * sin
    print(
        f"[verify] q uses cos[..., -q_len:, :] : "
        f"{torch.equal(q_ref, rc['q_post_rope'])}"
    )
    print(
        f"[verify] k uses FULL cos (len {cos.shape[-2]}) : "
        f"{torch.equal(k_ref, rc['k_post_rope'])}"
    )
    # and the wrong slicing (q with full cos head) would differ:
    q_wrong = q * cos[..., :T, :] + rotate_half(q) * sin[..., :T, :]
    print(
        f"[verify] q with cos[..., :q_len, :] (WRONG) maxdiff vs ref: "
        f"{_maxdiff(q_wrong, rc['q_post_rope'])}"
    )

    # (iv) bidirectional: redo layer-0 attention with an explicit causal mask and
    # show it DIFFERS from the reference output -> the reference is not causal.
    from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward

    # v is NOT normed: v_proj output goes straight to transpose(1, 2).
    v_cat = torch.cat(vproj_calls, dim=1).view(B, ctx_len + T, -1, hd).transpose(1, 2)
    qh, kh = rc["q_post_rope"], rc["k_post_rope"]
    free, _ = eager_attention_forward(attn, qh, kh, v_cat, None, scaling=attn.scaling)
    S = ctx_len + T
    causal = torch.zeros(B, 1, T, S, device=qh.device, dtype=qh.dtype)
    idx = torch.arange(S, device=qh.device)
    causal.masked_fill_(
        idx[None, None, None, :] > (ctx_len + idx[:T])[None, None, :, None],
        float("-inf"),
    )
    masked, _ = eager_attention_forward(
        attn, qh, kh, v_cat, causal, scaling=attn.scaling
    )
    print(
        f"[verify] attn free-vs-causal maxdiff={_maxdiff(free, masked)} "
        f"(nonzero => reference block attention is BIDIRECTIONAL)"
    )
    print(
        f"[verify] free-attn reproduces dumped d.l0_attn_preo: "
        f"{_maxdiff(free.reshape(B, T, -1), D['l0_attn_preo'])}"
    )
    print(
        f"[verify] module.is_causal={attn.is_causal}; v is NOT normed "
        f"(no v_norm module: {not hasattr(attn, 'v_norm')})"
    )
    print("[verify] --- end ---\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--ctx-len", type=int, default=16, help="aux/context length")
    ap.add_argument("--block-size", type=int, default=0, help="0 = config.block_size")
    ap.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    dev = "cuda"

    dflash = load_dflash_module(args.ckpt)
    cfg = build_config(args.ckpt, args.attn_impl)
    hidden = cfg.hidden_size
    T = args.block_size or cfg.block_size
    ctx_len = args.ctx_len
    B = 1
    n_aux = len(cfg.dflash_config["target_layer_ids"])

    print(
        f"[ref] hidden={hidden} T={T} ctx_len={ctx_len} n_aux={n_aux} "
        f"attn_impl={args.attn_impl}"
    )

    # ---------------- model ----------------
    with torch.device("meta"):
        model = dflash.DFlashDraftModel(cfg)
    model = model.to_empty(device="cpu")

    from safetensors.torch import load_file

    sd = load_file(os.path.join(args.ckpt, "model.safetensors"))
    own = set(model.state_dict().keys())
    draft_sd = {k: v for k, v in sd.items() if k in own}
    skipped = sorted(k for k in sd if k not in own)
    missing = sorted(own - set(draft_sd))
    print(
        f"[ref] loaded {len(draft_sd)} tensors; skipped (DSpark heads, not part "
        f"of DFlashDraftModel): {skipped}"
    )
    assert not missing, f"UNINITIALIZED PARAMS: {missing}"
    model.load_state_dict(draft_sd, strict=True)

    # TRAP (cost me one bisect): ``to_empty()`` leaves NON-PERSISTENT buffers as
    # uninitialized garbage, and ``rotary_emb.inv_freq`` is exactly that -- it is
    # not in the checkpoint, so ``load_state_dict`` never restores it.  The model
    # then runs with a random RoPE basis that differs per process.  Re-instantiate
    # the rotary embedding so ``inv_freq`` is recomputed from the config, and
    # assert no other non-persistent buffer is left dangling.
    # Rebuilt AFTER the dtype cast so that inv_freq keeps its fp32 precision
    # (``Module.to(dtype)`` would otherwise downcast the buffer to bf16).
    rotary_cls = type(model.rotary_emb)
    model = model.to(device=dev, dtype=torch.bfloat16).eval()
    model.rotary_emb = rotary_cls(cfg).to(dev)
    assert model.rotary_emb.inv_freq.dtype == torch.float32
    buf_names = {n for n, _ in model.named_buffers()}
    dangling = sorted(
        n
        for n in buf_names - set(draft_sd)
        if not n.startswith("rotary_emb.")  # freshly reconstructed above
    )
    assert not dangling, f"UNINITIALIZED NON-PERSISTENT BUFFERS: {dangling}"
    # markov head lives on DSparkDraftModel, not DFlashDraftModel; load it
    # standalone so we can dump the bias term without the target's lm_head.
    markov_w1 = sd["markov_head.markov_w1.weight"].to(dev)  # [vocab, rank]
    markov_w2 = sd["markov_head.markov_w2.weight"].to(dev)  # [vocab, rank]

    # ---------------- seeded inputs ----------------
    g = torch.Generator(device="cpu").manual_seed(AUX_SEED)
    # DFlashDraftModel.forward wants target_hidden = cat of the selected target
    # layers' hidden states along the LAST axis -> [B, ctx_len, n_aux*hidden].
    # (extract_context_feature: torch.cat(selected_states, dim=-1))
    aux_hidden = (
        AUX_SCALE
        * torch.randn(B, ctx_len, n_aux * hidden, generator=g, dtype=torch.float32)
    ).to(dev, torch.bfloat16)

    mask_id = int(cfg.dflash_config["mask_token_id"])
    # spec_generate's block is [anchor_token, MASK, MASK, ...] of width T.
    anchor_id = 100
    token_ids = torch.full((B, T), mask_id, dtype=torch.long)
    token_ids[:, 0] = anchor_id
    noise_embedding = seeded_embed(token_ids, hidden).to(dev, torch.bfloat16)
    token_ids = token_ids.to(dev)

    # position_ids must span ctx + block (spec_generate passes
    # position_ids[:, past_len : start + block_size] with an empty draft cache).
    position_ids = torch.arange(ctx_len + T, device=dev).unsqueeze(0)

    # ---------------- capture ----------------
    D: dict[str, torch.Tensor] = {}
    kproj_calls, qproj_calls, vproj_calls = [], [], []
    handles = []

    def save(name):
        def h(m, i, o):
            D[name] = (o[0] if isinstance(o, tuple) else o).detach().clone()

        return h

    handles.append(model.fc.register_forward_hook(save("fc_out")))
    handles.append(
        model.hidden_norm.register_forward_hook(save("ctx_after_hidden_norm"))
    )
    handles.append(model.norm.register_forward_hook(save("final_norm_out")))

    l0 = model.layers[0]
    handles.append(l0.input_layernorm.register_forward_hook(save("l0_input_ln_out")))
    handles.append(l0.self_attn.q_norm.register_forward_hook(save("l0_q_after_qnorm")))
    handles.append(l0.self_attn.k_norm.register_forward_hook(save("l0_k_after_knorm")))
    handles.append(l0.self_attn.o_proj.register_forward_hook(save("l0_attn_out")))
    handles.append(
        l0.self_attn.o_proj.register_forward_hook(
            lambda m, i, o: D.__setitem__("l0_attn_preo", i[0].detach().clone())
        )
    )
    handles.append(
        l0.self_attn.k_proj.register_forward_hook(
            lambda m, i, o: kproj_calls.append(o.detach().clone())
        )
    )
    handles.append(
        l0.self_attn.q_proj.register_forward_hook(
            lambda m, i, o: qproj_calls.append(o.detach().clone())
        )
    )
    handles.append(
        l0.self_attn.v_proj.register_forward_hook(
            lambda m, i, o: vproj_calls.append(o.detach().clone())
        )
    )
    for li, lyr in enumerate(model.layers):
        handles.append(lyr.register_forward_hook(save(f"l{li}_hidden_out")))

    # capture post-RoPE q/k for layer 0 by wrapping the module-level function
    rope_calls = []
    orig_rope = dflash.apply_rotary_pos_emb

    def rope_spy(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        qe, ke = orig_rope(q, k, cos, sin, position_ids, unsqueeze_dim)
        if len(rope_calls) == 0:
            rope_calls.append(
                {
                    "q_pre_rope": q.detach().clone(),
                    "k_pre_rope": k.detach().clone(),
                    "cos": cos.detach().clone(),
                    "sin": sin.detach().clone(),
                    "q_post_rope": qe.detach().clone(),
                    "k_post_rope": ke.detach().clone(),
                }
            )
        return qe, ke

    dflash.apply_rotary_pos_emb = rope_spy

    # ---------------- ONE forward ----------------
    with torch.inference_mode():
        out = model(
            position_ids=position_ids,
            attention_mask=None,  # <- spec_generate passes nothing: fully bidirectional
            noise_embedding=noise_embedding,
            target_hidden=aux_hidden,
            past_key_values=None,
            use_cache=False,
            is_causal=False,  # <- exactly what spec_generate passes
        )

    dflash.apply_rotary_pos_emb = orig_rope
    for h in handles:
        h.remove()

    # markov bias term (draft-owned weights only; needs no lm_head)
    with torch.inference_mode():
        prev_ids = token_ids  # teacher-forced "previous token" placeholder
        markov_latent = torch.nn.functional.embedding(prev_ids, markov_w1)
        markov_bias = torch.nn.functional.linear(markov_latent, markov_w2)

    # ---------------- assemble dump ----------------
    assert len(kproj_calls) == 2, f"expected 2 k_proj calls, got {len(kproj_calls)}"
    rc = rope_calls[0]

    dump = {
        # ---- INPUTS (identical-input contract) ----
        "inputs.aux_hidden": aux_hidden,
        "inputs.noise_embedding": noise_embedding,
        "inputs.token_ids": token_ids,
        "inputs.position_ids": position_ids,
        # ---- 7.2 (a) ----
        "a.fc_out": D["fc_out"],
        "a.ctx_after_hidden_norm": D["ctx_after_hidden_norm"],
        # ---- 7.2 (b) : BEFORE qk-norm, two separate k_proj calls ----
        "b.l0_k_proj_ctx": kproj_calls[0],
        "b.l0_k_proj_noise": kproj_calls[1],
        "b.l0_q_proj_noise": qproj_calls[0],
        "b.l0_v_proj_ctx": vproj_calls[0],
        "b.l0_v_proj_noise": vproj_calls[1],
        # ---- 7.2 (c) : AFTER qk-norm on the CONCAT ----
        "c.l0_k_after_knorm": D["l0_k_after_knorm"],
        "c.l0_q_after_qnorm": D["l0_q_after_qnorm"],
        # ---- extra: rope boundary ----
        "r.l0_q_pre_rope": rc["q_pre_rope"],
        "r.l0_k_pre_rope": rc["k_pre_rope"],
        "r.l0_q_post_rope": rc["q_post_rope"],
        "r.l0_k_post_rope": rc["k_post_rope"],
        "r.cos": rc["cos"],
        "r.sin": rc["sin"],
        # ---- 7.2 (d) ----
        "d.l0_attn_preo": D["l0_attn_preo"],
        "d.l0_attn_out": D["l0_attn_out"],
        # ---- extra: per-layer hidden_out for layer bisect ----
        **{
            f"h.l{i}_hidden_out": D[f"l{i}_hidden_out"]
            for i in range(len(model.layers))
        },
        "h.l0_input_ln_out": D["l0_input_ln_out"],
        # ---- 7.2 (e) ----
        "e.final_norm_out": D["final_norm_out"],
        "e.model_return": out.detach().clone(),
        # ---- markov bias (no lm_head available) ----
        "m.markov_latent": markov_latent,
        "m.markov_bias": markov_bias,
    }
    meta = {
        "ckpt": args.ckpt,
        "attn_impl": args.attn_impl,
        "B": B,
        "ctx_len": ctx_len,
        "block_size": T,
        "n_aux": n_aux,
        "hidden": hidden,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "rope_type": model.rotary_emb.rope_type,
        "rope_attention_scaling": float(model.rotary_emb.attention_scaling),
        "rope_parameters": dict(cfg.rope_parameters),
        "target_layer_ids": list(cfg.dflash_config["target_layer_ids"]),
        "extract_context_feature_offset": 1,
        "mask_token_id": mask_id,
        "anchor_token_id": anchor_id,
        "aux_seed": AUX_SEED,
        "embed_seed_base": EMBED_SEED_BASE,
        "embed_scale": EMBED_SCALE,
        "aux_scale": AUX_SCALE,
        "torch": torch.__version__,
        "SUBSTITUTED": (
            "noise_embedding is a SEEDED PSEUDO-EMBEDDING, not the target's "
            "embed_tokens; aux_hidden is seeded gaussian, not real target hidden "
            "states. Compare ATOM against inputs.* from this file."
        ),
    }

    if args.verify:
        verify(model, dflash, D, kproj_calls, vproj_calls, rc, ctx_len, T, cfg)

    path = os.path.join(args.out, f"ref_dflash_{args.attn_impl}.pt")
    torch.save({"meta": meta, "tensors": {k: v.cpu() for k, v in dump.items()}}, path)
    print(f"[ref] saved {path}")
    for k, v in dump.items():
        print(f"  {k:34s} {tuple(v.shape)} {v.dtype}")


if __name__ == "__main__":
    main()
