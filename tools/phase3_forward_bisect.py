# SPDX-License-Identifier: MIT
"""Phase 3+4 of the DFlash numerics bisect: the forward graph, stage by stage.

Builds `KimiK3DFlash` through ATOM's real path at TP=1 (same as
`tools/phase1_coverage.py`), loads the reference dump produced by
`tools/ref_dflash.py`, and drives ATOM's own modules on the REFERENCE's exact
inputs at every checkpoint, so each row of the table is a SINGLE-STEP
comparison (ref input -> ATOM op -> ref output) rather than an accumulated one.

Run:
  docker run --rm --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined --group-add video --ipc=host \
    --shm-size 16G -v /home/rbrugaro/code/ATOM-k3-dflash:/work -w /work \
    -v /data/workloads-inference/hf_hub_cache:/data/workloads-inference/hf_hub_cache:ro \
    -e AITER_LOG_LEVEL=WARNING -e PYTHONPATH=/work -e HIP_VISIBLE_DEVICES=4 \
    rocm/atom-dev:nightly_202610051453 python3 /work/tools/phase3_forward_bisect.py
"""

import copy
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "/work")

CKPT = (
    "/data/workloads-inference/hf_hub_cache/models--RadixArk--Kimi-K3-DSpark/"
    "snapshots/3c5bac301d9cf392706189d82ed947feca6c2f0f"
)
REF = "/work/tools/ref_dump/ref_dflash_eager.pt"

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29573")

ROWS = []


def rec(name, a, b, note=""):
    from atom.utils.debug_helper.compare import cos_max

    cos, mad, rel = cos_max(a.detach().float().cpu(), b.detach().float().cpu())
    flag = "OK " if cos > 0.9990 else ("?? " if cos > 0.99 else "XX ")
    ROWS.append((name, cos, mad, rel, flag, note))
    print(f"  [{flag}] {name:42s} cos={cos:.8f} maxdiff={mad:.4e} rel={rel:.3e} {note}")
    return cos


def hdr(s):
    print(f"\n{'=' * 90}\n{s}\n{'=' * 90}")


def main():
    torch.cuda.set_device(0)
    from atom.distributed.pp_comm import init_pp_aware_dist_env

    init_pp_aware_dist_env(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        global_rank=0,
        world_size=1,
        distributed_init_method="tcp://127.0.0.1:29573",
        backend="nccl",
    )

    from atom.config import (
        CompilationLevel,
        Config,
        SpeculativeConfig,
        set_current_atom_config,
    )
    from atom.model_loader.loader import load_model
    from atom.models.kimi_k3_dspark import _linear_out
    from atom.spec_decode.drafter import support_draft_model_arch_dict
    from atom.utils import resolve_obj_by_qualname

    spec = SpeculativeConfig(model=CKPT, method="dspark", num_speculative_tokens=5)
    draft_hf = spec.draft_model_hf_config
    cfg = Config(model=CKPT, speculative_config=spec, enforce_eager=True)
    draft_cfg = copy.copy(cfg)
    draft_cfg.hf_config = draft_hf
    draft_cfg.compilation_config = copy.copy(cfg.compilation_config)
    draft_cfg.compilation_config.level = CompilationLevel.NO_COMPILATION
    draft_cfg.compilation_config.static_forward_context = {}
    set_current_atom_config(draft_cfg)

    model_class = resolve_obj_by_qualname(
        support_draft_model_arch_dict[draft_hf.architectures[0]]
    )
    torch.set_default_dtype(draft_cfg.torch_dtype)  # bf16, as ModelRunner does
    with torch.device("cuda"):
        model = model_class(draft_cfg, layer_offset=0)
    load_model(model, CKPT, draft_hf, None, False)
    model.eval()

    # ---------------------------------------------------------------- ref
    d = torch.load(REF, map_location="cpu", weights_only=False)
    meta, R = d["meta"], d["tensors"]
    R = {k: v.cuda() for k, v in R.items()}
    ctx_len = meta["ctx_len"]
    T = meta["block_size"]
    hidden = meta["hidden"]
    hd = meta["head_dim"]
    nq = meta["num_attention_heads"]
    nkv = meta["num_key_value_heads"]
    S = ctx_len + T
    print(
        f"\nref meta: ctx_len={ctx_len} T={T} hidden={hidden} hd={hd} "
        f"nq={nq} nkv={nkv} attn_impl={meta['attn_impl']}"
    )
    print(
        f"default dtype: {torch.get_default_dtype()}  model dtype sample: "
        f"{model.norm.weight.dtype} / {model.layers[0].self_attn.q_norm.weight.dtype}"
    )

    l0 = model.layers[0]
    sa0 = l0.self_attn

    def f2(name, cols):
        """ref tensor [1, N, cols] -> [N, cols]"""
        return R[name].reshape(-1, cols)

    # ================================================== 1. project_context
    hdr("STEP 1  project_context = hidden_norm(fc(aux))")
    aux = R["inputs.aux_hidden"].reshape(ctx_len, -1)
    fc_atom = _linear_out(model.fc(aux))
    rec("1a fc(aux)            vs a.fc_out", fc_atom, f2("a.fc_out", hidden))
    ctx_atom = model.project_context(aux)
    rec(
        "1b project_context    vs a.ctx_after_hidden_norm",
        ctx_atom,
        f2("a.ctx_after_hidden_norm", hidden),
    )
    # single-step: hidden_norm on the REF fc_out
    hn_atom = model.hidden_norm(f2("a.fc_out", hidden).clone())
    rec(
        "1c hidden_norm(ref fc) vs a.ctx_after_hidden_norm",
        hn_atom,
        f2("a.ctx_after_hidden_norm", hidden),
    )
    ref_ctx = f2("a.ctx_after_hidden_norm", hidden)

    # ================================================== 2. projections
    hdr("STEP 2  layer-0 input_layernorm + fused qkv split")
    noise = R["inputs.noise_embedding"].reshape(T, hidden)
    ln_atom = l0.input_layernorm(noise.clone())
    rec(
        "2a input_layernorm(noise) vs h.l0_input_ln_out",
        ln_atom,
        f2("h.l0_input_ln_out", hidden),
    )
    ref_ln = f2("h.l0_input_ln_out", hidden)

    q_a, k_a, v_a = sa0._qkv(ref_ln.clone())
    rec(
        "2b q_proj(noise)      vs b.l0_q_proj_noise",
        q_a,
        f2("b.l0_q_proj_noise", nq * hd),
    )
    rec(
        "2c k_proj(noise)      vs b.l0_k_proj_noise",
        k_a,
        f2("b.l0_k_proj_noise", nkv * hd),
    )
    rec(
        "2d v_proj(noise)      vs b.l0_v_proj_noise",
        v_a,
        f2("b.l0_v_proj_noise", nkv * hd),
    )

    _, kc_a, vc_a = sa0._qkv(ref_ctx.clone())
    rec(
        "2e k_proj(ctx)        vs b.l0_k_proj_ctx",
        kc_a,
        f2("b.l0_k_proj_ctx", nkv * hd),
    )
    rec(
        "2f v_proj(ctx)        vs b.l0_v_proj_ctx",
        vc_a,
        f2("b.l0_v_proj_ctx", nkv * hd),
    )

    # ================================================== 3. qk norm
    hdr("STEP 3  q_norm / k_norm  (ATOM norms the two k halves SEPARATELY)")
    ref_q_proj = f2("b.l0_q_proj_noise", nq * hd)
    ref_k_ctx = f2("b.l0_k_proj_ctx", nkv * hd)
    ref_k_noise = f2("b.l0_k_proj_noise", nkv * hd)
    qn_atom = sa0.q_norm(ref_q_proj.clone())
    rec(
        "3a q_norm(ref q_proj) vs c.l0_q_after_qnorm",
        qn_atom,
        R["c.l0_q_after_qnorm"].reshape(T, nq * hd),
    )
    kn_ctx = sa0.k_norm(ref_k_ctx.clone())
    kn_noise = sa0.k_norm(ref_k_noise.clone())
    kn_cat = torch.cat([kn_ctx, kn_noise], dim=0)
    ref_k_norm = R["c.l0_k_after_knorm"].reshape(S, nkv * hd)
    rec(
        "3b cat(k_norm(ctx),k_norm(noise)) vs c.l0_k_after_knorm(concat)",
        kn_cat,
        ref_k_norm,
    )
    nb = int((kn_cat.view(torch.uint16) != ref_k_norm.view(torch.uint16)).sum())
    print(f"       bitwise-unequal elements: {nb} / {kn_cat.numel()}")

    # ================================================== 4. RoPE
    hdr("STEP 4  RoPE  (ATOM rotary_emb on the REF pre-rope q/k, ref positions)")
    # ref r.* are [B, H, S, D]; flatten to [S, H*D]
    ref_q_pre = (
        R["r.l0_q_pre_rope"][0].permute(1, 0, 2).reshape(T, nq * hd).contiguous()
    )
    ref_k_pre = (
        R["r.l0_k_pre_rope"][0].permute(1, 0, 2).reshape(S, nkv * hd).contiguous()
    )
    ref_q_post = R["r.l0_q_post_rope"][0].permute(1, 0, 2).reshape(T, nq * hd)
    ref_k_post = R["r.l0_k_post_rope"][0].permute(1, 0, 2).reshape(S, nkv * hd)
    # consistency: pre-rope q/k must equal the qk-norm outputs (transposed)
    rec(
        "4a ref q_pre_rope == ref q_after_qnorm (layout check)",
        ref_q_pre,
        R["c.l0_q_after_qnorm"].reshape(T, nq * hd),
    )
    rec("4b ref k_pre_rope == ref k_after_knorm (layout check)", ref_k_pre, ref_k_norm)

    # block q/k at absolute positions ctx_len..ctx_len+T-1
    pos_blk = torch.arange(ctx_len, S, device="cuda", dtype=torch.long)
    qb = ref_q_pre.clone()
    kb = ref_k_pre[ctx_len:].clone()
    qb, kb = sa0.rotary_emb(pos_blk, qb, kb)
    rec("4c rope(q, pos 16..22) vs r.l0_q_post_rope", qb, ref_q_post)
    rec("4d rope(k_noise, 16..22) vs r.l0_k_post_rope[16:]", kb, ref_k_post[ctx_len:])

    # context k at absolute positions 0..ctx_len-1 (what write_context_kv does)
    pos_ctx = torch.arange(0, ctx_len, device="cuda", dtype=torch.long)
    kc = ref_k_pre[:ctx_len].clone()
    dummy = torch.empty(ctx_len, nq * hd, device="cuda", dtype=kc.dtype)
    _, kc = sa0.rotary_emb(pos_ctx, dummy, kc)
    rec("4e rope(k_ctx, 0..15)  vs r.l0_k_post_rope[:16]", kc, ref_k_post[:ctx_len])

    # wrong-slice detector (what a cos[..., :q_len] bug would look like)
    q_wrong = ref_q_pre.clone()
    kw = ref_k_pre[:T].clone()
    q_wrong, _ = sa0.rotary_emb(torch.arange(0, T, device="cuda"), q_wrong, kw)
    wd = (q_wrong.float() - ref_q_post.float()).abs().max().item()
    print(
        f"       [control] rope(q at pos 0..6) maxdiff vs ref = {wd:.4e} "
        f"(a slice bug would read ~this, not ~0)"
    )

    # ================================================== 5. attention
    hdr("STEP 5  layer-0 attention (paged pool + mha_v3, non-causal, ctx_lens=23)")
    from atom.model_ops.attentions.mha_kv_pool import MhaKvPool

    BLOCK = 128
    pool = MhaKvPool(
        layers=len(model.layers),
        block_size=BLOCK,
        num_kv_heads=nkv,
        head_dim=hd,
        kv_dtype=torch.bfloat16,
        blocks_per_entry=1,
    )
    pool.allocate(8, "cuda")
    page = 3  # deliberately not page 0
    slots_all = torch.arange(S, device="cuda", dtype=torch.int64) + page * BLOCK
    block_tables = torch.tensor([[page]], dtype=torch.int32, device="cuda")
    ctx_lens_t = torch.tensor([S], dtype=torch.int32, device="cuda")

    ref_v_all = torch.cat(
        [f2("b.l0_v_proj_ctx", nkv * hd), f2("b.l0_v_proj_noise", nkv * hd)], dim=0
    )
    k_cache, v_cache = pool.fa_views(0)
    sa0._write_block_kv(
        ref_k_post.contiguous().view(-1, nkv, hd),
        ref_v_all.contiguous().view(-1, nkv, hd),
        slots_all,
        k_cache,
        v_cache,
    )
    torch.cuda.synchronize()
    md = SimpleNamespace(block_tables=block_tables, context_lens=ctx_lens_t)
    o_atom = sa0._attend_non_causal(ref_q_post.contiguous(), md, k_cache, v_cache)
    rec(
        "5a mha_v3(ref q/k/v)  vs d.l0_attn_preo", o_atom, f2("d.l0_attn_preo", nq * hd)
    )

    # dense torch control over the SAME ref values (is mha_v3 or the ref at fault?)
    Kd = ref_k_post.float().view(S, nkv, hd)
    Vd = ref_v_all.float().view(S, nkv, hd)
    Qd = ref_q_post.float().view(T, nq, hd)
    dense = torch.zeros(T, nq, hd, device="cuda")
    for h in range(nq):
        sc = (Qd[:, h] @ Kd[:, h // (nq // nkv)].T) * sa0.scaling
        dense[:, h] = torch.softmax(sc, -1) @ Vd[:, h // (nq // nkv)]
    rec(
        "5b dense torch ref    vs d.l0_attn_preo",
        dense.reshape(T, -1),
        f2("d.l0_attn_preo", nq * hd),
    )

    o_proj_atom = _linear_out(sa0.o_proj(f2("d.l0_attn_preo", nq * hd).clone()))
    rec(
        "5c o_proj(ref preo)   vs d.l0_attn_out",
        o_proj_atom,
        f2("d.l0_attn_out", hidden),
    )

    # ================================================== 6. full stack
    hdr("STEP 6  full 5-layer stack through ATOM's real forward + final norm")
    import atom.utils.forward_context as fc_mod

    kv_data = {}
    for i in range(len(model.layers)):
        kc_i, vc_i = pool.fa_views(i)
        kc_i.zero_()
        vc_i.zero_()
        kv_data[f"layer_{i}"] = SimpleNamespace(k_cache=kc_i, v_cache=vc_i)

    slot_blk = slots_all[ctx_len:]
    slot_ctx = slots_all[:ctx_len]
    fake_md = SimpleNamespace(
        block_tables=block_tables,
        context_lens=ctx_lens_t,
        slot_mapping=slot_blk,
    )
    fake_ctx = SimpleNamespace(attn_metadata=fake_md, kv_cache_data=kv_data)
    orig_gfc = fc_mod.get_forward_context
    fc_mod.get_forward_context = lambda: fake_ctx
    try:
        # context rows: ATOM's own project_context output, its own write path
        for i, lyr in enumerate(model.layers):
            lyr.self_attn.write_context_kv(ctx_atom, pos_ctx, slot_ctx, fake_ctx)
        torch.cuda.synchronize()
        h_atom = noise.clone()
        for i, lyr in enumerate(model.layers):
            h_atom = lyr(pos_blk, h_atom)
            rec(
                f"6{chr(97+i)} layer {i} hidden_out  vs h.l{i}_hidden_out",
                h_atom,
                f2(f"h.l{i}_hidden_out", hidden),
            )
        fin = model.norm(h_atom.clone())
        rec(
            "6f final norm          vs e.final_norm_out",
            fin,
            f2("e.final_norm_out", hidden),
        )
        rec(
            "6g final norm          vs e.model_return",
            fin,
            f2("e.model_return", hidden),
        )

        # per-layer single-step: feed each layer the REF's previous hidden
        print("\n  -- single-step (each layer fed the REF's input hidden) --")
        for i, lyr in enumerate(model.layers):
            hin = (
                noise.clone() if i == 0 else f2(f"h.l{i-1}_hidden_out", hidden).clone()
            )
            hout = lyr(pos_blk, hin)
            rec(f"6s{i} layer {i} single-step", hout, f2(f"h.l{i}_hidden_out", hidden))

        # Localize 6s0: is layer 0's extra drift the ATOM-projected context?
        print("\n  -- control: context KV written from the REF's ctx hidden --")
        for i in range(len(model.layers)):
            kc_i, vc_i = pool.fa_views(i)
            kc_i.zero_()
            vc_i.zero_()
        for lyr in model.layers:
            lyr.self_attn.write_context_kv(ref_ctx.clone(), pos_ctx, slot_ctx, fake_ctx)
        torch.cuda.synchronize()
        h2 = noise.clone()
        for i, lyr in enumerate(model.layers):
            h2 = lyr(pos_blk, h2)
            rec(
                f"6r{i} layer {i} hidden_out (ref ctx)",
                h2,
                f2(f"h.l{i}_hidden_out", hidden),
            )
        rec(
            "6rf final norm (ref ctx)",
            model.norm(h2.clone()),
            f2("e.final_norm_out", hidden),
        )
    finally:
        fc_mod.get_forward_context = orig_gfc

    # ================================================== 7. markov head
    hdr("STEP 7  markov head")
    tok = R["inputs.token_ids"].reshape(-1)
    lat = model.markov_head.markov_w1(tok)
    rec(
        "7a markov_w1[ids]     vs m.markov_latent",
        lat,
        R["m.markov_latent"].reshape(T, -1),
    )
    bias, _ = model.markov_head(tok)
    rec(
        "7b markov bias (fp32) vs m.markov_bias",
        bias,
        R["m.markov_bias"].reshape(T, -1),
    )
    bias_bf16 = torch.nn.functional.linear(
        R["m.markov_latent"].reshape(T, -1), model.markov_head.markov_w2.weight
    )
    rec(
        "7c markov bias (bf16, ref recipe) vs m.markov_bias",
        bias_bf16,
        R["m.markov_bias"].reshape(T, -1),
    )
    print(
        f"       fused_sample={model.markov_head.fused_sample}  "
        f"w2 stored as {type(model.markov_head.markov_w2).__name__} "
        f"shape={tuple(model.markov_head.markov_w2.weight.shape)}"
    )

    # _sample_block semantics: logits_k <- U_k + B(x_{k-1}, .), x_k = argmax
    V = model.vocab_size
    g = torch.Generator(device="cuda").manual_seed(31337)
    base = torch.randn(
        1, T, V, generator=g, device="cuda", dtype=torch.float32
    ).bfloat16()
    anchor = tok[:1].clone()
    got = model._sample_block(base, anchor)
    # explicit reference loop
    prev = anchor.clone()
    want = []
    w1 = model.markov_head.markov_w1.weight
    w2 = model.markov_head.markov_w2.weight
    for k in range(T):
        b = w1[prev].float() @ w2.float().t()
        nxt = (base[:, k] + b).argmax(-1)
        want.append(nxt)
        prev = nxt
    want = torch.stack(want, dim=1)
    same = bool(torch.equal(got, want))
    print(
        f"  [{'OK ' if same else 'XX '}] 7d _sample_block == explicit "
        f"U_k + B(x_(k-1),.) greedy loop: {same}  got={got.tolist()} want={want.tolist()}"
    )
    ROWS.append(
        (
            "7d _sample_block greedy semantics",
            1.0 if same else 0.0,
            0.0,
            0.0,
            "OK " if same else "XX ",
            "exact id match",
        )
    )

    # ================================================== summary
    hdr("SUMMARY")
    print(f"{'checkpoint':48s} {'cos':>12s} {'maxdiff':>12s} {'rel':>10s}  flag")
    first = None
    for name, cos, mad, rel, flag, note in ROWS:
        print(f"{name:48s} {cos:12.8f} {mad:12.4e} {rel:10.3e}  {flag}")
        if first is None and cos < 0.999:
            first = name
    print(f"\nFIRST checkpoint below cos 0.999: {first or 'NONE -- all clean'}")


if __name__ == "__main__":
    sys.exit(main())
