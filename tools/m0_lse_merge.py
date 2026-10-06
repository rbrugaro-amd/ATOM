"""Does mha_v3's `softmax_lse` use the convention ATOM's DCP merge assumes?

Appendix D measured that `flash_attn_with_kvcache` RETURNS an lse and that its
OUTPUT is accurate. It never checked the lse's numerical convention -- natural
log vs log2, whether `softmax_scale` is folded in, whether the max-subtraction
is undone. The entire TP8/DCP8 merge rests on that convention, and a wrong one
produces a plausible-looking but wrong attention with no error.

This closes it on ONE GPU with NO server: split a paged KV into W disjoint
round-robin token shards (exactly how DCP shards it), run the kernel once per
shard, merge the partials with ATOM's own `correct_attn_out` kernel, and
compare against a single full-KV call.

`correct_attn_out` is the AG+RS backend's combine step and is where the
convention lives; running it W times locally and summing is that backend's math
with the collectives deleted. A pass here means `dcp_lse_merge` is fed what it
expects.

    python3 tools/m0_lse_merge.py
"""

import torch
from aiter.ops.triton.attention.mha_v3 import flash_attn_with_kvcache

from atom.model_ops.dcp_ops import correct_attn_out

DEV = "cuda"
PAGE = 128
torch.manual_seed(0)


def _read(k_cache, v_cache, q, page_table, seqlens, scale, want_lse):
    out = flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=seqlens,
        softmax_scale=scale,
        causal=False,
        page_table=page_table,
        return_softmax_lse=want_lse,
    )
    return out if want_lse else (out, None)


def trial(hd=64, n_kv=16, gqa=4, T=5, ctx_list=(300, 777, 131), W=8, nb=512):
    """One (full read) vs (W shard reads + merge) comparison."""
    n_q = n_kv * gqa
    B = len(ctx_list)
    scale = hd**-0.5

    # The GLOBAL cache, laid out as one dense page run per request so the
    # reference read is trivially right.
    maxp = (max(ctx_list) + PAGE - 1) // PAGE
    k_cache = torch.randn(nb, PAGE, n_kv, hd, device=DEV, dtype=torch.bfloat16)
    v_cache = torch.randn(nb, PAGE, n_kv, hd, device=DEV, dtype=torch.bfloat16)
    q = torch.randn(B, T, n_q, hd, device=DEV, dtype=torch.bfloat16)
    page_table = torch.arange(B * maxp, dtype=torch.int32, device=DEV).reshape(B, maxp)
    seqlens = torch.tensor(ctx_list, dtype=torch.int32, device=DEV)

    full, _ = _read(k_cache, v_cache, q, page_table, seqlens, scale, False)

    # Now the DCP view of the same cache: rank r owns global positions
    # p with p % W == r, packed densely into its own pages.
    rows = B * T
    merged = torch.zeros(rows, n_q, hd, device=DEV, dtype=torch.float32)
    shard_out, shard_lse, owned = [], [], []
    for r in range(W):
        local_lens = [(L + W - 1 - r) // W for L in ctx_list]
        owned.append(local_lens)
        lmaxp = max(1, (max(local_lens) + PAGE - 1) // PAGE)
        kr = torch.zeros(B * lmaxp, PAGE, n_kv, hd, device=DEV, dtype=torch.bfloat16)
        vr = torch.zeros_like(kr)
        ptab = torch.arange(B * lmaxp, dtype=torch.int32, device=DEV).reshape(B, lmaxp)
        for s in range(B):
            for i, p in enumerate(range(r, ctx_list[s], W)):
                src_pg = int(page_table[s, p // PAGE])
                dst = int(ptab[s, i // PAGE])
                kr[dst, i % PAGE] = k_cache[src_pg, p % PAGE]
                vr[dst, i % PAGE] = v_cache[src_pg, p % PAGE]
        # Floor at one dummy row exactly as the proposer does: the kernel
        # cannot run a zero-page row, and the true count travels separately.
        lens_t = torch.tensor(
            [max(1, x) for x in local_lens], dtype=torch.int32, device=DEV
        )
        o, lse = _read(kr, vr, q, ptab, lens_t, scale, True)
        # out [B,T,H,D] -> [B*T,H,D]; lse [B,H,T] -> [B*T,H]
        shard_out.append(o.reshape(rows, n_q, hd).float())
        shard_lse.append(lse.transpose(1, 2).reshape(rows, n_q).float())

    lses = torch.stack(shard_lse)  # [W, B*T, H]
    # Mask the rows a rank genuinely owns nothing of. Its lse comes back as a
    # FINITE 0.0 over the dummy row, which no NaN scrub can catch.
    own_t = torch.tensor(owned, device=DEV)  # [W, B]
    lses = lses.masked_fill(
        own_t.view(W, B, 1).expand(W, B, T).reshape(W, rows, 1) == 0, float("-inf")
    )

    for r in range(W):
        corrected, _ = correct_attn_out(shard_out[r].clone(), lses, r)
        merged += corrected

    ref = full.reshape(rows, n_q, hd).float()
    return (merged - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)


print(f"{'hd':>4} {'n_kv':>5} {'T':>3} {'W':>3} {'rel_err':>9}  status")
for hd in (64, 128):
    for n_kv in (16, 2):
        for T in (5, 7):
            for W in (2, 8):
                try:
                    r = trial(hd=hd, n_kv=n_kv, gqa=64 // n_kv, T=T, W=W)
                    ok = "PASS" if r < 3e-2 else "MISMATCH"
                    print(f"{hd:>4} {n_kv:>5} {T:>3} {W:>3} {r:>9.5f}  {ok}")
                # Blind except on purpose: this is a kernel probe, so any
                # failure IS a result worth printing, and aborts here are raw
                # HIP / gl.static_assert errors with no common base class.
                except Exception as e:  # noqa: BLE001
                    print(
                        f"{hd:>4} {n_kv:>5} {T:>3} {W:>3} {'-':>9}  "
                        f"FAIL: {type(e).__name__}: {str(e)[:90]}"
                    )
