"""Phase 6: is the WRITE->READ round trip through ATOM's own MhaKvPool exact?

Write  : aiter.reshape_and_cache_flash(...)   into MhaKvPool.fa_views(layer)
Read   : aiter.ops.triton.attention.mha_v3.flash_attn_with_kvcache(causal=False,
         page_table=..., cache_seqlens=<Tensor>)
Target : a dense torch reference over the SAME values at the SAME positions.

Five tests, increasing strength:
  T1 layout identity   -- fa_views aliases the pool, is contiguous, per-layer disjoint
  T2 write/read identity (no attention) -- bit-exact index-encoded scatter
  T3 attention round trip -- mha_v3 vs dense gather reference
  T4 slot arithmetic   -- non-page-aligned context lens, anchor on a page boundary
  T5 multi-layer       -- all layers, distinct values, no cross-layer aliasing

Mirrors atom/models/kimi_k3_dflash.py `_write_block_kv` / `_attend_non_causal`
call-for-call (argument order, "auto" dtype, 1-element fp32 unit scales).
"""

import sys

import torch

sys.path.insert(0, "/work")

import aiter
from aiter.ops.triton.attention.mha_v3 import flash_attn_with_kvcache

from atom.model_ops.attentions.mha_kv_pool import MhaKvPool

DEV = "cuda"
KVD = torch.bfloat16
LAYERS = 5
BLOCK = 128
HD = 64
GQA = 4
T = 5
BLOCKS = 96

FAIL = []


def rec(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    if not ok:
        FAIL.append(name)
    print(f"  [{tag}] {name}{(' -- ' + detail) if detail else ''}")


def hdr(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}")


# ---------------------------------------------------------------- bf16 codec
# Every 16-bit pattern with exponent in 1..254 is an exact, finite, distinct
# bf16 value: 2*254*128 = 65024 of them. A bijection from a linear element
# index onto those makes any permutation of ANY axis detectable by value alone.
CODE_SPACE = 2 * 254 * 128


def encode(u: torch.Tensor) -> torch.Tensor:
    """int64 index tensor -> exact distinct bf16 values (bijective on <CODE_SPACE)."""
    if int(u.max()) >= CODE_SPACE:
        raise ValueError(f"index {int(u.max())} outside the {CODE_SPACE} exact codes")
    sign = torch.div(u, 254 * 128, rounding_mode="floor")
    rest = u % (254 * 128)
    e = torch.div(rest, 128, rounding_mode="floor") + 1  # 1..254
    m = rest % 128
    bits = ((sign << 15) | (e << 7) | m).to(torch.uint16)
    return bits.view(torch.bfloat16)


def decode(x: torch.Tensor) -> torch.Tensor:
    bits = x.contiguous().view(torch.uint16).to(torch.int64)
    sign = (bits >> 15) & 1
    e = (bits >> 7) & 0xFF
    m = bits & 0x7F
    return sign * (254 * 128) + (e - 1) * 128 + m


def byte_range(t):
    return t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()


def make_pool(nh):
    p = MhaKvPool(
        layers=LAYERS,
        block_size=BLOCK,
        num_kv_heads=nh,
        head_dim=HD,
        kv_dtype=KVD,
        blocks_per_entry=1,
    )
    p.allocate(BLOCKS, DEV)
    return p


UNIT = torch.ones(1, dtype=torch.float32, device=DEV)


def write_kv(k, v, slot_mapping, k_cache, v_cache):
    """Exactly `KimiK3DFlashAttention._write_block_kv`."""
    aiter.reshape_and_cache_flash(
        k, v, k_cache, v_cache, slot_mapping, "auto", UNIT, UNIT
    )


def attend(q, k_cache, v_cache, cache_seqlens, block_tables, scaling):
    """Exactly `KimiK3DFlashAttention._attend_non_causal`."""
    return flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        softmax_scale=scaling,
        causal=False,
        page_table=block_tables,
    )


# =================================================================== TEST 1
def test1_layout(nh):
    hdr(f"TEST 1  LAYOUT IDENTITY  (num_kv_heads={nh})")
    pool = make_pool(nh)
    raw_k = pool._views["k"]
    raw_v = pool._views["v"]
    ranges = {}
    for L in range(LAYERS):
        k, v = pool.fa_views(L)
        ok_shape = tuple(k.shape) == (BLOCKS, BLOCK, nh, HD) and tuple(v.shape) == (
            BLOCKS,
            BLOCK,
            nh,
            HD,
        )
        rec(
            f"L{L} shape == [blocks,block_size,kv_heads,head_dim]",
            ok_shape,
            f"k{tuple(k.shape)} v{tuple(v.shape)}",
        )
        rec(
            f"L{L} contiguous",
            k.is_contiguous() and v.is_contiguous(),
            f"k_stride={k.stride()}",
        )
        rec(
            f"L{L} zero-copy over pool storage",
            k.data_ptr() == raw_k[L].data_ptr()
            and v.data_ptr() == raw_v[L].data_ptr()
            and k.untyped_storage().data_ptr() == raw_k.untyped_storage().data_ptr(),
            f"fa k ptr={k.data_ptr():#x} raw k ptr={raw_k[L].data_ptr():#x}",
        )
        rec(f"L{L} dtype bf16", k.dtype is KVD and v.dtype is KVD)
        ranges[("k", L)] = byte_range(k)
        ranges[("v", L)] = byte_range(v)

    items = list(ranges.items())
    overlaps = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (na, (a0, a1)), (nb, (b0, b1)) = items[i], items[j]
            if a0 < b1 and b0 < a1:
                overlaps.append((na, nb))
    rec("all 5 layers x {k,v} byte-disjoint", not overlaps, str(overlaps[:4]))

    # Positive aliasing check: writing layer L must not perturb any other layer.
    k0, _ = pool.fa_views(0)
    k0.fill_(1.0)
    dirty = [L for L in range(1, LAYERS) if pool.fa_views(L)[0].abs().sum().item() != 0]
    dirty += [L for L in range(LAYERS) if pool.fa_views(L)[1].abs().sum().item() != 0]
    rec(
        "write to k[L0] leaves every other (layer, field) at zero",
        not dirty,
        str(dirty),
    )
    del pool
    torch.cuda.empty_cache()


# =================================================================== TEST 2
def test2_write_identity(nh, layer, slots_py, label=""):
    """Index-encoded scatter: assert every (block, token, head, chan) landed
    exactly where slot_mapping said. Bit-exact, no tolerance."""
    pool = make_pool(nh)
    k_cache, v_cache = pool.fa_views(layer)
    n = len(slots_py)
    if n * nh * HD > CODE_SPACE:
        raise ValueError("too many tokens to encode uniquely")
    u = torch.arange(n * nh * HD, device=DEV, dtype=torch.int64).view(n, nh, HD)
    k = encode(u)
    # V gets a disjoint half of the code space so a k/v swap is detectable.
    v = encode(CODE_SPACE - 1 - u)
    slot_mapping = torch.tensor(slots_py, dtype=torch.int64, device=DEV)
    write_kv(k, v, slot_mapping, k_cache, v_cache)
    torch.cuda.synchronize()

    bad = []
    for which, src, cache in (("k", k, k_cache), ("v", v, v_cache)):
        exp = torch.zeros_like(cache)
        for i, s in enumerate(slots_py):
            exp[s // BLOCK, s % BLOCK] = src[i]
        mism = (cache.view(torch.uint16) != exp.view(torch.uint16)).nonzero()
        if mism.numel():
            bad.append((which, mism, cache, exp, src))
    name = f"T2 write/read identity nh={nh} layer={layer} n={n} {label}"
    if not bad:
        rec(name, True, f"{n * nh * HD * 2} elements bit-exact")
    else:
        which, mism, cache, exp, src = bad[0]
        print(
            f"  mismatching {which} elements: {mism.shape[0]}; first 8 (blk,tok,h,c):"
        )
        for row in mism[:8].tolist():
            b, t, h, c = row
            got = decode(cache[b, t, h, c].reshape(1)).item()
            want = decode(exp[b, t, h, c].reshape(1)).item()
            # where did the value we got actually come from?
            src_lin = got if which == "k" else CODE_SPACE - 1 - got
            si, sh, sc = src_lin // (nh * HD), (src_lin // HD) % nh, src_lin % HD
            print(
                f"    at blk={b} tok={t} h={h} c={c}: got code {got} "
                f"(= src token {si} head {sh} chan {sc}), wanted code {want}"
            )
        rec(name, False, f"{mism.shape[0]} mismatches")
    del pool
    torch.cuda.empty_cache()


# =================================================================== TEST 3/4
def dense_ref(k_cache, v_cache, q, ctx, page_table, scale, nh):
    """Gather cache rows per request via the page table; plain non-causal softmax."""
    B, t, nq, _ = q.shape
    out = torch.zeros(B, t, nq, HD, device=DEV, dtype=torch.float32)
    for s in range(B):
        L = ctx[s]
        pos = torch.arange(L, device=DEV)
        pages = page_table[s, pos // BLOCK].long()
        off = pos % BLOCK
        K = k_cache[pages, off].float()  # [L, nh, hd]
        V = v_cache[pages, off].float()
        for h in range(nq):
            hk = h // (nq // nh)
            sc = (q[s, :, h].float() @ K[:, hk].T) * scale  # [t, L]
            out[s, :, h] = torch.softmax(sc, -1) @ V[:, hk]
    return out


def roundtrip(nh, layer, ctx_lens, anchor_note="", seed=0):
    """Populate the pool through reshape_and_cache_flash at exactly the slots
    the proposer computes, then read it back through mha_v3 and compare."""
    torch.manual_seed(seed)
    pool = make_pool(nh)
    k_cache, v_cache = pool.fa_views(layer)
    B = len(ctx_lens)
    nq = nh * GQA
    maxp = (max(ctx_lens) + BLOCK - 1) // BLOCK
    # Deliberately non-identity, non-monotonic page ids: a layout bug that
    # happens to work for block_table == arange would survive otherwise.
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    perm = torch.randperm(BLOCKS, generator=g)[: B * maxp]
    page_table = perm.reshape(B, maxp).to(torch.int32).to(DEV)

    # --- write EVERY context row through the production writer ---------------
    slots, kvals, vvals = [], [], []
    for s in range(B):
        for p in range(ctx_lens[s]):
            # the production slot formula (dspark_proposer, dcp_world_size == 1)
            slots.append(int(page_table[s, p // BLOCK]) * BLOCK + p % BLOCK)
    n = len(slots)
    kvals = torch.randn(n, nh, HD, device=DEV, dtype=KVD)
    vvals = torch.randn(n, nh, HD, device=DEV, dtype=KVD)
    slot_mapping = torch.tensor(slots, dtype=torch.int64, device=DEV)
    write_kv(kvals, vvals, slot_mapping, k_cache, v_cache)
    torch.cuda.synchronize()

    # --- slot arithmetic check: every written row readable at its (blk, off) --
    off = 0
    slot_bad = 0
    for s in range(B):
        for p in range(ctx_lens[s]):
            pass
    idx_slot = slot_mapping
    got_k = k_cache.view(-1, nh, HD)[idx_slot]
    slot_bad = int((got_k.view(torch.uint16) != kvals.view(torch.uint16)).sum())
    rec(
        f"T4 slot arithmetic nh={nh} ctx={ctx_lens} {anchor_note}",
        slot_bad == 0,
        f"{slot_bad} bad elements over {n} rows "
        f"(slot = page*{BLOCK} + pos%{BLOCK})",
    )
    del off

    # --- read through mha_v3 exactly as _attend_non_causal does ---------------
    q = torch.randn(B, T, nq, HD, device=DEV, dtype=KVD)
    scale = HD**-0.5
    cache_seqlens = torch.tensor(ctx_lens, dtype=torch.int32, device=DEV)
    out = attend(q, k_cache, v_cache, cache_seqlens, page_table, scale)
    o = out[0] if isinstance(out, (tuple, list)) else out
    torch.cuda.synchronize()
    ref = dense_ref(k_cache, v_cache, q, ctx_lens, page_table, scale, nh)
    rel = (o.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)

    # Second reference, built from the SOURCE tensors rather than the cache, so
    # a write bug the gather reference would inherit cannot hide here.
    ref2 = torch.zeros_like(ref)
    base = 0
    for s in range(B):
        L = ctx_lens[s]
        K = kvals[base : base + L].float()
        V = vvals[base : base + L].float()
        base += L
        for h in range(nq):
            hk = h // GQA
            sc = (q[s, :, h].float() @ K[:, hk].T) * scale
            ref2[s, :, h] = torch.softmax(sc, -1) @ V[:, hk]
    rel2 = (o.float() - ref2).abs().max().item() / max(ref2.abs().max().item(), 1e-6)
    del pool
    torch.cuda.empty_cache()
    return rel, rel2


def main():
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    print(f"geometry: layers={LAYERS} block={BLOCK} head_dim={HD} T={T} gqa={GQA}")

    for nh in (2, 16):
        test1_layout(nh)

    hdr("TEST 2  WRITE/READ IDENTITY (index-encoded, bit-exact, no attention)")
    for nh in (2, 16):
        cap = CODE_SPACE // (nh * HD)
        # contiguous inside one page
        test2_write_identity(
            nh,
            0,
            list(range(10 * BLOCK, 10 * BLOCK + min(T, cap))),
            "contiguous-in-page",
        )
        # straddling a page boundary (the anchor+1+T span of a real request)
        s0 = 7 * BLOCK + BLOCK - 2
        test2_write_identity(
            nh, 2, [s0 + i for i in range(min(T, cap))], "straddles-page-boundary"
        )
        # scattered across non-adjacent, non-monotonic pages
        scattered = [
            93 * BLOCK + 5,
            3 * BLOCK + 127,
            50 * BLOCK,
            3 * BLOCK + 0,
            93 * BLOCK + 6,
        ]
        test2_write_identity(nh, 4, scattered[: min(5, cap)], "scattered-pages")
        # a big write: many tokens, fills the code space
        big = list(range(min(cap, 60) * 1))
        big = [i for i in range(min(cap, 60))]
        test2_write_identity(nh, 1, big, "dense-prefix")

    hdr("TEST 5 + TEST 2  MULTI-LAYER: distinct values per layer, no aliasing")
    for nh in (2, 16):
        pool = make_pool(nh)
        n = min(CODE_SPACE // (nh * HD), 40)
        slots = [17 * BLOCK + 120 + i for i in range(n)]  # straddles 17->18
        slot_mapping = torch.tensor(slots, dtype=torch.int64, device=DEV)
        srcs = {}
        for L in range(LAYERS):
            kc, vc = pool.fa_views(L)
            u = torch.arange(n * nh * HD, device=DEV, dtype=torch.int64).view(n, nh, HD)
            u = (u + L * 7919) % CODE_SPACE
            k = encode(u)
            v = encode((CODE_SPACE - 1 - u) % CODE_SPACE)
            srcs[L] = (k, v)
            write_kv(k, v, slot_mapping, kc, vc)
        torch.cuda.synchronize()
        allbad = []
        for L in range(LAYERS):
            kc, vc = pool.fa_views(L)
            k, v = srcs[L]
            gk = kc.view(-1, nh, HD)[slot_mapping]
            gv = vc.view(-1, nh, HD)[slot_mapping]
            nbk = int((gk.view(torch.uint16) != k.view(torch.uint16)).sum())
            nbv = int((gv.view(torch.uint16) != v.view(torch.uint16)).sum())
            if nbk or nbv:
                allbad.append((L, nbk, nbv))
        rec(
            f"T5 all {LAYERS} layers bit-exact after interleaved writes (nh={nh})",
            not allbad,
            str(allbad),
        )
        # untouched rows of every layer must still be zero
        leak = []
        for L in range(LAYERS):
            kc, vc = pool.fa_views(L)
            mask = torch.ones(BLOCKS * BLOCK, dtype=torch.bool, device=DEV)
            mask[slot_mapping] = False
            if kc.view(-1, nh, HD)[mask].abs().sum().item() != 0:
                leak.append((L, "k"))
            if vc.view(-1, nh, HD)[mask].abs().sum().item() != 0:
                leak.append((L, "v"))
        rec(f"T5 no writes outside slot_mapping (nh={nh})", not leak, str(leak))
        del pool
        torch.cuda.empty_cache()

    hdr("TEST 3 + TEST 4  ATTENTION ROUND TRIP / SLOT ARITHMETIC")
    print(
        f"  {'nh':>3} {'layer':>5} {'ctx_lens':>26} {'rel(vs cache)':>14} "
        f"{'rel(vs source)':>15}  status"
    )
    cases = [
        (2, 0, [300, 777, 128], "anchor-on-page-boundary(128)"),
        (2, 3, [1000, 517, 133, 6], "non-multiples"),
        (16, 0, [300, 777, 128], "anchor-on-page-boundary(128)"),
        (16, 4, [1000, 517, 133, 6], "non-multiples"),
        (2, 2, [BLOCK + 1 + T, 2 * BLOCK + 1 + T, 6], "anchor exactly at page end"),
        (16, 1, [BLOCK + 1 + T, 2 * BLOCK + 1 + T, 6], "anchor exactly at page end"),
    ]
    for nh, L, ctx, note in cases:
        rel, rel2 = roundtrip(nh, L, ctx, note)
        ok = rel < 2e-2 and rel2 < 2e-2
        if not ok:
            FAIL.append(f"T3 nh={nh} L={L} ctx={ctx}")
        print(
            f"  {nh:>3} {L:>5} {ctx!s:>26} {rel:>14.5f} {rel2:>15.5f}  "
            f"{'PASS' if ok else 'MISMATCH'}   {note}"
        )

    hdr("TEST 6  SLOT -1 (pad / unaddressable rows the proposer emits) IS SKIPPED")
    for nh in (2, 16):
        pool = make_pool(nh)
        kc, vc = pool.fa_views(0)
        slots_py = [5 * BLOCK + 3, -1, 5 * BLOCK + 4, -1, 9 * BLOCK]
        n = len(slots_py)
        u = torch.arange(n * nh * HD, device=DEV, dtype=torch.int64).view(n, nh, HD)
        k, v = encode(u), encode(CODE_SPACE - 1 - u)
        sm = torch.tensor(slots_py, dtype=torch.int64, device=DEV)
        try:
            write_kv(k, v, sm, kc, vc)
            torch.cuda.synchronize()
            exp_k = torch.zeros_like(kc)
            exp_v = torch.zeros_like(vc)
            for i, s in enumerate(slots_py):
                if s >= 0:
                    exp_k[s // BLOCK, s % BLOCK] = k[i]
                    exp_v[s // BLOCK, s % BLOCK] = v[i]
            nb = int((kc.view(torch.uint16) != exp_k.view(torch.uint16)).sum()) + int(
                (vc.view(torch.uint16) != exp_v.view(torch.uint16)).sum()
            )
            rec(f"T6 slot -1 rows written nowhere (nh={nh})", nb == 0, f"{nb} bad")
        except Exception as e:  # noqa: BLE001
            rec(
                f"T6 slot -1 rows written nowhere (nh={nh})",
                False,
                f"{type(e).__name__}: {str(e)[:100]}",
            )
        del pool
        torch.cuda.empty_cache()

    hdr("VERDICT")
    if FAIL:
        print(f"ROUND TRIP BROKEN -- {len(FAIL)} failing checks:")
        for f in FAIL:
            print(f"  - {f}")
    else:
        print("ROUND TRIP EXACT -- every check passed.")


if __name__ == "__main__":
    main()
