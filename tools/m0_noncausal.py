"""Can aiter do a NON-CAUSAL paged GQA decode at qlen=5, head_dim=64?

This is the shape the DFlash block pass needs: T=5 queries per request, each
attending [0, anchor] ++ the T block rows, bidirectionally. The MLA draft gets
this from get_mla_metadata_v1(..., is_causal=False); the GQA analogue is
get_pa_metadata_v1(..., is_causal) + pa_persistent_fwd(mask=0).
"""

import aiter
import torch

DEV = "cuda"
torch.manual_seed(0)
BF16 = torch.bfloat16


def trial(hd, n_kv, gqa, T, ctx_list, block_size=128, nb=256, causal=False):
    n_q = n_kv * gqa
    B = len(ctx_list)
    x = 16 // torch.tensor([], dtype=BF16).element_size()
    k_cache = torch.randn(nb, n_kv, hd // x, block_size, x, device=DEV).to(BF16)
    v_cache = torch.randn(nb, n_kv, hd, block_size, device=DEV).to(BF16)
    q = torch.randn(B * T, n_q, hd, device=DEV).to(BF16)

    ctx = torch.tensor(ctx_list, dtype=torch.int32, device=DEV)
    pages = [(c + block_size - 1) // block_size for c in ctx_list]
    kv_indptr = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(pages), 0)), dtype=torch.int32, device=DEV
    )
    kv_indices = torch.arange(int(kv_indptr[-1]), dtype=torch.int32, device=DEV) % nb
    qo_indptr = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=DEV)

    sizes = aiter.get_pa_metadata_info_v1(B, n_kv)
    bufs = [torch.empty(sz, dtype=dt, device=DEV) for (sz, dt) in sizes]
    work_meta, work_indptr, work_info, red_indptr, red_final, red_partial = bufs

    aiter.get_pa_metadata_v1(
        qo_indptr,
        kv_indptr,
        ctx,
        gqa,
        n_kv,
        causal,
        work_meta,
        work_indptr,
        work_info,
        red_indptr,
        red_final,
        red_partial,
        kv_granularity=max(block_size, 16),
        block_size=block_size,
        max_seqlen_qo=T,
        uni_seqlen_qo=T,
        fast_mode=True,
        max_split_per_batch=-1,
    )
    out = torch.empty_like(q)
    aiter.pa_persistent_fwd(
        Q=q,
        K=k_cache,
        V=v_cache,
        output=out,
        max_qlen=T,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        context_lens=ctx,
        K_QScale=None,
        V_QScale=None,
        work_indptr=work_indptr,
        work_info=work_info,
        reduce_indptr=red_indptr,
        reduce_final_map=red_final,
        reduce_partial_map=red_partial,
        softmax_scale=hd**-0.5,
        mask=1 if causal else 0,
    )
    torch.cuda.synchronize()

    # torch reference: NON-causal over the full ctx (or bottom-right causal)
    ref = torch.zeros(B * T, n_q, hd, device=DEV)
    for s in range(B):
        L = ctx_list[s]
        base = int(kv_indptr[s])
        for h in range(n_q):
            hk = h // gqa
            K = torch.stack(
                [
                    k_cache[
                        int(kv_indices[base + p // block_size]),
                        hk,
                        :,
                        p % block_size,
                        :,
                    ]
                    .reshape(-1)
                    .float()[:hd]
                    for p in range(L)
                ]
            )
            V = torch.stack(
                [
                    v_cache[
                        int(kv_indices[base + p // block_size]), hk, :, p % block_size
                    ].float()
                    for p in range(L)
                ]
            )
            for t in range(T):
                sc = (q[s * T + t, h].float() @ K.T) * (hd**-0.5)
                if causal:
                    sc[L - T + t + 1 :] = float("-inf")
                ref[s * T + t, h] = torch.softmax(sc, -1) @ V
    return (out.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)


print(f"{'hd':>4} {'T':>3} {'mode':>10} {'n_kv':>5} {'rel_err':>9}  status")
for hd in (128, 64):
    for T, causal in ((5, False), (5, True), (1, False)):
        for n_kv in (2, 16):
            try:
                r = trial(hd, n_kv, 4, T, [300, 777, 128], causal=causal)
                st = "PASS" if r < 2e-2 else "MISMATCH"
                print(
                    f"{hd:>4} {T:>3} {'causal' if causal else 'NONcausal':>10} {n_kv:>5} {r:>9.5f}  {st}"
                )
            # Blind except on purpose: this is a kernel probe, so any failure IS a
            # result worth printing, and aborts here are raw HIP / gl.static_assert
            # errors with no common base class to narrow to.
            except Exception as e:  # noqa: BLE001
                print(
                    f"{hd:>4} {T:>3} {'causal' if causal else 'NONcausal':>10} {n_kv:>5} {'-':>9}  FAIL: {type(e).__name__}: {str(e)[:95]}"
                )
