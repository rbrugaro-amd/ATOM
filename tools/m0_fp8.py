"""M0 extension: gluon paged decode at head_dim=64 with fp8 KV + PER-TOKEN scales."""

import aiter
import torch
from aiter.ops.triton.gluon.pa_decode_gluon import pa_decode_gluon

DEV = "cuda"
torch.manual_seed(0)
FP8 = aiter.dtypes.fp8
FP8_MAX = torch.finfo(FP8).max
print("fp8 dtype:", FP8, "max:", FP8_MAX)


def trial(hd, bs, n_kv, gqa, ctx_list, parts, nb=96):
    n_q = n_kv * gqa
    n_seq = len(ctx_list)
    x = 16  # fp8 -> 16 elems per 16B
    # dense ground truth, then quantize per (block, head, token)
    Kf = torch.randn(nb, n_kv, bs, hd, device=DEV)
    Vf = torch.randn(nb, n_kv, bs, hd, device=DEV)
    ks = (Kf.abs().amax(-1, keepdim=True) / FP8_MAX).clamp_min(1e-8)  # [nb,n_kv,bs,1]
    vs = (Vf.abs().amax(-1, keepdim=True) / FP8_MAX).clamp_min(1e-8)
    Kq = (Kf / ks).to(FP8)
    Vq = (Vf / vs).to(FP8)
    # pack into cache layouts: K [nb,n_kv,hd//x,bs,x], V [nb,n_kv,hd,bs]
    k = (
        Kq.permute(0, 1, 3, 2)
        .reshape(nb, n_kv, hd // x, x, bs)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
    )
    v = Vq.permute(0, 1, 3, 2).contiguous()
    q = torch.randn(n_seq, n_q, hd, device=DEV).to(torch.bfloat16)
    ctx = torch.tensor(ctx_list, dtype=torch.int32, device=DEV)
    mbt = (max(ctx_list) + bs - 1) // bs
    bt = (
        torch.arange(n_seq * mbt, dtype=torch.int32, device=DEV).reshape(n_seq, mbt)
        % nb
    )
    scale = hd**-0.5
    qgs = gqa
    o = torch.empty(n_seq, n_q, hd, device=DEV, dtype=torch.bfloat16)
    shape = (n_seq, n_kv, parts, qgs)
    es = torch.empty(shape, dtype=torch.float32, device=DEV)
    ml = torch.empty(shape, dtype=torch.float32, device=DEV)
    tmp = torch.empty(*shape, hd, dtype=torch.bfloat16, device=DEV)
    pa_decode_gluon(
        output=o,
        query=q,
        key_cache=k,
        value_cache=v,
        context_lengths=ctx,
        block_tables=bt,
        softmax_scale=scale,
        query_length=1,
        max_context_partition_num=parts,
        context_partition_size=256,
        compute_type=FP8,
        key_scale=ks,
        value_scale=vs,
        exp_sums=es,
        max_logits=ml,
        temporary_output=tmp,
        ps=True,
    )
    torch.cuda.synchronize()
    # reference from the DEQUANTIZED cache
    Kd = Kq.float() * ks
    Vd = Vq.float() * vs
    ro = torch.zeros(n_seq, n_q, hd, device=DEV)
    rl = torch.zeros(n_seq, n_q, device=DEV)
    for s in range(n_seq):
        L = ctx_list[s]
        for h in range(n_q):
            hk = h // gqa
            idx = [(int(bt[s, p // bs]), p % bs) for p in range(L)]
            K = torch.stack([Kd[b, hk, t] for b, t in idx])
            V = torch.stack([Vd[b, hk, t] for b, t in idx])
            sc = (q[s, h].float() @ K.T) * scale
            m = sc.max()
            e = torch.exp(sc - m)
            ro[s, h] = (e / e.sum()) @ V
            rl[s, h] = m + torch.log(e.sum())
    pl = ml + torch.log(es.clamp_min(1e-30))
    got = torch.logsumexp(pl, dim=2).reshape(n_seq, n_kv * qgs)
    return (o.float() - ro).abs().max().item() / max(ro.abs().max().item(), 1e-6), (
        got - rl
    ).abs().max().item()


print(
    f"{'hd':>4} {'blk':>4} {'parts':>6} {'ctx':>20} {'out_rel':>9} {'lse_err':>9}  status"
)
for hd, blk, parts, ctxs in [
    (64, 128, 4, [300, 128, 777]),
    (128, 128, 4, [300, 128, 777]),
    (64, 128, 8, [300, 128, 777]),
    (64, 128, 4, [64, 1000, 2048]),
]:
    try:
        r, l = trial(hd, blk, 2, 4, ctxs, parts)
        print(
            f"{hd:>4} {blk:>4} {parts:>6} {ctxs!s:>20} {r:>9.5f} {l:>9.5f}  {'PASS' if r<6e-2 and l<5e-2 else 'MISMATCH'}"
        )
    # Blind except on purpose: this is a kernel probe, so any failure IS a
    # result worth printing, and aborts here are raw HIP / gl.static_assert
    # errors with no common base class to narrow to.
    except Exception as e:  # noqa: BLE001
        print(
            f"{hd:>4} {blk:>4} {parts:>6} {ctxs!s:>20} {'-':>9} {'-':>9}  FAIL: {type(e).__name__}: {str(e)[:110]}"
        )
