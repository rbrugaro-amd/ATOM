"""M0: can aiter's gluon paged decode run at head_dim=64?

Shape under test is the DFlash draft at TP8/DCP1: 2 KV heads/rank, 8 Q heads/rank
(gqa_ratio 4), qlen=1, block_size 128. head_dim=128 is the control.
Checks (a) it runs, (b) output matches a torch reference, (c) the LSE is
recoverable from exp_sums/max_logits -- which is what the DCP merge needs.
"""
import torch, aiter
from aiter.ops.triton.gluon.pa_decode_gluon import pa_decode_gluon

DEV = "cuda"
torch.manual_seed(0)


def build(head_dim, dtype, num_blocks=64, block_size=128, n_kv=2, gqa=4, n_seq=3):
    n_q = n_kv * gqa
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    k = torch.randn(num_blocks, n_kv, head_dim // x, block_size, x, device=DEV).to(dtype)
    v = torch.randn(num_blocks, n_kv, head_dim, block_size, device=DEV).to(dtype)
    q = torch.randn(n_seq, n_q, head_dim, device=DEV).to(dtype)
    ctx = torch.tensor([300, 128, 777], dtype=torch.int32, device=DEV)[:n_seq]
    max_bt = (int(ctx.max()) + block_size - 1) // block_size
    bt = torch.arange(n_seq * max_bt, dtype=torch.int32, device=DEV).reshape(n_seq, max_bt) % num_blocks
    return q, k, v, ctx, bt, n_q, n_kv, block_size, x


def reference(q, k, v, ctx, bt, n_kv, block_size, x, head_dim, scale):
    """Gather the paged cache back into dense [ctx, head_dim] and attend."""
    n_seq, n_q, _ = q.shape
    gqa = n_q // n_kv
    out = torch.zeros_like(q, dtype=torch.float32)
    lse = torch.zeros(n_seq, n_q, device=DEV, dtype=torch.float32)
    for s in range(n_seq):
        L = int(ctx[s])
        for h in range(n_q):
            hk = h // gqa
            K = torch.empty(L, head_dim, device=DEV, dtype=torch.float32)
            V = torch.empty(L, head_dim, device=DEV, dtype=torch.float32)
            for p in range(L):
                blk, off = int(bt[s, p // block_size]), p % block_size
                K[p] = k[blk, hk, :, off, :].reshape(-1).float()[:head_dim]
                V[p] = v[blk, hk, :, off].float()
            s_ = (q[s, h].float() @ K.T) * scale
            m = s_.max()
            e = torch.exp(s_ - m)
            out[s, h] = (e / e.sum()) @ V
            lse[s, h] = m + torch.log(e.sum())
    return out, lse


def run(head_dim, dtype=torch.bfloat16):
    q, k, v, ctx, bt, n_q, n_kv, bs, x = build(head_dim, dtype)
    n_seq = q.shape[0]
    scale = head_dim ** -0.5
    qgs = 1 * (n_q // n_kv)          # query_length * gqa_ratio
    parts = 4
    o = torch.empty_like(q)
    shape = (n_seq, n_kv, parts, qgs)
    exp_sums = torch.empty(shape, dtype=torch.float32, device=DEV)
    max_logits = torch.empty(shape, dtype=torch.float32, device=DEV)
    tmp = torch.empty(*shape, head_dim, dtype=q.dtype, device=DEV)

    pa_decode_gluon(
        output=o, query=q, key_cache=k, value_cache=v,
        context_lengths=ctx, block_tables=bt, softmax_scale=scale,
        query_length=1, max_context_partition_num=parts, context_partition_size=256,
        compute_type=dtype, exp_sums=exp_sums, max_logits=max_logits,
        temporary_output=tmp, ps=True,
    )
    torch.cuda.synchronize()

    ref_o, ref_lse = reference(q, k, v, ctx, bt, n_kv, bs, x, head_dim, scale)
    err = (o.float() - ref_o).abs().max().item()
    rel = err / max(ref_o.abs().max().item(), 1e-6)

    # LSE from the per-partition softmax state, masked to live partitions.
    npart = torch.ceil(ctx.float() / 256).clamp(max=parts).to(torch.int64)
    pid = torch.arange(parts, device=DEV).view(1, 1, parts, 1)
    live = pid < npart.view(n_seq, 1, 1, 1)
    part_lse = torch.where(live, max_logits + torch.log(exp_sums.clamp_min(1e-30)),
                           torch.full_like(exp_sums, float("-inf")))
    got_lse = torch.logsumexp(part_lse, dim=2).reshape(n_seq, n_kv * qgs)
    lse_err = (got_lse - ref_lse).abs().max().item()
    return err, rel, lse_err


print(f"{'head_dim':>9} {'dtype':>9} {'max_abs':>10} {'rel':>10} {'lse_err':>10}  status")
for hd in (128, 64):
    try:
        e, r, l = run(hd)
        ok = "PASS" if r < 2e-2 and l < 5e-2 else "MISMATCH"
        print(f"{hd:>9} {'bf16':>9} {e:>10.5f} {r:>10.5f} {l:>10.5f}  {ok}")
    except Exception as ex:
        print(f"{hd:>9} {'bf16':>9} {'-':>10} {'-':>10} {'-':>10}  FAIL: {type(ex).__name__}: {str(ex)[:160]}")
