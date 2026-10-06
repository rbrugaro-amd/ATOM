"""Does aiter's Triton mha_v3 give a NON-CAUSAL paged GQA decode at qlen=5, head_dim=64?

If yes, the DFlash block pass needs no qlen=1 fan-out: one row per request,
T queries, bidirectional over [0,anchor]++T block rows.
"""
import torch
from aiter.ops.triton.attention.mha_v3 import flash_attn_with_kvcache
DEV="cuda"; torch.manual_seed(0)

def trial(hd, n_kv, gqa, T, ctx_list, page=128, nb=256, causal=False, lse=False):
    n_q = n_kv*gqa; B=len(ctx_list)
    # standard FA paged layout: [num_blocks, page_size, num_kv_heads, head_dim]
    k_cache = torch.randn(nb, page, n_kv, hd, device=DEV, dtype=torch.bfloat16)
    v_cache = torch.randn(nb, page, n_kv, hd, device=DEV, dtype=torch.bfloat16)
    q = torch.randn(B, T, n_q, hd, device=DEV, dtype=torch.bfloat16)
    cache_seqlens = torch.tensor(ctx_list, dtype=torch.int32, device=DEV)
    maxp = (max(ctx_list)+page-1)//page
    page_table = (torch.arange(B*maxp, dtype=torch.int32, device=DEV).reshape(B,maxp)) % nb
    scale = hd**-0.5
    out = flash_attn_with_kvcache(q=q, k_cache=k_cache, v_cache=v_cache,
                                  cache_seqlens=cache_seqlens, softmax_scale=scale,
                                  causal=causal, page_table=page_table,
                                  return_softmax_lse=lse)
    o = out[0] if isinstance(out,(tuple,list)) else out
    torch.cuda.synchronize()
    ref = torch.zeros(B,T,n_q,hd, device=DEV)
    for s in range(B):
        L=ctx_list[s]
        K=torch.stack([k_cache[int(page_table[s,p//page]), p%page].float() for p in range(L)])  # [L,n_kv,hd]
        V=torch.stack([v_cache[int(page_table[s,p//page]), p%page].float() for p in range(L)])
        for h in range(n_q):
            hk=h//gqa
            for t in range(T):
                sc=(q[s,t,h].float() @ K[:,hk].T)*scale
                if causal: sc[L-T+t+1:] = float("-inf")
                ref[s,t,h]=torch.softmax(sc,-1) @ V[:,hk]
    return (o.float()-ref).abs().max().item()/max(ref.abs().max().item(),1e-6)

print(f"{'hd':>4} {'T':>3} {'mode':>10} {'n_kv':>5} {'lse':>4} {'rel_err':>9}  status")
for hd in (64,128):
    for T,c in ((5,False),(5,True),(7,False)):
        for n_kv in (2,16):
            try:
                r=trial(hd,n_kv,4,T,[300,777,128],causal=c,lse=(not c))
                print(f"{hd:>4} {T:>3} {'causal' if c else 'NONcausal':>10} {n_kv:>5} {str(not c):>4} {r:>9.5f}  {'PASS' if r<3e-2 else 'MISMATCH'}")
            except Exception as e:
                print(f"{hd:>4} {T:>3} {'causal' if c else 'NONcausal':>10} {n_kv:>5} {'-':>4} {'-':>9}  FAIL: {type(e).__name__}: {str(e)[:80]}")
