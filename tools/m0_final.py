import torch
from aiter.ops.triton.gluon.pa_decode_gluon import pa_decode_gluon
DEV="cuda"; torch.manual_seed(0)

def trial(hd, bs, n_kv, gqa, ctx_list, parts, dtype=torch.bfloat16, nb=96):
    n_q=n_kv*gqa; n_seq=len(ctx_list)
    x=16//torch.tensor([],dtype=dtype).element_size()
    k=torch.randn(nb,n_kv,hd//x,bs,x,device=DEV).to(dtype)
    v=torch.randn(nb,n_kv,hd,bs,device=DEV).to(dtype)
    q=torch.randn(n_seq,n_q,hd,device=DEV).to(dtype)
    ctx=torch.tensor(ctx_list,dtype=torch.int32,device=DEV)
    mbt=(max(ctx_list)+bs-1)//bs
    bt=torch.arange(n_seq*mbt,dtype=torch.int32,device=DEV).reshape(n_seq,mbt)%nb
    scale=hd**-0.5; qgs=gqa
    o=torch.empty_like(q); shape=(n_seq,n_kv,parts,qgs)
    es=torch.empty(shape,dtype=torch.float32,device=DEV)
    ml=torch.empty(shape,dtype=torch.float32,device=DEV)
    tmp=torch.empty(*shape,hd,dtype=q.dtype,device=DEV)
    pa_decode_gluon(output=o,query=q,key_cache=k,value_cache=v,context_lengths=ctx,
        block_tables=bt,softmax_scale=scale,query_length=1,max_context_partition_num=parts,
        context_partition_size=256,compute_type=dtype,exp_sums=es,max_logits=ml,
        temporary_output=tmp,ps=True)
    torch.cuda.synchronize()
    # reference
    ro=torch.zeros_like(q,dtype=torch.float32); rl=torch.zeros(n_seq,n_q,device=DEV)
    for s in range(n_seq):
        L=ctx_list[s]
        for h in range(n_q):
            hk=h//gqa
            K=torch.stack([k[int(bt[s,p//bs]),hk,:,p%bs,:].reshape(-1).float()[:hd] for p in range(L)])
            V=torch.stack([v[int(bt[s,p//bs]),hk,:,p%bs].float() for p in range(L)])
            sc=(q[s,h].float()@K.T)*scale; m=sc.max(); e=torch.exp(sc-m)
            ro[s,h]=(e/e.sum())@V; rl[s,h]=m+torch.log(e.sum())
    # LSE over ALL partitions (no ceil(ctx/256) mask)
    pl=ml+torch.log(es.clamp_min(1e-30))
    got=torch.logsumexp(pl,dim=2).reshape(n_seq,n_kv*qgs)
    return (o.float()-ro).abs().max().item()/max(ro.abs().max().item(),1e-6), (got-rl).abs().max().item()

print(f"{'hd':>4} {'blk':>4} {'parts':>6} {'ctx':>22} {'out_rel':>9} {'lse_err':>9}  status")
cases=[(64,128,4,[300,128,777]),(128,128,4,[300,128,777]),(64,128,8,[300,128,777]),
       (64,128,4,[64,1000,2048]),(64,128,1,[300,128,777]),(64,64,4,[300,128,777])]
for hd,blk,parts,ctxs in cases:
    try:
        r,l=trial(hd,blk,2,4,ctxs,parts)
        print(f"{hd:>4} {blk:>4} {parts:>6} {str(ctxs):>22} {r:>9.5f} {l:>9.5f}  {'PASS' if r<2e-2 and l<5e-2 else 'MISMATCH'}")
    except Exception as e:
        print(f"{hd:>4} {blk:>4} {parts:>6} {str(ctxs):>22} {'-':>9} {'-':>9}  FAIL: {str(e)[:90]}")
