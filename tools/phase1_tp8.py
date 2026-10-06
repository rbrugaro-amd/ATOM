# SPDX-License-Identifier: MIT
"""Phase 1b of the DFlash numerics bisect: verify the TP=8 SHARDED load.

Spawns 8 real processes (one per GPU), inits ATOM's real distributed env at
tensor_model_parallel_size=8, builds `KimiK3DFlash` through the same real
config path as `tools/phase1_coverage.py`, and runs ATOM's REAL loader
(`atom.model_loader.loader.load_model`). Then, per rank:

  1. qkv_proj  == cat(q[512r:], k[128r:], v[128r:])  -- and the explicit WRONG
     hypothesis (a contiguous slice of the TP1 [4096;1024;1024] fusion)
  2. o_proj    == o[:, 512r:512(r+1)]  (and which rank's column shard it is)
  3. gate_up   == cat(gate[1792r:], up[1792r:])  + swap/interleave detectors
  4. down_proj == down[:, 1792r:1792(r+1)]
  5. replicated tensors: bitwise == ckpt AND byte-identical across all ranks
  6. all_gather reconstruction: reassembled logical weight == ckpt, bitwise

Nothing here reimplements ATOM's sharding: every comparison is ATOM's loaded
parameter against a slice of the raw safetensors tensor.
"""

import copy
import hashlib
import json
import os
import sys

import torch
import torch.distributed as dist

CKPT = (
    "/data/workloads-inference/hf_hub_cache/models--RadixArk--Kimi-K3-DSpark/"
    "snapshots/3c5bac301d9cf392706189d82ed947feca6c2f0f"
)
WORLD = 8
PORT = "29583"
# DCP=1 writes the original filenames and must reproduce them BYTE-FOR-BYTE;
# any other DCP gets its own suffix so the banked DCP1 oracle is never
# overwritten by a DCP8 run.
DCP = int(os.environ.get("DCP", "1"))
OUT = (
    "/work/tools/phase1_tp8_rank%d.json"
    if DCP == 1
    else ("/work/tools/phase1_tp8_dcp" + str(DCP) + "_rank%d.json")
)


def _eq(a, b):
    """Bitwise equality of two same-dtype tensors (NaN-safe: compares bytes)."""
    a = a.detach().cpu().contiguous()
    b = b.detach().cpu().contiguous()
    if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
        return False
    return bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))


def _sha(t):
    return hashlib.sha256(
        t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def run(rank: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = PORT
    torch.cuda.set_device(rank)

    from atom.distributed.pp_comm import init_pp_aware_dist_env

    init_pp_aware_dist_env(
        tensor_model_parallel_size=WORLD,
        pipeline_model_parallel_size=1,
        global_rank=rank,
        world_size=WORLD,
        distributed_init_method=f"tcp://127.0.0.1:{PORT}",
        backend="nccl",
        decode_context_parallel_size=DCP,
    )

    from aiter.dist.parallel_state import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        get_tp_group,
    )

    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()
    assert tp_size == WORLD, f"tp_size={tp_size}"
    assert tp_rank == rank, f"tp_rank={tp_rank} rank={rank}"

    from atom.config import (
        CompilationLevel,
        Config,
        SpeculativeConfig,
        set_current_atom_config,
    )
    from atom.model_loader.loader import load_model
    from atom.spec_decode.drafter import support_draft_model_arch_dict
    from atom.utils import resolve_obj_by_qualname

    spec = SpeculativeConfig(model=CKPT, method="dspark", num_speculative_tokens=5)
    draft_hf = spec.draft_model_hf_config
    cfg = Config(
        model=CKPT,
        speculative_config=spec,
        enforce_eager=True,
        tensor_parallel_size=WORLD,
        decode_context_parallel_size=DCP,
    )
    draft_cfg = copy.copy(cfg)
    draft_cfg.hf_config = draft_hf
    draft_cfg.compilation_config = copy.copy(cfg.compilation_config)
    draft_cfg.compilation_config.level = CompilationLevel.NO_COMPILATION
    draft_cfg.compilation_config.static_forward_context = {}
    set_current_atom_config(draft_cfg)

    model_class = resolve_obj_by_qualname(
        support_draft_model_arch_dict[draft_hf.architectures[0]]
    )
    torch.set_default_dtype(draft_cfg.torch_dtype)
    with torch.device("cuda"):
        model = model_class(draft_cfg, layer_offset=0)

    import atom.model_loader.loading_core as lc

    captured = {}
    orig = lc._report_coverage

    def spy(**kw):
        captured.update(kw)
        return orig(**kw)

    lc._report_coverage = spy
    loaded = load_model(model, CKPT, draft_hf, None, False)
    lc._report_coverage = orig

    params = captured["params_dict"]
    dropped = captured["dropped_ckpt_keys"]
    unloaded = sorted(set(params) - set(loaded))

    from safetensors import safe_open

    f = safe_open(os.path.join(CKPT, "model.safetensors"), framework="pt")

    def C(n):
        return f.get_tensor(n)

    def P(n):
        return params[n].detach().cpu()

    hd = draft_hf.head_dim
    nq = draft_hf.num_attention_heads
    nkv = draft_hf.num_key_value_heads
    inter = draft_hf.intermediate_size
    hidden = draft_hf.hidden_size
    nlayers = draft_hf.num_hidden_layers

    # The draft shards its HEADS at the EFFECTIVE TP, tp_eff = tp // dcp: under
    # DCP the KV is sharded by token across the same GPUs, and sharding heads
    # on top of that leaves (head, token) pairs stored by nobody. At DCP8
    # tp_eff is 1, so every rank must hold the FULL q/k/v weight.
    tp_eff = tp_size // DCP
    qkv_rank = tp_rank // DCP
    q_sh = nq // tp_eff * hd  # 512 at DCP1, 4096 at DCP8
    kv_sh = nkv // tp_eff * hd  # 128 at DCP1, 1024 at DCP8
    # o_proj and the MLP have NO head axis to replicate: both stay sharded by
    # the FULL TP group at every DCP. o_sh is 512 always.
    o_sh = nq // tp_size * hd
    i_sh = inter // tp_size  # 1792

    rep = {
        "geometry": {
            "hd": hd,
            "nq": nq,
            "nkv": nkv,
            "inter": inter,
            "hidden": hidden,
            "nlayers": nlayers,
            "q_sh": q_sh,
            "kv_sh": kv_sh,
            "o_sh": o_sh,
            "i_sh": i_sh,
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "dcp": DCP,
            "tp_eff": tp_eff,
            "qkv_rank": qkv_rank,
        },
        "coverage": {
            "n_params": len(params),
            "n_loaded": len(loaded),
            "dropped": dropped,
            "unloaded": unloaded,
        },
        "checks": [],
        "fail": [],
        "shapes": {n: list(params[n].shape) for n in sorted(params)},
    }

    def record(name, ok, detail=""):
        rep["checks"].append([name, bool(ok), detail])
        if not ok:
            rep["fail"].append([name, detail])

    # ---------------- per-layer sharded checks -------------------------
    for li in range(nlayers):
        qr = C(f"layers.{li}.self_attn.q_proj.weight")
        kr = C(f"layers.{li}.self_attn.k_proj.weight")
        vr = C(f"layers.{li}.self_attn.v_proj.weight")
        orf = C(f"layers.{li}.self_attn.o_proj.weight")
        gr = C(f"layers.{li}.mlp.gate_proj.weight")
        ur = C(f"layers.{li}.mlp.up_proj.weight")
        dr = C(f"layers.{li}.mlp.down_proj.weight")

        # ---- 1. qkv_proj ----
        pq = P(f"layers.{li}.self_attn.qkv_proj.weight")
        record(
            f"L{li}.qkv.shape",
            tuple(pq.shape) == (q_sh + 2 * kv_sh, hidden),
            f"got {tuple(pq.shape)} want {(q_sh + 2 * kv_sh, hidden)}",
        )
        if tuple(pq.shape) == (q_sh + 2 * kv_sh, hidden):
            qb = pq[:q_sh]
            kb = pq[q_sh : q_sh + kv_sh]
            vb = pq[q_sh + kv_sh :]
            record(f"L{li}.qkv.q", _eq(qb, qr[q_sh * qkv_rank : q_sh * (qkv_rank + 1)]))
            record(
                f"L{li}.qkv.k", _eq(kb, kr[kv_sh * qkv_rank : kv_sh * (qkv_rank + 1)])
            )
            record(
                f"L{li}.qkv.v", _eq(vb, vr[kv_sh * qkv_rank : kv_sh * (qkv_rank + 1)])
            )
            # which shard did each block actually get? (permutation probe)
            qm = [j for j in range(tp_eff) if _eq(qb, qr[q_sh * j : q_sh * (j + 1)])]
            km = [j for j in range(tp_eff) if _eq(kb, kr[kv_sh * j : kv_sh * (j + 1)])]
            vm = [j for j in range(tp_eff) if _eq(vb, vr[kv_sh * j : kv_sh * (j + 1)])]
            rep["checks"].append(
                [f"L{li}.qkv.blockmatch", True, f"q->{qm} k->{km} v->{vm}"]
            )
            # k/v swapped?
            if _eq(kb, vr[kv_sh * qkv_rank : kv_sh * (qkv_rank + 1)]) and _eq(
                vb, kr[kv_sh * qkv_rank : kv_sh * (qkv_rank + 1)]
            ):
                rep["fail"].append([f"L{li}.qkv.KV_SWAPPED", ""])
            # WRONG hypothesis: contiguous slice of the TP1 [q;k;v] fusion
            fused = torch.cat([qr, kr, vr], dim=0)
            w = q_sh + 2 * kv_sh
            alt = _eq(pq, fused[w * qkv_rank : w * (qkv_rank + 1)])
            rep["checks"].append(
                [
                    f"L{li}.qkv.ALT_tp1_contiguous_slice",
                    True,
                    f"matches_wrong_layout={alt}",
                ]
            )
            # At tp_eff == 1 there is only ONE shard, so "a contiguous slice
            # of the TP1 fusion" IS the correct layout and matching it is not
            # a finding. The detector only discriminates once the weight is
            # genuinely split.
            if alt and tp_eff > 1:
                rep["fail"].append(
                    [f"L{li}.qkv.IS_TP1_CONTIGUOUS_SLICE", "wrong layout matched"]
                )
            del fused

        # ---- 2. o_proj ----
        po = P(f"layers.{li}.self_attn.o_proj.weight")
        record(
            f"L{li}.o.shape",
            tuple(po.shape) == (hidden, o_sh),
            f"got {tuple(po.shape)} want {(hidden, o_sh)}",
        )
        if tuple(po.shape) == (hidden, o_sh):
            record(
                f"L{li}.o.colshard",
                _eq(po, orf[:, o_sh * tp_rank : o_sh * (tp_rank + 1)]),
            )
            om = [
                j
                for j in range(tp_size)
                if _eq(po, orf[:, o_sh * j : o_sh * (j + 1)].contiguous())
            ]
            rep["checks"].append(
                [
                    f"L{li}.o.colmatch",
                    om == [tp_rank],
                    f"o_proj cols match rank(s) {om}; q heads are rank {tp_rank}",
                ]
            )
            if om != [tp_rank]:
                rep["fail"].append(
                    [f"L{li}.o.QHEAD_MISALIGN", f"cols->{om} but q->{tp_rank}"]
                )

        # ---- 3. gate_up_proj ----
        pg = P(f"layers.{li}.mlp.gate_up_proj.weight")
        record(
            f"L{li}.gate_up.shape",
            tuple(pg.shape) == (2 * i_sh, hidden),
            f"got {tuple(pg.shape)} want {(2 * i_sh, hidden)}",
        )
        if tuple(pg.shape) == (2 * i_sh, hidden):
            g_b, u_b = pg[:i_sh], pg[i_sh:]
            record(
                f"L{li}.gate_up.gate",
                _eq(g_b, gr[i_sh * tp_rank : i_sh * (tp_rank + 1)]),
            )
            record(
                f"L{li}.gate_up.up", _eq(u_b, ur[i_sh * tp_rank : i_sh * (tp_rank + 1)])
            )
            gm = [j for j in range(tp_size) if _eq(g_b, gr[i_sh * j : i_sh * (j + 1)])]
            um = [j for j in range(tp_size) if _eq(u_b, ur[i_sh * j : i_sh * (j + 1)])]
            rep["checks"].append(
                [f"L{li}.gate_up.blockmatch", True, f"g->{gm} u->{um}"]
            )
            if _eq(g_b, ur[i_sh * tp_rank : i_sh * (tp_rank + 1)]) and _eq(
                u_b, gr[i_sh * tp_rank : i_sh * (tp_rank + 1)]
            ):
                rep["fail"].append([f"L{li}.gate_up.SWAPPED", ""])
            # WRONG hypothesis: contiguous slice of the TP1 [gate(14336);up(14336)]
            fused = torch.cat([gr, ur], dim=0)
            w = 2 * i_sh
            alt = _eq(pg, fused[w * tp_rank : w * (tp_rank + 1)])
            rep["checks"].append(
                [
                    f"L{li}.gate_up.ALT_tp1_contiguous_slice",
                    True,
                    f"matches_wrong_layout={alt}",
                ]
            )
            if alt and tp_rank != 0:
                rep["fail"].append(
                    [f"L{li}.gate_up.IS_TP1_CONTIGUOUS_SLICE", "wrong layout matched"]
                )
            # interleave detector: row-strided (gate/up alternating by shard)
            inter_alt = _eq(
                pg,
                torch.cat(
                    [
                        gr[i_sh * tp_rank : i_sh * (tp_rank + 1)],
                        gr[i_sh * tp_rank : i_sh * (tp_rank + 1)],
                    ],
                    dim=0,
                ),
            )
            rep["checks"].append(
                [f"L{li}.gate_up.ALT_gate_duplicated", True, f"matched={inter_alt}"]
            )
            del fused

        # ---- 4. down_proj ----
        pd = P(f"layers.{li}.mlp.down_proj.weight")
        record(
            f"L{li}.down.shape",
            tuple(pd.shape) == (hidden, i_sh),
            f"got {tuple(pd.shape)} want {(hidden, i_sh)}",
        )
        if tuple(pd.shape) == (hidden, i_sh):
            record(
                f"L{li}.down.colshard",
                _eq(pd, dr[:, i_sh * tp_rank : i_sh * (tp_rank + 1)]),
            )
            dm = [
                j
                for j in range(tp_size)
                if _eq(pd, dr[:, i_sh * j : i_sh * (j + 1)].contiguous())
            ]
            rep["checks"].append([f"L{li}.down.colmatch", dm == [tp_rank], f"->{dm}"])
            if dm != [tp_rank]:
                rep["fail"].append(
                    [f"L{li}.down.MISALIGN", f"cols->{dm} want {tp_rank}"]
                )

    # ---------------- 5. replicated tensors ----------------------------
    REPL = [
        "fc.weight",
        "hidden_norm.weight",
        "norm.weight",
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
    ]
    for li in range(nlayers):
        REPL += [
            f"layers.{li}.self_attn.q_norm.weight",
            f"layers.{li}.self_attn.k_norm.weight",
            f"layers.{li}.input_layernorm.weight",
            f"layers.{li}.post_attention_layernorm.weight",
        ]
    shas = {}
    for n in REPL:
        p = P(n)
        c = C(n)
        record(
            f"REPL {n} shape_full",
            tuple(p.shape) == tuple(c.shape),
            f"got {tuple(p.shape)} ckpt {tuple(c.shape)}",
        )
        record(f"REPL {n} bitwise_vs_ckpt", _eq(p, c))
        shas[n] = _sha(p)

    # cross-rank identity of replicated tensors
    gathered = [None] * tp_size
    dist.all_gather_object(gathered, shas, group=get_tp_group().cpu_group)
    for n in REPL:
        same = all(g[n] == gathered[0][n] for g in gathered)
        record(f"REPL {n} identical_across_ranks", same)

    # ---------------- 6. all_gather reconstruction ---------------------
    pg_group = get_tp_group().device_group
    recon_fail = []

    def gather(t):
        t = t.cuda().contiguous()
        buf = [torch.empty_like(t) for _ in range(tp_size)]
        dist.all_gather(buf, t, group=pg_group)
        return [b.cpu() for b in buf]

    for li in range(nlayers):
        gq = gather(params[f"layers.{li}.self_attn.qkv_proj.weight"].detach())
        if DCP > 1:
            # Head replication, proved bitwise: at tp_eff < tp_size several
            # ranks hold the SAME qkv shard, so there is nothing to reassemble
            # -- the oracle is that the DCP group's shards are IDENTICAL.
            for r in range(tp_size):
                record(
                    f"REPLICATED L{li}.qkv rank{r}",
                    _eq(gq[r], gq[(r // DCP) * DCP]),
                    f"rank {r} must equal rank {(r // DCP) * DCP}",
                )
            # And the DISTINCT shards still reassemble the checkpoint.
            gq = [gq[j * DCP] for j in range(tp_eff)]
        qcat = torch.cat([g[:q_sh] for g in gq], 0)
        kcat = torch.cat([g[q_sh : q_sh + kv_sh] for g in gq], 0)
        vcat = torch.cat([g[q_sh + kv_sh :] for g in gq], 0)
        if tp_rank == 0:
            for nm, got, want in (
                ("q_proj", qcat, C(f"layers.{li}.self_attn.q_proj.weight")),
                ("k_proj", kcat, C(f"layers.{li}.self_attn.k_proj.weight")),
                ("v_proj", vcat, C(f"layers.{li}.self_attn.v_proj.weight")),
            ):
                ok = _eq(got, want)
                record(f"RECON L{li}.{nm}", ok)
                if not ok:
                    recon_fail.append(f"L{li}.{nm}")
        del gq, qcat, kcat, vcat

        go = gather(params[f"layers.{li}.self_attn.o_proj.weight"].detach())
        if tp_rank == 0:
            record(
                f"RECON L{li}.o_proj",
                _eq(torch.cat(go, 1), C(f"layers.{li}.self_attn.o_proj.weight")),
            )
        del go

        gg = gather(params[f"layers.{li}.mlp.gate_up_proj.weight"].detach())
        if tp_rank == 0:
            record(
                f"RECON L{li}.gate_proj",
                _eq(
                    torch.cat([g[:i_sh] for g in gg], 0),
                    C(f"layers.{li}.mlp.gate_proj.weight"),
                ),
            )
            record(
                f"RECON L{li}.up_proj",
                _eq(
                    torch.cat([g[i_sh:] for g in gg], 0),
                    C(f"layers.{li}.mlp.up_proj.weight"),
                ),
            )
        del gg

        gd = gather(params[f"layers.{li}.mlp.down_proj.weight"].detach())
        if tp_rank == 0:
            record(
                f"RECON L{li}.down_proj",
                _eq(torch.cat(gd, 1), C(f"layers.{li}.mlp.down_proj.weight")),
            )
        del gd
        torch.cuda.empty_cache()

    with open(OUT % tp_rank, "w") as fh:
        json.dump(rep, fh, indent=1)

    nfail = len(rep["fail"]) + sum(1 for c in rep["checks"] if not c[1])
    print(
        f"[rank {tp_rank}] checks={len(rep['checks'])} "
        f"failed={nfail} dropped={len(dropped)} unloaded={len(unloaded)}",
        flush=True,
    )
    dist.barrier()

    def _load(r):
        with open(OUT % r) as fh:
            return json.load(fh)

    if tp_rank == 0:
        print("\n==== TP8 SUMMARY ====", flush=True)
        for r in range(tp_size):
            d = _load(r)
            bad = [c for c in d["checks"] if not c[1]]
            print(
                f"rank {r}: {len(d['checks']) - len(bad)}/{len(d['checks'])} checks OK; "
                f"explicit-fail list={d['fail']}"
            )
            for c in bad:
                print(f"    FAIL {c[0]}  {c[2]}")
        d0 = _load(0)
        print("\nper-rank param shapes (rank0):")
        for n, s in d0["shapes"].items():
            print(f"   {n:52s} {s}")
        print("\ninformational block-match / alt-layout probes (rank0 + rank3):")
        for r in (0, 3):
            dd = _load(r)
            for c in dd["checks"]:
                if "blockmatch" in c[0] or "ALT_" in c[0] or "colmatch" in c[0]:
                    print(f"   r{r} {c[0]:46s} {c[2]}")
    dist.barrier()
    dist.destroy_process_group()


def main():
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    procs = []
    for r in range(WORLD):
        p = mp.Process(target=run, args=(r,))
        p.start()
        procs.append(p)
    codes = []
    for p in procs:
        p.join()
        codes.append(p.exitcode)
    print("exit codes:", codes)
    return 0 if all(c == 0 for c in codes) else 1


if __name__ == "__main__":
    sys.exit(main())
