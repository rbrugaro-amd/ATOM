# SPDX-License-Identifier: MIT
"""Phase 1 of the DFlash numerics bisect: rule out weight loading.

Builds `KimiK3DFlash` through ATOM's real path (SpeculativeConfig ->
hf_config_override -> DSparkProposer-style draft atom_config) and runs ATOM's
REAL loader (`atom.model_loader.loader.load_model`) against the checkpoint at
tensor_parallel_size=1, then reports:

  (a) checkpoint tensors dropped / never mapped
  (b) model parameters never written
  (c) whether the only intentional drops are `skip_weight_prefixes`
  (d) per-parameter bitwise comparison against the raw safetensors tensor,
      after undoing the qkv / gate_up / fc fusions.

No forward pass is run.
"""

import copy
import json
import os
import sys

import torch

CKPT = (
    "/data/workloads-inference/hf_hub_cache/models--RadixArk--Kimi-K3-DSpark/"
    "snapshots/3c5bac301d9cf392706189d82ed947feca6c2f0f"
)

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29571")


def init_dist():
    from atom.distributed.pp_comm import init_pp_aware_dist_env

    init_pp_aware_dist_env(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        global_rank=0,
        world_size=1,
        distributed_init_method="tcp://127.0.0.1:29571",
        backend="nccl",
    )


def main():
    torch.cuda.set_device(0)
    init_dist()

    from atom.config import Config, SpeculativeConfig, set_current_atom_config
    from atom.model_loader.loader import load_model
    from atom.spec_decode.drafter import support_draft_model_arch_dict
    from atom.utils import resolve_obj_by_qualname

    # --- real config path -------------------------------------------------
    # SpeculativeConfig.__post_init__ calls hf_config_override, which is what
    # normalizes dspark_* and renames the arch to K3DFlashModel.
    spec = SpeculativeConfig(model=CKPT, method="dspark", num_speculative_tokens=5)
    draft_hf = spec.draft_model_hf_config
    print("arch after override :", draft_hf.architectures)
    for f in (
        "dspark_target_layer_ids",
        "dspark_noise_token_id",
        "dspark_markov_rank",
        "dspark_block_size",
        "dspark_with_draft",
    ):
        print(f"  {f:28s} = {getattr(draft_hf, f, '<MISSING>')!r}")

    # A Config whose hf_config is the draft's, mirroring what
    # DSparkProposer._build_draft_model constructs for the standalone flavor.
    cfg = Config(model=CKPT, speculative_config=spec, enforce_eager=True)
    from atom.config import CompilationLevel

    draft_cfg = copy.copy(cfg)
    draft_cfg.hf_config = draft_hf
    draft_cfg.compilation_config = copy.copy(cfg.compilation_config)
    draft_cfg.compilation_config.level = CompilationLevel.NO_COMPILATION
    draft_cfg.compilation_config.static_forward_context = {}
    set_current_atom_config(draft_cfg)

    model_class = resolve_obj_by_qualname(
        support_draft_model_arch_dict[draft_hf.architectures[0]]
    )
    print("model class         :", model_class)
    # ModelRunner.__init__ does exactly this before building the model
    # (model_runner.py:737-738); without it every RMSNorm / nn.Embedding comes
    # out fp32 and the dtype column below would be a harness artifact.
    print("config.torch_dtype  :", draft_cfg.torch_dtype)
    torch.set_default_dtype(draft_cfg.torch_dtype)
    with torch.device("cuda"):
        model = model_class(draft_cfg, layer_offset=0)

    # --- the REAL loader --------------------------------------------------
    import atom.model_loader.loading_core as lc

    captured = {}
    orig = lc._report_coverage

    def spy(**kw):
        captured.update(kw)
        return orig(**kw)

    lc._report_coverage = spy

    loaded = load_model(model, CKPT, draft_hf, None, False)
    lc._report_coverage = orig

    params_dict = captured["params_dict"]
    dropped = captured["dropped_ckpt_keys"]

    # --- (a)/(b)/(c) coverage --------------------------------------------
    from safetensors import safe_open

    f = safe_open(os.path.join(CKPT, "model.safetensors"), framework="pt")
    ckpt_keys = sorted(f.keys())

    print("\n==== COVERAGE ====")
    print(f"checkpoint tensors        : {len(ckpt_keys)}")
    print(f"model parameters          : {len(params_dict)}")
    print(f"loaded_weights_record     : {len(loaded)}")

    skip = list(getattr(model, "skip_weight_prefixes", []))
    # ckpt tensors that the rewriter refused outright (never even read)
    from atom.model_loader.weight_names import CheckpointNameRewriter

    rw = CheckpointNameRewriter(
        weights_mapper=None,
        weights_mapping=getattr(model, "weights_mapping", {}),
        skip_weight_prefixes=skip,
        mtp_remap=None,
        spec_decode=False,
        num_hidden_layers=draft_hf.num_hidden_layers,
        n_routed_experts=None,
        fuse_shared_expert=lambda a, b: False,
    )
    refused, kept = [], {}
    for k in ckpt_keys:
        r = rw.rewrite(k)
        if r is None:
            refused.append(k)
        else:
            kept[k] = r

    print("\n(a) checkpoint tensors NOT mapped to any parameter:")
    print(f"  refused by rewriter (never read): {len(refused)}")
    for k in refused:
        why = next((p for p in skip if k.startswith(p)), "NOT-IN-SKIPLIST")
        print(f"      {k:60s}  <- skip_prefix {why}")
    print(f"  dropped by dispatcher (no param slot): {len(dropped)}")
    for o, m in dropped:
        print(f"      {o}  ->  {m}")

    unloaded = sorted(set(params_dict) - set(loaded))
    print(f"\n(b) model parameters NEVER written: {len(unloaded)}")
    for n in unloaded:
        print(f"      {n}  shape={tuple(params_dict[n].shape)}")

    unexpected_refusals = [k for k in refused if not any(k.startswith(p) for p in skip)]
    print(f"\n(c) skip_weight_prefixes = {skip}")
    print(f"    refusals NOT explained by the skip list: {unexpected_refusals}")

    # --- (2) numeric equivalence -----------------------------------------
    print("\n==== NUMERIC EQUIVALENCE (ATOM param vs raw safetensors) ====")
    hd = draft_hf.head_dim
    nq = draft_hf.num_attention_heads
    nkv = draft_hf.num_key_value_heads
    q_rows, kv_rows = nq * hd, nkv * hd
    inter = draft_hf.intermediate_size
    hidden = draft_hf.hidden_size
    tlids = list(draft_hf.dspark_target_layer_ids)

    results = []

    def cmp(label, atom_t, ref_t, note=""):
        a = atom_t.detach().cpu()
        b = ref_t.cpu()
        ok_shape = tuple(a.shape) == tuple(b.shape)
        ok_dtype = a.dtype == b.dtype
        bitwise, mad = False, float("nan")
        if ok_shape:
            # Value comparison always runs; `bitwise` additionally requires the
            # dtypes to match, so an fp32-held norm reads "exact value, wrong
            # container" rather than a bare FAIL.
            mad = (a.float() - b.float()).abs().max().item()
            if ok_dtype:
                bitwise = bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))
            else:
                bitwise = mad == 0.0
        results.append((label, ok_shape, ok_dtype, bitwise, mad, note))

    def P(n):
        return params_dict[n]

    def C(n):
        return f.get_tensor(n)

    cmp("fc.weight (full)", P("fc.weight"), C("fc.weight"))
    cmp("hidden_norm.weight", P("hidden_norm.weight"), C("hidden_norm.weight"))
    cmp("norm.weight", P("norm.weight"), C("norm.weight"))
    cmp(
        "markov_head.markov_w1.weight",
        P("markov_head.markov_w1.weight"),
        C("markov_head.markov_w1.weight"),
    )
    cmp(
        "markov_head.markov_w2.weight",
        P("markov_head.markov_w2.weight"),
        C("markov_head.markov_w2.weight"),
    )

    # fc block order: block i must equal columns [i*hidden : (i+1)*hidden]
    fc_atom = P("fc.weight").detach().cpu()
    fc_ref = C("fc.weight").cpu()
    print("\n-- fc.weight block-order check (5 blocks of 7168 input cols) --")
    for i in range(len(tlids)):
        sl = slice(i * hidden, (i + 1) * hidden)
        a = fc_atom[:, sl]
        best = None
        for j in range(len(tlids)):
            b = fc_ref[:, j * hidden : (j + 1) * hidden]
            if torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
                best = j
                break
        print(
            f"   ATOM fc block {i} (target_layer_id {tlids[i]}) "
            f"== ckpt block {best}"
        )

    for li in range(draft_hf.num_hidden_layers):
        pq = P(f"layers.{li}.self_attn.qkv_proj.weight").detach().cpu()
        cmp(
            f"L{li} qkv[q]",
            pq[:q_rows],
            C(f"layers.{li}.self_attn.q_proj.weight"),
            "rows 0:4096",
        )
        cmp(
            f"L{li} qkv[k]",
            pq[q_rows : q_rows + kv_rows],
            C(f"layers.{li}.self_attn.k_proj.weight"),
            "rows 4096:5120",
        )
        cmp(
            f"L{li} qkv[v]",
            pq[q_rows + kv_rows :],
            C(f"layers.{li}.self_attn.v_proj.weight"),
            "rows 5120:6144",
        )
        pg = P(f"layers.{li}.mlp.gate_up_proj.weight").detach().cpu()
        cmp(
            f"L{li} gate_up[0]=gate",
            pg[:inter],
            C(f"layers.{li}.mlp.gate_proj.weight"),
            "first half",
        )
        cmp(
            f"L{li} gate_up[1]=up",
            pg[inter:],
            C(f"layers.{li}.mlp.up_proj.weight"),
            "second half",
        )
        # cross check: is gate/up swapped?
        g_ref = C(f"layers.{li}.mlp.gate_proj.weight").cpu()
        u_ref = C(f"layers.{li}.mlp.up_proj.weight").cpu()
        swapped = torch.equal(
            pg[:inter].view(torch.uint8), u_ref.view(torch.uint8)
        ) and torch.equal(pg[inter:].view(torch.uint8), g_ref.view(torch.uint8))
        if swapped:
            print(f"   !!! L{li} gate/up SWAPPED")
        for nm in (
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ):
            cmp(f"L{li} {nm}", P(f"layers.{li}.{nm}"), C(f"layers.{li}.{nm}"))

    print(
        f"\n{'param':40s} {'shape':>6s} {'dtype':>6s} {'bitwise':>8s} {'maxabs':>10s}"
    )
    bad = 0
    for label, ok_s, ok_d, bw, mad, note in results:
        flag = "OK" if (ok_s and ok_d and bw) else "FAIL"
        if flag == "FAIL":
            bad += 1
        print(
            f"{label:40s} {ok_s!s:>6s} {ok_d!s:>6s} {bw!s:>8s} "
            f"{mad:10.3e}  {flag} {note}"
        )
    print(f"\n{len(results) - bad}/{len(results)} comparisons bitwise-equal")

    # --- (3) dtype / shape table -----------------------------------------
    print("\n==== PARAM dtype/shape vs ckpt ====")
    for n in sorted(params_dict):
        p = params_dict[n]
        print(f"  {n:50s} {tuple(p.shape)!s:22s} {p.dtype}")
    print("\n==== CKPT tensors ====")
    for k in ckpt_keys:
        t = f.get_slice(k)
        print(f"  {k:50s} {tuple(t.get_shape())!s:22s} {t.get_dtype()}")

    summary = {
        "refused": refused,
        "dropped": dropped,
        "unloaded": unloaded,
        "failures": [r[0] for r in results if not (r[1] and r[2] and r[3])],
    }
    with open("/work/tools/phase1_result.json", "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    sys.exit(main())
