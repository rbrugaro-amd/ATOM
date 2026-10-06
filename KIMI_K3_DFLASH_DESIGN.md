# Kimi-K3 + RadixArk DFlash draft — verified design

**Base commit:** `aa4e7a04d` (origin/main, 2026-10-05). Supersedes the analysis in
`KIMI_K3_DFLASH_DRAFT.md`, which was written against a tree 232 commits behind main.

---

## Appendix A (read first) — R2 is CLOSED

The design below lists "the exact semantics of the reference dual-source attention"
as its #2 risk, stating the SpecForge reference was not read. It **was** read, from
the checkpoint itself:
`/data/workloads-inference/hf_hub_cache/models--RadixArk--Kimi-K3-DSpark/snapshots/3c5bac301d9cf392706189d82ed947feca6c2f0f/dflash.py`
(the checkpoint ships `dflash.py` + `dspark.py`). Resolved:

**(i) Operator order is qk-norm → RoPE → cache-write.** `Qwen3DFlashAttention.forward`:

```python
q = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)
q = self.q_norm(q).transpose(1, 2)                  # q-norm BEFORE rope
k_ctx   = self.k_proj(target_hidden)
k_noise = self.k_proj(hidden_states)                # SAME weights both sources
k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
k = self.k_norm(k).transpose(1, 2)                  # k-norm on the CONCAT, before rope
v = v.transpose(1, 2)                               # v is NOT normed
q, k = apply_rotary_pos_emb(q, k, cos, sin)         # rope after norm
if past_key_values is not None:
    k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)   # cache AFTER rope
```

**(ii) The pool must therefore store post-norm, post-RoPE K.** The cache update is
strictly after `apply_rotary_pos_emb`.

**(iii) The `norm(cat) == cat(norm)` factorization the design relies on is SOUND.**
`k_norm` is `Qwen3RMSNorm(self.head_dim)` — a row-wise op over the last axis — so
splitting the concat into a paged context term and a dense block term is exact, not an
approximation. This is what licenses writing context K to the pool at
`write_context_kv` time.

**(iv) At inference the block is fully bidirectional.** `spec_generate` calls the model
with `attention_mask=None` and `is_causal=False`; `self.is_causal = False` is set in
`__init__`. The captured-tensor `create_dflash_block_mask` (anchor sampling /
`block_keep_mask`) is a **training-time** construct — it is what the module docstring
says rules out the FA4 backend, and it does not appear on the inference path. So D4's
decomposition (paged prefix term ⊕ dense `T x T` block term, merged by LSE) matches the
reference exactly.

**Still open from R2:** nothing load-bearing. Note `apply_rotary_pos_emb` is overridden
locally and slices `cos[..., -q_len:, :]` for q while using the full `cos` for k — the
positions must span `ctx + block`, with q taking the last `q_len`. Port that slicing
deliberately.

### Two further facts verified directly, not in the design below

- **`k` convention has no off-by-one.** `KimiK3DSpark.forward_spec`
  (`atom/models/kimi_k3_dspark.py`) builds `[anchor, MASK, ...]` of width `T` and
  returns `T` draft tokens, with an explicit comment that `dflash.py`'s
  `spec_generate` slice `[:, -block_size+1:]` is a demo artifact and "NOT what any
  production implementation does. Do not 'fix' the sampler to match it." So ATOM `k`
  == SGLang `--speculative-dspark-block-size`. Trained width 7; we run k=5.
- **Checkpoint inventory:** 62 tensors, all BF16, no biases. `fc.weight [7168, 35840]`
  (= 5 x 7168, confirming fusion width is `len(target_layer_ids)`, NOT
  `num_target_layers: 93`). `markov_head.markov_w{1,2}` both `[163840, 256]`.
  `confidence_head.proj` `[1, 7424]` / `[1]` — training-only, skip.

---

# Serving Kimi-K3 with the SpecForge/SGLang `RadixArk/Kimi-K3-DSpark` (DFlash) draft in ATOM

**Design document — upstream-conforming, DCP8-first.**
Base commit: `aa4e7a04d` (`[Perf] MegaMoE: DP pad-row masking and configurable decode capacity (#2463)`).
Everything below was re-read at that commit; aiter sources were read from `/home/rbrugaro/code/aiter` (see §8 R7 for the pin caveat).

---

## 0. One-paragraph summary

The new draft is a **GQA/MHA** draft (`model_type: "qwen3"`, no `kv_lora_rank`) attached to an **MLA+KDA** target. `attn_family()` resolves it to `Family.MHA` (`atom/utils/selector.py:86-90`), `AiterBackend.DRAFT_OWNS_KV_POOL = True` (`atom/model_ops/attentions/aiter_attention.py:136`), so it gets a **sibling KV pool** via `DraftKvBuilder` — a path that has *never* been exercised under DCP, and that today **cannot even allocate** against an MLA target (§4.4). The project splits cleanly into (a) a DCP-neutral port that is ~80% of the work, and (b) a DCP8 step that is additive **only if four decisions are made now** (§3). The DCP8 step needs **no new aiter kernel** under the composition in §5, but that conclusion rests on one unverified hardware fact (gluon paged decode at `head_dim=64`), with a stated fallback that also needs no new kernel.

---

## 1. Verdict on the claimed DCP defects

Three claims went to adversarial review. **Two were refuted 3-0, one survived 3-0.** Correcting the record matters, because two of them would have sent us patching code that is not on our path.

### 1.1 `blocktable-column` — **REFUTED (3/3, high/medium confidence)**

> *Claim: `AiterAttentionMetadataBuilder.prepare_mtp_decode` computes `slot = block_tables[seq, (ctx-1)//B]*B + (ctx-1)%B` with no DCP term, therefore a GQA/MHA draft using it indexes the wrong block-table column under DCP.*

The arithmetic quote is accurate — the kernel at `atom/model_ops/attentions/aiter_attention.py:86-131` really does compute `block_col = last_pos // block_size` (line 120) with no `dcp_world_size`, no rank, no owner mask. **The attribution is wrong.** `prepare_mtp_decode` is called on the **target** runner's builder (`builder = self.runner.attn_metadata_builder`, `atom/spec_decode/eagle_proposer.py:845`), never on the draft's — `DraftKvBuilder` explicitly "does NOT drive prepare_decode/prepare_prefill; it piggybacks on the target builder's metadata flow during propose" (`atom/spec_decode/draft_kv.py:51-53`). With a Kimi-K3 (MLA) target the builder is `KimiAiterMLAGDNMetadataBuilder(…, AiterMLAMetadataBuilder)`, whose `prepare_mtp_decode` *is* DCP-aware. So this function is not reachable on our path.

**But do not file it as a non-issue.** It is dormant, not correct: the whole MHA family is DCP-blind (`grep -ci dcp atom/model_ops/attentions/aiter_attention.py` → **0**; same file, `prepare_decode` at :1171 has no DCP branch either). The moment anyone gives a GQA target a DCP path, this line becomes a live bug. **Design consequence:** our draft must *not* route its slots through this function (§3.3).

### 1.2 `ownership-mask-branch` — **REFUTED (3/3)**, but it is the sharpest warning in the whole evidence base

> *Claim: the DCP ownership mask in `eagle_proposer.py` is inside `if has_flat_kv and "slot_mapping" not in workinfos`, and the Aiter MHA backend DOES return `slot_mapping`, so for a GQA draft the mask is skipped.*

Both premises are **literally true** — verified:

```python
# atom/spec_decode/eagle_proposer.py:846
if has_flat_kv and "slot_mapping" not in workinfos:
    ...
    if getattr(builder, "dcp_world_size", 1) > 1:
        S = getattr(builder, "cp_kv_cache_interleave_size", 1)
        ctx = attn_metadata.context_lens[:running_bs]
        owned = (((ctx - 1) // S) % builder.dcp_world_size) == builder.dcp_rank
        attn_metadata.slot_mapping[:] = torch.where(owned, raw_slots, -1)   # :855-867
```
```python
# atom/model_ops/attentions/aiter_attention.py:943
workinfos = {"slot_mapping": slot_mapping}
```

It is refuted as a *bug* only because the two conditions are mutually exclusive today: the builder that returns `slot_mapping` is the MHA target builder, and an MHA target cannot run DCP. The skip is also *deliberate* — `aiter_attention.py:903-906` says it exists so "EagleProposer skips its token-granular (MLA physical block_size==1) flat-kv slot derivation, which would yield a bare block id for `B > 1`."

**The warning:** the only DCP ownership mask in `spec_decode/` lives *beside* the thing that produces the slot, gated on that producer staying silent. If our draft ever grows its own metadata step that returns a `slot_mapping` — the natural thing to do — the mask silently disappears and the failure mode is *acceptance length only* (§7.4). This is design decision **D3** in §3.

### 1.3 `no-dcp-in-mha` — **SURVIVED (0/3 refuted, high confidence)**

> *Claim: `atom/model_ops/attention_mha.py` has zero DCP references; there is no DCP read path for GQA/MHA; a GQA draft under DCP>1 is silently wrong rather than crashing.*

Verified at head: `grep -ci dcp` gives **0** for `atom/model_ops/attention_mha.py`, **0** for `atom/model_ops/attentions/aiter_attention.py`, **0** for `atom/spec_decode/draft_kv.py`, against **263** for `atom/model_ops/attention_mla.py`. Meanwhile the *write* side already shards for MHA, because `AiterAttentionMetadataBuilder` inherits `CommonAttentionBuilder._write_prefill_slots` which branches on `self.dcp_world_size > 1` (`atom/model_ops/attentions/backends.py:795-806`). And `DraftKvBuilder.get_kv_transfer_tensors` already declares `tp_replication_factor=1` with the comment "A paged MHA/GQA draft is a per-rank shard" (`atom/spec_decode/draft_kv.py:201-204`).

**So: the write path already believes the GQA draft is sharded, and the read path has no idea DCP exists.** No config assert blocks the combination (the DCP block at `atom/config.py:2202-2228` checks only `tp % dcp == 0`, gfx950 for spec-decode, TBO, interleave). This is the real work item, and §5 answers it.

**Prior art:** no issue or PR in ROCm/ATOM has ever addressed a non-MLA draft under DCP. The closest is merged vLLM-plugin PR **#2033**, whose `_dcp_local_slots()` fixes exactly this class of bug on the plugin side ("Upstream computes `pos // block_size`, which is the wrong interpretation once the pool is round-robin sharded"), and merged **#1746**, which established the `((ctx-1)//S) % W == rank` mask as required and applied it to MLA only. Separately, **#2106** measured −1.70 pp acceptance from a draft-KV misaddressing bug whose *only* symptom was acceptance. That is our expected failure signature.

---

## 2. Decision: **DCP-sharded tokens, DCP-replicated KV heads.** (Not "replicated pool.")

The question "sharded or replicated draft pool?" is a false binary, because a GQA cache has **two** axes and DCP collides with only one of them.

### 2.1 Why a plain "sharded" GQA pool is mathematically broken at TP8/DCP8

Under `-tp 8 -dcp 8`, the DCP group *is* the TP group — DCP "reuses the GPUs of TP group" and does not change world size (`atom/config.py:1016-1017`). The draft pool's head sharding is by TP only:

```python
# atom/model_ops/attentions/mha_kv_pool.py:186-196 (from_hf_config)
heads = hf_config.num_key_value_heads
if heads >= world_size:
    remainder, per_rank = heads % world_size, heads // world_size
```
called with `world_size=model_runner.world_size` (`atom/spec_decode/draft_kv.py:37`), i.e. `tensor_parallel_size`.

16 KV heads / 8 ranks = **2 heads per rank, a different pair on each**. Layer DCP token round-robin on top and rank *r* holds `{heads 2r, 2r+1} × {tokens ≡ r (mod 8)}`. The pair `(head h, token p)` with `owner(h) ≠ p mod 8` is stored by **nobody**. No LSE merge can recover it — the merge sums partials over disjoint *key* sets for the *same* head, and here the heads differ. MLA is immune only structurally: its cache is one latent row per token, replicated across TP, so DCP's token shard is the only shard.

### 2.2 The decision

> **The draft's QKV projections are built at `tp_eff = tensor_parallel_size // decode_context_parallel_size`; its KV pool is sized at `num_key_value_heads // tp_eff`; its KV is sharded across the DCP group by token only; `o_proj` stays at full TP and consumes the post-merge head slice.**

At TP8/DCP1: `tp_eff = 8` → 8 q heads, 2 kv heads per rank, no merge. Standard.
At TP8/DCP8: `tp_eff = 1` → **all 64 q heads and all 16 kv heads on every rank**, tokens round-robin across 8, partial attention merged by `dcp_lse_merge`, whose AG+RS backend ends in `reduce_scatter` over the head axis — handing each rank 64/8 = 8 heads, exactly what `o_proj` at full TP8 wants.

This is precisely MLA's QREP shape, and the helper already exists and is reusable verbatim:

```python
# atom/model_ops/attention_mla.py:395-416
def qrep_tp_override(tp_size: int) -> dict:
    ...
    assert tp_size % dcp_size == 0, f"QREP needs tp ({tp_size}) divisible by dcp ({dcp_size})"
    return {"override_tp_size": tp_size // dcp_size,
            "override_tp_rank": get_tensor_model_parallel_rank() // dcp_size}
```

**One difference that must be written down:** for MLA, QREP is an *optimization* gated on `dcp_config.enable_query_replication`. For a GQA draft, head replication across the DCP group is **mandatory**, not optional — §2.1. The draft must therefore use its own unconditional override, not `qrep_tp_override`'s config-gated one (or we lift the gate for drafts).

### 2.3 Why this respects the single-PAGE-index-space invariant

`plan_pools` raises `ValueError(f"more than one PAGE entry class: {paged}")` (`atom/model_ops/attentions/pool_layout/sub_pool_spec.py:232-234`), and `DraftKvBuilder.sub_pool_specs` joins the target's class by name:

```python
# atom/spec_decode/draft_kv.py:102-104
"""`page_pool` puts the draft in the target's entry class, so the two
contributions sum into one per-block cost instead of a second pool."""
return [page_pool(self.kv_pool.entry_bytes)]
```

Our decision changes **only `entry_bytes`**, never the class count, never the block-id space, never `blocks_per_entry`. `tp_replication_factor=1` (`draft_kv.py:203`) stays correct: the pool is still a per-rank shard, just sharded on a different axis.

### 2.4 The memory result — and it is the good news

Per-block cost for 5 layers, `head_dim=64`, fp8 KV, scheduler block `B`, `h = 16 // tp_eff` KV heads per rank (`mha_kv_pool.py:56-60` for k/v, `:78-82` for the fp32 scale planes, `entry_arena.py:243-258` for the 256 B alignment):

| | KV heads/rank | k+v bytes/block | scales bytes/block | **entry_bytes** (B=128) |
|---|---|---|---|---|
| TP8, DCP1 | 2 | `2·(5·2·64·B)` | `2·(5·2·4·B)` | **172 032 B** (~168 KiB) |
| TP8, DCP8 | 16 | `2·(5·16·64·B)` | `2·(5·16·4·B)` | **1 376 256 B** (~1.31 MiB) |

8× per block — **but** under DCP each sequence occupies 1/8 as many blocks (`BlockManager.num_pool_blocks` divides through `get_dcp_local_seq_lens`, `atom/model_engine/block_manager.py:572-592`). The two factors cancel exactly:

> **The draft's KV footprint per rank is DCP-invariant.** Head replication (×W) exactly cancels token sharding (÷W). Capacity planning done at DCP1 carries to DCP8 unchanged.

What does **not** cancel: the fp32 scale planes are `4 B` per (head, token) against `64 B` of fp8 cache, i.e. **1/16 of the draft pool**, not the 1/64 the `mha_kv_pool.py:69-74` docstring quotes for `head_dim=128`. And the draft's ~168 KiB/block at B=128 is charged against the *same* block budget the K3 MLA target draws from, so `num_kvcache_blocks` drops for both (`sub_pool_spec.py:237`, `count = max(0, remaining // spec.entry_bytes)`). Budget this explicitly in M1.

---

## 3. The load-bearing decisions that must be made NOW

These four are the answer to "what makes DCP8 additive rather than a rewrite". Each is cheap to encode at M0 and expensive to retrofit.

### D1 — Parameterize head sharding by `tp_eff = tp // dcp` from the first line of model code
**Lives in:** `atom/models/kimi_k3_dflash.py` (new), and the `world_size=` argument at `atom/spec_decode/draft_kv.py:37`.
**Decision:** the draft's `QKVParallelLinear` takes `override_tp_size=tp//dcp`; `o_proj` is a `RowParallelLinear` at full TP; the pool is built with `world_size=tp//dcp`.
**If decided the other way** (hardcode `tp_size` like `atom/models/qwen3.py:79-85` does): at DCP8 the pool is double-sharded, the cache is unreconstructible (§2.1), and fixing it means re-sharding the weight loader, the pool geometry, the `o_proj` input width, *and* re-tuning the kernel's `gqa_ratio`. That is the "almost like starting from scratch" the user fears. At DCP1 the two formulations produce **identical numbers**, so there is zero cost to being right now.

### D2 — The draft owns exactly one explicit attention kernel path; it does not go through `_dispatch_decode`
**Lives in:** `atom/models/kimi_k3_dflash.py` + a new draft-specific entry on `PagedAttentionImpl` in `atom/model_ops/attention_mha.py`.
**Decision:** the draft calls one named kernel wrapper that returns `(out, lse)`. `PagedAttentionImpl._dispatch_decode` (`atom/model_ops/attention_mha.py:960-1019`) picks between four kernels based on env vars and `kv_cache_block_size`, and **three of the four cannot produce an LSE**. Under DCP we need LSE unconditionally.
**If decided the other way:** the draft's correctness depends on `ATOM_USE_UNIFIED_ATTN` and `--block-size`; at `--block-size 128` with unified on, dispatch lands on `paged_attention_unified`, which both asserts causality (`aiter/ops/triton/attention/unified_attention.py:158`, `assert causal, "Only causal attention is supported"`) and returns no LSE. You would discover this at DCP8 bring-up, not at M0.

### D3 — The DCP ownership filter lives **inside** whatever computes the slot, never beside it
**Lives in:** wherever we produce the draft's context-write slots — today `atom/spec_decode/eagle_proposer.py:684-686` hands the draft the target's buffer, which is already DCP-correct by inheritance.
**Decision:** **do not** give the draft a `prepare_mtp_decode` that returns `workinfos["slot_mapping"]`. Either (a) keep inheriting the target's `slot_mapping` (the `draft_uses_mha` branch), or (b) if a private step is ever needed, put the `(pos//S) % W == rank` test in the same kernel that computes the slot.
**If decided the other way:** you reproduce exactly claim 1.2 — the mask at `eagle_proposer.py:855-867` is skipped because `"slot_mapping" in workinfos`, and the only symptom is a few points of acceptance length (#2106's measured signature). This is the single most likely way to ship a silently-wrong DCP8.

### D4 — The draft's own block K/V is **ephemeral**, not paged
**Lives in:** `atom/models/kimi_k3_dflash.py`.
**Decision:** only the *target-context* rows (`k_proj(target_hidden)`) go into the paged pool via `write_context_kv`. The block's own `k_proj(noise)` / `v_proj(noise)` stay as dense `[B, T, H_kv, D]` tensors and are consumed by a small dense intra-block attention merged in with `merge_attn_states` (§5.2).
**Why now:** this is what makes the bidirectional (`is_causal=False`) block attention expressible on **existing** kernels, and it removes an entire DCP question (who owns the speculative positions' slots — a question the K3 MLA draft answers with `ctx_lens = anchor + (1 + T)` at `dspark_proposer.py:801` and a hand-rolled round-robin at `:802-826`).
**If decided the other way** (write noise rows into the pool like the MLA draft): you need non-causal paged attention over `ctx + T`, which on GQA means a new aiter kernel (§5.1), *and* you inherit a second DCP slot-ownership computation, *and* under DCP the T noise rows land on only one rank each so every rank's block pass sees a different 1/8 of its own block.

---

## 4. Full change surface, in dependency order

Legend: **[N]** DCP-neutral (useful at DCP1, unchanged by DCP8) · **[R]** DCP-required · **[P]** DCP-prepared (written once at M0 with the DCP shape, inert at DCP1).

### 4.1 `atom/config.py` — architecture normalizer **[N]**

Today the standalone-DSpark branch is keyed on a suffix the new checkpoint does **not** match:

```python
# atom/config.py:1459-1462 (SpeculativeConfig.hf_config_override)
arch = (getattr(hf_config, "architectures", None) or [""])[0]
if arch.endswith("DSparkModel"):
    _normalize_draft_dspark_config(hf_config)
    return
```

`"DSparkDraftModel"` ends with `"DraftModel"`, not `"DSparkModel"` — so it falls through to the MTP rewrite path and dies. Changes:

1. **Add a DFlash branch before the suffix test**, discriminated on the *presence of `dflash_config`* rather than on the arch string (SpecForge reuses `DSparkDraftModel` for every DFlash draft, so the string alone is not a model identity).
2. **Rename the architecture** to a unique ATOM key, e.g. `hf_config.architectures = ["K3DFlashModel"]`. This resolves the collision with the ABC `DSparkDraftModel` in `atom/models/dspark_draft.py:32` — a registry key that is also an unrelated class name is a trap for the next reader, and the `support_draft_model_arch_dict` lookup at `atom/spec_decode/drafter.py:170` is a bare dict index that would silently bind the wrong thing if SpecForge ships a second DFlash draft.
3. **Lift the nested `dflash_config`** onto the canonical names, mirroring `_normalize_draft_dspark_config` (`atom/config.py:1191-1240`): `dspark_target_layer_ids = dflash_config["target_layer_ids"]` (`[7,23,51,67,83]`), `dspark_noise_token_id = dflash_config["mask_token_id"]` (`163824`), `dspark_markov_rank` from the markov head width, `dspark_with_draft = True`.
4. **Map top-level `block_size: 7` → `dspark_block_size = 7`.** `DSparkProposer._resolve_mtp_k` (`atom/spec_decode/dspark_proposer.py:372-388`) then yields `dspark_block_size = 7` (trained block width) and `mtp_k = num_speculative_tokens = 5` (verify horizon) — the V4-Pro shape, which is what the user's `--num-speculative-tokens 5` asks for. ⚠️ **Check**: `hf_config_override` at `:1472` routes *any* config carrying `dspark_block_size` to `DeepseekV4DSparkModel`; our branch must `return` before that, and the limit assertions at `atom/config.py:2405-2430` must be re-read against `dspark_block_size=7, num_spec=5`.
5. **Rope isolation.** The draft carries its own `rope_parameters` (yarn, factor 16.0, `original_max_position_embeddings` 65536, theta 10000). The target uses theta 50000 / factor 32. Assert at normalize time that the draft config's rope fields are populated from the draft checkpoint and never from `--model`'s config; the draft model constructs its own `get_rope` from `draft_cfg`, never from `atom_config`.
6. **`_PLAIN_CONFIG_MODEL_TYPES`** (`atom/config.py:755`) — *not* needed: `model_type` is `"qwen3"`, which transformers has a config class for. Good. But that also means `AutoConfig` will *drop unknown top-level keys into the config object*, which is how `dflash_config` survives — verify this on the real checkpoint rather than assuming (cheap: one `AutoConfig.from_pretrained` + `print`).

Also add a **hard guard** that is currently missing engine-wide: if `decode_context_parallel_size > 1` and the *draft* resolves to a non-MLA family, require the DFlash path (i.e. refuse Eagle3-llama + DCP). Today nothing rejects it; the MHA/GQA + DCP combination configures cleanly and produces garbage (§1.3). **[P]**

### 4.2 `atom/models/kimi_k3_dflash.py` (new) — the draft **[N core, P sharding]**

Model after `atom/models/kimi_k3_dspark.py` (which subclasses `DSparkDraftModel`, `kimi_k3_dspark.py:596`) and reuse the whole `dspark_draft.py` contract — `write_context_kv` / `project_context` / `context_layers` / `block_backbone` (`atom/models/dspark_draft.py:32-80`). What differs:

- **`fc` → `project_context`.** `fc.weight [7168, 35840]` = `len(target_layer_ids) * hidden`. Same role as K3-DSpark's `context_proj.weight [7168, 35840]` (`kimi_k3_dspark.py:38`) — a 1:1 port, including `hidden_norm` / `norm` ↔ `context_norm` / `final_norm`.
- **Attention is GQA, not MLA.** 64 q heads / 16 kv heads / `head_dim 64`, per-head `RMSNorm` on q and on k. Built with `QKVParallelLinear` + `RowParallelLinear` per the `atom/models/qwen3.py:88-104` pattern, **but with `override_tp_size=tp//dcp`** (D1).
- **Dual-source K/V.** `K = cat([k_proj(ctx_hidden), k_proj(noise)])`, same weights both sides. Because QK-norm is a per-head RMSNorm over `head_dim` (a row-wise op), `norm(cat(A,B)) == cat(norm(A), norm(B))` — so the normalized context K can be written to the pool at `write_context_kv` time and read back identically. **State this invariant in a comment**; it is the thing that makes the paged formulation equal the reference's `cat` formulation. ⚠️ Open question O1 (§8): the exact *order* of qk-norm vs RoPE vs cache-write must be read off the reference, not inferred.
- **`use_mla = False` and a `base_attention` attribute** on each attention module — `DraftKvBuilder._module_kinds` keys the pool walk on exactly that (`atom/spec_decode/draft_kv.py:159-163`). Get this wrong and `row_counts()[DRAFT_KV_ROWS]` raises the `KeyError` promised at `draft_kv.py:76-78`.
- **No `embed_tokens`, no `lm_head`** — shared from the target, as K3-DSpark already does via `skip_weight_prefixes` (`kimi_k3_dspark.py:54-55`). `confidence_head` skipped (training-only).
- **`markov_head.markov_w1/w2 [163840, 256]`** — reuse the existing block sampler (`tests/test_dspark_markov_sample.py` covers it).
- **NEVER add `@support_torch_compile`** to this file unless you also take the COMPILE BOUNDARY discipline (CLAUDE.md Critical Rules). The K3 DSpark draft is built outside the target's compile level (`dspark_proposer.py:308-312`); follow it.

### 4.3 `atom/model_ops/attention_mha.py` — an LSE-returning draft decode + the DCP merge **[P → R]**

Today the file has **zero** DCP references and three of four kernels cannot emit an LSE. Changes:

1. **Stop discarding the LSE on the persistent path** (one line, DCP-neutral, free):
   ```python
   # atom/model_ops/attention_mha.py:674-695 — today ends `return output`
   aiter.pa_persistent_fwd(..., mask=1)   # :692
   ```
   aiter already computes it: `final_lse = torch.empty((total_s, nhead), ...)` then `return logits, final_lse` (`/home/rbrugaro/code/aiter/aiter/ops/attention.py:584, 612-614`). Unusable for *our* draft (head_dim 64, §5.1) but it is the right shape for every future GQA DCP consumer, and it costs nothing.
2. **New `paged_attention_lse(...)`**: the gluon paged decode, plus an ATOM-side Triton reduce that folds the per-partition softmax state the kernel already writes — ATOM already allocates `exp_sums` / `max_logits` at `attention_mha.py:587-588` and passes them at `:620-621`:
   `lse[token, head] = max_logits + log(exp_sums)`, logsumexp'd over the partition axis. ⚠️ **Must mask to `cdiv(local_ctx, CONTEXT_PARTITION_SIZE)` partitions** — the gluon kernel bare-`return`s for partitions past the context over a `torch.empty` buffer (`aiter/ops/triton/gluon/pa_decode_gluon.py:3634-3636`, `:703-705`), and ATOM's split count is context-independent by design (`atom/model_ops/base_attention.py:81-84`).
3. **`_dcp_decode` for GQA**, mirroring `atom/model_ops/attention_mla.py:1337-1360` but *without* the Q all-gather (D1 already replicated the heads): run `paged_attention_lse` on the local shard, then `dcp_lse_merge(o, lse, self.dcp_group, backend=...)` (`atom/model_ops/dcp_ops.py:794`). The merge is already head-agnostic over `[B,H,D]`+`[B,H]`; `correct_attn_out`'s `assert (N & (N-1)) == 0` (`dcp_ops.py:201`) passes at W=8, and the A2A backend's `assert h_total % n_ranks == 0` (`dcp_ops.py:694-697`) passes at H=64, N=8. **[R]**
4. **Keep per-token fp8 scales off `unified_attention`** — `_reject_per_token_scales_on_unified` (`attention_mha.py:182-196`) already guards it; our draft path bypasses unified anyway (D2).

### 4.4 `atom/spec_decode/draft_kv.py` — the pool's block size and head count **[N then R]**

Two changes, both small, both blocking:

1. **Block size — blocking at M0, DCP-neutral.** Today:
   ```python
   # atom/spec_decode/draft_kv.py:89-92
   self._kv_pool = self._make_pool(
       layers=self.row_counts()[DRAFT_KV_ROWS],
       target_block_size=self.model_runner.attn_metadata_builder.block_size,
   )
   ```
   For a Kimi-K3 target that builder is `KimiAiterMLAGDNMetadataBuilder` → `AiterMLAMetadataBuilder`, whose `block_size` is `ATOM_MLA_PAGE_SIZE` or **1** (default `"1"`, `atom/utils/envs.py:280`). `MhaKvPool.__init__` then raises ("SHUFFLE packs 16 elements … has to divide both head_dim and block_size", `mha_kv_pool.py:121-127`), and even past that `allocate_kv_cache_tensors` raises because `pool.block_size != runner.block_size` (`draft_kv.py:130-137`). **Today an MHA draft on an MLA target cannot allocate at all.** The fix is to pass `runner.block_size` — which is what `draft_kv.py:131`'s own invariant already declares ("a draft is allocated one entry per scheduler block … so all three are one number"). Raising `ATOM_MLA_PAGE_SIZE` is *not* an escape: `DSparkProposer._init_block_persistent_buffers` asserts persistent mode, which requires `page_size <= 1` (`atom/spec_decode/dspark_proposer.py:242-255`).
2. **Head count — `world_size=` at `draft_kv.py:37`** must become `tp // dcp` (D1). **[R]**, but encode the expression at M0 so DCP8 flips one number.
3. `blocks_per_entry` stays **1**. It is the kernel-page ratio, not a DCP lever (`mha_kv_pool.py:111-119, 135`); using it to replicate blocks would need `AiterBackend.make_kv_pool` to forward it (`aiter_attention.py:173-187` does not), a W-expanded block table, and an 8× entry charge — all to buy something head replication already gives us for free (§2.4).
4. `tp_replication_factor=1` (`:203`) — **unchanged**, still correct.

### 4.5 `atom/spec_decode/dspark_proposer.py` — backend-dispatch the six MLA hard-wirings **[N]**

`_propose_with_draft` / `_build_paged_block_metadata` are written straight against MLA. Each must move behind a small per-flavor seam (a `BlockMetadataProvider` chosen off `attn_family(draft_hf)`), **not** an `if use_mla:` sprinkled through:

| # | MLA hard-wiring | Cite | GQA replacement |
|---|---|---|---|
| 1 | `impl = self.model.layers[0].self_attn.mla_attn.impl`; `padded_num_heads` / `dcp_kernel_num_heads` | `dspark_proposer.py:218-224` | GQA impl exposes `num_heads`; no head padding, no gather width |
| 2 | `assert should_use_persistent_mode(...)` (requires `ATOM_MLA_PAGE_SIZE<=1`) | `:242-255` | not applicable; GQA has its own capture-safety argument (D2: one fixed kernel) |
| 3 | `get_mla_metadata_info_v1` / `get_mla_metadata_v1` buffers, incl. literal `1,  # nhead_kv` | `:268-284`, `:920-945` | gluon needs no persistent work plan; drop the buffers entirely for this flavor |
| 4 | `_resolve_dtype_q` → `attn_metadata.dtype_q` (read only by `attention_mla.py:2993,3007,3017`) | `:390-448`, `:782` | unused by MHA; must not be set |
| 5 | `attn_metadata.causal = False` (read only at `attention_mla.py:2478`) | `:760` | MHA ignores it — the non-causality is expressed structurally (D4 / §5.2) |
| 6 | token-unit `kv_indptr` + `kv_indices_generate_triton(..., block_size, ...)` | `:875`, `:905-911` | GQA is **page**-unit: `cumsum(cdiv(ctx, block_size))`, ratio `self.block_ratio` (cf. `aiter_attention.py:1225-1231, 1393-1399`) |

Also **reuse, do not re-derive, the DCP math**. `_build_paged_block_metadata` hand-rolls round-robin at `dspark_proposer.py:802-834` (`virtual_block = block_size * dcp_world_size`, `local_off = (pos % virtual_block) // W`, `ctx_lens = (global + W-1-r)//W`) instead of calling `dcp_owner_rank` / `dcp_local_index` from `atom/distributed/dcp_layout.py:12-33`. It is correct **only at S=1**, which config enforces (`atom/config.py:2247-2256` asserts `speculative_config is None` when `interleave_size > 1`). Factor the GQA path onto the shared helpers so the next interleave change touches one place. **[P]**

### 4.6 `atom/model_ops/attentions/aiter_attention.py` **[P]**

- **Do not** extend `prepare_mtp_decode` (§1.1/D3). Instead add a comment at `:899-906` recording that the formula is non-DCP and that this builder is unreachable under DCP, so the next reader does not "fix" it into our path.
- `attn_block_size` (`:159-173`) already returns `scheduler_block_size` under `ATOM_USE_UNIFIED_ATTN`; our draft pool bypasses it entirely after the §4.4 change.
- If and only if someone later wants a *GQA target* under DCP: `prepare_decode` (`:1171-1220`) needs local lengths, local `kv_indptr`, a `g_kv_indptr`, and DCP-aware slots. **Out of scope for this PR. Say so in the PR body.**

### 4.7 `atom/spec_decode/drafter.py` registry **[N]**

One line next to `"K3DSparkModel": "atom.models.kimi_k3_dspark.KimiK3DSpark"` (`drafter.py:134`):
`"K3DFlashModel": "atom.models.kimi_k3_dflash.KimiK3DFlash"`.
Key must be the **renamed** arch from §4.1, never the raw `"DSparkDraftModel"`.

### 4.8 Plugin registries **[N]**

Mirror the native registration in all three places that exist for K3-DSpark, so plugin mode does not diverge (the vLLM plugin is where PR #2033's `_dcp_local_slots` already lives, and divergence there is how the two DCP implementations drifted in the first place):
- `atom/plugin/vllm/register.py:42` (`"K3DSparkModel": …KimiK3DSparkVllm`)
- `atom/plugin/vllm/model_wrapper.py:167` (`… :KimiK3DSparkDraft`)
- `atom/plugin/vllm/models/kimi_k3_dflash.py` (new, mirroring `atom/plugin/vllm/models/kimi_k3_dspark.py:15,64`)

Note `atom/plugin/vllm/attention/metadata.py`'s `AiterMhaMetadataBuilderForVllm` has **no** DCP handling either — same gap as native. Out of scope; note it.

### 4.9 Tests **[N]**

CPU-only config/sizing tests, following the existing conventions (`tests/test_dspark.py`, `tests/test_draft_kv_builder.py`, `tests/test_mha_kv_pool.py`, `tests/test_sub_pool_spec.py`, `tests/test_speculative_config_limits.py` — all run with no GPU, mocking AITER and `torch.cuda`):

1. `test_dflash_config_normalize` — `architectures: ["DSparkDraftModel"] + dflash_config` → `["K3FlashModel"]`, `dspark_target_layer_ids == [7,23,51,67,83]`, `dspark_noise_token_id == 163824`, `dspark_block_size == 7`, `dspark_with_draft is True`, and the draft's rope stays theta 10000 / factor 16 with the target at 50000/32.
2. `test_dflash_attn_family_is_mha` — `attn_family(draft_cfg) is Family.MHA` and `get_attn_backend(...).DRAFT_OWNS_KV_POOL is True`.
3. `test_dflash_draft_pool_block_size` — the pool is built at the **scheduler** block size, and the pre-fix `attn_metadata_builder.block_size==1` path raises. This is the regression test for §4.4.
4. `test_dflash_pool_heads_under_dcp` — parametrized over `(tp, dcp) ∈ {(8,1),(8,8),(8,2)}`: asserts `num_kv_heads == 16 // (tp//dcp)` and that `entry_bytes * blocks_per_seq` is **DCP-invariant** (§2.4). This is the test that makes D1 unforgettable.
5. `test_dflash_mtp_k` — `--num-speculative-tokens 5` with `dspark_block_size 7` → `mtp_k == 5`, block width 7.
6. A DCP slot-ownership unit test over `dcp_owner_rank` / `dcp_local_index` for the draft's write path, asserting that `-1` is emitted on exactly `W-1` of every `W` ranks (the `PAD_SLOT_ID` contract, `atom/model_ops/attentions/backends.py:50-52`).

### 4.10 CI **[N then R]**

- Accuracy: add a `kimi-k3 … dflash5` entry next to the existing `kimi-k3-mxfp4-1p1d-tp8-dcp8-dspark3-…` in `.github/benchmark/models_atomesh.yaml` (the only DCP + separate-draft config that ships today). **Gate it on M4, not earlier** — #2129 documents that the existing K3-DSpark DCP8 accuracy cell "has never been green" and has its own history.
- A DCP1 TP8 accuracy cell first (M2 exit), which is the one that must be green before DCP8 is attempted.

---

## 5. Does DCP-for-GQA need a new aiter kernel?

**Answer: no — under the decomposition below, and subject to one unverified hardware fact. If that fact goes the wrong way, there is still a no-new-kernel fallback that costs 2× draft KV memory.** I will not soften this: the honest confidence is ~75% on "no new aiter kernel", not 100%.

### 5.1 What aiter actually offers for GQA, verified in source

| ATOM entry | aiter kernel | LSE? | non-causal? | `head_dim=64`? |
|---|---|---|---|---|
| `paged_attention_unified` (`attention_mha.py:510`) | `unified_attention` | `skip_reduce` only, and only when the 3D branch runs **and** `NUM_SEGMENTS>1` (`unified_attention.py:296-365`) — autotuner-dependent, returns dummy aliases otherwise | **No.** `assert causal, "Only causal attention is supported"` (`unified_attention.py:158`) | yes (generic) |
| `paged_attention_triton` (`:554`) | gluon `pa_decode` | **Derivable** — `exp_sums`/`max_logits` are caller-owned and already allocated (`attention_mha.py:587-588, 620-621`) | no flag; causal masking baked in (`pa_decode_gluon.py:865-889, 1854-1906`) | `HEAD_SIZE_POW2` constexpr ⇒ *probably* yes — **unverified** |
| `paged_attention_asm` (`:638`) | `pa_fwd_asm` | no | no | **No.** `AITER_CHECK(head_size == 128, …)` (`csrc/py_itfs_cu/asm_pa.cu:191-194`) |
| `paged_attention_persistent_asm` (`:666`) | `pa_persistent_fwd` | **Yes** — `return logits, final_lse` (`aiter/ops/attention.py:612-614`), ATOM discards it at `:695` | **Yes** — `mask: int = 0` (`attention.py:568`), `get_pa_metadata_v1(..., is_causal, ...)` (`attention.py:943-949`); ATOM hardcodes `mask=1` (`:692`) and `True` (`aiter_attention.py:485, 1417`) | **No.** `pa_ps_fwd` has no assert but never passes a head dim to the kernel — `PsKernelArgs` (`asm_pa.cu:364-399`) carries `kv_nheads`, strides, `GQA`, `mtp`, and no head dim; `dim` is used only for `k_scalar = log2E/sqrt(dim)` (`asm_pa.cu:361-363`), so head width is baked into the `.co` |

Two cruel facts: **the one kernel with both an LSE and a non-causal flag is ASM-only at `head_dim=128`**, and **the one kernel ATOM dispatches to by default at `--block-size 128` refuses non-causal outright**.

### 5.2 The composition that avoids a new kernel

Three moves, all ATOM-side:

**(a) D4 — the block's own K/V never enters the pool.** The key set splits into two disjoint parts: the paged **prefix** (target-context rows, one per verified token) and the **T noise rows**. LSE merge is exact for any disjoint partition of the key set, so:
```
out = merge_attn_states( attn(Q_noise, KV_prefix), attn(Q_noise, KV_block) )
```
`merge_attn_states` already exists at `atom/model_ops/attentions/triton_merge_attn_states.py:15-24` over `[tokens, heads, head_size]`.

**(b) The `KV_block` term is dense and tiny.** `T=7`, 64 heads, `head_dim 64`, full (bidirectional) `T×T` — a single small Triton/`torch` op returning `(out, lse)`. This is where `is_causal=False` is *expressed*, structurally, instead of being requested from a kernel that will not provide it.

**(c) The `KV_prefix` term runs at `qlen=1`, fanned out.** The block's T queries each attend to **all** prefix keys. Reshape the block pass from `B` rows of `qlen=T` into `B*T` rows of `qlen=1`, each with `context_lens = ctx` and the same block-table row (a `repeat_interleave` of `block_tables`). At `qlen=1`, causal masking is a no-op, so the causal-only gluon kernel computes exactly the right thing — and it stays inside `gluon_decode_over_limit` (`atom/model_ops/base_attention.py:103-116`; `PA_GLUON_MAX_QUERY_LEN = 4`, and `query_length*query_group_size = 1*4 = 4 ≤ 64`), which `qlen=7` would **not** (`pa_decode_gluon.py:4812`, `assert query_length <= 4`).

**Cost of (c):** the prefix KV is read T times instead of once. Worked example at `bs=64`, `ctx=8k`, 5 layers, fp8, and the DCP-invariant 2-heads-equivalent per rank: `5 · 1024 · 16 · 64 · 2 B = 10.5 MiB`/seq → 671 MiB per block pass, ×5 ≈ 3.3 GiB → ~0.7 ms at ~5 TB/s, of which ~0.55 ms is the fan-out waste. One block pass per verify cycle. Measure it in M2; if it bites, the mitigation is (b')+(c') below.

**Under DCP this composes in a fixed order:** merge the **DCP partials of the paged term first** (`dcp_lse_merge`, which also reduce-scatters heads), then merge in the **dense block term** — which is computed identically on every rank, so merging it before the reduce-scatter would count it W times. Both steps are LSE merges and associative *given the right order*; write that order down in a comment, it is the kind of thing that produces a −1 pp acceptance bug.

### 5.3 The fallback if gluon cannot do `head_dim=64`

**Pad the head dim 64 → 128** and use `pa_persistent_fwd` with `mask=0`:
- zero-pad K (scores unchanged), zero-pad V (upper half of `out` is zero, sliced away);
- `pa_ps_fwd` **ignores** `softmax_scale` and hardcodes `k_scalar = log2E/sqrt(dim)` with `dim = 128` (`asm_pa.cu:361-363`) — compensate by pre-scaling Q by `sqrt(128/64) = sqrt(2)`;
- `required_qTile = gqa_ratio * max_qlen = 4 * 1 = 4 ≤ 64` (`asm_pa.cu:445-460`): fine;
- `mask=0` gives true non-causal, so (b)/(c) collapse back into one `qlen=T` call if desired;
- `final_lse` comes back free.

**Cost:** 2× draft KV (`entry_bytes` doubles; the DCP-invariance of §2.4 still holds), and a dependency on `ATOM_USE_UNIFIED_ATTN` + `kv_cache_block_size == 256` to reach that dispatch arm — which conflicts with the production `--block-size 128`. So this fallback is *viable but ugly*, and if we land on it, requesting a `head_dim=64` PS `.co` from the aiter team is the right follow-up, not a blocker.

### 5.4 Bottom line for the user

> **No new aiter kernel is required for either DCP1 or DCP8**, under §5.2. The DCP cross-rank merge itself is already built, already head-agnostic, and already tested (`dcp_lse_merge`, `atom/model_ops/dcp_ops.py:794`; `correct_attn_out`, `:113-192`). What is missing is entirely ATOM-side: an LSE-returning GQA decode wrapper, a `_dcp_decode` for `PagedAttentionImpl`, and head-replicated draft KV.
> **The one thing that could change the project's size** is if the gluon paged decode cannot run at `head_dim=64` on gfx950 — §8 R1. That is a 30-minute GPU experiment and it should be the **first thing done**, before any code is written.

---

## 6. Staged milestones

| | Milestone | Exit criterion | DCP |
|---|---|---|---|
| **M0** | **Kernel feasibility spike.** One standalone script: gluon `pa_decode` at `head_dim=64`, `block_size=128`, fp8 SHUFFLE, `qlen=1`, `gqa_ratio=4`; check it runs, check `exp_sums`/`max_logits` reconstruct the LSE against a torch reference, check the unwritten-partition masking rule. | Numbers match torch to fp8 tolerance, or the §5.3 fallback is selected. **No model code written before this passes.** | neutral |
| **M1** | **Config + registry + pool sizing, CPU-only.** §4.1, §4.4(1), §4.7, §4.8, and tests §4.9 items 1-5. | `python -m pytest tests/ -k dflash` green on CPU; `DraftKvBuilder` builds a pool at the scheduler block size against a K3 target config; `entry_bytes` budget reported and accepted against the K3 MLA pool. | neutral |
| **M2** | **DCP1 correctness.** The draft model, weight loading, §4.5 flavor seam, §4.3(2) LSE wrapper, §5.2 composition. TP8, DCP1, `--block-size 128`, fp8 KV. | Greedy losslessness vs no-spec (§7.3) **bitwise-equal token streams** on 200 prompts, and acceptance length within 10% of the model card. | neutral |
| **M3** | **DCP1 production shape.** `--cudagraph-mode FULL`, prefix caching, LMCache offload, `max-num-seqs 64`, `-k 5` / block 7. | GSM8K via `/run-atom-workload` within CI threshold; benchmark sweep recorded. | neutral |
| **M4** | **DCP8.** Flip `world_size → tp//dcp` (§4.4(2)), add `_dcp_decode` (§4.3(3)), add the §4.1 DCP guard. | gsm8k at TP8/DCP8 **equal to M3's DCP1 number within noise**, and acceptance length equal to M3's within 0.2 tokens. Acceptance is the DCP-correctness oracle (§7.4). | **gated** |
| **M5** | Plugin parity + CI cells (§4.8, §4.10). | CI cells green twice consecutively. | both |

M0-M3 is ~80% of the work and is **entirely useful at DCP1**. Nothing in M0-M3 is thrown away by M4 **provided D1-D4 are honoured** — M4 is then: one `world_size` expression, one `_dcp_decode` method, one merge-order comment, one guard.

---

## 7. Correctness strategy

### 7.1 Weight-loading equivalence (cheapest, do first)
Load the checkpoint in ATOM and in HF/SpecForge reference side by side on CPU; assert every ATOM parameter is bitwise-equal to the reference tensor it claims to be, after the fusions (`q/k/v → qkv_proj`, `gate/up → gate_up_proj`, the `fc [7168, 35840]` split across the 5 target layers). Assert the skip list (`confidence_head`, any `embed_tokens`) is exactly what we intended and that **nothing was silently left uninitialized** — this is the #1 source of "runs but drafts garbage". Also assert the draft's rope cache ≠ the target's (theta 10000/factor 16 vs 50000/32) — a single shared `get_rope` cache would be invisible except in acceptance.

### 7.2 Numerics bisect (`/dump-bisect-debug`)
Dump per-layer intermediates from the ATOM draft and the reference for one fixed `(target_hidden, noise)` pair and bisect. The specific checkpoints to compare, in order, because each isolates one of the things that is *new* here:
1. `project_context(fc(aux_concat))` output after `hidden_norm`;
2. layer-0 `k_proj(ctx)` and `k_proj(noise)` **before** QK-norm;
3. layer-0 K **after** QK-norm on the concatenation (this is where the `norm(cat)==cat(norm)` factorization is proved, not assumed — O1);
4. layer-0 attention output with the §5.2 two-term merge vs the reference's single `cat` attention (this is where `is_causal=False` is proved);
5. final `norm` output and the markov head logits.

### 7.3 Losslessness vs greedy no-spec
Speculative decoding must be output-identical to non-speculative greedy decoding. Run the same 200 prompts at `temperature=0` with `--method dspark -k 5` and with no draft, and diff the token streams. **Any** divergence is a verify/rejection-sampler bug, not a draft-quality issue, and it must be zero before acceptance is measured. This is the strongest single test and it does not depend on knowing the reference's acceptance numbers.

### 7.4 Acceptance length — the DCP oracle
This is the **only** observable that distinguishes a correct DCP draft from a mis-sharded one. #2106 measured −1.70 / −1.76 pp acceptance from draft-KV misaddressing whose *only* symptom was acceptance ("in bounds, no fault, wrong rows, and only acceptance moves"). Therefore:

> **Acceptance length is an M4 exit criterion, not a performance metric.** Record mean accepted tokens/step at M3 (DCP1); M4 must reproduce it within 0.2. A DCP8 run that is "fast and produces fluent text" with 0.5 fewer accepted tokens is a *correctness failure*, and it is the exact failure D3 produces.

Instrument it in the proposer (the existing DSpark path already reports it) and log it per-rank, not just rank-0 — a per-rank divergence localizes the ownership bug immediately.

### 7.5 Test conventions
Config/sizing tests go in `tests/`, CPU-only, mocking AITER and `torch.cuda`, in the style of `tests/test_dspark.py`, `tests/test_draft_kv_builder.py`, `tests/test_mha_kv_pool.py`, `tests/test_sub_pool_spec.py`. GPU workloads go through `/run-atom-workload` (which encodes stop → start → workload-in-bg → `wait_infer_drain` → stop). Remember `rm -rf /root/.cache/atom/*` before every restart and `AITER_LOG_LEVEL=WARNING`; verify the server with `rocm-smi --showmemuse`, not `curl /health` (CLAUDE.md).

---

## 8. Risks and open questions, ranked

**R1 (highest) — Does gluon `pa_decode` run at `head_dim=64`, `block_size=128`, fp8 SHUFFLE on gfx950?**
If no, §5.2 collapses and we take the §5.3 head-pad fallback (2× draft KV, and a `--block-size 256` dependency that fights the production config). *Close:* M0, one script, ~30 min of GPU time. **Do this before writing any model code.**

**R2 — The exact semantics of the reference's dual-source attention.** Specifically: (i) is RoPE applied to the context K at its absolute position before or after QK-norm; (ii) is the context K written to the pool pre- or post-norm; (iii) is the intra-block attention truly full `T×T` or full-plus-self-exclusion. I have **not** read the SpecForge reference — everything in §4.2 is inferred from the task brief plus the ATOM K3-DSpark analogue (`atom/models/kimi_k3_dspark.py:26-30`). *Close:* read `RadixArk/Kimi-K3-DSpark`'s modeling file + §7.2 bisect checkpoints 2-4.

**R3 — Shared-PAGE-class capacity.** The draft adds ~168 KiB/block at B=128 (§2.4) to the one PAGE class, and `num_kvcache_blocks = remaining // entry_bytes` (`sub_pool_spec.py:237`) shrinks for the **target** too. At long contexts with LMCache offload this may be the binding constraint. *Close:* M1, pure arithmetic — print `entry_bytes` for target and draft and the resulting block count, no GPU needed. I did not have a Kimi-K3 config in-tree to quote the target's per-block cost against.

**R4 — The fan-out cost in §5.2(c).** T× prefix KV re-read, estimated ~0.7 ms/block-pass at bs64/ctx8k. If it is worse than estimated, mitigations are: raise `PA_GLUON_MAX_QUERY_LEN` handling by splitting into ⌈T/4⌉ groups of `qlen≤4` (causal within a group, strictly-upper correction merged in), or the §5.3 non-causal ASM path. *Close:* M2 with `/capture-trace`.

**R5 — Verify-window / block-width interaction.** `dspark_block_size=7` with `mtp_k=5` is the V4-Pro shape and the limit assertions at `atom/config.py:2405-2430` were written for it, but I did not read them line by line against this combination. *Close:* M1 test §4.9(5) plus reading `:2405-2430`.

**R6 — `--cudagraph-mode FULL` capture safety for the draft.** The MLA draft's whole persistent-buffer edifice (`dspark_proposer.py:255-290`) exists because "a recording made at the warmup context would carry that context's split plan into every replay." The gluon path decides its split count from the grid, **not** from the context (`atom/model_ops/base_attention.py:81-84`) — which is exactly why it is capture-safe and why D2 (one fixed kernel) matters. But I have not verified that the gluon launch makes no other host-side, context-dependent decision. *Close:* M3, capture at a short context and replay at a long one, compare against eager.

**R7 — aiter pin.** All aiter citations are from the `/home/rbrugaro/code/aiter` checkout, not from the pinned wheel (no `python` on PATH in this sandbox). The `mask=0` parameter, `return logits, final_lse`, the `head_size == 128` assert and `assert causal` must be re-confirmed against the pinned build. *Close:* one `python -c "import inspect, aiter; print(inspect.signature(aiter.pa_persistent_fwd))"` on the target image.

**R8 (low, but file it) — the unguarded combination.** Nothing today rejects `--decode-context-parallel-size 8` with a GQA target or an Eagle3-llama draft; it configures cleanly and is silently wrong (§1.3). Our §4.1 guard closes it for drafts. The *target* side (`AiterAttentionMetadataBuilder.prepare_decode`, `aiter_attention.py:1171-1220`) stays unguarded and should get its own small PR.

---

## 9. What I did not verify, stated plainly

- The SpecForge reference implementation (R2) — the dual-source attention description in §4.2 is a faithful restatement of the brief, not of code I read.
- Any GPU execution. Every kernel claim is a source-reading claim.
- The Kimi-K3 target's per-block MLA pool cost, so §2.4's draft overhead is absolute bytes, not a percentage.
- Whether `AutoConfig` preserves the nested `dflash_config` dict through `get_hf_config` (§4.1 item 6).
- Whether `pa_decode_gluon`'s C++/FlyDSL PS reduce tolerates the uninitialized `exp_sums`/`max_logits` partitions, which bears on §4.3(2)'s masking rule.

---

## Appendix B — M0 RESULT: R1 is CLOSED (measured, not reasoned)

Run on MI355X / gfx950 in `rocm/atom-dev:nightly_202610051453` (same day as base
commit `aa4e7a04d`). Script: `tools/m0_gluon_hd64.py`. Called
`aiter.ops.triton.gluon.pa_decode_gluon` directly at the DFlash draft's DCP1 shape
(2 KV heads, 8 Q heads, gqa_ratio 4, `query_length=1`), bf16, vs a dense torch
reference built by gathering the paged cache.

| head_dim | block | parts | context lens | out rel-err | LSE err | status |
|---:|---:|---:|---|---:|---:|---|
| 64 | 128 | 4 | 300/128/777 | 0.00386 | **0.00000** | PASS |
| 128 | 128 | 4 | 300/128/777 | 0.00285 | 0.00000 | PASS (control) |
| 64 | 128 | 8 | 300/128/777 | 0.00331 | 0.00000 | PASS |
| 64 | 128 | 4 | 64/1000/2048 | 0.00355 | 0.00000 | PASS |
| 64 | 128 | **1** | 300/128/777 | 0.00259 | **78.196** | **LSE MISSING** |
| 64 | 64 | 4 | 300/128/777 | 0.00333 | 0.00000 | PASS |

**Conclusions, all measured:**

1. **`head_dim=64` is fully supported** at `block_size=128` on gfx950. Accuracy is
   indistinguishable from the `head_dim=128` control. There is no `head_size` assert
   anywhere in `pa_decode_gluon` (upstream `ROCm/aiter@main`): its guards are
   `query_group_size <= 64`, `query_length <= 4`, tensor ranks and dtypes.
   **The §5.3 head-padding fallback is NOT needed. §5.2 stands.**
2. **The LSE is exactly recoverable** as
   `logsumexp_p(max_logits[...,p,...] + log(exp_sums[...,p,...]))` — error 0.00000,
   i.e. bit-level agreement with the reference. The DCP merge has its input.
3. **CORRECTION to §4.3(2).** The design says to mask to
   `cdiv(local_ctx, CONTEXT_PARTITION_SIZE)` partitions. **That rule is wrong.** The
   kernel divides each request's context *evenly* across `max_context_partition_num`;
   it does not emit fixed 256-token chunks. With `L=300, parts=4` all four partitions
   are live. Masking to `ceil(300/256)=2` is what produced the initial mismatch.
   **Use all `max_context_partition_num` partitions, unmasked.**
4. **NEW CONSTRAINT — `max_context_partition_num >= 2` is mandatory on the DCP path.**
   At `parts=1` the wrapper takes `one_shot = max_context_partition_num <= 1` and never
   writes `exp_sums` / `max_logits`; the buffers keep their `torch.empty` garbage
   (LSE error 78.2 above). The attention *output* is still correct, so this fails
   silently in exactly the acceptance-only way §7.4 warns about. The draft's
   `_dcp_decode` must floor the split count at 2, and a unit test must pin it.
5. **Dispatch caveat.** At `n_seq=1` the call routed to a non-large-block kernel that
   `gl.static_assert`s `KV_BLOCK_SIZE in [16, 64]` and fails at block 128. Single-sequence
   decode at `--block-size 128` therefore needs checking before it can be relied on
   (bs=1 is a real cudagraph capture size). Not on the critical path; test at M2.

### M0 extension — fp8 KV with PER-TOKEN scales (`tools/m0_fp8.py`)

fp8 = `torch.float8_e4m3fn`, scales 4D `[blocks, kv_heads, block_size, 1]` (what
ATOM produces after its `unsqueeze(-1)`), reference computed from the dequantized cache.

| head_dim | block | parts | context lens | out rel-err | LSE err | status |
|---:|---:|---:|---|---:|---:|---|
| 64 | 128 | 4 | 300/128/777 | 0.03983 | 0.01974 | PASS |
| 128 | 128 | 4 | 300/128/777 | 0.03767 | 0.01081 | PASS (control) |
| 64 | 128 | 8 | 300/128/777 | 0.03663 | 0.01358 | PASS |
| 64 | 128 | 4 | 64/1000/2048 | 0.04571 | 0.02296 | PASS |

fp8 error (~4%) is quantization noise and matches the `head_dim=128` control exactly;
the LSE survives fp8 with ~0.02 absolute error. **fp8 + per-token scales at head_dim=64
is confirmed working.** The production `--kv_cache_dtype fp8` path is clear.

**Residual R1 risk: none.** Still untested: `query_length > 1`, which §5.2(c)
deliberately avoids by fanning out to `qlen=1`.

---

## Appendix C — §5.2 IS REFUTED. The corrected block formulation.

Three adversarial panels ran against the design's own load-bearing claims before
any model code was written. **Two refuted at high confidence.** Corrections:

### C.1 Write-once context rows: CONFIRMED (not refuted)

The reference is write-once. `spec_generate` slices the next step's context to
only the newly accepted tokens
(`target_hidden = extract_context_feature(...)[:, : acceptance_length + 1, :]`)
and `past_key_values_draft.crop(start)` keeps verified rows while discarding only
the block's noise rows. `k_ctx` is never recomputed for a row already cached, no
cached row is ever rewritten, and rope-at-absolute-position (applied BEFORE
`cache.update`) makes a row valid on every later step. ATOM's
`compute_draft_kv` already encodes the same invariant.

### C.2 The paged-prefix (+) dense-block LSE merge: REFUTED

Three independent kills:
1. **No LSE exists on the GQA side.** Every entry point in `attention_mha.py`
   returns output only (`paged_attention_unified` :551, `paged_attention_triton`
   :636, `paged_attention_asm` :663, `paged_attention_persistent_asm` :696).
   `merge_attn_states` needs `[NUM_HEADS, NUM_TOKENS]` fp32 LSEs it cannot get.
2. **The paged branch is causally masked** -- `causal=True` is a literal at
   `attention_mha.py:540, 834, 863, 946` and `mask=1` at :692. Bottom-right
   causal with `L_q=T` over `L_k = anchor+1` silently drops the last `T-1`
   CONTEXT keys for row 0.
3. **The split is unnecessary.** `_build_paged_block_metadata` already realizes
   the reference's single attention over the cat by WRITING the block's T rows
   into the pool: `global_lens = anchor_positions + (1 + T)`
   (`dspark_proposer.py:801`), docstring "Unlike V4's, this block WRITES the KV
   it attends."

### C.3 The qlen=1 fan-out: REFUTED AS SPECIFIED, but rescued by C.2

The masking semantics are right -- `pa_decode_gluon` sets
`is_causal = query_length > 1` (`pa_decode_gluon.py:4804`), so qlen=1 is launched
*literally* non-causal, not merely masked to a no-op. What breaks is the LSE:
the fan-out multiplies `num_seqs` by T, and `dense_decode_splits` then returns
**1** for any batch above ~25 (DCP1) or ~3 (DCP8). At splits=1 the wrapper takes
`one_shot` and never writes `exp_sums`/`max_logits` -- the garbage Appendix B
measured (LSE err 78.196).

**But under C.2 no merge is needed, so no LSE is needed at DCP1.** The two
refutations cancel.

### C.4 Can a non-causal qlen=T GQA decode avoid the fan-out? NO -- measured.

`tools/m0_noncausal.py` on gfx950 aborts in kernel selection. Enumerating
`/app/aiter-test/hsa/gfx950/pa/pa_asm.csv` (51 kernels) says why:

| | count |
|---|---|
| `Msk=0` (non-causal), `ps=0`, `blk=16`, `quant=0` | 26 |
| `Msk=1` (causal), `ps=0`, `blk=16` | 10 |
| `Msk=1`, `ps=1`, `blk=256` or `blk=1024` | 15 |
| **`Msk=0` AND `ps=1` (non-causal persistent)** | **0** |
| **any `blkSz=128`** | **0** |

`pa_decode_gluon` also asserts `query_length <= 4`, so T=5 cannot use it at all.
**A non-causal paged GQA decode at qlen=5 does not exist in aiter today.**

### C.5 The corrected formulation

> Write the block's own T rows into the sibling pool (exactly as the MLA draft
> does, `ctx_lens = anchor + 1 + T`), then fan the block pass out to `B*T` rows
> of `qlen=1` and issue ONE gluon paged decode per row. qlen=1 is natively
> non-causal, so every draft position attends `[0, anchor] ++ all T block rows`
> bidirectionally -- the reference's single attention over `cat([k_ctx, k_noise])`,
> with no merge, no LSE, and no new kernel.

Consequences to carry into the code:
- **`ctx_lens` is `anchor + 1 + T`**, NOT `anchor + 1`. The block rows are in the
  pool and must be attended. (One agent argued `anchor`; `dspark_proposer.py:799`
  settles it -- "Each request's KV spans [0, anchor] (context) ++ the T block rows".)
- **Preallocate the fanned-out metadata.** `repeat_interleave` allocates, and a
  cudagraph replay reads captured addresses. Needs `_blk_fan_block_tables
  [max_bs*T, cols]` and `_blk_fan_ctx_lens [max_bs*T]` filled in place, mirroring
  `_blk_slots` / `_blk_ctx_lens` in `_init_block_persistent_buffers`.
- **Cost: the prefix is read T times per block pass.** ~0.7 ms at ctx 8k, but
  linear in context: ~2.7 ms at 32k and ~10.7 ms at 128k, which is the agentic
  range this project targets. This is the price of having no non-causal qlen=T
  GQA kernel, and it is the strongest argument for requesting one from the aiter
  team (non-causal, persistent, blk=128, head_dim 64) as a follow-up.
- **At DCP8 the LSE comes back as a requirement** (cross-rank merge), so the
  split count must be floored at 2 there -- `max(2, dense_decode_splits(B*T, ...))`
  -- with a unit test pinning it, since the output alone is correct at splits=1.
- The gate arithmetic in §5.2(c) was wrong: `gluon_decode_over_limit` computes
  `group_p2 = 1 * max(16, 4) = 16`, not 4. Still under the 64 limit.

---

## Appendix D — C.4 WAS WRONG. A non-causal paged GQA decode DOES exist.

Appendix C.4 concluded "a non-causal paged GQA decode at qlen=5 does not exist in
aiter today". **That is false.** It enumerated the ASM kernel table
(`hsa/gfx950/pa/pa_asm.csv`) and the gluon path and generalized from those two to
"aiter". It never tested the Triton `mha_v3` path, which is exactly where vLLM's
ROCm backend routes this case.

`aiter.ops.triton.attention.mha_v3.flash_attn_with_kvcache` takes `causal`,
`page_table` AND `return_softmax_lse`. Measured on MI355X/gfx950 in
`nightly_202610051453` (`tools/m0_mhav3.py`), q `[B, T, H, D]`, paged KV in the
standard FA layout `[num_blocks, page_size, num_kv_heads, head_dim]`, vs a dense
torch reference:

| head_dim | T | mode | kv heads | LSE | rel err |
|---:|---:|---|---:|---|---:|
| 64 | 5 | **non-causal** | 2 | yes | 0.00250 |
| 64 | 5 | **non-causal** | 16 | yes | 0.00321 |
| 64 | 5 | causal | 2 / 16 | no | 0.0025 / 0.0030 |
| 64 | **7** | **non-causal** | 2 / 16 | yes | 0.0026 / 0.0031 |
| 128 | 5 / 7 | non-causal | 2 / 16 | yes | 0.0020 - 0.0045 |

All PASS. vLLM does the same thing in
`vllm/v1/attention/backends/rocm_aiter_fa.py`: the causal branch goes to
`unified_attention` (which hard-asserts `causal`), and the NON-causal branch is
routed to `mha_v3.flash_attn_with_kvcache(causal=attn_metadata.causal)`.

### What this changes

- **The qlen=1 fan-out of C.5 is NOT needed.** One row per request, T queries,
  bidirectional over `[0, anchor] ++ T block rows` -- the reference's single
  attention, directly.
- **The T-times prefix re-read disappears**, with it the ~2.7 ms @32k /
  ~10.7 ms @128k penalty C.5 accepted, and with it the main argument for
  requesting a new aiter kernel.
- **The DCP LSE problem dissolves**: `return_softmax_lse=True` hands
  `dcp_lse_merge` its input directly. No `max(2, splits)` floor, no reliance on
  the gluon per-partition softmax state.
- **k=7 is viable** whenever we want it; the kernel does not care.
- The `query_length <= 4` gluon assert and the `paged_attention_unified` causal
  hardcoding both become irrelevant -- neither kernel is on this path.

### The one open integration question

The kernel wants the FA paged layout `[num_blocks, page_size, num_kv_heads,
head_dim]`. ATOM's `MhaKvPool` produces `[nb, nh, hd//x, bs, x]` for K and
`[nb, nh, hd, bs]` for V (SHUFFLE/NHD). Either a view/transpose at the call site
or a pool layout choice has to reconcile them, and `write_context_kv_kv` must
write whichever layout the read side expects. Resolve this before the proposer
seam, since it decides what `make_kv_pool` is asked for.

### Cross-engine corroboration (vLLM / SGLang)

- **vLLM has first-class DFlash support**: `vllm/model_executor/models/qwen3_dflash.py`
  is architecturally our exact case (GQA DFlash draft), with
  `DFlashDraftModel -> qwen3_dflash.DFlashQwen3ForCausalLM` in the registry.
- **Naming caution:** vLLM maps the string `"DSparkDraftModel"` to **DeepSeek-V4**,
  not to a Qwen3 DFlash draft. Our checkpoint declares that same string with
  `model_type: qwen3`. Our rename to `K3DFlashModel` avoids the collision; if we
  ever want to align with vLLM, their name for this shape is `DFlashDraftModel`.
- **Both engines write context K/V into the paged cache** rather than materializing
  `torch.cat([proj(target), proj(noise)])` -- vLLM `precompute_and_store_context_kv`,
  SGLang `write_target_hidden_kv`. The explicit cat and the block mask live only
  in SpecForge TRAINING. This confirms the `write_context_kv` approach.
- SpecForge's training mask is `mask_draft = is_draft & (q_block_id == kv_block_id)`
  -- block-diagonal with NO causal term, confirming full bidirectionality.
- **vLLM enforces `num_speculative_tokens == block_size`.** We currently allow
  k=5 against a trained block of 7; worth revisiting once acceptance is measured.
- **Known hazards both projects hit:** KV-cache group explosion for a 5-layer
  draft bucket against the target's layer count (vLLM RFC #58638 names Kimi-K3 +
  DSpark explicitly), and DCP being actively broken for this combination in BOTH
  engines (vLLM #59184, #57992; SGLang #34920). Our DCP caution was well placed.

---

## Appendix E — KV LAYOUT DECISION (resolved)

**Decision: give the DRAFT pool a FLASH-layout view and a flash writer, and run
its KV in bf16. Do not touch the target's layout.**

### Why not the alternatives

- **(A) zero-copy views of the existing SHUFFLE tensors: IMPOSSIBLE for K.**
  `mha_v3.flash_attn_with_kvcache` reads every axis through its own stride (so
  non-contiguous views are fine) but hard-unpacks exactly FOUR axes
  `(block, token, kv_head, head_dim)`. ATOM's SHUFFLE K is
  `[nb, nkvh, hd//x, page, x]` -- `head_dim` is split across two non-adjacent
  axes, so no permute of a 5-D tensor can produce the required 4-D order. V
  happens to work; K cannot. A `reshape` would silently copy.
- **(C) per-step copy:** real cost, and the one existing conversion in-tree
  calls `.contiguous()`. Rejected.
- **(D) a kernel that reads SHUFFLE non-causally:** candidates exist but none is
  a drop-in. `mha_batch_prefill_func` takes `causal=False` and reads ATOM's exact
  SHUFFLE layout, but `block_size` must be 16 or 1024 (we run 128).
  `paged_attention_rocm` with `mtp>1` is non-causal by construction and reads
  ATOM's layout, but its MTP arithmetic (`MTP_PER_THREAD=2 x
  MTP_PARALLEL_THREADS=2 = 4`) looks hostile to our T=5 and needs a probe.
  Worth revisiting as an optimization; not the path to a first working model.

### Why bf16 for the draft pool

`mha_v3` accepts fp8 KV only when **q is also fp8**, and its descales are
strictly 2-D `(batch, num_kv_heads)` -- per-tensor per kv-head. ATOM's fp8 pool
carries per-(kv head, token) scales, which the kernel rejects by assert. Separately,
the only aiter writers that emit FLASH layout (`reshape_and_cache_flash`,
`fused_qk_rope_reshape_and_cache(flash_layout=True)`) are per-tensor-scale only;
`reshape_and_cache_with_pertoken_quant` hard-unpacks a 5-D x-packed K and cannot
write FA layout at all.

bf16 sidesteps both constraints. The cost is 2x on the DRAFT pool only -- 1360 ->
2720 B/token/rank, i.e. ~348 MB/rank at 128k context. The target stays fp8.
fp8 draft KV is a follow-up once a per-token FA writer exists.

### Landed

`MhaKvPool.fa_views(layer)` -> `[blocks, block_size, kv_heads, head_dim]` for K
and V. **Verified zero-copy** (same `data_ptr` as `kv_views`), contiguous, and
`entry_bytes` unchanged at 327680 for the bf16 draft geometry. The pool charges
per byte (`mha_kv_fields`), so element order costs nothing and only the reader's
requirement decides it.

### Remaining for M2

1. `AiterBackend.make_kv_pool` / `draft_kv.py`: let the draft request bf16 +
   FLASH layout independently of the target's `kv_cache_dtype`.
2. `write_context_kv_kv`: dispatch to `reshape_and_cache_flash` when the draft
   pool is FA-layout (the per-token-quant writer cannot express it).
3. The block-pass READ: the draft must bypass `PagedAttentionImpl`'s dispatch,
   which is causal at every arm, and call `mha_v3.flash_attn_with_kvcache(
   causal=False, page_table=..., cache_seqlens=...)` directly.
   NOTE: `cache_seqlens` must be a Tensor -- a `page_table` alone does not route
   to the decode kernel and the call falls into the prefill kernel and asserts.
4. The proposer seam: the six MLA hard-wirings in `dspark_proposer.py`.
