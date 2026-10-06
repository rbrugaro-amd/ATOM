> # ⚠️ SUPERSEDED — DO NOT USE AS A SOURCE OF TRUTH
>
> Written against a tree **232 commits behind** `origin/main`. Multiple claims in
> here are now known to be WRONG, including the file/symbol names it builds on
> (`eagle3_kv_builder.py` / `Eagle3DraftBuilder` no longer exist; the fork is not
> `draft_is_mla`), the image-version claim, and `draft_graph.py`'s absence.
>
> **Read `RESUME.md` and `KIMI_K3_DFLASH_DESIGN.md` instead.** Kept only as a
> record of the starting hypothesis.

# Enabling `RadixArk/Kimi-K3-DSpark` (DFlash/GQA draft) in ATOM

Working notes for the `rbrugaro/k3-dflash-draft` branch.

**Goal:** let ATOM serve Kimi-K3 with the SGLang-side DSpark draft
(`RadixArk/Kimi-K3-DSpark`) in addition to the vLLM-side one
(`Inferact/Kimi-K3-DSpark`) that `atom/models/kimi_k3_dspark.py` already supports.

**Motivation:** at k=5 on the MI355X C32 agentic recipe, ATOM measures a mean
acceptance length of **3.17**, materially below what the SGLang recipe gets at the
same k. The two stacks are not running the same draft model — they are running two
independently trained checkpoints that merely share the "DSpark" name.

---

## 1. The two checkpoints

Both are on the MI355X box under `/data/workloads-inference/hf_hub_cache/`.

| | `RadixArk/Kimi-K3-DSpark` (SGLang) | `Inferact/Kimi-K3-DSpark` (ATOM/vLLM) |
|---|---|---|
| `architectures` | `DSparkDraftModel` | `K3DSparkModel` |
| `model_type` | `qwen3` | `k3_dspark` |
| Attention | **GQA**, 64 q / 16 kv heads, `head_dim` 64, **QK-norm** | **MLA**, `q_lora_rank` 1536, `kv_lora_rank` 512, nope 128 / rope 64 |
| Layers | 5 | 5 |
| Params / size | 2.25 B, 4.5 GB bf16, 62 tensors | ~3.56 B, 7.1 GB bf16, 68 tensors |
| Aux target taps | `[7, 23, 51, 67, 83]` (under `dflash_config`) | `[2, 23, 47, 71, 89]` (top level) |
| `mask_token_id` | `163824` (under `dflash_config`) | `163837` (top level) |
| RoPE | θ=10000, YaRN ×16 from 65536 | θ=50000, YaRN ×32 from 32768 |
| Trained block | 7 (`block_size` top level) | 7 (not recorded in config) |
| Markov head | `markov_rank` 256, `vanilla` | `markov_rank` 256, `vanilla` |
| Trained by | SpecForge, hidden states from a live **SGLang** target | TorchSpec, hidden states from a live **vLLM** target |

Same algorithm (block-parallel draft + low-rank Markov head + confidence head),
different backbone, different training stack, different target-engine numerics.

### Published acceptance length (both at k=7, from the model cards)

| Workload | RadixArk | Inferact |
|---|---:|---:|
| SWE-Rebench / SWE-bench Pro | **4.66** | **3.35** |
| GSM8K | 5.42 | 5.64 |
| MATH-500 | 4.13 | 3.82 |
| MT-Bench | 3.93 | 3.14 |
| AIME26 | 2.99 | 2.72 |

The agentic/SWE row is the one that matters for this submission, and it is where
the gap is widest. Two caveats before treating the delta as fully settled:

- The estimators differ. RadixArk quotes SGLang's histogram-native per-request
  `acc_len` averaged within question then across questions; ATOM's counter is not
  defined identically.
- Each draft was trained on hidden states streamed from the engine now serving it.
  A cross-engine draft is off-policy by construction, so RadixArk-under-ATOM will
  not necessarily reproduce RadixArk-under-SGLang.

### Try this first, before any code

Both checkpoints were trained at `block_size=7`, and both the InferenceX ATOM
recipe and this repo's C32 config run at `--num-speculative-tokens 5`. Running
below the trained block width truncates a block the draft was trained to emit
whole. Bumping the existing Inferact draft to **k=7** is a one-line config change
(plus resizing `--cudagraph-capture-sizes` to 2 × 32 × (1 + 7) = 512) and costs
nothing to measure. The Inferact card's 3.35 at k=7 vs our measured 3.17 at k=5 is
consistent with this being worth real acceptance length.

---

## 2. Why it does not load today

Three hard stops, in the order they fire:

1. **Config loading.** `model_type: qwen3` is not in `_PLAIN_CONFIG_MODEL_TYPES`
   (`atom/config.py:617`), so the draft config does not take the
   "bare `PretrainedConfig`, keep every field verbatim" path that `k3_dspark` takes
   at `atom/config.py:701`. *Verify what actually survives* — transformers usually
   retains unknown kwargs as attributes, but `head_dim` and the nested
   `dflash_config` are the ones to check.
2. **Normalizer never runs.** `atom/config.py:1223` gates on
   `arch.endswith("DSparkModel")`. `DSparkDraftModel` ends with `DraftModel`, so
   `_normalize_draft_dspark_config` (`atom/config.py:959`) is skipped and none of
   the `dspark_*` canonical fields get set. Even if it did run, it reads
   `target_layer_ids` and `mask_token_id` from the **top level**; RadixArk nests
   both under `dflash_config`.
3. **No model class.** `support_draft_model_arch_dict`
   (`atom/spec_decode/drafter.py:112`) has no `DSparkDraftModel` entry →
   `KeyError` at `drafter.py:158`. Same gap in
   `atom/plugin/vllm/model_wrapper.py:167` and `atom/plugin/vllm/register.py:42`.

### The good news: the KV plumbing already exists

This was the piece I expected to dominate the effort, and it is already upstream.
`DSparkProposer._load_model` (`atom/spec_decode/dspark_proposer.py:203`) forks on
the draft's attention type:

```python
draft_is_mla = bool(getattr(draft_hf, "kv_lora_rank", None))
if not draft_is_mla:
    from atom.spec_decode.eagle3_kv_builder import Eagle3DraftBuilder
    self.runner.eagle3_draft_builder = Eagle3DraftBuilder(self.runner, draft_hf)
```

An MLA draft binds into the **target's** paged pool as extra rows (its 576-wide
latent is identical to the target's full-attention layers). A non-MLA draft gets a
**sibling pool** via `Eagle3DraftBuilder` (`atom/spec_decode/eagle3_kv_builder.py`),
which sizes itself from `draft_hf.num_key_value_heads`, `draft_hf.head_dim`, and
`draft_hf.num_hidden_layers` — all three present at the top level of RadixArk's
config (16 / 64 / 5). RadixArk has no `kv_lora_rank`, so it takes the sibling-pool
branch automatically.

Note this landed **after** `rocm/atom-dev:nightly_202609251613`, the image the
mlperf-endpoints C32 recipe pins. That image does not have the fork. Any bring-up
needs a newer image or a source build.

---

## 3. Checkpoint layout mapping

62 tensors, single-file BF16. Shapes confirmed by reading the safetensors header.

| RadixArk tensor | Shape | ATOM `KimiK3DSpark` equivalent |
|---|---|---|
| `fc.weight` | `[7168, 35840]` | `context_proj.weight` — **same 35840 = 5 × 7168** |
| `hidden_norm.weight` | `[7168]` | `context_norm.weight` |
| `norm.weight` | `[7168]` | `final_norm.weight` |
| `layers.{i}.self_attn.q_proj.weight` | `[4096, 7168]` | — (MLA has `q_a_proj`/`q_b_proj`) |
| `layers.{i}.self_attn.k_proj.weight` | `[1024, 7168]` | — |
| `layers.{i}.self_attn.v_proj.weight` | `[1024, 7168]` | — |
| `layers.{i}.self_attn.{q,k}_norm.weight` | `[64]` | — **no MLA equivalent; per-head QK-norm** |
| `layers.{i}.self_attn.o_proj.weight` | `[7168, 4096]` | `o_proj` (`[7168, 8192]` in MLA) |
| `layers.{i}.mlp.{gate,up}_proj.weight` | `[14336, 7168]` | same → fuse to `gate_up_proj` |
| `layers.{i}.mlp.down_proj.weight` | `[7168, 14336]` | same |
| `layers.{i}.{input,post_attention}_layernorm.weight` | `[7168]` | same |
| `markov_head.markov_w{1,2}.weight` | `[163840, 256]` | **identical name and shape — reusable as-is** |
| `confidence_head.proj.{weight,bias}` | `[1, 7424]`, `[1]` | SKIP (training-only; 7424 = 7168 + 256) |

No `embed_tokens`, no `lm_head` — both shared from the target, same as Inferact.

**Key correction to the config:** `num_target_layers: 93` in RadixArk's config is
the *target's depth*, not the aux-tap count. The aux count is
`len(dflash_config.target_layer_ids)` = 5, which is what `fc.weight`'s 35840
confirms. ATOM's `KimiK3DSpark` sizes its `context_proj` as
`target_hidden_size * num_target_layers` (`kimi_k3_dspark.py:621`) — that
expression is correct for Inferact (`num_target_layers: 5`) and **wrong** here.
Size the fusion from `len(target_layer_ids)` instead.

---

## 4. Implementation plan

Ordered so each step is independently testable.

### Step 1 — config plumbing

- Key the plain-config path on **architectures**, not `model_type`
  (`atom/config.py:701`). `qwen3` is a legitimate target `model_type` and must not
  be blanket-routed to the draft path. Something like: if
  `architectures[0]` ends in `DraftModel`/`DSparkModel`, take the verbatim
  `PretrainedConfig.from_dict` route.
- Widen the normalizer gate at `atom/config.py:1223` to catch `DSparkDraftModel`.
- In `_normalize_draft_dspark_config`, fall back to `dflash_config.target_layer_ids`
  and `dflash_config.mask_token_id` when the top-level fields are absent.
- Map the top-level `block_size: 7` onto `dspark_block_size` so `_resolve_mtp_k`
  (`dspark_proposer.py:212`) knows the trained width. Current K3 path has no block
  width in config and leans entirely on `--num-speculative-tokens`.
- Set `dspark_with_draft = True` as the MLA path does.

**Test:** construct the config standalone and assert the `dspark_*` fields and
`head_dim`/`num_key_value_heads` come out right. No GPU needed.

### Step 2 — the model module

New `atom/models/kimi_k3_dflash.py`, subclassing `DSparkDraftModel`
(`atom/models/dspark_draft.py`) exactly as `kimi_k3_dspark.py` does. Reuse
`DSparkMarkovHead` from `kimi_k3_dspark.py` unchanged.

Implement the three contract methods:
- `project_context(aux_concat)` → `hidden_norm(fc(aux))`, fusion width from
  `len(target_layer_ids)`.
- `context_layers` → the 5 GQA layers.
- `write_context_kv(aux_concat, positions)` → scatter into the **sibling** pool
  rather than the target's latent rows. Follow the Eagle3 MHA path, not the
  Inferact scatter at `dspark_proposer.py:373`.

Backbone specifics that differ from every existing ATOM K3 draft:
- GQA with per-head `q_norm`/`k_norm` (Qwen3 style) — no MLA analogue, needs
  writing.
- The draft's **own** RoPE (θ=10000, YaRN ×16 from `original_max_position_embeddings`
  65536). Do not inherit the target's θ=50000/×32. This is a silent-wrong-answer
  class of bug: it loads clean and just degrades acceptance.
- `skip_weight_prefixes` must cover `confidence_head.`, as the MLA version does at
  `kimi_k3_dspark.py:598`.
- Fuse `gate_proj`/`up_proj` into `MergedColumnParallelLinear`, same as the MLA
  module.

### Step 3 — registration

- `atom/spec_decode/drafter.py:112` → `"DSparkDraftModel": "atom.models.kimi_k3_dflash.KimiK3DFlash"`
- `atom/plugin/vllm/model_wrapper.py:167` and `_DSPARK_DRAFT_ARCHS` at `:172`
- `atom/plugin/vllm/register.py:42`

### Step 4 — bring-up

- `--method dspark --draft-model <RadixArk path> --num-speculative-tokens 7`.
- Confirm `Eagle3DraftBuilder` is selected (it should be automatic — no
  `kv_lora_rank`) and that the sibling pool is sized sanely.
- Draft cudagraph capture for the new backbone shape
  (`atom/spec_decode/draft_graph.py`).
- Start **TP8 / DCP1 / no LMCache / no ReplaySSM** to isolate the draft. Only then
  layer back the C32 recipe's DCP8 + LMCache offload + `ATOM_ENABLE_REPLAYSSM=1`.
  The sibling pool interacts with all three and debugging them together is a trap.

### Step 5 — validate

- Acceptance length at k=7 and k=5, against the Inferact baseline (3.17 at k=5).
- **Lossless check.** Per `utility/endpoints_rules.md` §2 in mlperf-endpoints, the
  verifier must not accept tokens the target would not have produced; output must
  be token-for-token identical to no-speculation greedy. Diff greedy output with
  and without the draft before trusting any acceptance number.
- Do **not** set `--spec-decode-acceptance-length` during this work. It forces a
  fixed mean AL and makes the rejection sampler ignore real draft/target
  agreement — it would fabricate exactly the number being measured.

---

## 5. Effort and the honest alternative

With `Eagle3DraftBuilder` already upstream, the remaining work is a model module
plus config plumbing — tractable, not the multi-week engine port it would have been
without it. The GQA + QK-norm backbone and the RoPE handling are the two places to
spend care.

That said, the architectural difference is deliberate, not accidental: Inferact
chose MLA so the draft's 576-wide latent unifies with the target's KV pool, which
is what makes KV offload and P/D disaggregation work without a second page format.
A GQA draft gives that up. If the goal is acceptance length on the submission
recipe rather than a general capability, the cheaper asks are (a) k=7 on the
existing draft, and (b) a retrained MLA-native draft from Inferact/AMD.

---

## Repo setup

Worktree of `ROCm/ATOM` branched from `origin/main` at `ae38bd9c`, with
`rbrugaro-amd/ATOM` added as remote `fork` and set as this branch's push target:

```
git -C ~/code/ATOM worktree list
git push        # goes to fork/rbrugaro-k3-dflash-draft
```

The fork was 0 ahead / 232 behind upstream at the time of branching, so upstream
main was used as the base.
