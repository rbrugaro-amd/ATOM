# RESUME — Kimi-K3 + RadixArk DFlash draft

Base: origin/main `aa4e7a04d`. Image `rocm/atom-dev:nightly_202610051453`.
Full decision record (with measured numbers): `KIMI_K3_DFLASH_DESIGN.md` App. A-E.

## State: WORKING. First forward pass done 2026-10-06.
`radixark smoke`: accept_length 2.5, acceptance_rate 0.3, no errors, 128 tokens.
(2 prompts only -- NOT a meaningful acceptance number yet. Run `accept`/`swebench`.)

871 unit tests green, lint clean, nothing committed (all in working tree).

## FIXED BLOCKERS (do not re-introduce)
5. `page_unit_bytes` must be `self.attn_metadata_builder.paged_pool_bytes(1)`, NOT
   `plan.entry_bytes[plan.paged_class]` (model_runner.py ~1849). The merged PAGE
   class is target+draft, but the KDA state image is tucked into the TARGET's
   regions only, which is all `_page_unit_regions` enumerates.
6. `reshape_and_cache_flash` arg6/arg7 (k_scale/v_scale) are TYPED tensors -- None
   is rejected even for a bf16 "auto" write. Pass a 1-element fp32 ones tensor.
7. `KimiK3DFlashAttention.write_context_kv` was lost in a bulk patch once; the
   decoder layer calls it. Keep both methods (attention + decoder layer).

## OLD BLOCKER (resolved)

```
RuntimeError: a PAGE unit is 2097152 B but this pool gives a logical block
24 rows x 73728 B = 1769472 B; the two disagree about block granularity
  atom/model_ops/attentions/pool_layout/page_unit_geometry.py:142 _page_unit_regions
  via atom/model_ops/attentions/gdn_attn.py:773 _checkpoint_copy_plan  (KDA state ckpt path)
```
2097152 - 1769472 = 327680 = EXACTLY the draft pool's `entry_bytes`.
The draft's bytes ARE in the PAGE unit (by design: `DraftKvBuilder.sub_pool_specs`
-> `page_pool` merges into the target's entry class) but are NOT in the
24-MLA-row logical-block figure `page_unit_geometry` reconstructs.
Fix likely belongs in how `page_unit_geometry` derives its logical block, not
in the draft. See vLLM RFC #58638, which names this exact pairing.

## OPERATIONS — how to start / restart / tear down

Everything goes through `tools/dflash_ab.sh <arm> <workload>`; it starts the
container, waits for readiness, runs the client, and tears down on success.

    arms      : radixark (new GQA draft) | inferact (existing MLA draft) | nospec
    workloads : smoke | lossless | accept | swebench | perf

    ./tools/dflash_ab.sh radixark smoke

Env knobs: `K=7` (default 5), `RUNDIR=...` (default /tmp/dflash_ab/<arm>),
`READY_TIMEOUT=2400`, `KEEP_ON_FAIL=0` (default 1 keeps a failed container so
its logs survive -- `docker rm -f` destroys the only evidence).

### Hard-coded in tools/dflash_ab.sh (change there, not on the CLI)
    IMAGE    rocm/atom-dev:nightly_202610051453   (matches origin/main aa4e7a04d)
    MODEL    /data/.../models--moonshotai--Kimi-K3/snapshots/f831ab66814297da540d832a5235f8e904f29d06
    RADIXARK /data/.../models--RadixArk--Kimi-K3-DSpark/snapshots/3c5bac301d9cf392706189d82ed947feca6c2f0f
    INFERACT /data/.../models--Inferact--Kimi-K3-DSpark/snapshots/cf6b8244620e7ea4b0651d214f28e89eac75bed6
    (all under /data/workloads-inference/hf_hub_cache/)
Serve shape: TP8, DCP1, fp8 KV (draft forced bf16), block 128, max-num-seqs 64,
prefix caching OFF, no LMCache. The repo is bind-mounted at /work, so a source
edit takes effect on the next container start -- no rebuild.

### Watching it
    docker ps -a --format '{{.Names}}\t{{.Status}}' | grep k3-ab
    docker logs -f k3-ab-radixark 2>&1 | tr '\r' '\n'      # tqdm uses \r
    docker logs k3-ab-radixark 2>&1 | tr '\r' '\n' | grep -E "^[A-Za-z_.]*(Error|Exception):" | sort -u
    rocm-smi --showmemuse | awk -F': ' '/VRAM%/{print $NF+0; exit}'

### Gotchas that cost time already
- Load is ~4.5 min and goes SILENT around shard 12/96. That is not a hang.
- Readiness = VRAM% > 0 AND /health. `curl /health` alone returns OK with no
  model loaded (CLAUDE.md).
- `docker run -d` prints the container id, NOT server output; the script streams
  real logs via a backgrounded `docker logs -f`. That streamer dies with its
  parent shell -- re-run it by hand if you killed the script.
- A failed run leaves the container up on purpose. `docker rm -f k3-ab-<arm>`
  before relaunching, or the name collides.
- Only 8 of the box's 40 GPUs are used (TP8).

### Teardown
    docker rm -f k3-ab-radixark k3-ab-inferact k3-ab-nospec 2>/dev/null

### Unit tests (no server, needs GPU only for the aiter import)
    docker run --rm --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined \
      --group-add video --ipc=host --shm-size 8G -v $PWD:/work -w /work \
      -e AITER_LOG_LEVEL=WARNING rocm/atom-dev:nightly_202610051453 \
      python3 -m pytest tests/ -q -k "dflash or draft or dspark or mha or pool"
Host lint: `black --check . && ruff check .` (both on PATH at ~/.local/bin).


## Four bring-up bugs already found+fixed (do not re-introduce)
1. `share_with_target`: it is `target_base.model.embed_tokens` / `target_base.lm_head`.
2. warmup runs BEFORE `allocate_kv_cache()` -> `kv_cache_data` is None; `_caches`
   returns (None,None) and forward returns zeros (that pass is memory profiling).
   Key on the POOL, not `is_dummy_run` (the DP-align dummy has a bound pool).
3. `DSparkMarkovHead.sample_next(prev_ids, logits, out)` writes ids into `out`
   and returns the markov embedding -- it does NOT return ids.
4. `build_kv_cache_tensor` must use the POOL's dtype (draft is bf16 under an
   fp8 target) and bind `fa_views`, not `kv_views`.

## Design decisions that are settled (do not relitigate)
- Draft pool: FLASH layout (`MhaKvPool.fa_views`, zero-copy) + **bf16**.
  mha_v3 takes fp8 only if q is fp8 too, and its descales are per-(batch,kv_head);
  omitting them under fp8 does NOT raise -- it warns and uses 1.0 (silent wrong numerics).
- Read: `mha_v3.flash_attn_with_kvcache(causal=False, page_table=..., cache_seqlens=<TENSOR>)`.
  `cache_seqlens` MUST be a Tensor or it routes to the prefill kernel and asserts.
  This is the ONLY aiter entry non-causal at qlen>1 over a paged GQA cache.
- Write: `aiter.reshape_and_cache_flash(k,v,k_cache,v_cache,slot,"auto",None,None)`.
  The block WRITES the KV it attends; `ctx_lens = anchor + 1 + T`.
- Bypass `PagedAttentionImpl` entirely: it is causal at every arm.
- Norms in the model (norm THEN rope); `q_norm=None,k_norm=None` to Attention.
- k=5. ATOM k == SGLang block_size (no off-by-one; see kimi_k3_dspark forward_spec comment).
- DCP8 is a LATER milestone (M4). DCP+non-MLA-draft is broken in vLLM and SGLang too.

## NEXT: validation chain (wired; smoke done, rest pending)
1. `./tools/dflash_ab.sh radixark lossless` then
   `tools/dflash_ab_compare.py lossless --a /tmp/dflash_ab/nospec --b /tmp/dflash_ab/radixark`
   (no-spec reference ALREADY BANKED at /tmp/dflash_ab/nospec/). Must be identical.
2. `accept` on radixark + inferact -> `dflash_ab_compare.py accept` (the uplift).
3. `swebench` both arms. 4. `perf` both arms.
NEVER pass --spec-decode-acceptance-length: it fabricates the number being measured.
