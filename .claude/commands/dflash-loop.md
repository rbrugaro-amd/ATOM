---
description: One unattended step of the Kimi-K3 DFlash bring-up — reads RESUME.md as its cursor, does the next thing, writes the result back. Designed to be driven by /loop.
---

# dflash-loop — one step of the K3 DFlash bring-up

Read `RESUME.md` first. It is the **only** cursor; this prompt is stateless and
fires verbatim on every wake, so never assume the state described below is still
current — RESUME.md wins wherever the two disagree.
`KIMI_K3_DFLASH_DESIGN.md` §6 holds the milestone exit criteria;
Appendix F replaced M2's gate (use the Appendix F gate, **not** §7.3).

Do **one** step per wake, then write the result back and reschedule.

## Where the project is

M0, M1, M2 are **done and passed** (M2 closed 2026-10-06: GSM8K 1319 samples,
radixark 0.9484 vs nospec 0.9393, ~1.0 sigma, no regression).

**ACCEPTANCE IS SOLVED** (2026-10-06). Root cause found, confirmed causal by a
two-armed A/B, and confirmed on GSM8K. RESUME.md has the full trail.

ATOM fed the radixark draft the RAW block-local `prefix_sum` while the draft was
trained (by SpecForge, on a live SGLang target) on the softmax MIXTURE of
Kimi-K3's attn-residual bank + prefix. Feeding the mixture
(`ATOM_K3_AUX_AGGREGATE=1`):

| GSM8K K=5, 1319 samples | before | after | vs inferact |
|---|---|---|---|
| acceptance | 55.61% | **62.04%** (+6.43 pp) | 62.54% -> **parity (-0.50 pp)** |
| toks/fwd | 3.78 | **4.10** | 4.13 |
| accuracy | 0.9484 | 0.9500 | within noise (1.20 sigma) |

It is CORRECT that the same flag makes **inferact WORSE** (-2.61 pp): that arm
was trained on vLLM/HF hidden states, so it wants today's raw prefix. The
opposite-direction response on the two arms is what proved causality.

**DO NOT make the aggregate unconditional.** The remaining work is to ROUTE the
feature by draft provenance — and that is **Rita's design call**, not a loop
decision. See "DECISION NEEDED" in RESUME.md. Until she answers, the fix stays
behind the env flag; proceed with the items below, which are unblocked.

## Priority order — take the first item that is not done

0. **BLOCKED ON RITA — the provenance-routing decision.** Do not implement a
   guess. Do not spend wakes relitigating it. If she has answered (check
   RESUME.md "DECISION NEEDED"), implement exactly what she chose, re-run
   `radixark gsm8k` + `inferact gsm8k` to confirm both arms land right, and
   remove the env flag.

1. **Acceptance debug — DONE, see above. Priority 1 is CLOSED.**
   What remains here is OPTIONAL and low value; prefer item 2.
   - residual **-0.50 pp** vs inferact: inside the noise of this measurement.
     Do NOT chase it; it is not a defect.
   - rung a (`K=7`) is still unrun. It is now only interesting for comparing to
     the card's 5.4176, not for explaining a deficit. Low priority.
   - The old ladder (rungs b-e) is spent: profiling showed the gap was quality
     not cost; the SGLang compare found the root cause; the bisect found no bug.

   ### Historical ladder (kept for the citation trail; do NOT re-run)

   ### THE NUMERICS BISECT IS DONE AND FOUND NO BUG. Do not redo it.
   RESUME.md "NUMERICS BISECT" has the full table. Proven clean, each with a
   real test left in `tools/`: weight loading at **TP1 and TP8** (1355 checks,
   0 failures, all-gather reconstruction bitwise), the **KV pool write->read
   round trip** (~90 checks bit-exact at zero tolerance — so the
   `non_causal`->`fa_views` mis-striding theory is **DISPROVEN**), the **forward
   operator graph** (no checkpoint below cos 0.99987, with a negative control
   that would have caught a rope-slice bug at maxdiff 14.1), rope/mscale,
   block positions, fc order, aux capture, and the markov sampler.
   **The draft implementation is not the problem.** Re-running any of these is
   a wasted wake.

   Ground truth is banked: `tools/ref_dump/ref_dflash_eager.pt` +
   `tools/ref_dflash.py`. Reuse it; do not regenerate.

   ### Remaining ladder, one rung per wake, record each result in RESUME.md

   **rung 0 — THE LEADING HYPOTHESIS, do this first.** RESUME.md
   "ROOT-CAUSE HYPOTHESIS" has the full citation trail. Short version: Kimi-K3's
   residual stream is a *bank of per-12-layer snapshots + a running prefix*
   mixed by softmax; SGLang's dspark aux tap emits the **mixture**
   (`aggregate_stream`), ATOM emits the **raw prefix** and throws the bank away
   (`kimi_k3.py:1653`). radixark was trained on SGLang hidden states, inferact
   on vLLM/HF ones — which is exactly why only radixark is short.
   Do step 1 (offline cos of `prefix_sum` vs `aggregate_stream` at the 5 taps,
   NO GPU-hours) before step 2 (env-gated A/B). **Do NOT edit `kimi_k3.py` — it
   is `@support_torch_compile`**; the bank is already in the layer's return
   tuple, so do it in the proposer's `_aux_capture_spec` hook.
   **Falsifier: radixark should rise to ~62-63% and inferact should get WORSE.
   If both move the same way, the hypothesis is wrong — say so and move on.**

   a. `K=7 ./tools/dflash_ab.sh radixark gsm8k` — the draft is trained at block
      width 7 and we run k=5. A knob, not a rewrite. **This is also the only
      run that makes our number directly comparable to the card's published
      GSM8K acc_len of 5.4176 (same benchmark, same n=1319).** And it tests the
      unverified assumption behind the whole "port-specific deficit" argument:
      that both drafts degrade identically from k=7 to k=5. If we land near
      5.4, there is no deficit and the hunt is over.
   b. **Profiling.** `./tools/dflash_ab.sh <arm> trace`, then
      `python3 tools/analyze_trace_summary.py /tmp/dflash_ab/trace/<arm>/rank_0/*.pt.trace.json.gz`.
      Separates "the draft proposes worse tokens" from "the draft costs more per
      step". Run radixark and inferact and compare draft-step annotations. Do
      not hand-roll zcat analysis — the analyzer already extracts
      prefill/decode/draft steps and diffs draft steps across iterations.
   c. **Compare against SGLang upstream, which runs this same checkpoint in
      production and is where the card's 5.4176 came from.** The full reference
      implementation is checked out locally:
      `/home/rbrugaro/code/sglang/python/sglang/srt/models/dflash.py`,
      `srt/speculative/dflash_worker_v2.py`, `dflash_info_v2.py`,
      `dflash_utils.py`, `srt/models/dspark.py`.
      Compare ONLY things that can move acceptance: what goes in the block and
      at which positions, how context KV is written and when it is reused across
      steps, the verify/accept bookkeeping, whether the **confidence head** is
      used for adaptive block length (ATOM skips it — the ckpt sets
      `enable_confidence_head: true`), and anything SGLang does per drafting
      step that ATOM does once (or vice versa). Ignore kernel/backend
      differences that only affect speed.
   d. **Proposer plumbing under REAL batches** — the one thing the bisect could
      not reach, because it hand-supplied a single-request slot map. Multi-request
      batches, non-page-aligned anchors, `anchor+[1..T]` bookkeeping ACROSS
      drafting steps, KV staleness/reuse. Cheap probe: GSM8K at concurrency 1 vs
      32 (`NUM_CONCURRENT` in `tools/dflash_ab.sh:run_gsm8k`). Batch-dependent
      breakage shows as acceptance FALLING with batch size.
   e. **fp8 / real-activation regime.** The bisect used seeded gaussian aux
      states and a 0.02-scale FAKE embedding, so it validated the operator graph
      only — magnitudes are unrepresentative and any quantization-regime bug is
      invisible to it. The target runs fp8 KV while the draft pool is forced
      bf16.

   ### Separate from acceptance: a known tok/s overhead, not draft quality
   Review finding 1 (`dspark_proposer.py:960`): the GQA early-return sits after
   MLA-only `kv_indptr`/`kv_indices` work, so every draft step pays a cumsum, a
   borrow/zero/restore of the runner's live `block_tables`, and a
   `kv_indices_generate_triton` launch the MHA draft never reads. Move the return
   to just after `:912`; gate `_blk_kv_indices` (`:185`, ~268 MB/rank of
   never-read VRAM). **This changes the speed column, not the acceptance
   column** — do not count it as progress on the gap.

2. **Validation chain — COMPLETE. All 5 workloads, both arms. Do not re-run.**
   `lossless` PASS; `gsm8k` parity + accuracy in noise; `accept` parity;
   **`swebench` radixark WINS +6.4%**; **`perf` radixark WINS +1.9%**
   (single-stream latency, NOT throughput — see RESUME.md before quoting it).
   radixark leads on both realistic workloads and is at parity on math/chat.
   Full table in RESUME.md "VALIDATION CHAIN: COMPLETE".

3. **M3** — DCP1 production shape. **Smaller than it reads:**
   `--cudagraph-mode FULL` and `--max-num-seqs 64` are **already in effect** and
   always have been. The only real deltas are **prefix caching** and **LMCache**.
   - **prefix caching: ESCALATED, not done, and NOT this branch's problem.**
     Acceptance/accuracy unchanged (61.92% vs 62.04%), but the hit rate was
     0.0% — and the discriminating run showed **`nospec` is ALSO 0.0%** with no
     draft loaded at all. So spec decode does not suppress the cache and the
     draft is cleared. Prefix caching appears not to hit on Kimi-K3 at all,
     which is an ATOM-wide question (or an unwired metric), out of scope here.
     **Do NOT spend further wakes on it.** The draft-pool/cache hazard stays
     untested because it is unreachable at a 0% hit rate. Details + the one
     remaining discriminator in RESUME.md.
   - LMCache offload: not started. Needs `kv_transfer_config` (today `'{}'`).
   Harness knob: `PREFIX_CACHING=1` -> `--enable_prefix_caching`; default stays
   off so banked results do not shift.

4. **M4** — DCP8. The oracle is **acceptance length within 0.2 of M3's**, not
   accuracy: a DCP8 run that is fast and fluent with 0.5 fewer accepted tokens
   is a correctness failure (§7.4). Then **M5** — plugin parity + CI cells.

5. **PR blockers — DEFERRED.** These gate *opening the PR*, not the work above.
   Do not spend a wake on them while 1-4 have anything open; pick them up only
   when the queue above is clear or something there is hard-blocked.
   - `model_runner.py:1856` `page_unit_bytes=self.attn_metadata_builder.paged_pool_bytes(1)`
     breaks **DeepSeek-V4 / V4.1-Flash startup** — base `paged_pool_bytes`
     returns 0, those backends do not override it, and
     `PagedStateCheckpointSpec.__post_init__` rejects `<= 0`. The K3 motivation
     is right, the expression is too narrow: use the target builder's own PAGE
     declaration. Needs a DSV4 checkpoint to validate against — if none is
     available, leave it and move on rather than guessing.
   - `ruff check --fix tools/m0_*.py` plus the `BLE001` blind-excepts by hand.
     Those files are new on this branch, so their findings land in diff context
     and will fail CI; the ~543 repo-wide pre-existing ones will not.

## Rules

- **The box has 8 GPUs; TP8 is the whole box.** Never run two arms at once.
  Check `docker ps --format '{{.Names}}' | grep k3-ab` before starting any
  GPU rung; if one is live, do NOT start another — but do not idle either:
  **rung (c), the SGLang comparison, is pure code reading and is safe to do
  while a GPU run is in flight.** Prefer it over sleeping. Every other rung is
  serial on the whole box.
- Drive every workload through `./tools/dflash_ab.sh <arm> <workload>`,
  backgrounded. Do not block on it — schedule the next wake for when it should
  realistically be done. Load alone is ~4.5 min and **goes silent around shard
  12/96; that is not a hang.** Readiness is VRAM% > 0 **and** /health, never
  /health alone.
- On any server or GPU error, run `/debug-guide` before anything else. Never
  retry a failed run unchanged — change something or escalate. A failed run
  deliberately leaves its container up (`KEEP_ON_FAIL=1`); read its logs before
  `docker rm -f`, which destroys the only evidence.
- **Never pass `--spec-decode-acceptance-length`** — it fabricates the number
  being measured.
- The kineto exporter is async: `/stop_profile` returns while the per-rank
  write+gzip takes 10-60 s. The harness blocks on `wait_trace_export`; do not
  bypass it.
- The target is **not run-to-run deterministic**. Before calling any
  spec-vs-nospec text difference a bug, check the control. RESUME.md records why
  exact-text equality is unachievable on this stack.
- After every step, update RESUME.md: record the measured number, tick what
  passed, and append any new blocker to FIXED BLOCKERS so it is never
  re-introduced. A step with no number recorded did not happen.
- Do not relitigate the settled decisions section in RESUME.md.

## Stop condition

Call `ScheduleWakeup` with `stop: true` — after writing the reason into a
`## NEEDS HUMAN` section at the top of RESUME.md — if any of these hold:

- M5 is complete, or the PR is ready to open;
- the same step has failed on three consecutive wakes;
- the acceptance ladder (1a-1d) is exhausted and the gap is still unexplained —
  that is a design question, not a debugging one;
- a decision is needed that is Rita's to make (abandoning the GQA draft,
  changing the trained block width, accepting a regression).
