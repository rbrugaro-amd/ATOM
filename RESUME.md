# RESUME — Kimi-K3 + RadixArk DFlash draft

## BENCH-ONLY CHANGES — REVERT BEFORE UPSTREAM

Added 2026-10-06 so the model card's metric can be measured at the card's
concurrency. The card defines acc_len per REQUEST ("averaged within each
question and then equally across questions"); ATOM aggregates into ONE GLOBAL
distribution with no per-request attribution, which forced the harness to
serialise traffic and difference the globals (bs=1 — itself a mismatch vs the
card). This adds per-request attribution ALONGSIDE the global counters.

**`grep -rn "BENCH-ONLY" atom tools tests` finds the whole change.** Every added
block is fenced by `# BENCH-ONLY (per-request acceptance): REVERT BEFORE
UPSTREAM` ... `# end BENCH-ONLY`.

**Default OFF.** Gated on `ATOM_PER_REQUEST_SPEC_STATS=1`. It is not free:
`update_spec` is per-sequence-per-decode-step, and enabled it adds two dict
lookups plus two int adds there. Disabled, the cost is one bool load at the
scheduler call site and one at the top of the per-request block — no
allocation, no logging, nothing O(live requests). `tools/dflash_ab.sh` sets it
automatically for the `gsm8k_card` workload only, so no other banked workload
runs with it on.

**Memory leak, how it is handled** (this is the hazard that would kill a
production server):
- `EngineStats.finish_spec_request()` pops the live counters, called from
  `Scheduler._mark_finished` — the single retirement point for any sequence
  that ever ran a decode step (the other two `status = FINISHED` sites,
  `_reject_aborted_waiting` and the unschedulable branch, retire sequences that
  never decoded, so they are never in the dict).
- Finished records go into a `deque(maxlen=20000)`
  (`ATOM_PER_REQUEST_SPEC_STATS_CAP`) — bounded by construction, and the
  payload reports `finished_truncated` when it wraps.
- The live dict has an INDEPENDENT cap (`ATOM_PER_REQUEST_SPEC_STATS_LIVE_CAP`,
  default 8192), evicting the oldest insertion in O(1). So it cannot grow
  without bound even if some future path retires a sequence without going
  through `_mark_finished`. Both bounds are covered by tests.

**Preemption:** counters deliberately CONTINUE across preempt/resume (same
`seq.id`/`external_request_id`). One question stays one sample, which is what
"averaged within each question" means and what the global estimator already
does; resetting would split a question into two samples and bias the mean.

**Globals are untouched:** `acceptance_rate` / `average_tokens_per_forward` /
`distribution` are byte-identical to before — verified against the pristine
backup (`tests/test_per_request_spec_stats.py`,
`test_globals_unchanged_by_per_request_accounting`).

### Files touched and how to undo

Pristine copies of every file below: `tools/revert_backup/<path with `/` ->
`_`>.orig`.

| file | what was added |
|---|---|
| `atom/model_engine/engine_stats.py` | `import os`; 4 `__slots__` entries + `per_request_spec_enabled`; init block; `req_key=` param on `update_spec` + the accounting block; `finish_spec_request()`; `per_request_spec_statistics()`; clear in `reset_spec` |
| `atom/model_engine/scheduler.py` | `req_key=` at the single `update_spec` call site (~:3214); `finish_spec_request()` in `_mark_finished` |
| `atom/model_engine/engine_utility.py` | `result["per_request"]` in `_handle_get_mtp_statistics` only (NOT in `collect_metrics`, so `/metrics` is unchanged) |
| `atom/model_engine/llm_engine.py` | merge `per_request` across DP ranks in `get_mtp_statistics` (that aggregator whitelists keys, so the payload would otherwise be silently dropped) |
| `tools/dflash_ab_client.py` | `--concurrency`/`CONC` (default 32), `run_gsm8k_card_concurrent`, `per_request_table`, `_strip_per_request`, `_ask_gsm8k_card` split out of `run_gsm8k_card` |
| `tools/dflash_ab.sh` | forwards `ATOM_PER_REQUEST_SPEC_STATS=1`; passes `--concurrency "${CONC:-32}"` |
| `tests/test_per_request_spec_stats.py` | NEW FILE, entirely bench-only |

To undo:

    cd <repo>
    for f in atom/model_engine/engine_stats.py atom/model_engine/scheduler.py \
             atom/model_engine/engine_utility.py atom/model_engine/llm_engine.py \
             tools/dflash_ab_client.py tools/dflash_ab.sh; do
      cp "tools/revert_backup/$(echo $f | tr '/' '_').orig" "$f"
    done
    rm -f tests/test_per_request_spec_stats.py
    rm -rf tools/revert_backup
    grep -rn "BENCH-ONLY" atom tools tests      # must print nothing
    black --check . && ruff check .

NOTE the backups are of the files as they stood on this branch (they contain
the branch's other, non-bench changes). Restoring them undoes ONLY the
bench-only edit, not the DFlash work.

### Using it

    # card metric at the card's concurrency (needs the flag; the harness sets it)
    LIMIT=1319 FEWSHOT=5 CONC=32 K=7 ATOM_K3_AUX_AGGREGATE=1 \
      RUNDIR=/tmp/dflash_ab/card_c32/radixark ./tools/dflash_ab.sh radixark gsm8k_card

    # the old, server-patch-free estimator, for comparability with what is banked
    CONC=1 ... ./tools/dflash_ab.sh radixark gsm8k_card

Both report `acc_len_card_style` AND `acc_len_global`. At CONC>1 the client
ALSO prints `acc_len_global_server_delta`, recomputed from the server's own
global counters over the run window: if that disagrees with `acc_len_global`,
questions went unjoined and the card-style mean is on a subset. Check
`joined N/N` in the client output, and watch for the
`server exposes no per-request acceptance table` warning — an unflagged server
produces `acc_len_card_style: null` rather than a silently wrong number.

## BENCH-ONLY CHANGES — REVERT BEFORE UPSTREAM

**Added 2026-10-06.** Per-request speculative-acceptance attribution, so the
model card's estimator ("averaged within each question and then equally across
questions") can be measured at the card's **concurrency 32** instead of at the
concurrency-1 the old differencing workaround forced. This is a MEASUREMENT
affordance. It must not be upstreamed.

**`grep -rn "BENCH-ONLY" atom/ tools/` finds the entire change.** Every added
block is wrapped in a `# BENCH-ONLY (per-request acceptance): REVERT BEFORE
UPSTREAM` … `# end BENCH-ONLY` pair.

**Pristine pre-edit copies live in `tools/revert_backup/`**, named after the
original path with `/` replaced by `_` and `.orig` appended.

| file | what was added | backup |
|---|---|---|
| `atom/model_engine/engine_stats.py` | `import os`; 5 `__slots__` entries; the `per_request_spec_enabled` flag + `_per_req_accepted` / `_per_req_steps` / `_per_req_done` state in `__init__`; the optional `req_key=` parameter and the accumulate block in `update_spec`; new `finish_spec_request()` and `per_request_spec_statistics()`; a clear in `reset_spec` | `tools/revert_backup/atom_model_engine_engine_stats.py.orig` |
| `atom/model_engine/scheduler.py` | `req_key=` at the single `update_spec` call site (~:3214); `engine_stats.finish_spec_request(...)` in `_mark_finished` | `tools/revert_backup/atom_model_engine_scheduler.py.orig` |
| `atom/model_engine/engine_utility.py` | `result["per_request"] = ...` in `_handle_get_mtp_statistics` only — deliberately NOT in `spec_statistics()`, so `/metrics` keeps its exact present shape | `tools/revert_backup/atom_model_engine_engine_utility.py.orig` |
| `atom/model_engine/llm_engine.py` | merge of the per-rank `per_request` payloads in `get_mtp_statistics` (that aggregator whitelists keys, so without this the payload is silently dropped on the DP path) | `tools/revert_backup/atom_model_engine_llm_engine.py.orig` |
| `tools/dflash_ab_client.py` | `--concurrency` / `CONC` (default 32); `run_gsm8k_card_concurrent`, `_ask_gsm8k_card`, `per_request_table`, `_strip_per_request`; both estimators still reported | `tools/revert_backup/tools_dflash_ab_client.py.orig` |
| `tools/dflash_ab.sh` | forwards `ATOM_PER_REQUEST_SPEC_STATS=1` (auto-on for `gsm8k_card` only); passes `--concurrency "${CONC:-32}"` | `tools/revert_backup/tools_dflash_ab.sh.orig` |

### Exact undo

    cd <repo>
    for f in atom/model_engine/engine_stats.py atom/model_engine/scheduler.py \
             atom/model_engine/engine_utility.py atom/model_engine/llm_engine.py \
             tools/dflash_ab_client.py tools/dflash_ab.sh; do
      cp "tools/revert_backup/$(echo "$f" | tr '/' '_').orig" "$f"
    done
    rm -rf tools/revert_backup
    grep -rn "BENCH-ONLY" atom/ tools/        # must print nothing
    grep -rn "ATOM_PER_REQUEST_SPEC_STATS" .  # must print nothing

(The `.orig` files are the branch state immediately before this change — i.e.
they still contain every other DFlash edit on this branch. Restoring them undoes
*only* the per-request work.)

### Hazards, and what was done about each

- **Memory leak.** `Scheduler._mark_finished` is the single retirement point for
  any sequence that ever ran a decode step (the other two `status = FINISHED`
  sites — `_reject_aborted_waiting` and the unschedulable drop — retire requests
  that never decoded, so they are never in the dict). It now calls
  `finish_spec_request`, which POPS the live counters and parks the record in a
  `deque(maxlen=20000)`. Two independent backstops: the live dict is capped at
  `ATOM_PER_REQUEST_SPEC_STATS_LIVE_CAP` (8192, oldest-insertion evicted in
  O(1)) in case a future path ever retires without that hook, and the finished
  ring is bounded by construction and reports `finished_truncated` so a client
  can tell. Verified: 5000 never-retired requests leave the live dict at exactly
  the cap with the globals still exact.
- **Hot path.** `update_spec` runs per sequence per decode step. Enabled it adds
  two dict lookups and two int adds — no allocation, no logging, nothing O(live
  requests). Disabled it is one bool load *at the scheduler call site* (the flag
  is checked there as well as inside `update_spec`), so the key expression is
  not even evaluated.
- **Existing numbers.** Strictly additive: nothing in the global path reads the
  per-request state. Proven against the pristine backup — see below.
- **Preemption / restart.** Counters deliberately CONTINUE across a preempt+
  resume. The seq keeps its id, so the sample covers the whole question, which
  is what "averaged within each question" means and what the global estimator
  already does. Resetting would split one question into two samples and bias the
  per-question mean toward the shorter fragment.
- **Gating.** **DEFAULT OFF**, `ATOM_PER_REQUEST_SPEC_STATS=1`. It is not free
  when on, so it is gated; `tools/dflash_ab.sh` turns it on automatically for
  `gsm8k_card` and for nothing else, so no other banked workload shifts.

### Evidence (no GPU; the stats object driven directly)

- 40 synthetic requests, 538 interleaved decode steps (shuffled = concurrency
  >1): per-request sums equal the globals exactly — accepted 1371 == 1371,
  steps 538 == 538, 40/40 requests present. Global acc_len 3.5483 vs card-style
  3.6631, i.e. the two estimators differ as they should.
- Globals byte-identical across (pre-edit code path) / (flag off, keys passed) /
  (flag on): `{"acceptance_rate": 0.5096654275092937, "distribution": {0:81,
  1:83, 2:100, 3:90, 4:102, 5:82}, "total_accepted_tokens": 1371,
  "total_draft_tokens": 2690}`.
- 5000 steps replayed through BOTH the pristine
  `tools/revert_backup/atom_model_engine_engine_stats.py.orig` and the edited
  module: `spec_statistics()`, `total_accepted` and `total_steps` identical.
- Targeted suite in the pinned image with `PYTHONPATH=/work`:
  **1133 passed, 82 skipped**. `black --check` and `ruff check` clean on all six
  touched files.

### How to run it

    CONC=32 FEWSHOT=5 LIMIT=1319 RUNDIR=/tmp/dflash_ab/card_c32/radixark \
      ATOM_K3_AUX_AGGREGATE=1 K=7 ./tools/dflash_ab.sh radixark gsm8k_card

`CONC=1` still works and still uses the OLD global-differencing estimator, so
the banked conc-1 numbers remain reproducible against an unpatched server.
The summary now carries `per_request_source`, `n_joined` and
`acc_len_global_server_delta` — if `n_joined` < `n_questions`, or if
`acc_len_global_server_delta` disagrees with `acc_len_global`, the attribution
is incomplete and the card-style number must not be quoted.

## NEEDS HUMAN — loop STOPPED 2026-10-06 (cron job 33714f3d deleted)

**Everything unblocked is done.** The branch works and delivers uplift. Every
remaining item needs a decision only Rita can make, so the loop stopped rather
than guess. Nothing is half-finished; no run is in flight; GPUs are free.

### What the branch now delivers (all measured, all banked)
| workload | radixark + fix | inferact | result |
|---|---|---|---|
| `lossless` | 6/6 identical | — | PASS |
| `gsm8k` accuracy | 0.9500 | 0.9507 | both within noise of nospec 0.9393 |
| `gsm8k` acceptance | 62.04% / 4.10 | 62.54% / 4.13 | parity |
| `swebench` | **2.9751 / 79.58 tok/s** | 2.7961 / 77.21 | **radixark +6.4%** |
| `perf` (single-stream) | **3.0464 / 111.41 tok/s** | 2.9604 / 109.34 | **radixark +1.9%** |

### The 4 decisions waiting on you, cheapest first

1. **Provenance routing (blocks landing the fix).** The aggregate must not be
   unconditional — it costs inferact 6%. Provenance is already detectable with
   no new flag (radixark has `dflash_config`; inferact has `_torchspec_version`).
   Is a config sniff acceptable, or do you want an explicit CLI flag? What is
   the default for an unknown checkpoint? This changes the target<->draft
   feature contract. **Everything above was measured provenance-correct, so the
   numbers are what shipping looks like whichever mechanism you pick.**

2. **DSV4 / V4.1-Flash startup regression (blocks the PR).** `model_runner.py:1856`
   `page_unit_bytes=paged_pool_bytes(1)` returns 0 for those backends, which
   `PagedStateCheckpointSpec` rejects. Fix shape is known (use the target
   builder's own PAGE declaration); it needs a DSV4 checkpoint to validate
   against, which this box does not have.

3. **LMCache offload (M3's last item) — NOT started, deliberately.** There are
   reference recipes for GLM-5.2 and MiniMax-M3 but **none for Kimi-K3**, and
   nothing anywhere for a SIBLING DRAFT POOL under offload. The draft declares
   `draft.{role}` regions at `tp_replication_factor=1` and disables whole-object
   collapse (`draft_kv.py:258`) — a path nobody has exercised. Picking a backend,
   layout and sizing is a design choice, not a mechanical port. I did not want to
   invent a plausible-looking config unattended.

4. **M4 / DCP8.** Explicitly gated, and DCP + a non-MLA draft is broken in vLLM
   and SGLang too (vllm#59184, #57992; sglang#34920). Worth a deliberate go/no-go.

### Out of scope, escalate elsewhere
**Prefix caching returns a 0.0% hit rate on Kimi-K3 even with NO draft loaded**
(nospec control run). Not a DFlash issue; the branch is cleared. It is either an
ATOM-wide prefix-caching problem on this model or an unwired metric. One
discriminator remains: with a server live, send the same prompt twice and watch
the rate.

### Two caveats to carry if these numbers go anywhere external
- `swebench` is a 20-problem subset; `perf`/`accept` use only 5 distinct prompts.
  The DIRECTION is consistent across four independent measurements, which is why
  I believe it — but the individual magnitudes are not tight. **GSM8K (n=1319) is
  the only number I would quote unqualified.**
- Exact-text losslessness is unachievable on this stack (the target does not
  reproduce itself run-to-run). The `lossless` PASS is on a self-validating
  low-entropy prompt set, not the design's original 200-prompt gate. See
  Appendix F.

## ROOT CAUSE: **CONFIRMED CAUSAL** by two-armed A/B (2026-10-06)

Pre-registered rule was "radixark RISES **and** inferact FALLS". **Both held.**
`accept`, 20 req x 512 tok, K=5, TP8, one arm at a time; only the env differs.
Both runs verified armed (`feed_aggregate=True`, nvb=[1,2,5,6,7]) — not no-ops.

| arm | OFF (raw prefix) | ON (`ATOM_K3_AUX_AGGREGATE=1`) | d accept_len | d rate |
|---|---|---|---|---|
| **radixark** (SpecForge/SGLang-trained) | 2.7046 / 0.3409 | **2.8996 / 0.3799** | **+0.195 (+7.2%)** | **+3.90 pp** |
| **inferact** (TorchSpec/vLLM-trained) | 3.0917 / 0.4183 | **2.9610 / 0.3922** | **-0.131 (-4.2%)** | **-2.61 pp** |

The OPPOSITE-direction response is what makes this causal and not merely
"the aggregate is a nicer feature".

### REPORT IT HONESTLY: "84% of the gap closed" is MISLEADING
gap 0.3871 -> 0.0614 is 84% closed, but decomposed:
- radixark **RISING** = 0.1950 = **50%** of the gap  <- the actual win
- inferact **FALLING** = 0.1307 = **34%** of the gap  <- the CONTROL DEGRADING
**The aggregate recovers about HALF of radixark's deficit, not 84%.** Quote the
+7.2% / +3.90 pp on radixark. A residual ~2.1% arm gap remains unexplained.

### DECISION NEEDED (Rita's call) — do NOT make the aggregate unconditional
It **costs inferact 6%**. The feature must be routed by DRAFT PROVENANCE.
Good news: provenance is detectable from the draft config with no new flag —
- radixark: `dflash_config` present, SpecForge/SGLang -> wants the **aggregate**
- inferact: `_torchspec_version: "0.1.0"` present, TorchSpec/vLLM -> wants the
  **raw prefix** (today's behaviour)
Open design questions: is a provenance sniff acceptable, or should this be an
explicit CLI/config flag on the draft? What is the default for an unknown
checkpoint? This changes the target<->draft feature contract, so it wants a
deliberate answer rather than a sniff bolted in.

### GSM8K CONFIRMATION: **DONE, CONFIRMED** (2026-10-06). Acceptance CLOSED.
`ATOM_K3_AUX_AGGREGATE=1`, radixark, K=5, 1319 samples, concurrency 32.
Run verified armed (`feed_aggregate=True`, nvb=[1,2,5,6,7]).

| GSM8K K=5 | baseline (raw prefix) | **+ aggregate** | delta |
|---|---|---|---|
| acceptance | 55.61% | **62.04%** | **+6.43 pp (+11.6% rel)** |
| toks/fwd | 3.78 | **4.10** | +0.32 |
| accuracy | 0.9484 +/- 0.0061 | **0.9500 +/- 0.0060** | +1.20 sigma vs nospec = **within noise** |
| gap vs inferact (62.54% / 4.13) | **-6.93 pp** | **-0.50 pp** | **parity** |

**Both predictions, registered BEFORE the run, held:** ~62% acceptance (actual
62.04%), and accuracy unmoved because the aggregate changes what the DRAFT sees,
never what the target verifies (actual 1.20 sigma).

Note the `accept`-workload run understated this: it showed +7.2% accept_length
on 5 prompts at bs=1, where the strong metric shows +11.6% relative acceptance
and full parity with inferact. **Quote the GSM8K numbers, not the `accept` ones.**

### OLD IN-FLIGHT NOTE (superseded by the result above)
`ATOM_K3_AUX_AGGREGATE=1 FEWSHOT=5 RUNDIR=/tmp/dflash_ab/gsm8k_agg/radixark
./tools/dflash_ab.sh radixark gsm8k`, **K=5** so it is directly comparable to
the banked baseline. Turns the weak `accept` signal into the strong metric
(1319 samples, concurrency 32, real traffic).
Baseline to beat: **acc 0.9484, acceptance 55.61%, 3.78 toks/fwd**.
**PREDICTION, stated before the result:** if the `accept` gain (+11.4% relative
on rate) transfers, acceptance lands ~62%, i.e. level with inferact's 62.54%
and the card-implied 63.1%. Accuracy must stay within noise of 0.9393 +/- 0.0066
(nospec) — the aggregate changes what the DRAFT sees, never what the target
verifies, so **any accuracy move outside noise would indicate the change is not
as contained as believed** and must be investigated, not waved through.
Check the arming line in `server.log` before trusting the result.

## M3 step 1 (prefix caching): ORACLE HELD, but **HAZARD NOT EXERCISED**

| | baseline (PC off) | **PC on** | |
|---|---|---|---|
| acceptance | 62.04% / 4.10 | **61.92% / 4.10** | -0.12 pp = noise |
| accuracy | 0.9500 | **0.9477 +/- 0.0061** | noise |
| **prefix cache hit rate** | n/a | **0.0%** | **cache NEVER hit** |

Flags verified armed: `'enable_prefix_caching': True` in the engine kwargs,
`feed_aggregate=True`, no disabling warning anywhere in the log.

**Read this correctly: it is a PASS on the oracle and an INCONCLUSIVE on the
test.** The hazard (target serves a cached prefix block without recompute ->
`compute_draft_kv` never writes the draft's context rows -> block pass attends
stale rows) CANNOT occur at a 0% hit rate. Nothing about draft-pool/prefix-cache
interaction has actually been validated. Do NOT record M3's prefix-caching item
as done.

### RESOLVED: the 0% is NOT a DFlash issue. The branch is cleared.
Discriminating run done — **`nospec` (no draft loaded at all) is ALSO 0.0%**,
with `'enable_prefix_caching': True` verified in its engine kwargs.

| arm | prefix caching | hit rate |
|---|---|---|
| radixark (spec) | ON, verified | 0.0% |
| **nospec (no draft)** | ON, verified | **0.0%** |

So spec decode does NOT suppress the cache and the draft's sibling pool does NOT
defeat block reuse. **This branch is not responsible and should not chase it.**

**ESCALATE, do not spend loop wakes on it:** prefix caching appears to deliver a
0% hit rate on Kimi-K3 even with no draft, on a workload (GSM8K 5-shot, 1319
prompts sharing a ~500-token fixed few-shot prefix) that should hit nearly every
time. It is either an ATOM-wide prefix-caching problem on this model or an
unwired hit-rate metric. Either way it is out of scope for DFlash.
Remaining discriminator if anyone picks it up: with a server live, send the SAME
prompt twice and watch the rate. Moves => metric was fine, the workload was not
sharing blocks. Stays 0% => caching itself is not working.

**Consequence for M3:** its prefix-caching item **cannot be validated by this
branch**. The draft-pool/cache hazard remains untested (it needs a nonzero hit
rate to exist at all), but it is also unreachable until prefix caching works.
Mark M3's prefix-caching item ESCALATED, not done, and move on to LMCache.

### (original, now answered) open question: why is the hit rate 0%?
1319 GSM8K 5-shot prompts share the same long fixed few-shot prefix (~500+
tokens = ~4 blocks at block_size 128), so a near-100% hit rate was expected.
Candidates, none yet checked:
- prefix caching is effectively inert when spec decode is on (would be a
  PRODUCTION-RELEVANT finding well beyond this branch);
- the draft's sibling pool sharing the target's PAGE entry class defeats block
  reuse;
- the hit-rate metric is simply not wired on this path (measurement gap, no bug).

**Discriminating next test (cheap, one run):** same command with the **nospec**
arm. If nospec hits, spec decode is what suppresses the cache. If nospec is also
0%, it is the workload or the metric, not the draft. Run that before drawing any
conclusion.

## (superseded) in-flight note for M3 step 1
`PREFIX_CACHING=1 ATOM_K3_AUX_AGGREGATE=1 RUNDIR=/tmp/dflash_ab/m3_pc/radixark
./tools/dflash_ab.sh radixark gsm8k`, K=5.

**M3 is SMALLER than it looks.** `--cudagraph-mode FULL` and `--max-num-seqs 64`
are **already in effect** and have been for every result banked so far
(`cudagraph_mode=<CUDAGraphMode.FULL` in the server logs). The only real M3
deltas are **prefix caching** and **LMCache offload**. Doing them one at a time.

**The hazard being tested, stated before the result.** The draft's sibling KV
pool rides the TARGET's block ids (`DraftKvBuilder.sub_pool_specs` joins the
target's PAGE entry class). If the target serves a cached prefix block WITHOUT
recomputing it, `compute_draft_kv` never writes the draft's context rows for
those positions and the block pass attends whatever those pool rows last held.
GSM8K 5-shot is the ideal probe: all 1319 prompts share the same long few-shot
prefix, so the cache hit rate is high by construction.
**ORACLE: acceptance 62.04% / accuracy 0.9500 must HOLD.** A drop in acceptance
with accuracy intact is the signature of exactly this bug — and is precisely the
"only acceptance moves" failure mode this project already chased once.

Harness: `PREFIX_CACHING=1` -> `--enable_prefix_caching` (verified present at
`atom/model_engine/arg_utils.py:277`). Default stays `--no-enable_prefix_caching`
so nothing already banked shifts under a re-run.

## VALIDATION CHAIN: **COMPLETE** — all 5 workloads, both arms

| workload | radixark (+fix) | inferact | verdict |
|---|---|---|---|
| `lossless` | 6/6 identical, 0 divergent | — | **PASS** |
| `gsm8k` acc | 0.9500 | 0.9507 | both within noise of nospec 0.9393 |
| `gsm8k` acceptance | **62.04% / 4.10** | 62.54% / 4.13 | parity (-0.50 pp) |
| `accept` | 2.8996 / 37.99% | 2.9610 / 39.22% | parity (weak metric) |
| **`swebench`** | **2.9751 / 39.50% / 79.58 tok/s** | 2.7961 / 35.92% / 77.21 | **radixark WINS +6.4%** |
| **`perf`** (single-stream) | **3.0464 / 111.41 tok/s** | 2.9604 / 109.34 | **radixark WINS +1.9%** |

**radixark leads on both realistic workloads** (swebench +6.4%, perf +1.9%) and
is at parity on math/chat. The branch delivers uplift.

### `perf` RESULT — label it correctly
`perf` is `prompts_generic(64)` @ 512 tok run **SEQUENTIALLY**, so it is
**single-stream latency at bs=1, NOT server throughput.** Do not quote its
`tok/s` beside a concurrency benchmark. For spec decode this is the most
relevant regime (the win is largest at low batch), but it is not throughput.
Also `prompts_generic(64)` is only **5 distinct prompts** (`base[i%5]`).
The two arms emitted DIFFERENT token counts (32601 vs 32149 — generations hit
EOS at different points), so **compare the RATE, not the wall time**.
A real throughput number needs a concurrent client the harness does not have;
`scripts/run_benchmark.sh` (ISL/OSL/concurrency) is the in-repo tool, drivable
via `docker exec` the way `run_gsm8k` is.

### (superseded) in-flight note for `perf`

**LABEL THIS CORRECTLY when recording it.** `perf` is `prompts_generic(64)` at
512 tok run **SEQUENTIALLY** (`run_completions` is a plain for-loop), so it is
**single-stream latency at bs=1, NOT server throughput.** Do not quote its
`tok/s` as a throughput figure. For speculative decoding this is a legitimate
and arguably the most important metric — spec decode's win is largest at low
batch — but it is the wrong number to put next to a concurrency benchmark.
Also note `prompts_generic(64)` is still only **5 distinct prompts** (`base[i%5]`),
so it is 64 sequential samples of 5 texts, not 64 different workloads.

A real throughput number needs a concurrent client; the harness has none. If one
is ever wanted, `scripts/run_benchmark.sh` (ISL/OSL/concurrency) is the in-repo
tool for it, driven via `docker exec` the way `run_gsm8k` is.

### swebench: **DONE — radixark now BEATS inferact. The branch premise holds.**
20 real SWE-bench_Verified problems (6000-char statements), 512 tok, K=5,
provenance-correct (radixark aggregate ON — verified armed; inferact OFF).
No fallback warning, so these are REAL swebench prompts.

| | inferact | **radixark + fix** | delta |
|---|---|---|---|
| accept_length | 2.7961 | **2.9751** | **+0.179 (+6.4%)** |
| acceptance | 35.92% | **39.50%** | **+3.58 pp** |
| tok/s | 77.21 | **79.58** | **+2.37 (+3.1%)** |

Quality: 0 empty, 0 under-20-word generations on both arms. Comparator PASS.

**This is the uplift the branch exists to deliver**, and it shows up on the
workload this project actually targets (agentic/long-context), not just on math.
Note radixark LEADS here (+6.4%) while it only reached parity on GSM8K
(-0.50 pp) — the GQA draft's advantage is larger on agentic text.

Both arms ~2.8-3.0 here vs ~4.1 on GSM8K. That is EXPECTED, not a regression:
the card's own table shows the same shape (SWE-Rebench 4.66 @ block 7, but
mid-length output buckets 2.56-3.13). Speculation is hardest on open-ended
agentic text.

### (superseded) in-flight note for the swebench run
`/tmp/dflash_ab/swe/{radixark,inferact}/`. Run in the **provenance-correct**
configuration, i.e. what shipping looks like whatever routing mechanism is
chosen: **radixark with `ATOM_K3_AUX_AGGREGATE=1`, inferact WITHOUT it.**
That reflects the established fact of WHICH feature each arm wants; it does not
preempt the open question of HOW to route.
This is the first radixark number on long, realistic agentic prompts (20 real
SWE-bench_Verified problem statements, truncated to 6000 chars, 512 tok). The
acceptance numbers so far are all on short math/chat prompts.

**Pre-flight done, and it matters:** `prompts_swebench` has a SILENT FALLBACK to
generic prompts on any load failure, which would report generic results under a
SWE-bench label. Verified on the host before burning GPU time: real prompts load
(`Repo: astropy/astropy ... Issue: Modeling's separability_matrix ...`).
Re-check this if the dataset or host env ever changes.
Caveat: the loader emitted "sending unauthenticated requests to the HF Hub", so
it is NOT purely offline — a network blip could flip it to the fallback. Watch
for the `WARN: SWE-bench load failed` line in the console.

### Still open after this
- Residual ~2.1% radixark-vs-inferact gap with the aggregate on.
- **K=7 never run** (rung a) — still the only run comparable to the card's 5.4176.
- The `accept` workload is a WEAK absolute metric (5 distinct prompts x4, bs=1,
  server-lifetime cumulative). Read the SIGN as the result and the magnitude as
  indicative. Confirm on GSM8K before quoting any number externally.

## Original hypothesis (kept for the citation trail; now CONFIRMED above)

**ATOM feeds the radixark draft a DIFFERENT FEATURE than it was trained on.**

Kimi-K3's residual stream is not a plain pre-norm stack: `attn_res_block_size=12`
over 93 layers, so the stream is a **bank of per-12-layer snapshots + a running
prefix**, recombined by a softmax mixture.

| side | what the aux tap emits | cite |
|---|---|---|
| **ATOM** | the RAW block-local `prefix_sum`; the bank is destructured and **thrown away** (`prefix_sum, *pendings, _block_residual = output`) | `atom/models/kimi_k3.py:1653-1659` |
| **SGLang** | `aggregate_stream(hidden_states, attn_res.block_residual, nvb, next.self_attention_res_proj, next.self_attention_res_norm)` | `sglang/srt/models/kimi_k3.py:3145-3156` |

SGLang's docstring is the whole finding — it built a dedicated function FOR THIS:
> `sglang/srt/layers/attn_residual.py:346-348` — "Pre-norm aggregated stream
> value (softmax mixture, no output norm): the K3 analogue of the residual
> stream, **for dspark aux capture** -- the raw wire only carries the current
> block's running prefix."

So for taps [7,23,51,67,83] ATOM's draft sees only the contributions of layers
[0..7], [12..23], [48..51], [60..67], [72..83].

**Why it is radixark-specific — the model cards say so outright:**
- radixark: "trained with **SpecForge** using hidden states from a live
  **SGLang** target engine" -> trained on `aggregate_stream`. ATOM feeds
  `prefix_sum`. **MISMATCH.** 55.61% vs card-implied 63.1%.
- inferact: "target hidden states extracted from **vLLM** itself",
  `_torchspec_version: 0.1.0` -> HF/vLLM convention = the plain residual stream,
  which is exactly what ATOM emits. **MATCH.** 62.54% vs card-implied 63.4%.

**Why the whole numerics bisect missed it:** every forward check fed ATOM the
REFERENCE's aux tensor. `dflash.py` contains only the DRAFT — it cannot define
what the TARGET should emit. "aux capture clean" verified ATOM against ATOM's
own intended (HF) semantics, not against SpecForge's training distribution.
This is a feature-distribution mismatch, not a math bug, which is why 1355
weight checks, a bit-exact pool round trip and a cos-0.9999 forward bisect all
passed.

### TWO EARLIER HYPOTHESES IN THIS FILE ARE WRONG — do not act on them
1. **Confidence head: DEAD.** SGLang's default `SGLANG_RAGGED_VERIFY_MODE=static`
   returns `None` from `build_confidence_head` (`sglang/srt/models/dspark.py:455-457`,
   default at `srt/environ.py:1360`) and skips the weights at load. The config's
   `enable_confidence_head` is never read. The card was measured with it INACTIVE.
   ATOM skipping it is correct and matches.
2. **The delta-of-deltas ratio argument ASSUMED both drafts degrade identically
   from k=7 to k=5.** That is an assumption, not a fact. On per-token rates
   inferact already reproduces its card inside ATOM, so the isolated deficit is
   radixark's ~7.5 pp; the gap to 5.4176 is mostly just k=5 vs block 7, plus the
   unit mismatch (SGLang `--speculative-num-draft-tokens N` means `gamma=N-1`;
   ATOM `--num-speculative-tokens N` means `T=N`).
   The metric itself is IDENTICAL on both sides — both are the mean over
   request-steps of `1 + accepted_drafts` (`engine_stats.py:205-228` vs
   SGLang `metrics_reporter.py:464-476`). No bug hides in the metric.

### rung 0 step 1 RESULT (2026-10-06): **HYPOTHESIS SURVIVES — MEASURED**
`ATOM_K3_AUX_PROBE=1 ./tools/dflash_ab.sh radixark smoke`, TP8, real activations.
step=2 is the real 15-token prefill, step=3 a real decode-verify step.

| tap | nvb | cos (prefill/decode) | rel L2 | prefix mass | **bank mass** |
|---|---|---|---|---|---|
| 7  | 1 | 0.99999 / 0.99999 | 0.060 / 0.054 | 0.940 | 0.060 |
| **23** | 2 | **0.616 / 0.690** | 0.803 / 0.722 | 0.310 | **0.690 / 0.665** |
| 51 | 5 | 0.850 / 0.900 | 0.642 / 0.503 | 0.943 | 0.058 |
| 67 | 6 | 0.912 / 0.944 | 0.459 / 0.315 | 0.860 | 0.141 |
| **83** | 7 | **0.759 / 0.819** | 0.664 / 0.572 | 0.411 | **0.589 / 0.515** |

(min-over-rows cos dips to 0.268 at tap 67, 0.371 at tap 83.)
Observed `nvb` == predicted `[1,2,5,6,7]`. Decision rule was fixed in advance
(cos < ~0.99 + real bank mass => survives); 4 of 5 taps clear it easily.
At taps 23 and 83 the MAJORITY of the softmax mass sits on bank rows that ATOM
throws away.

**Cross-check against self-confirmation:** the probe reused ATOM's OWN
`apply_attn_res` kernel (`score_weight` is `res_norm.weight * res_proj.weight`
pre-folded at load), with `out_norm_weight=None` = SGLang's "mixture, no output
norm". Only the softmax `probs` were recomputed in plain torch (the fused kernel
keeps them in registers). At tap 7 the independent bank mass (0.0539) and the
kernel's rel L2 (0.0541) agree to 3 digits -> neither path is a transcription error.

### CORRECTIONS this measurement forced
- **"Divergence grows with depth" was WRONG.** It is largest at tap **23**
  (nvb=2) and smallest at tap **51** (nvb=5, only 5.8% bank mass despite 5 bank
  entries). The driver is the LEARNED per-layer scoring weight, not nvb.
- **Tap 7 is effectively a no-op** (cos 0.99999): its single bank entry is the
  layer-0 prefix, nearly collinear with the layer-7 prefix. So the mismatch is
  real at **4 of 5** taps, not 5.

### Probe implementation (measurement-only, left in tree)
`dspark_proposer.py`: `_arm_k3_aux_probe` + `_k3_probe_report`, gated on
`ATOM_K3_AUX_PROBE=1`. Wrapped `extract` returns the UNTOUCHED prefix; buffers
written in place (cudagraph-safe) and read in `propose()` outside the compiled
region; tap dispatch via an int attribute set at arm time, not `id()`, to stay
Dynamo-traceable. `kimi_k3.py` NOT touched. Env-unset path is one `is not None`
check against a None class attribute — no tensor op, no wrapping.
`tools/dflash_ab.sh` forwards the env, and its trace branch now uses
`EXTRA_ENVS+=` instead of `=` (it used to clobber earlier entries).
871 unit tests pass, black+ruff clean.

### nvb per tap — ALL FIVE taps differ (divergence does NOT track depth; see above)
`prev_valid_blocks = _cdiv(layer_idx, attn_res_block_size)` — **CEILING**
division (`sglang/srt/models/kimi_k3.py:2568`), and `nvb = next_layer.prev_valid_blocks`
(`:3147`). With `attn_res_block_size=12`:

| tap | next | nvb = cdiv(next,12) | aggregate_stream does |
|---|---|---|---|
| 7 | 8 | **1** | mixture over 1 bank entry + prefix |
| 23 | 24 | **2** | mixture over 2 + prefix |
| 51 | 52 | **5** | mixture over 5 + prefix |
| 67 | 68 | **6** | mixture over 6 + prefix |
| 83 | 84 | **7** | mixture over 7 + prefix |

**No tap has `nvb == 0`**, so the `if nvb == 0: return prefix_sum` short-circuit
(`attn_residual.py:350`) never fires — ATOM's raw prefix differs from SGLang's
mixture at EVERY tap, by more at deeper ones. (Floor division would have given
nvb=0 at tap 7 and made that tap identical; it is cdiv, so it does not. Do not
re-derive this with `//`.)

### How to test, cheapest first
1. **Offline, no GPU-hours.** On one real prompt dump both `prefix_sum` and
   `aggregate_stream(prefix_sum, block_residual, nvb=cdiv(i+1,12),
   layers[i+1].self_attention_res_proj, layers[i+1].self_attention_res_norm)`
   for i in {7,23,51,67,83}. cos < ~0.99 at the deep taps confirms a real
   distribution shift. ATOM already has every weight
   (`kimi_k3.py:1541-1563`) and the bank is already in the layer's return
   tuple — it is only discarded at `:1653`.
2. **Env-gated A/B.** `ATOM_K3_AUX_AGGREGATE=1` makes
   `DSparkProposer._aux_capture_spec` (`dspark_proposer.py:499-516`) install an
   `extract` that aggregates using layer `i+1`'s proj/norm.
   **Do NOT edit `kimi_k3.py` — it is `@support_torch_compile`** (CLAUDE.md
   critical rule). The proposer hook already receives `(output, block)` and can
   reach `runner.model`'s layer list, and the bank is already in `output`.
   Then `./tools/dflash_ab.sh radixark accept`.
   **Falsifier:** expect radixark -> ~62-63% and inferact -> WORSE. If BOTH move
   the same way, the hypothesis is wrong.
3. All `target_layer_ids` are <= 83 < 92, so the last-layer
   `output_attn_res_proj` branch is never needed — only the `next_layer` branch.

### rung 0 step 2 RESULT (2026-10-06): **THE FALSIFIER — HYPOTHESIS CONFIRMED CAUSAL**
`ATOM_K3_AUX_AGGREGATE=1` makes the dspark aux tap return the K3 aggregate
(SGLang's `aggregate_stream` feature) instead of the raw block prefix. Same
harness, same workload (`accept`, 20 requests / 512 tok), same K=5, TP8, one arm
at a time; ONLY the env differs.

| arm | env | accept_length | acceptance_rate |
|---|---|---|---|
| radixark (SGLang-trained) | OFF (prefix) | 2.7046 | 0.3409 |
| radixark | **AGGREGATE** | **2.8996 (+7.2%)** | **0.3799 (+11.4%)** |
| inferact (vLLM/HF-trained) | OFF (prefix) | 3.0917 | 0.4183 |
| inferact | **AGGREGATE** | **2.9610 (-4.2%)** | **0.3922 (-6.2%)** |

**The two arms moved in OPPOSITE directions**, which was the pre-registered
decision rule. radixark improves on the SGLang feature it was trained on;
inferact degrades on it, exactly as a vLLM/HF-trained draft should. The
cross-arm gap collapses from -12.5% to -2.1% (`dflash_ab_compare.py`). This is
a feature-distribution mismatch, confirmed causally, not a correlation.

Caveat kept honest: radixark reached 0.3799, not the ~0.41 guessed in advance,
so the aggregate recovers roughly HALF the arm gap, not all of it. `accept` is
5 distinct prompts x4 at bs=1 with a server-lifetime cumulative metric, so treat
the magnitude as indicative and the SIGN as the result. The remaining radixark
deficit is a separate question (block-7 vs K=5, GQA draft quality).

## LADDER STATUS
- **rung b, profiling: DONE** (below). - **rung c, SGLang compare: DONE**
  (it produced the ROOT-CAUSE HYPOTHESIS above).
- **rung 0 step 1: DONE — HYPOTHESIS SURVIVES** (table above). The probe is in
  tree and reusable: `ATOM_K3_AUX_PROBE=1`.
- **rung 0 step 2: DONE — CONFIRMED CAUSAL** (result table above).
  `ATOM_K3_AUX_AGGREGATE=1` (behavioural: the extract RETURNS the aggregate),
  two runs, `accept` on both arms, banked in
  `/tmp/dflash_ab/agg/{radixark,inferact}/`. Both arms moved in OPPOSITE
  directions. Do NOT re-run; do NOT make the aggregate unconditional — it costs
  inferact 6%. **NEXT: route the feature per draft provenance, then rung a (K=7).**
  **Validity confirmed from the server log** (guard against a silent no-op run):
  `K3 aggregate tap armed on [7, 23, 51, 67, 83] (nvb=[1, 2, 5, 6, 7],
  report=False, feed_aggregate=True)` and `ATOM_K3_AUX_AGGREGATE=1` present on
  the container. Always check this line before trusting an agg result — an env
  that silently fails to arm looks exactly like "the fix did nothing".
- **THE FALSIFIER (original wording, do not soften).** Flip the tap from `prefix_sum` to the
  aggregate behind a second env (e.g. `ATOM_K3_AUX_AGGREGATE=1`), reusing the
  probe's existing `apply_attn_res` call, then run `accept` on BOTH arms.
  - **radixark should RISE** (~34% -> ~41%+ on `accept`; the GSM8K equivalent is
    55.6% -> ~62%).
  - **inferact should get WORSE** — it was trained on vLLM/HF hidden states, so
    feeding it the SGLang mixture should hurt it.
  - **If BOTH move the same way, the divergence is real but is NOT the cause —
    say so and move on to rung a (K=7).** A one-armed improvement is not
    enough; the two-armed opposite response is what makes this causal rather
    than correlational.
  Keep the default (env-unset) path unchanged so nothing else on the branch
  shifts underneath the comparison.

## rung b RESULT — the gap is a QUALITY story, not a COST story

Traces captured both arms (`gz=8/8, orphan_json=0`; the async export wait
worked). `tools/analyze_trace_summary.py`, trace workload = 8 prompts x 128 tok:

| | radixark (GQA) | inferact (MLA) | delta |
|---|---|---|---|
| draft step mean | **157.1 us** | **125.0 us** | **+25.7%** |
| decode step mean | **5.35 ms** | **4.93 ms** | **+8.5%** |
| decode iterations (same 1024 tok) | 416 | 407 | +2.2% |
| prefill total | 1443.8 ms | 1439.9 ms | +0.3% (same target, as expected) |

**The draft step is only ~2.9% of a decode iteration** (157 us vs 5.35 ms), so
a 25.7%-more-expensive draft CANNOT move acceptance — it moves tok/s only.
Acceptance appears here exactly where it should: radixark needs 416 iterations
for the same 1024 tokens where inferact needs 407.
=> **Profiling does not explain the acceptance gap, and rules itself out as the
explanation.** This is consistent with, and leaves intact, the aux-feature
root-cause hypothesis above.

Two secondary notes:
- The +25.7% draft-step cost is the predicted signature of review finding 1
  (the dead MLA `kv_indptr`/`kv_indices` work the GQA path runs and never reads).
- The **+8.5% DECODE cost is new and unexplained**. Both arms run the identical
  target verify at qlen=6, so it should not differ. Candidates: the dead
  metadata work landing outside the `propose_dspark` annotation, or the bf16
  draft pool's larger PAGE-class footprint shrinking `num_kvcache_blocks`.
  Not acceptance-relevant; record it, do not chase it yet.

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
- The box has **8 GPUs, not 40** (`rocm-smi` and `rocminfo` both report 8;
  `/dev/dri/renderD*` shows 63 nodes, which is what the old "40" came from and
  is not a GPU count). TP8 therefore uses the WHOLE box: arms cannot be run in
  parallel on disjoint device sets, and the validation chain is strictly serial.

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

## LOSSLESSNESS: PASSED 2026-10-06 (after fixing the test, which was broken)

    tools/dflash_ab_compare.py lossless --a /tmp/dflash_ab/v2/nospec \
                                        --b /tmp/dflash_ab/v2/radixark
    -> 6/6 comparable, 0 divergent, PASS. acceptance 92.9% (accept_length 5.65/6).

High acceptance is what gives this test its power: ~93% of emitted tokens came
through the draft+verify path, not plain decode, and were still byte-identical
to no-spec.

### THE TARGET IS NOT RUN-TO-RUN DETERMINISTIC. Do not re-learn this the hard way.

The first lossless run reported 20/20 divergent and looked like a catastrophic
verifier bug. It was not. **The baseline does not reproduce itself.** Measured,
with NO draft loaded, prefix caching off, inside ONE server process:

| prompt | distinct first tokens / 10 identical `max_tokens=1, temperature=0` reqs |
|---|---|
| "Explain the difference between a process and a thread." | **3** |
| "Summarize the tradeoffs of speculative decoding..."     | **4** |
| "The capital of France is" / "2 + 2 =" / "def add(a,b):" | 1 (stable) |

The sampler is a true argmax (`atom/model_ops/sampler.py:50`,
`temperatures <= SAMPLER_EPS`). The noise is in the FORWARD, and a near-tied
top-1 converts it into a different token, which then forks the whole suffix.
Low-entropy prompts are stable; open-ended ones are coin flips. The old
`lossless` workload used `prompts_generic` -- five open-ended prompts -- so it
could only ever report FAIL, no matter what the draft did.

A SECOND, structural reason exact-text equality cannot be assumed: the spec arm
samples from a forward at `qlen=k+1=6`, the no-spec arm at `qlen=1`. Different
MLA split plan (`max_split_per_batch=-1` is unpinned, `aiter_mla.py:963`), a
different KDA branch (`kimi_k3.py:1302` vs `:1371`), a taller lm_head GEMM.
**Spec decode is lossless w.r.t. the target's VERIFY-step argmax, not its
DECODE-step argmax.** `.claude/skills/dump-bisect-debug/SKILL.md:380` says this
is "not fixable at the ATOM layer -- needs aiter to ship batch-invariant kernels".

**Consequence for the design doc:** §7.3's "bitwise-equal token streams" and the
M2 gate built on it are stronger than this stack can deliver, for reasons that
predate this branch. That gate can never go green here. It is DEMOTED to a cheap
regression guard; **GSM8K accuracy vs the no-spec baseline is the real M2 exit
criterion** (an aggregate metric is immune to individual near-tie flips).

### How the test was fixed (tools/dflash_ab_client.py, dflash_ab_compare.py)
- `prompts_lossless()`: six low-entropy prompts, each screened for stability over
  8 reps at `max_tokens=64` **on a no-spec server with no draft loaded** --
  selection on the CONTROL, never on the spec-vs-nospec delta.
- horizon 256 -> 64: the longer the greedy chain, the likelier it crosses a
  near-tie, and one flip forks everything after it.
- every prompt now runs `LOSSLESS_REPS=4` times **within each arm**, recording
  `self_consistent`. The comparator comes with its own control now: prompts that
  flip under their own arm are EXCLUDED and printed, not scored.
- a prompt with no `self_consistent` key counts as unproven, not as stable, so
  the pre-fix banked runs cannot silently pass.
- excluding too much yields INCONCLUSIVE, not PASS -- "nothing left to compare"
  must never read as green.

Banked: `/tmp/dflash_ab/v2/{nospec,radixark}/`. The pre-fix dirs
`/tmp/dflash_ab/{nospec,radixark,rerun}` are from the broken test; ignore them.

## M2 EXIT CRITERION: **PASSED** 2026-10-06 (GSM8K, 1319 samples, 5-shot)

| arm | flexible-extract | strict-match |
|---|---|---|
| nospec   | **0.9393 ± 0.0066** | 0.9386 |
| radixark | **0.9484 ± 0.0061** | 0.9484 |

delta +0.0091 against a combined sigma of 0.0090 => **~1.0 sigma, no regression**
(the skill's rule is "within 1 sigma = no regression"). The spec arm is
nominally HIGHER, which is noise, not an improvement -- spec decode cannot raise
accuracy. That it lands on the high side is still reassuring: a broken verifier
degrades, it never improves.

**Real acceptance, on real traffic: 55.6%, avg 3.78 toks/fwd** (max 6) over
175000 draft tokens. This is the number to quote -- NOT the 92.9% from the
lossless set (deliberately predictable text) nor the 26% from the old
open-ended generic prompts.

    FEWSHOT=5 RUNDIR=/tmp/dflash_ab/gsm8k/<arm> ./tools/dflash_ab.sh <arm> gsm8k

Two non-obvious things the gsm8k workload had to solve (both encoded in
`tools/dflash_ab.sh:run_gsm8k`, do not re-derive):
- lm_eval builds a tokenizer from `model=`, and `moonshotai/Kimi-K3` is a bare
  repo id NOT in the hub cache, so offline it hits the network and dies. Keep
  `model=` as the SERVED name (it goes in the request body) and add
  `tokenizer=<local snapshot path>`.
- the HF cache is bind-mounted `:ro`, and `datasets` takes a FileLock next to
  the dataset *before reading it* -- read-only is fatal even for a pure read.
  gsm8k is 4.5 MB and is staged to a writable copy under /tmp.

## NEXT: ACCEPTANCE DEBUG, then finish validation, then M3-M4

M0/M1/M2 are closed. The validation chain is no longer the blocker -- **the
8.5% acceptance deficit vs `inferact` is**, because uplift is the whole premise
of the branch. Close it before M3.

**Now: acceptance debug.** One rung per sitting, record the number here:
1. `K=7 ./tools/dflash_ab.sh radixark gsm8k` -- trained at block width 7, run at
   k=5. A knob, not a rewrite. Try first.          measured: ____
2. Review finding 1 (`dspark_proposer.py:960`) -- dead per-step MLA metadata
   work. This is tok/s OVERHEAD, not draft quality; it moves the speed column,
   not the acceptance column. Say which you are attacking. measured: ____
3. `trace` both arms (see below) -- this is what separates "proposes worse
   tokens" from "costs more per step".              measured: ____
4. If 1-3 do not explain it: `dump-bisect-debug` on the draft forward. Prime
   suspect is review finding 2 -- nothing tests the `non_causal` -> `fa_views`
   binding, and a silent mis-strided 5-D SHUFFLE read looks EXACTLY like an
   acceptance shortfall with no error.             measured: ____

**Then: finish validation.** `swebench` both arms (20-problem subset via
`SWE_SUBSET_N`, real SWE-bench_Verified statements truncated to 6000 chars;
raise to 50 only to tighten the acceptance estimate -- it is not an accuracy
test), then `perf` both arms.

**Then: M3** (cudagraph FULL, prefix caching, LMCache, max-num-seqs 64) **and
M4** (DCP8; the oracle is acceptance length within 0.2 of M3's, not accuracy).

DONE, do not re-run except to confirm a fix: ~~lossless~~ · ~~GSM8K (M2 exit)~~
· ~~`accept` both arms~~ (`inferact` has been brought up; it takes a different
config-normalization path, which is no longer an unknown).

The PR blockers below (DSV4 `page_unit_bytes`, `ruff` on `tools/m0_*.py`) are
**deferred** -- they gate opening the PR, not the work above.

### `trace` workload (wired, follows the capture-trace skill)
    ./tools/dflash_ab.sh radixark trace     # then the same for nospec/inferact
    python3 tools/analyze_trace_summary.py /tmp/dflash_ab/trace/<arm>/rank_0/*.pt.trace.json.gz
`analyze_trace_summary.py` already extracts prefill/decode/draft step
annotations and compares draft steps across iterations -- do not hand-roll zcat
analysis. Gotchas already encoded in the harness:
- `--torch-profiler-dir` and `ATOM_PROFILER_MORE=0` go on the SERVER; the
  profiler lives in the model-runner workers, so client-side env does nothing.
- **the kineto exporter is ASYNC.** `/stop_profile` returns immediately while
  the per-rank write+gzip takes 10-60 s. The harness now blocks on
  `wait_trace_export` before teardown -- without it `docker rm -f` destroys
  every trace. Completion = `.gz` present AND no orphan `.json` (gz size alone
  is not a valid signal).
- warmup runs OUTSIDE the profiled window (first request pays compile +
  CUDAGraph capture, ~50 s vs ~2 s steady state, and would bury the decode).
NEVER pass --spec-decode-acceptance-length: it fabricates the number being measured.

## HEADLINE: both drafts are CORRECT; the new one is ~8.5% WORSE on acceptance

GSM8K, 1319 samples, 5-shot, TP8/DCP1, k=5, concurrency 32:

| arm | accuracy (flexible-extract) | acceptance | toks/fwd |
|---|---|---|---|
| nospec   | 0.9393 ± 0.0066 | -- | -- |
| radixark (new GQA DFlash) | 0.9484 ± 0.0061 | **55.61%** | **3.78** |
| inferact (existing MLA)   | 0.9507 ± 0.0060 | **62.54%** | **4.13** |

**Accuracy: all three within noise.** Both drafts are correct; neither
regresses. This is M2 confirmed for radixark independently of the lossless test.

**Acceptance: inferact wins by 6.9 pp / +0.35 toks/fwd (-8.5% relative).**

### The two arms are NOT on equal footing -- three handicaps, one is a knob
| | radixark | inferact |
|---|---|---|
| trained block width | **7** | not constrained |
| running `num_spec_tokens` | **5** (= 5 of its trained 7) | 5 |
| draft KV pool | **bf16 forced**, 2x bytes/token | fp8, same as target |
| attention path | `mha_v3` Triton, bypasses `PagedAttentionImpl` | tuned MLA ASM/persistent |

The bf16 pool and the Triton path are deliberate, documented (Appendix E:
`mha_v3` cannot express per-token fp8 scales). **k=5 against a trained 7 is
just a knob** -- `K=7 ./tools/dflash_ab.sh radixark gsm8k`. Add the dead
per-step MLA metadata work (review finding 1) and the tok/s gap is partly
explained overhead rather than draft quality.

## ACCEPTANCE: the new draft currently LOSES to the existing one

`accept` workload (20 requests, but only **5 DISTINCT** prompts repeated 4x,
512 tok, greedy, bs=1 sequential):

| arm | accept_length | acceptance | tok/s |
|---|---|---|---|
| inferact (existing MLA DSpark draft) | **3.09** | **41.8%** | 83.05 |
| radixark (new GQA DFlash draft)      | **2.70** | **34.1%** | 76.36 |

**-12.5% acceptance and slower.** The project's whole premise is uplift; there
is none yet on this workload. GSM8K gave radixark 55.6% on real traffic; the
inferact GSM8K comparison is the decisive one.

### Read the `accept` number with care -- it is a WEAK measurement
- `prompts_generic(20)` is `base[i % 5]`: **5 distinct texts**, each 4x.
- `accept_length` is read from `/debug/mtp_stats` as the **absolute
  server-lifetime** value (`dflash_ab_client.py:280` uses `after`, never
  `after - before`), so it includes warmup forwards. Only safe because each arm
  gets a fresh server.
- bs=1 sequential, not production concurrency.
It is valid as a RELATIVE A/B (both arms see byte-identical prompts). For an
absolute number quote GSM8K instead.

### Hypothesis worth testing next: k=7
We run `k=5` against a draft **trained at block_size 7**, and Appendix D notes
vLLM enforces `num_speculative_tokens == block_size`, flagging it as "worth
revisiting once acceptance is measured". Acceptance is now measured and is
underwhelming, so this is the first thing to try: `K=7 ./tools/dflash_ab.sh
radixark gsm8k`.

## NUMERICS BISECT (in progress) — hunting the -8.5% acceptance deficit

Following `.claude/skills/dump-bisect-debug/SKILL.md`. The target is the
**delta-of-deltas**: the two model cards say radixark and inferact are EQUAL on
GSM8K at k=7 (5.4176 vs 5.44, ratio 0.995); we measure ratio 0.915 at k=5.
Both drafts are trained at block 7, so k cancels in the ratio. That ~8.5% is
specific to our port. Symptom class per design 7.4 / PR #2106: "in bounds, no
fault, wrong rows, and ONLY acceptance moves".

### ELIMINATED (do not re-investigate)
| suspect | verdict | how |
|---|---|---|
| **Weight loading @ TP1** | **CLEAN** | real loader; 62 ckpt tensors -> 45 params; 0 dropped, 0 uninitialized; **60/60 bitwise-equal, max abs diff 0.0** |
| `fc` block permutation | **CLEAN** | weight blocks identity-ordered AND feed order traced: `arm_aux_capture` asserts strictly-increasing `layer_ids`, so ATOM concatenates [7,23,51,67,83] exactly as the reference's `extract_context_feature` |
| qkv / gate_up fusion order @ TP1 | **CLEAN** | explicit swap-detector fired on zero layers |
| markov_w1/w2 transpose | **CLEAN** | `[163840,256]` both sides, bitwise, no transpose |
| **KV pool write->read round trip** | **CLEAN** | ~90 checks bit-exact: index-encoded values compared as raw uint16 at ZERO tolerance, randomly permuted page tables, all 5 layers, nh=2 and nh=16 |
| `fa_views` zero-copy / strides | **CLEAN** | same `data_ptr`, contiguous, `[blocks,block_size,kv_heads,head_dim]` |
| cross-layer pool aliasing | **CLEAN** | per-layer ranges disjoint; filling layer 0 left all others zero |
| slot arithmetic / page boundaries / `-1` pads | **CLEAN** | ctx 300/777/1000/517/133/6, anchor on a page boundary |
| aux-capture convention | **shared with inferact** | neither draft overrides `target_aux_capture_spec`; both use the generic `_extract_layer_hidden`. Cannot explain a radixark-specific delta |
| rope config (YaRN-16 @ theta 10000) | **CLEAN** | ckpt declares `original_max_position_embeddings: 65536`, so `setdefault(config.max_position_embeddings)` (= 1048576) never fires |
| `block_backbone` / `_block_positions` | **shared with inferact** | both build `[anchor, MASK x(T-1)]` identically; both take the `_with_draft` branch |
| `_block_page_idx` row drop | **unreachable here** | `in_table = page_idx < block_tables.shape[1]` only fails near `max_model_len`; GSM8K ctx ~1k |
| GQA q->kv head mapping @ TP8 | **CLEAN** (by hand) | rank r holds q heads [8r,8r+8), kv [2r,2r+2); mha_v3 maps local i -> i//4, so global 8r+i -> 2r+i//4 |

| **Weight loading @ TP8** | **CLEAN** | 8 REAL processes, ATOM's real loader, **1355 checks / 0 failures**. Per-rank fused qkv is `[512 q ; 128 k ; 128 v]`; permutation probe matched exactly ONE of 8 candidates (its own rank) so the test discriminates; wrong-layout and gate/up-swap detectors never fired; NCCL all_gather reconstruction of all 35 logical weights bitwise-equal to ckpt; `fc` genuinely replicated at full 35840 width |
| **Rope mscale** | **CLEAN** | ATOM's `YaRNScalingRotaryEmbedding.mscale` == reference `attention_scaling` == **1.2772588722239782** exactly (= 0.1·ln16+1). aiter's `"yarn"` branch folds mscale into cos/sin, which is why DFlash needs no `self.scaling*mscale²` (inferact takes the equivalent DeepSeek-convention route) |
| **Anchor position convention** | **CLEAN — ATOM IS RIGHT** | the reference's `position_ids` span `ctx+T`; q takes the LAST q_len. At ctx=16: context 0..15 (anchor 15), block 16..22 = `anchor+[1..T]` — exactly `_blk_positions`. The flagged "anchor token in slot 0 roped at anchor+1" is CORRECT, and is NOT a shared bug |
| Appendix A operator order | **CONFIRMED** | reference agent verified all claims, 3 numerically: qk-norm→RoPE→cache-write; v NOT normed; `norm(cat)==cat(norm)` **bitwise** (maxdiff 0.0); fully bidirectional at inference (causal mask would shift by 12.2; free attention reproduces the dump at maxdiff 0.0); q uses `cos[...,-q_len:,:]` while k uses full cos (wrong slice shifts q by 14.1) |

### Reference ground truth — BANKED, reuse it
`tools/ref_dflash.py` + `tools/ref_dump/ref_dflash_eager.pt` (and `_sdpa.pt`
cross-check). Bitwise reproducible across processes. Load with
`torch.load(..., weights_only=False)`. B=1, ctx_len=16, T=7.
**`inputs.noise_embedding` is a SEEDED FAKE embedding** (the draft has no
embedding table; the real one is the 1T target's) — any comparison MUST feed
that exact tensor, not re-embed ids. `aux_hidden` is seeded gaussian, so this
validates the OPERATOR GRAPH, not the real-activation/fp8 numerical regime.

### Facts the design doc was MISSING (now recorded)
- **YaRN `attention_scaling` = 1.2772588722239782** multiplies cos/sin. ATOM
  gets this right via aiter, but it was never written down.
- `extract_context_feature` uses **offset=1**: `hidden_states[layer_id+1]` =
  the OUTPUT of decoder layers [7,23,51,67,83], concatenated on the LAST axis
  in that order. This fixes the `fc.weight` column order — and matches ATOM.
- Appendix A minor error: "no biases" is wrong, `confidence_head.proj.bias [1]`
  exists (on the skipped training-only head, so harmless).
- Appendix A's quote elides `v = cat([v_ctx, v_noise])`, which could misread as
  "v is single-source". It is dual-source like k, just unnormed.
- **Reference buffer trap:** `rotary_emb.inv_freq` / `original_inv_freq` are
  NON-PERSISTENT buffers absent from the safetensors. A `meta`+`to_empty()`
  load runs with a garbage RoPE basis that differs per process. A
  "all checkpoint tensors consumed" check will NOT catch this.

| **Forward operator graph (Phase 3-4)** | **CLEAN** | `tools/phase3_forward_bisect.py`, TP1, single-step (ref input -> ATOM op -> ref output, so no row inherits upstream error). **No checkpoint below cos 0.99987.** NEGATIVE CONTROL: roping q at 0..6 instead of 16..22 gives maxdiff **14.1**, so the test discriminates. mha_v3 non-causal paged read cos 0.99998 with a dense torch control at 0.99999 (same bf16 floor -> mha_v3 is not the error source). Fused qkv split bit-perfect. `_sample_block` produces IDENTICAL ids to an explicit fp32 argmax loop (and exercised the FUSED `dspark_markov_argmax` kernel, since `fused_sample=True` in this image) |

### BISECT CONCLUSION: no bug in the draft
Weights (TP1+TP8), KV pool round trip, forward operator graph, rope, block
positions, aux capture, fc order, and the markov head/sampler are ALL clean.
The -8.5% is not in the draft implementation.

### OPEN (ranked by remaining risk, none yet implicating the draft)
1. **Proposer slot/position plumbing under REAL `attn_metadata`** — multi-request
   batches, non-page-aligned anchors, `anchor+[1..T]` bookkeeping ACROSS
   drafting steps, KV reuse/staleness. The bisect hand-supplied a single-request
   slot map, so this is genuinely untested. Cheap probe: run GSM8K at
   concurrency 1 vs 32; batch-dependent plumbing breakage shows as acceptance
   falling with batch size.
2. **fp8 / real-activation regime.** The reference inputs are seeded gaussian
   aux states + a 0.02-scale FAKE embedding, so magnitudes are unrepresentative.
   Validates the operator graph ONLY; a quantization-regime bug is invisible.
3. Target-coupled half: `share_with_target` (embed/lm_head), real `base_logits`,
   the verify path.
4. TP8 FORWARD (weights proven at TP8; head sharding / o_proj all-reduce /
   replicated fc not numerically exercised).
- If that comes back clean, the remaining explanation is MEASUREMENT CONDITIONS,
  not a bug: the cards were taken on SGLang with
  `--speculative-draft-attention-backend trtllm_mha` and the **confidence head
  active for adaptive block length**, which ATOM deliberately skips
  (`enable_confidence_head: true` in the ckpt config). Plus per-question acc_len
  averaging and chat-template prompting. Say this explicitly rather than
  hunting forever.

### Harness traps (cost real time; encoded in tools/phase1_*.py)
- **`-e PYTHONPATH=/work` is mandatory.** The image ships a STALE `atom` in
  site-packages and `python3 tools/foo.py` puts `/work/tools` on `sys.path[0]`,
  not `/work` -- without it you silently test old code and get a convincing
  false result.
- Build the draft under `torch.set_default_dtype(config.torch_dtype)` as
  `ModelRunner.__init__` does, or all 24 RMSNorm weights and both markov tables
  silently come out fp32 (values exact, container wrong).

## PR BLOCKER (verified): this branch breaks DeepSeek-V4 / V4.1-Flash startup

`model_runner.py:1856` changed
`page_unit_bytes=int(plan.entry_bytes[plan.paged_class])` to
`page_unit_bytes=self.attn_metadata_builder.paged_pool_bytes(1)`.

- `paged_pool_bytes` BASE returns **0** (`backends.py:429`) = "I allocate my own".
- Only `aiter_mla`, `aiter_attention`, `qwen4_exp` override it. **DeepSeek-V4 and
  V4.1 do NOT**, and `deepseek_v4_attn.py:1457` says so deliberately.
- Both DO return `StateTransfer.copy(...)` (`deepseek_v4_attn.py:962`,
  `deepseek_v41/backend.py:235`), so they DO reach this branch.
- `PagedStateCheckpointSpec.__post_init__` rejects `value <= 0`
  (`atom/model_engine/page_unit_checkpoint.py:53`).

=> DSV4 / V4.1-Flash raise at startup, no draft involved. `recipes/DeepSeek-V4.1-Flash.md`
ships, so this family is live. The K3 motivation (item 5 above) is RIGHT; the
expression is too narrow. Use the target builder's own PAGE **declaration**,
which equals `paged_pool_bytes(1)` for carving builders (so K3 is unchanged)
and stays correct for self-allocating ones. NOT yet fixed -- needs a DSV4
checkpoint to validate against.

### Three other review findings (lower severity, not yet fixed)
1. `dspark_proposer.py:960` -- the GQA early-return sits AFTER the MLA-only
   `kv_indptr`/`kv_indices` work, so every draft step runs a cumsum, a
   borrow/zero/restore of the runner's LIVE `block_tables`, and a
   `kv_indices_generate_triton` launch whose output the MHA draft never reads
   (`kimi_k3_dflash.py:279-290` passes only `page_table` + `cache_seqlens`).
   Move the return to just after `:912`. Also `_blk_kv_indices` (`:185`) is
   ~268 MB/rank of never-read VRAM taken out of the KV budget -- gate it.
2. **No test covers the `non_causal` -> `fa_views` binding**, the central new
   behaviour. Deleting `non_causal=non_causal` at `draft_kv.py:88` leaves every
   test green, while at runtime the pool binds 5-D SHUFFLE into a kernel that
   unpacks 4 axes -- a silent mis-strided read, i.e. acceptance collapse with no
   error. Cheap CPU test: real `MhaKvPool`, `allocate(2,"cpu")`, assert
   `k_cache.shape == (2,128,2,64)` for non_causal and `ndim == 5` otherwise.

## PR-TIME TODO (not blocking validation)
`ruff check .` reports ~543 errors repo-wide, almost all pre-existing. CI
(`.github/workflows/pre-checks.yaml:57`) runs with `--exit-zero` and reviewdog
`-filter-mode=diff_context -fail-on-error=true`, so only findings INSIDE THE DIFF
fail. The pre-existing ones are safe; **`tools/m0_*.py` are new on this branch**
(commit 77ee92148) and their ~20 findings WILL be in diff context and WILL fail.
Fix before opening the PR: `ruff check --fix tools/m0_*.py` plus the handful of
`BLE001` blind-except warnings by hand. `tools/dflash_ab_*.py` are already clean.

## TRACE: propose_dspark kernel sequence CONFIRMED + a timing CORRECTION

Traces persisted in-workspace (gitignored, 574 MB):
`traces/radixark_aggON_2026-10-06/rank_{0..7}/*.pt.trace.json.gz`
`traces/inferact_2026-10-06/rank_{0..7}/*.pt.trace.json.gz`

### Kernel sequence inside `propose_dspark` — exactly as designed
424 steps, **deterministic 128 kernels each** (min=max=128, cudagraph replay).
**The GQA attention is `_fwd_kernel_splitK` + `_splitK_reduce`** — the Triton
pair inside `aiter.ops.triton.attention.mha_v3.flash_attn_with_kvcache`.

| kernel | calls/step | role |
|---|---|---|
| `_masked_embedding_kernel` | 1 | embed `[anchor, MASK x4]` |
| `hgemm_bf16_t16x96x128x3` | 1 | `fc` aux fusion (per block pass, NOT per layer) |
| `kn_entry_2c_sbhd_cached_indirect_inplace<OpCachedFwd>` | 5 | RoPE, per layer |
| `reshape_and_cache_flash_kernel<bf16,bf16,...>` | 5 | KV WRITE, per layer |
| **`_fwd_kernel_splitK`** | **5** | **GQA non-causal paged READ** |
| **`_splitK_reduce`** | **5** | its split-K reduction |
| `_dspark_markov_argmax_stage1/2` | 5 each | the T=5 sequential markov loop |
| `cross_device_reduce_1stage<bf16,8>` | 11 | TP all-reduce |

Write-then-read ordering confirmed. **No `pa_decode_gluon`, no
`unified_attention`** — the causal-only kernels we deliberately bypassed are
genuinely off the path.

### CORRECTION to the earlier profiling numbers
The `analyze_trace_summary.py` figures are **CPU launch** annotations. GPU truth:

| | radixark | inferact |
|---|---|---|
| `propose_dspark` CPU (launch) | 157.1 us | 125.0 us | (**+25.7%** <- previously quoted) |
| `propose_dspark` **GPU (real)** | **889.0 us** | **794.1 us** | (**+12.0%**) |
| draft share of a decode iteration | **~6.3%** | ~5.6% | (previously said 2.9%) |

The 128 kernels are serial on ONE stream and sum to 842 us against an 855 us
window, so 889 us is real serial GPU time, not queueing.
**Conclusion survives:** at ~6% of an iteration a 12%-costlier draft still
cannot explain a 6.4 pp acceptance gap — the gap was quality, not cost. But the
inter-arm draft cost is **+12%, not +25.7%**; the CPU figure inflated it with
launch overhead.
Caveat: the "decode" substring matches several annotation types, so ~6% is
approximate. The 889 vs 794 us comparison is exact (same annotation, both arms).

## K=7 RESULT + A METHODOLOGY LESSON (2026-10-06)

| arm | k | accuracy | acc_len | per-token |
|---|---|---|---|---|
| radixark (+fix) | 5 | 0.9500 | 4.10 | 62.04% |
| radixark (+fix) | **7** | **0.9515** | **4.45** | **49.30%** |
| inferact | 5 | 0.9507 | 4.13 | 62.54% |
| inferact | **7** | **0.9560** | **4.59** | **51.31%** |

At MATCHED k=7 vs the cards: radixark 4.45 vs 5.4176 (**-17.9%**, -13.8 pp
per-token); inferact 4.59 vs 5.44 (**-15.6%**, -12.1 pp). **BOTH arms short by
the same amount => methodology, not the DFlash port.**

### TWO CLAIMS IN THIS FILE WERE WRONG — corrected here
1. **"We are within ~1 pp of both cards" is VOID.** It compared OUR k=5
   per-token rate (62.0%) to the CARD's k=7 rate (63.1%). Per-token acceptance
   is **NOT k-invariant** — ours falls 62.0% -> 49.3% from k=5 to k=7 because
   positions 6-7 are much harder. Never normalise across different k again.
2. The predicted k=7 acc_len (~5.2-5.34) was wrong; actual 4.45.

### THE PROCESS LESSON — do this FIRST next time
**The radixark card states its estimator in plain text** and it was read early
and not acted on:
> "`acc_len` is SGLang's histogram-native request acceptance length, **averaged
> within each question and then equally across questions**."
We built a GLOBAL accepted/total ratio and compared it to a per-question-averaged
number, then compounded it across different k. Same class of error as the
losslessness test: **a measurement was built before it was validated against the
thing it would be compared to.** Match the reference's methodology BEFORE
collecting numbers, not after three rounds of reruns.
Note the A/B work was NOT invalidated — both arms used identical methodology, so
relative comparisons (and the root-cause finding) stand. Only the ABSOLUTE
card comparison was affected.

### Residual worth watching
radixark/inferact per-token ratio: ours k=5 **0.992**, ours k=7 **0.961**,
cards k=7 **0.996**. So radixark degrades MORE with horizon than inferact does —
a ~4% relative shortfall at the trained width (both drafts are block_size=7).
Much smaller than the original 8.5%, but not zero.

### IN FLIGHT: `gsm8k_card` workload (do NOT relaunch)
New additive workload matching the card on BOTH levers at once:
chat template (`/v1/chat/completions`, few-shot as turns) AND the per-question
estimator (concurrency 1, poll `/debug/mtp_stats` before/after each request and
difference — the ONLY way to get per-question acceptance out of ATOM, since
`engine_stats.py` keeps a global distribution with no per-request breakdown).
Reports card-style AND global acc_len from the SAME run, isolating the estimator
effect. K=7 both arms; radixark with the aggregate, inferact without.

## CARD-MATCHED MEASUREMENT (gsm8k_card, n=1319, K=7, conc 1) — 2026-10-06

| arm | prior (completions, global, c32) | global (chat, c1) | **card-style** | CARD | residual |
|---|---|---|---|---|---|
| radixark (+aggregate) | 4.45 | 4.6798 | **4.9916** | 5.4176 | **-7.86%** |
| inferact | 4.59 | 4.7404 | **5.0176** | 5.4400 | **-7.76%** |

Levers closed: radixark **56.0%**, inferact **50.3%**. The per-question
estimator is the bigger lever (+0.31 / +0.28) — exactly as the card's wording
implied, and it was readable from the card on day one.

### THE HEADLINE: the port reproduces the cards' RELATIVE result almost exactly
| | radixark - inferact |
|---|---|
| ours | **-0.0260** |
| cards | **-0.0224** |
| agreement | **0.0036 acc_len** |

**That is what the DFlash port is judged on, and it is now essentially exact.**
The absolute level is short by the SAME ~7.8% on BOTH arms — the same
"shared, not draft-specific" signature that correctly pointed at methodology
the first time. SEM +/-0.022, so the 0.426 residual is ~19 sigma: real, not noise.
Accuracy 0.9386 / 0.9378 matches banked nospec 0.9393 -> verifier intact.

### Residual ~7.8%: TWO shared candidates, neither tested
1. **Concurrency.** Cards ran at production concurrency; we are at 1 (forced by
   the missing per-request tracking — now being added). Lever A above
   CONFOUNDS chat template with the c1 batch regime and cannot be split with
   that instrument.
2. **Target quantization.** The C32 production recipe runs
   `--online_quant_config ptpc_fp8`; we serve bf16 weights with fp8 KV only. A
   differently quantized TARGET emits a different token distribution, changing
   how predictable it is — and would shift acceptance IDENTICALLY on both arms,
   which matches the observed signature. Untested.

Raw: `/tmp/dflash_ab/card/{radixark,inferact}/`. NOTE these two runs used the
concurrency-1 counter-differencing path; the harness has since gained
server-side per-request stats and a conc-32 default, so re-running will NOT
reproduce them byte-for-byte.

## BENCH-ONLY per-request change: **REVERTED** 2026-10-06 (saved, not lost)

Restored from `tools/revert_backup/*.orig`; `tests/test_per_request_spec_stats.py`
deleted. Verified: **0 `BENCH-ONLY` markers in live code**, and the four ATOM
engine files (`engine_stats`, `scheduler`, `engine_utility`, `llm_engine`) are
back to HEAD with no diff.

**Saved for reuse at `tools/bench_only_patch/`** — `.patched` full copies,
`.patch` reapply diffs (663 lines total), and the unit test. To reapply:
copy the `.patched` files back, or `patch -p0 < *.patch` against the `.orig`s.

**KEPT (not reverted): the aux-aggregate fix** in
`atom/spec_decode/dspark_proposer.py` (`ATOM_K3_AUX_AGGREGATE=1`). That is the
session's actual finding and must survive.

### Why it was abandoned
It never worked end to end: two conc-32 smokes both returned `n_joined 0/50`
with `per_request` ABSENT from `/debug/mtp_stats`, despite the env being
verified on the container via `docker inspect`. Root cause not isolated —
somewhere between `per_request_spec_statistics()` reporting `enabled` and the
DP merge in `llm_engine` dropping the key. The workflow review also found a
real leak: **preempted-then-aborted sequences bypass `finish_spec_request`**
(`_reject_aborted_waiting` sets FINISHED without going through `_mark_finished`),
so entries leak. Both would need fixing before it is trustworthy.
The instrument DID behave well in one respect: it returned `None` rather than a
plausible wrong number, which is why this cost a diagnosis and not a bad result.

## IN FLIGHT: KV OFFLOAD (LMCache) — first test

`LMCACHE=1` ported from the C32 production recipe
(AMD-MLPerf/mlperf-endpoints `configs/kimi-k3/mi355x/atom/server_C32.env`):
`--kv-transfer-config {"kv_connector":"lmcache_offload","kv_role":"offload"}`
plus `LMCACHE_LOCAL_CPU=True`, `LMCACHE_CHUNK_SIZE=1024`, `LMCACHE_NUMA_MODE=auto`,
`LMCACHE_MAX_LOCAL_CPU_SIZE=128`, `OFFLOAD_KV_FOR_HYBRID=1`,
`OFFLOAD_GPU_STAGING_CHUNKS=32`. Offload implies prefix caching, so LMCACHE=1
turns it on too.

Run: `LMCACHE=1 ATOM_K3_AUX_AGGREGATE=1 K=5 radixark gsm8k`.
**Baseline to hold: acceptance 62.04% / 4.10, accuracy 0.9500.**

### Deliberate deviations from the recipe, and why
- **NUMA pinning NOT ported.** The recipe sets `ATOM_NUMA_NODE=0,0,0,0,1,1,1,1`
  (8 ranks -> 2 nodes). **This box has 8 NUMA nodes**, so that mapping would
  crowd every rank onto nodes 0-1. Left on auto.
- **Online quant NOT ported** (`--online_quant_config ptpc_fp8`). One variable at
  a time; also note its `exclude_layer` lists `context_proj` (the MLA draft's
  fusion layer) but NOT `fc` (the DFlash draft's equivalent) — porting it as-is
  would quantize radixark's aux fusion. See the C32 notes above.
- DCP8 not enabled yet; that is the next step after offload.

### THE HAZARD being tested
The draft's sibling pool registers transfer regions as `draft.{role}` at
`tp_replication_factor=1` (`draft_kv.py:258-271`) — a PER-RANK SHARD offloaded
alongside the target's replicated regions. Nobody has exercised that. Wrong or
missing restore of draft rows = stale KV in the block pass = **acceptance-only**
failure. Watch acceptance, not just "it ran".

### Sizing / pre-flight
`LMCACHE_MAX_LOCAL_CPU_SIZE` is PER RANK: TP8 pins 8 x 128 = **1024 GB** host RAM.
Verified **2938 GB available** before enabling.

### Harness bug caught before the run (would have been a silent no-op)
`EXTRA_ENVS=()` is initialised AFTER the knob blocks, so an `EXTRA_ENVS+=` placed
above it is silently discarded — the run would look fine with offload never
enabled. The LMCache env appends are now below that line, with a comment
pinning the ordering requirement. `$LMCACHE_ARG` was also initially never passed
to the server command.

## KV OFFLOAD (LMCache): PASSES, but the RESTORE path is barely exercised

`LMCACHE=1 ATOM_K3_AUX_AGGREGATE=1 K=5 radixark gsm8k` (n=1319).

| | baseline (no offload) | + LMCache offload | delta |
|---|---|---|---|
| acceptance | 62.04% / 4.10 | **61.79% / 4.09** | -0.25 pp = noise |
| accuracy | 0.9500 | **0.9492 +/- 0.0060** | noise |

**The draft-pool offload path initialises and does not corrupt.** LMCacheEngine
came up with LocalCPUBackend, `kv_connector: lmcache_offload` accepted, and the
`draft.{role}` regions at `tp_replication_factor=1` survived with acceptance
intact. That was the unexercised hazard and it is clear.

### BUT: stores 1344, load attempts 24 — the cache is written, not read
`Stored 1024 out of total 1024 tokens, 0.0156 GB` x1344, vs `offload_loads_attempted 24`.
**Cause: `LMCACHE_CHUNK_SIZE=1024` vs GSM8K's ~500-token shared few-shot prefix.**
A chunk is only reusable if identical; the first 1024-token chunk always
contains question-specific tokens, so the hash differs every time and nothing is
reusable. This ALSO explains the 0.0% prefix-cache hit rate — and **disproves the
earlier hypothesis that the 0% was a missing offload backend.** It is the
workload, not the backend.

### What is therefore NOT yet validated
**The RESTORE path.** 24 load attempts is far too few to trust that per-rank
sharded `draft.{role}` regions come BACK correctly. A wrong restore is an
acceptance-only failure — exactly what this test was meant to catch, and it
cannot catch it at this hit rate.
To exercise restore, use a workload whose shared prefix EXCEEDS the chunk size:
long-context/agentic traffic (what the C32 recipe actually targets, at 1M
context), or repeated long prompts, or drop `LMCACHE_CHUNK_SIZE` well below the
shared-prefix length. **GSM8K is the wrong workload for offload reuse.**
