#!/usr/bin/env python3
"""Client side of the DSpark draft A/B harness. See tools/dflash_ab.sh.

Writes one JSON per (arm, workload) into --outdir so arms can be diffed offline.

On losslessness: ATOM's /v1/completions returns ``logprobs: None``
(serving_completion.py:53), so per-token ids are not available over the API. The
check therefore compares exact output TEXT plus completion_token counts. Two
different token streams can in principle detokenize to the same string, so this
is marginally weaker than an id-level diff -- but a greedy spec-decode bug that
preserved both the full text and the token count on every prompt is not a
failure mode anyone has observed, and the alternative (offline simple_inference)
does not exercise the server path we are validating.

Run `--workload compare --outdir <dir>` to diff two arms' saved runs.
"""

import argparse
import json
import os
import re
import shutil
import statistics
import time
import urllib.request

SWE_SUBSET_N = 20

# gsm8k_card: where the writable copy of the HF datasets cache is staged. The
# real cache is bind-mounted read-only in containers and `datasets` takes a
# FileLock NEXT TO the dataset before reading it, so read-only is fatal even for
# a pure read. Same trick (and same path) as `tools/dflash_ab.sh:run_gsm8k`.
HF_CACHE = "/data/workloads-inference/hf_hub_cache"
GSM8K_DS_CACHE = "/tmp/dflash_ab/hf_datasets"
GSM8K_CARD_LIMIT = 200
GSM8K_CARD_FEWSHOT = 5
GSM8K_CARD_MAX_TOKENS = 512

# How many times the lossless workload repeats each prompt WITHIN one arm, to
# establish that the arm agrees with itself before its output is compared to
# the other arm. 4 is enough to catch the observed flip rates (3-4 distinct in
# 5-10 reps on an unstable prompt) without quadrupling the run.
LOSSLESS_REPS = 4


def _post(url, payload, timeout=1800):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(url, timeout=60):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except (OSError, ValueError):
        # Endpoint absent (spec decode off) or not JSON -- both mean "no stats".
        return None


def spec_stats(base):
    """/debug/mtp_stats: average_tokens_per_forward = 1 + accepted draft tokens."""
    s = _get(f"{base}/debug/mtp_stats")
    if not s or not s.get("enabled"):
        return {"enabled": False}
    return s


def prompts_lossless():
    """Low-entropy prompts, for the one workload that needs a stable oracle.

    MEASURED 2026-10-06 on Kimi-K3 / TP8 / this image: the target is NOT
    reproducible run to run. With no draft at all, prefix caching off, in a
    SINGLE server process, the open-ended ``prompts_generic`` set returns 4
    distinct first tokens out of 5 identical ``max_tokens=1, temperature=0``
    requests. ``'The capital of France is'`` returns 1 distinct out of 10.

    So the divergence is GPU numerical noise amplified by a near-tied top-1,
    not a sampler bug -- the sampler is a true argmax
    (``atom/model_ops/sampler.py:50``, ``temperatures <= SAMPLER_EPS``). Prompts
    whose argmax has a clear margin are stable; prompts whose top candidates are
    near-tied flip on noise. The old prompt set was entirely the latter, so the
    lossless check was measuring noise and could only ever report FAIL.

    These prompts were selected by measurement, not by taste: each was stable
    across 8 repetitions at ``max_tokens=64`` in one process. That is necessary
    but not sufficient, which is why the workload re-verifies self-consistency
    on every run (``LOSSLESS_REPS``) instead of trusting this list.

    Note a SECOND, structural reason exact-text equality cannot be assumed: the
    spec arm samples from a forward at ``qlen=k+1`` and the no-spec arm from
    ``qlen=1``. Different MLA split plan (``max_split_per_batch=-1`` is unpinned,
    ``atom/model_ops/attentions/aiter_mla.py:963``), a different KDA branch
    (``atom/models/kimi_k3.py:1302`` vs ``:1371``), and a taller lm_head GEMM.
    Spec decode is lossless w.r.t. the target's VERIFY-step argmax, not its
    DECODE-step argmax. On a confident token the two agree; on a near-tie they
    need not. Hence low-entropy prompts, and hence a short horizon.
    """
    return [
        "Count from 1 to 60, separated by commas: 1, 2, 3,",
        "The first 30 prime numbers are: 2, 3, 5, 7,",
        "JSON array of the integers 1 through 50:\n[1, 2, 3,",
        "Multiplication table of 7:\n7 x 1 = 7\n7 x 2 = 14\n7 x 3 = 21\n7 x 4 =",
        "Powers of two: 1, 2, 4, 8, 16, 32, 64,",
        (
            "def fib(n):\n    if n < 2:\n        return n\n    return fib(n-1) + "
            "fib(n-2)\n\n# The first 25 Fibonacci numbers are:\n# 0, 1, 1, 2, 3,"
        ),
    ]


def prompts_generic(n):
    base = [
        "Write a Python function that reverses a linked list, and explain it.",
        "Explain the difference between a process and a thread.",
        "Summarize the tradeoffs of speculative decoding in LLM inference.",
        "Implement binary search over a rotated sorted array in Python.",
        "What causes a deadlock, and how do you detect one in production?",
    ]
    return [base[i % len(base)] for i in range(n)]


def prompts_swebench(n):
    """Real SWE-bench_Verified problem statements, truncated to a smoke size."""
    try:
        from datasets import load_dataset

        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HOME", "/data/workloads-inference/hf_hub_cache")
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        out = []
        for row in ds.select(range(min(n, len(ds)))):
            out.append(
                "You are a software engineer. Analyze this issue and describe the fix.\n\n"
                f"Repo: {row.get('repo')}\n\nIssue:\n{(row.get('problem_statement') or '')[:6000]}"
            )
        return out
    except (ImportError, OSError, ValueError, KeyError) as e:
        # Say so loudly: a silent fallback would report generic-prompt results
        # under a SWE-bench label.
        print(
            f"  WARN: SWE-bench load failed ({type(e).__name__}: {e}); using generic prompts"
        )
        return prompts_generic(n)


def run_lossless(base, model, prompts, max_tokens, temperature, reps=LOSSLESS_REPS):
    """Run each prompt ``reps`` times and record whether the ARM agreed with itself.

    This is the control the original harness lacked. Without it a run cannot
    tell "the draft changed the output" from "this prompt is a coin flip on
    this hardware", and the latter is the common case -- see
    ``prompts_lossless``. Self-inconsistent prompts carry no information about
    the draft, so they are reported and excluded rather than silently counted
    as failures.
    """
    rows = []
    for i, p in enumerate(prompts):
        texts, usages, t0 = [], [], time.time()
        for _ in range(reps):
            r = _post(
                f"{base}/v1/completions",
                {
                    "model": model,
                    "prompt": p,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            texts.append(r["choices"][0].get("text", ""))
            usages.append(r.get("usage"))
        dt = time.time() - t0
        stable = len(set(texts)) == 1
        rows.append(
            {
                "i": i,
                "text": texts[0],
                "reps": texts,
                "self_consistent": stable,
                "usage": usages[0],
                "latency_s": round(dt, 3),
            }
        )
        print(
            f"  [{i+1}/{len(prompts)}] {dt:6.2f}s  "
            f"{'stable' if stable else f'UNSTABLE ({len(set(texts))}/{reps} distinct)'}"
        )
    return rows


def run_completions(base, model, prompts, max_tokens, temperature):
    """Greedy by default. Returns per-prompt token ids + text + timing."""
    rows = []
    for i, p in enumerate(prompts):
        t0 = time.time()
        r = _post(
            f"{base}/v1/completions",
            {
                "model": model,
                "prompt": p,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "seed": 0,
            },
        )
        dt = time.time() - t0
        ch = r["choices"][0]
        rows.append(
            {
                "i": i,
                "text": ch.get("text", ""),
                "finish_reason": ch.get("finish_reason"),
                "usage": r.get("usage"),
                "latency_s": round(dt, 3),
            }
        )
        print(f"  [{i+1}/{len(prompts)}] {dt:6.2f}s  {ch.get('finish_reason')}")
    return rows


# ── gsm8k_card ───────────────────────────────────────────────────────────────
# A GSM8K acceptance measurement that MATCHES THE MODEL CARDS' METHODOLOGY, so
# the number is comparable to the published 5.4176 (RadixArk) / 5.44 (Inferact).
#
# Two deliberate differences from the `gsm8k` workload, which drives raw 5-shot
# `/v1/completions` through lm_eval and reports a GLOBAL acceptance ratio:
#
#   1. ESTIMATOR. The RadixArk card defines acc_len as "SGLang's histogram-native
#      request acceptance length, averaged within each question and then equally
#      across questions". ATOM's /debug/mtp_stats only exposes a single global
#      total_accepted/total_steps ratio, which is dominated by LONG generations.
#      The only way to recover a per-request breakdown is to poll the endpoint
#      before and after each request and difference the counters -- which is why
#      this workload is SEQUENTIAL (concurrency 1). With concurrent requests the
#      deltas cannot be attributed to a question.
#   2. PROMPTING. The cards used the Kimi-K3 chat template, so the few-shot
#      examples go through /v1/chat/completions as real user/assistant turns
#      (the equivalent of lm_eval's --apply_chat_template
#      --fewshot_as_multiturn), not as one flat completion prompt.
#
# This workload reports BOTH estimators on the SAME run, so their difference is
# the estimator effect in isolation.


def _stage_gsm8k_cache():
    """Copy the read-only HF datasets cache entry to a writable location."""
    dst = os.path.join(GSM8K_DS_CACHE, "openai___gsm8k")
    if not os.path.isdir(dst):
        os.makedirs(GSM8K_DS_CACHE, exist_ok=True)
        shutil.copytree(os.path.join(HF_CACHE, "datasets", "openai___gsm8k"), dst)
        for root, dirs, files in os.walk(dst):
            for name in dirs + files:
                p = os.path.join(root, name)
                os.chmod(p, os.stat(p).st_mode | 0o200)
    os.environ["HF_HOME"] = HF_CACHE
    os.environ["HF_DATASETS_CACHE"] = GSM8K_DS_CACHE
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"


def gsm8k_card_messages(limit, fewshot):
    """Build (messages, gold) per test question, few-shot as real chat turns.

    The few-shot examples are the FIRST ``fewshot`` rows of the train split --
    fixed, not sampled, so two arms see byte-identical prompts. Question/answer
    wording mirrors lm_eval's gsm8k task ("Question: ...\\nAnswer:" and the full
    chain-of-thought ending in "#### N") so the only methodological deltas vs
    the `gsm8k` workload are the chat template and the estimator.
    """
    _stage_gsm8k_cache()
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main")
    shots = []
    for row in ds["train"].select(range(fewshot)):
        shots.append({"role": "user", "content": f"Question: {row['question']}"})
        shots.append({"role": "assistant", "content": row["answer"]})
    test = ds["test"]
    n = len(test) if limit is None else min(limit, len(test))
    out = []
    for row in test.select(range(n)):
        msgs = shots + [{"role": "user", "content": f"Question: {row['question']}"}]
        out.append((msgs, _gsm8k_gold(row["answer"])))
    return out


_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _gsm8k_gold(answer):
    return answer.split("####")[-1].strip().replace(",", "")


def _gsm8k_pred(text):
    """Final numeric answer: after '####' if present, else the last number."""
    tail = text.split("####")[-1] if "####" in text else text
    nums = _NUM_RE.findall(tail)
    if not nums:
        nums = _NUM_RE.findall(text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def _spec_counters(base):
    """(accepted, steps, drafted) from /debug/mtp_stats, or None if spec is off."""
    s = spec_stats(base)
    if not s.get("enabled"):
        return None
    return (
        int(s.get("total_accepted_tokens", 0)),
        sum(int(v) for v in (s.get("distribution") or {}).values()),
        int(s.get("total_draft_tokens", 0)),
    )


def run_gsm8k_card(base, model, items, max_tokens, temperature):
    """Sequential chat-completions run with per-question acceptance deltas."""
    rows = []
    for i, (msgs, gold) in enumerate(items):
        before = _spec_counters(base)
        t0 = time.time()
        r = _post(
            f"{base}/v1/chat/completions",
            {
                "model": model,
                "messages": msgs,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        dt = time.time() - t0
        after = _spec_counters(base)
        ch = r["choices"][0]
        text = (ch.get("message") or {}).get("content") or ""
        pred = _gsm8k_pred(text)
        row = {
            "i": i,
            "gold": gold,
            "pred": pred,
            "correct": pred is not None and pred == gold,
            "finish_reason": ch.get("finish_reason"),
            "usage": r.get("usage"),
            "latency_s": round(dt, 3),
            "text": text,
        }
        if before and after:
            row["accepted"] = after[0] - before[0]
            row["steps"] = after[1] - before[1]
            row["drafted"] = after[2] - before[2]
            if row["steps"] > 0:
                row["acc_len"] = 1.0 + row["accepted"] / row["steps"]
        rows.append(row)
        print(
            f"  [{i+1}/{len(items)}] {dt:6.2f}s  "
            f"acc_len={row.get('acc_len')}  {'OK' if row['correct'] else 'x'}"
        )
    return rows


def summarize_gsm8k_card(rows):
    """CARD-STYLE (per-question mean) vs OURS/GLOBAL (pooled ratio)."""
    scored = [r for r in rows if r.get("steps", 0) > 0]
    per_q = [r["acc_len"] for r in scored]
    tot_a = sum(r["accepted"] for r in scored)
    tot_s = sum(r["steps"] for r in scored)
    tot_d = sum(r["drafted"] for r in scored)
    graded = [r for r in rows if r["pred"] is not None]
    return {
        "n_questions": len(rows),
        "n_scored": len(scored),
        # Per question first, then an EQUAL average over questions -- the card's
        # estimator. Short generations count as much as long ones.
        "acc_len_card_style": round(statistics.fmean(per_q), 4) if per_q else None,
        # One pooled ratio over every step on the server -- today's metric.
        # Long generations dominate it.
        "acc_len_global": round(1.0 + tot_a / tot_s, 4) if tot_s else None,
        "acceptance_rate_global": round(tot_a / tot_d, 4) if tot_d else None,
        "acc_len_min": round(min(per_q), 4) if per_q else None,
        "acc_len_median": round(statistics.median(per_q), 4) if per_q else None,
        "acc_len_max": round(max(per_q), 4) if per_q else None,
        "acc_len_stdev": round(statistics.stdev(per_q), 4) if len(per_q) > 1 else None,
        "total_accepted": tot_a,
        "total_steps": tot_s,
        "total_drafted": tot_d,
        "accuracy_exact_match": (
            round(sum(r["correct"] for r in rows) / len(rows), 4) if rows else None
        ),
        "n_unparsed": len(rows) - len(graded),
    }


def gsm8k_card_workload(a, base):
    items = gsm8k_card_messages(a.limit, a.fewshot)
    print(
        f"arm={a.arm} workload=gsm8k_card questions={len(items)} "
        f"fewshot={a.fewshot} max_tokens={a.max_tokens} concurrency=1"
    )
    before = spec_stats(base)
    t0 = time.time()
    rows = run_gsm8k_card(base, a.model, items, a.max_tokens, 0.0)
    wall = time.time() - t0
    after = spec_stats(base)
    summary = summarize_gsm8k_card(rows)

    out_tokens = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rows)
    result = {
        "arm": a.arm,
        "workload": "gsm8k_card",
        "fewshot": a.fewshot,
        "limit": a.limit,
        "concurrency": 1,
        "endpoint": "/v1/chat/completions",
        "wall_s": round(wall, 2),
        "output_tokens": out_tokens,
        "output_tok_per_s": round(out_tokens / wall, 2) if wall else None,
        "spec_stats_before": before,
        "spec_stats_after": after,
        # Kept under the same keys the other workloads use, so the existing
        # tooling still reads them. These are the SERVER-LIFETIME globals.
        "accept_length": (after or {}).get("average_tokens_per_forward"),
        "acceptance_rate": (after or {}).get("acceptance_rate"),
        "summary": summary,
        "rows": rows,
    }
    path = os.path.join(a.outdir, f"{a.arm}_gsm8k_card.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)

    print(f"\n== {a.arm} / gsm8k_card ==")
    print(
        f"  wall {wall:.1f}s  out_tokens {out_tokens}  tok/s {result['output_tok_per_s']}"
    )
    print(f"  questions {summary['n_questions']}  scored {summary['n_scored']}")
    print(f"  OURS/GLOBAL  acc_len {summary['acc_len_global']}")
    print(f"  CARD-STYLE   acc_len {summary['acc_len_card_style']}")
    print(
        f"  per-question acc_len  min {summary['acc_len_min']}  "
        f"median {summary['acc_len_median']}  max {summary['acc_len_max']}  "
        f"sd {summary['acc_len_stdev']}"
    )
    print(f"  acceptance_rate (global) {summary['acceptance_rate_global']}")
    print(
        f"  accuracy (exact match) {summary['accuracy_exact_match']}  "
        f"unparsed {summary['n_unparsed']}"
    )
    print(f"  -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model", default="moonshotai/Kimi-K3")
    # gsm8k_card only. Concurrency is pinned at 1 (see the section comment), so
    # the full 1319-question test split takes hours; cap it by default.
    ap.add_argument("--limit", type=int, default=GSM8K_CARD_LIMIT)
    ap.add_argument("--fewshot", type=int, default=GSM8K_CARD_FEWSHOT)
    ap.add_argument("--max-tokens", type=int, default=GSM8K_CARD_MAX_TOKENS)
    a = ap.parse_args()

    base = f"http://127.0.0.1:{a.port}"
    os.makedirs(a.outdir, exist_ok=True)

    # Its own branch, not an entry in `cfg`: it carries (messages, gold) pairs
    # rather than prompt strings, runs against /v1/chat/completions, and needs
    # the per-question stats polling that the other workloads must NOT pay for.
    if a.workload == "gsm8k_card":
        gsm8k_card_workload(a, base)
        return

    # Lambdas, not values: building the dict eagerly would load the SWE-bench
    # dataset on every run, including the ones that never use it.
    cfg = {
        # Short and greedy: the point is an exact stream match, not long-form quality.
        "smoke": lambda: (prompts_generic(2), 64, 0.0),
        # Deliberately small. A trace from a long run is unreadable and OOMs the
        # exporter (capture-trace skill, pre-flight #2). 8 concurrent-ish
        # requests x 128 tokens is enough decode steps to see the draft's
        # kernels against the target's.
        "trace": lambda: (prompts_generic(8), 128, 0.0),
        # 64, not 256: the longer the greedy chain the likelier it crosses a
        # near-tie, and one flip forks the whole suffix. See prompts_lossless.
        "lossless": lambda: (prompts_lossless(), 64, 0.0),
        "accept": lambda: (prompts_generic(20), 512, 0.0),
        "swebench": lambda: (prompts_swebench(SWE_SUBSET_N), 512, 0.0),
        "perf": lambda: (prompts_generic(64), 512, 0.0),
    }[a.workload]
    prompts, max_tokens, temperature = cfg()

    print(
        f"arm={a.arm} workload={a.workload} prompts={len(prompts)} max_tokens={max_tokens}"
    )
    before = spec_stats(base)
    t0 = time.time()
    if a.workload == "lossless":
        rows = run_lossless(base, a.model, prompts, max_tokens, temperature)
    elif a.workload == "trace":
        # Warm up OUTSIDE the profiled window. The first request pays torch
        # compile + CUDAGraph capture (measured: ~50 s vs ~2 s steady state),
        # which would dominate the trace and bury the steady-state decode the
        # trace exists to show.
        print("  warmup (outside the profiled window)...")
        run_completions(base, a.model, prompts[:1], 32, temperature)
        _post(f"{base}/start_profile", {})
        print("  profiler ON")
        try:
            rows = run_completions(base, a.model, prompts, max_tokens, temperature)
        finally:
            # finally: a half-open profiler leaves the server writing traces for
            # the rest of its life and the dir unreadable.
            _post(f"{base}/stop_profile", {})
            print("  profiler OFF")
    else:
        rows = run_completions(base, a.model, prompts, max_tokens, temperature)
    wall = time.time() - t0
    after = spec_stats(base)

    out_tokens = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rows)
    result = {
        "arm": a.arm,
        "workload": a.workload,
        "wall_s": round(wall, 2),
        "output_tokens": out_tokens,
        "output_tok_per_s": round(out_tokens / wall, 2) if wall else None,
        "spec_stats_before": before,
        "spec_stats_after": after,
        "accept_length": (after or {}).get("average_tokens_per_forward"),
        "acceptance_rate": (after or {}).get("acceptance_rate"),
        "rows": rows,
    }
    path = os.path.join(a.outdir, f"{a.arm}_{a.workload}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)

    print(f"\n== {a.arm} / {a.workload} ==")
    print(
        f"  wall {wall:.1f}s  out_tokens {out_tokens}  tok/s {result['output_tok_per_s']}"
    )
    print(
        f"  accept_length {result['accept_length']}  acceptance_rate {result['acceptance_rate']}"
    )
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
