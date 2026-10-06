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
import time
import urllib.request

SWE_SUBSET_N = 20


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model", default="moonshotai/Kimi-K3")
    a = ap.parse_args()

    base = f"http://127.0.0.1:{a.port}"
    os.makedirs(a.outdir, exist_ok=True)

    # Lambdas, not values: building the dict eagerly would load the SWE-bench
    # dataset on every run, including the ones that never use it.
    cfg = {
        # Short and greedy: the point is an exact stream match, not long-form quality.
        "smoke": lambda: (prompts_generic(2), 64, 0.0),
        "lossless": lambda: (prompts_generic(20), 256, 0.0),
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
