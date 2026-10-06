#!/usr/bin/env python3
"""Diff two arms of the DSpark draft A/B, and report the verdicts.

    tools/dflash_ab_compare.py lossless  --a /tmp/dflash_ab/nospec   --b /tmp/dflash_ab/radixark
    tools/dflash_ab_compare.py accept    --a /tmp/dflash_ab/inferact --b /tmp/dflash_ab/radixark
    tools/dflash_ab_compare.py perf      --a /tmp/dflash_ab/inferact --b /tmp/dflash_ab/radixark

Exit code is nonzero when a check FAILS, so this can gate the next milestone.
"""

import argparse
import json
import os
import sys


def load(d, workload):
    arm = os.path.basename(d.rstrip("/"))
    path = os.path.join(d, f"{arm}_{workload}.json")
    with open(path) as f:
        return json.load(f)


def cmp_lossless(a, b):
    """Spec decode must be output-identical to greedy non-speculative decoding.

    Any divergence is a verifier / rejection-sampler bug, not draft quality, and
    it invalidates every acceptance number measured after it.
    """
    ra, rb = a["rows"], b["rows"]
    if len(ra) != len(rb):
        print(f"FAIL: prompt count differs: {len(ra)} vs {len(rb)}")
        return False
    bad = []
    for x, y in zip(ra, rb, strict=True):
        ta, tb = x.get("text", ""), y.get("text", "")
        na = (x.get("usage") or {}).get("completion_tokens")
        nb = (y.get("usage") or {}).get("completion_tokens")
        if ta != tb or na != nb:
            # First divergent character localizes it better than a diff dump.
            k = next(
                (i for i, (ca, cb) in enumerate(zip(ta, tb, strict=False)) if ca != cb),
                min(len(ta), len(tb)),
            )
            bad.append(
                (
                    x["i"],
                    k,
                    na,
                    nb,
                    ta[max(0, k - 40) : k + 40],
                    tb[max(0, k - 40) : k + 40],
                )
            )
    print(f"\n== LOSSLESS  {a['arm']} vs {b['arm']} ==")
    print(f"  prompts: {len(ra)}   divergent: {len(bad)}")
    for i, k, na, nb, sa, sb in bad[:5]:
        print(f"  prompt {i}: first diff at char {k}, tokens {na} vs {nb}")
        print(f"     A: ...{sa!r}")
        print(f"     B: ...{sb!r}")
    ok = not bad
    print(
        "  VERDICT:", "PASS - token streams identical" if ok else "FAIL - NOT LOSSLESS"
    )
    return ok


def cmp_accept(a, b):
    """Acceptance length uplift. Higher is better; this is the whole point."""
    la, lb = a.get("accept_length"), b.get("accept_length")
    print(f"\n== ACCEPTANCE  {a['arm']} vs {b['arm']} ==")
    print(
        f"  {a['arm']:>10}: accept_length {la}  acceptance_rate {a.get('acceptance_rate')}"
    )
    print(
        f"  {b['arm']:>10}: accept_length {lb}  acceptance_rate {b.get('acceptance_rate')}"
    )
    if la is None or lb is None:
        print("  VERDICT: INCONCLUSIVE - an arm reported no spec stats")
        return False
    d = lb - la
    print(f"  delta: {d:+.4f}  ({100 * d / la:+.1f}%)")
    print("  VERDICT:", "UPLIFT" if d > 0 else "NO UPLIFT")
    return True


def cmp_perf(a, b):
    print(f"\n== PERF  {a['arm']} vs {b['arm']} ==")
    for r in (a, b):
        print(
            f"  {r['arm']:>10}: {r['output_tok_per_s']} tok/s  wall {r['wall_s']}s  "
            f"out {r['output_tokens']}  accept_length {r.get('accept_length')}"
        )
    pa, pb = a["output_tok_per_s"], b["output_tok_per_s"]
    if pa and pb:
        print(f"  delta: {pb - pa:+.2f} tok/s  ({100 * (pb - pa) / pa:+.1f}%)")
    return True


def cmp_swebench(a, b):
    """Smoke A/B: both arms must produce non-empty, non-degenerate output."""
    print(f"\n== SWE-BENCH SUBSET  {a['arm']} vs {b['arm']} ==")
    ok = True
    for r in (a, b):
        rows = r["rows"]
        empty = sum(1 for x in rows if not (x.get("text") or "").strip())
        short = sum(1 for x in rows if len((x.get("text") or "").split()) < 20)
        print(
            f"  {r['arm']:>10}: {len(rows)} prompts  empty {empty}  under-20-words {short}  "
            f"accept_length {r.get('accept_length')}  tok/s {r['output_tok_per_s']}"
        )
        if empty:
            ok = False
    agree = sum(
        1
        for x, y in zip(a["rows"], b["rows"], strict=True)
        if x.get("text") == y.get("text")
    )
    print(f"  identical outputs across arms: {agree}/{len(a['rows'])}")
    print(
        "  (both arms are lossless w.r.t. the same target, so greedy outputs SHOULD match)"
    )
    print("  VERDICT:", "PASS" if ok else "FAIL - empty generations")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workload", choices=["lossless", "accept", "perf", "swebench"])
    ap.add_argument("--a", required=True, help="baseline arm run dir")
    ap.add_argument("--b", required=True, help="candidate arm run dir")
    args = ap.parse_args()

    a = load(args.a, args.workload)
    b = load(args.b, args.workload)
    ok = {
        "lossless": cmp_lossless,
        "accept": cmp_accept,
        "perf": cmp_perf,
        "swebench": cmp_swebench,
    }[args.workload](a, b)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
