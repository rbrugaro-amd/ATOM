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
    """Spec decode must be output-identical to greedy non-speculative decoding
    ON PROMPTS WHERE EACH ARM AGREES WITH ITSELF.

    That qualifier is load-bearing and was missing originally. Kimi-K3 on this
    stack is not reproducible run to run: with no draft at all, in one process,
    an open-ended prompt returns several different first tokens across identical
    ``temperature=0`` requests (the sampler is a true argmax -- the noise is in
    the forward, and a near-tied top-1 turns it into a different token). A
    prompt that flips under the baseline cannot say anything about the draft,
    so comparing it to the other arm manufactures a FAIL out of GPU noise.

    Each arm therefore runs every prompt ``LOSSLESS_REPS`` times and records
    ``self_consistent``. Only prompts stable in BOTH arms are evidence; the rest
    are reported and excluded. If too few survive, the verdict is INCONCLUSIVE
    rather than PASS -- "nothing left to compare" is not a pass.
    """
    ra, rb = a["rows"], b["rows"]
    if len(ra) != len(rb):
        print(f"FAIL: prompt count differs: {len(ra)} vs {len(rb)}")
        return False

    # Older banked runs predate self_consistent; absent means "never checked",
    # which is exactly the state that produced the bogus verdict. Treat it as
    # unproven rather than defaulting it to True.
    def stable(r):
        return bool(r.get("self_consistent", False))

    excluded = [
        (x["i"], stable(x), stable(y))
        for x, y in zip(ra, rb, strict=True)
        if not (stable(x) and stable(y))
    ]
    pairs = [(x, y) for x, y in zip(ra, rb, strict=True) if stable(x) and stable(y)]

    bad = []
    for x, y in pairs:
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
    print(f"  prompts: {len(ra)}   comparable: {len(pairs)}   divergent: {len(bad)}")
    for i, sa, sb in excluded:
        why = []
        if not sa:
            why.append(f"{a['arm']} flipped")
        if not sb:
            why.append(f"{b['arm']} flipped")
        print(f"  EXCLUDED prompt {i}: not self-consistent ({', '.join(why)})")
    for i, k, na, nb, sa, sb in bad[:5]:
        print(f"  prompt {i}: first diff at char {k}, tokens {na} vs {nb}")
        print(f"     A: ...{sa!r}")
        print(f"     B: ...{sb!r}")

    # Require a real majority of the set to have been comparable. Excluding
    # everything and declaring PASS is the failure mode this guard exists for.
    min_comparable = max(1, (len(ra) + 1) // 2)
    if len(pairs) < min_comparable:
        print(
            f"  VERDICT: INCONCLUSIVE - only {len(pairs)}/{len(ra)} prompts were "
            f"self-consistent in both arms (need {min_comparable}). The baseline "
            "is too noisy on this hardware to license a losslessness claim."
        )
        return False
    ok = not bad
    print(
        "  VERDICT:",
        (
            f"PASS - token streams identical on all {len(pairs)} comparable prompts"
            if ok
            else "FAIL - NOT LOSSLESS"
        ),
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
    # This line used to claim "both arms are lossless w.r.t. the same target, so
    # greedy outputs SHOULD match". That is FALSE on this stack and would have
    # manufactured the same bogus failure the lossless check did: SWE-bench
    # prompts are open-ended, the target does not reproduce itself run to run on
    # open-ended prompts (see cmp_lossless), and 512-token continuations cross
    # many near-ties. Expect this count to be LOW, and read nothing into it.
    # Losslessness is established by `lossless`, on prompts chosen so the
    # baseline is stable; here the number of interest is accept_length.
    print("  (informational only -- see cmp_lossless; agreement here is NOT expected)")
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
