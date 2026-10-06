#!/usr/bin/env python3
"""Per-iteration segment timing + GPU idle-gap attribution for the DFlash A/B.

Why both a CPU and a GPU column: ATOM's `user_annotation` records the HOST
window (launch), `gpu_user_annotation` the DEVICE window. They differ by ~6x on
the draft step, and reading the CPU one as GPU cost is how `propose_dspark` got
reported as 157 us when it is really 889 us on device.
"""

import gzip, json, glob, re, sys, bisect
from collections import defaultdict

SEGMENTS = [
    "ATOM::prepare_model",
    "ATOM::run_model",
    "decode[",
    "ATOM::forward",
    "ATOM::postprocess",
    "draft_kv[",
    "propose_dspark[",
    "nccl:broadcast",
    "ATOM::_gate_staging_reuse",
    "ATOM::_mark_staging_h2d_enqueued",
]


def load(arm):
    p = glob.glob(f"traces/{arm}/rank_0/*.pt.trace.json.gz")[0]
    with gzip.open(p) as f:
        return json.load(f)["traceEvents"]


def seg_stats(ev):
    out = {}
    for seg in SEGMENTS:
        row = {}
        for cat, key in (("user_annotation", "cpu"), ("gpu_user_annotation", "gpu")):
            ds = [
                e["dur"]
                for e in ev
                if e.get("cat") == cat and str(e.get("name", "")).startswith(seg)
            ]
            row[key] = (sum(ds) / len(ds), len(ds)) if ds else (0.0, 0)
        out[seg] = row
    return out


def gap_analysis(ev, min_gap=20.0):
    """GPU idle gaps on the busiest stream, attributed to whatever the HOST was doing."""
    kern = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    # busiest stream only: mixing streams invents gaps that another stream filled
    bytid = defaultdict(list)
    for k in kern:
        bytid[k.get("tid")].append(k)
    tid = max(bytid, key=lambda t: sum(k["dur"] for k in bytid[t]))
    ks = sorted(bytid[tid], key=lambda e: e["ts"])
    cpu = sorted(
        [e for e in ev if e.get("cat") == "user_annotation"], key=lambda e: e["ts"]
    )
    cts = [e["ts"] for e in cpu]
    rt = sorted(
        [e for e in ev if e.get("cat") == "cuda_runtime"], key=lambda e: e["ts"]
    )
    rts = [e["ts"] for e in rt]

    gaps, busy = [], sum(k["dur"] for k in ks)
    for a, b in zip(ks, ks[1:]):
        g = b["ts"] - (a["ts"] + a["dur"])
        if g >= min_gap:
            gaps.append((a["ts"] + a["dur"], g, a["name"], b["name"]))
    span = (ks[-1]["ts"] + ks[-1]["dur"]) - ks[0]["ts"]

    attrib = defaultdict(lambda: [0, 0.0])  # host annotation spanning the gap
    runtime = defaultdict(lambda: [0, 0.0])  # HIP API call spanning the gap
    for t0, g, _, _ in gaps:
        t1 = t0 + g
        i = bisect.bisect_right(cts, t1)
        inner = None
        for e in cpu[max(0, i - 400) : i]:  # innermost (shortest) enclosing annotation
            if e["ts"] <= t0 and e["ts"] + e["dur"] >= t1:
                if inner is None or e["dur"] < inner["dur"]:
                    inner = e
        attrib[
            (
                re.sub(r"\[.*?\]", "[..]", inner["name"])
                if inner
                else "<no host annotation>"
            )
        ][0] += 1
        attrib[
            (
                re.sub(r"\[.*?\]", "[..]", inner["name"])
                if inner
                else "<no host annotation>"
            )
        ][1] += g
        j = bisect.bisect_right(rts, t1)
        best = None
        for e in rt[max(0, j - 400) : j]:
            if e["ts"] <= t0 + g * 0.5 <= e["ts"] + e["dur"]:
                if best is None or e["dur"] > best["dur"]:
                    best = e
        runtime[best["name"] if best else "<none>"][0] += 1
        runtime[best["name"] if best else "<none>"][1] += g
    return dict(
        tid=tid, span=span, busy=busy, gaps=gaps, attrib=attrib, runtime=runtime
    )


if __name__ == "__main__":
    arm = sys.argv[1]
    ev = load(arm)
    n_iter = len(
        [
            e
            for e in ev
            if e.get("cat") == "user_annotation"
            and str(e.get("name", "")).startswith("ATOM::forward")
        ]
    )
    print(f"\n######## {arm}   ({n_iter} decode iterations) ########")
    st = seg_stats(ev)
    print(f"\n{'segment':34s} {'CPU us':>9s} {'n':>5s} {'GPU us':>9s} {'n':>5s}")
    for seg, row in st.items():
        (c, cn), (g, gn) = row["cpu"], row["gpu"]
        print(f"{seg:34s} {c:9.1f} {cn:5d} {g:9.1f} {gn:5d}")

    ga = gap_analysis(ev)
    print(f"\n-- GPU stream {ga['tid']} --")
    print(
        f"   span {ga['span']/1000:.1f} ms, busy {ga['busy']/1000:.1f} ms "
        f"=> utilisation {100*ga['busy']/ga['span']:.1f}%, idle {(ga['span']-ga['busy'])/1000:.1f} ms"
    )
    tot = sum(g for _, g, _, _ in ga["gaps"])
    print(
        f"   gaps >=20us: {len(ga['gaps'])}  total {tot/1000:.1f} ms  "
        f"({tot/max(1,n_iter):.0f} us per iteration)"
    )
    print(
        f"\n   {'count':>6} {'tot_ms':>8} {'us/iter':>8}  host annotation spanning the gap"
    )
    for k, (c, s) in sorted(ga["attrib"].items(), key=lambda x: -x[1][1])[:8]:
        print(f"   {c:6d} {s/1000:8.1f} {s/max(1,n_iter):8.0f}  {k}")
    print(
        f"\n   {'count':>6} {'tot_ms':>8} {'us/iter':>8}  HIP API call covering the gap midpoint"
    )
    for k, (c, s) in sorted(ga["runtime"].items(), key=lambda x: -x[1][1])[:8]:
        print(f"   {c:6d} {s/1000:8.1f} {s/max(1,n_iter):8.0f}  {k}")
