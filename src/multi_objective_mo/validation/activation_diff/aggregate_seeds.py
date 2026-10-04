#!/usr/bin/env python3
"""Aggregate per-fineweb-seed act-diff summaries into one aggregate.json.

Layout produced by run_act_diff.sh (multi-seed):
    <act_diff_dir>/default/relevance_summary.txt        (primary seed, e.g. 42)
    <act_diff_dir>/seed43/relevance_summary.txt
    <act_diff_dir>/seed44/relevance_summary.txt
This writes <act_diff_dir>/aggregate.json with the mean/std/per-seed of the scored
channel (DIFF patchscope Weighted%) plus the primary seed's full logitlens/patchscope
breakdown. compute_validation_scores.py reads this (mean = the scored value), exactly
like cot_naturalness reads classify_summary.json.

Single source of truth: run_act_diff.sh calls this after its seed loop, and the
migration of pre-existing results calls it directly.
"""
import os, re, json, argparse, statistics as st

_OVERALL = "Overall (mean across all layers and positions)"

def parse_summary(path):
    """Parse the Overall 'difference' rows -> {logitlens:{relevant,weighted},
    patchscope:{...}, mean:{...}} or None if absent/unparseable. Mirrors
    compute_validation_scores.read_actdiff so numbers are identical."""
    if not os.path.isfile(path):
        return None
    text = open(path).read()
    idx = text.find(_OVERALL)
    block = text[idx:] if idx != -1 else text
    got = {}
    for line in block.splitlines():
        m = re.match(r"\s*difference\s+(logitlens|patchscope)\s+([\d.]+)%\s+([\d.]+)%", line)
        if m:
            got[m.group(1)] = {"relevant": float(m.group(2)), "weighted": float(m.group(3))}
    if not got:
        return None
    srcs = list(got.values())
    got["mean"] = {"relevant": sum(s["relevant"] for s in srcs) / len(srcs),
                   "weighted": sum(s["weighted"] for s in srcs) / len(srcs)}
    return got

def subdir_for(seed, primary):
    return "default" if str(seed) == str(primary) else f"seed{seed}"

def aggregate(act_diff_dir, seeds, primary, source="patchscope"):
    per_seed, breakdown_primary = {}, None
    for s in seeds:
        bd = parse_summary(os.path.join(act_diff_dir, subdir_for(s, primary), "relevance_summary.txt"))
        if bd is None:
            continue
        per_seed[str(s)] = bd[source]["weighted"]
        if str(s) == str(primary):
            breakdown_primary = bd
    if not per_seed:
        return None
    # If the primary seed's summary was missing, fall back to any present seed's breakdown.
    if breakdown_primary is None:
        for s in seeds:
            bd = parse_summary(os.path.join(act_diff_dir, subdir_for(s, primary), "relevance_summary.txt"))
            if bd is not None:
                breakdown_primary = bd
                break
    vals = list(per_seed.values())
    mean = sum(vals) / len(vals)
    std = st.stdev(vals) if len(vals) > 1 else 0.0
    return {
        "metric": f"{source}_diff_weighted",
        "source": source,
        "n_seeds": len(vals),
        "seeds": [int(x) for x in per_seed],
        "primary_seed": int(primary),
        "mean": mean,
        "std": round(std, 4),
        "stderr": round(std / (len(vals) ** 0.5), 4),
        "per_seed": per_seed,
        "breakdown": breakdown_primary,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-diff-dir", required=True, help="dir holding default/ + seed<N>/ subdirs")
    ap.add_argument("--seeds", required=True, help="space-separated seed list, e.g. '42 43 44'")
    ap.add_argument("--primary", required=True, help="the seed whose subdir is 'default'")
    ap.add_argument("--source", default="patchscope", choices=["patchscope", "logitlens", "mean"])
    a = ap.parse_args()
    out = aggregate(a.act_diff_dir, a.seeds.split(), a.primary, a.source)
    if out is None:
        print(f"[aggregate_seeds] WARNING: no per-seed summaries under {a.act_diff_dir} — no aggregate.json written")
        return
    dst = os.path.join(a.act_diff_dir, "aggregate.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[aggregate_seeds] wrote {dst}  mean={out['mean']:.2f}  n={out['n_seeds']}  per_seed={out['per_seed']}")

if __name__ == "__main__":
    main()
