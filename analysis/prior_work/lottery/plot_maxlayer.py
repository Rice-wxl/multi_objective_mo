#!/usr/bin/env python
"""Max-over-layers interpretability results per lottery organism, one bar per training recipe:
  --tool ao          AO detection accuracy, Wald 95% CI (clamped to [0, 1])   fig:lottery-ao-diff / -nondiff
  --tool logit_lens  mean quirk-token cumulative probability (MCP), ± SEM     fig:lottery-logitlens-diff / -nondiff
each for the diffing read (activation difference vs the base) and the non-diffing read (the organism alone).

    python analysis/prior_work/lottery/plot_maxlayer.py --tool ao --results results/prior_work/lottery --out analysis/out/lottery
Writes <out>/<tool>_maxlayer_{diff,nondiff}.{pdf,json}.
"""
import argparse
import math

import helpers as H

YLABEL = {"ao": "AO detection accuracy", "logit_lens": "MCP"}


def error(tool, r):
    """(below, above) error-bar lengths: AO = Wald 95% CI clamped to [0, 1]; logit lens = ± SEM."""
    if tool == "logit_lens":
        return r["sem"], r["sem"]
    p, half = r["max_layer"], 1.96 * math.sqrt(r["max_layer"] * (1 - r["max_layer"]) / r["n"])
    return p - max(0.0, p - half), min(1.0, p + half) - p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tool", choices=list(H.READS), required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = H.load(args.results)
    order = {r: i for i, r in enumerate(H.RECIPES)}
    for setup in H.READS[args.tool]:
        reads = {k: H.read(rec, args.tool, setup) for k, rec in data.items() if H.read(rec, args.tool, setup)}
        reads = dict(sorted(reads.items(), key=lambda kv: (list(H.FAMILIES).index(kv[0][0]), order[kv[0][1]])))
        path = f"{args.out}/{args.tool}_maxlayer_{setup}"
        H.bar_figure(path, {k: (r["max_layer"], error(args.tool, r)) for k, r in reads.items()}, YLABEL[args.tool],
                     ylim=(0, 1) if args.tool == "ao" else None)
        H.write_json(f"{path}.json", [{"family": f, "recipe": rc, "value": r["max_layer"], "best_layer": r["best_layer"]}
                                      for (f, rc), r in reads.items()])
        print(f"{H.READ_LABEL[(args.tool, setup)]} -> {path}.{{pdf,json}}")
        for (fam, recipe), r in reads.items():
            print(f"  {fam:13s} {recipe:20s} {r['max_layer']:.4f}  (layer {r['best_layer']})")


if __name__ == "__main__":
    main()
