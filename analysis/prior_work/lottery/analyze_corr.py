#!/usr/bin/env python
"""tab:lottery-within-corr: within-family Spearman ρ between each validation metric and each interpretability read
(AO / logit lens, diffing / non-diffing; max over layers) across the 19 lottery organisms. Both sides are demeaned
within each quirk family; 95% CIs are Fisher-z with the Bonett-Wright variance and n_eff = n - (k - 1) for the k
family means (the p is Spearman's own test on the demeaned values, as in the paper); * = survives
Benjamini-Hochberg over the 16 cells.

    python analysis/prior_work/lottery/analyze_corr.py --results results/prior_work/lottery --out analysis/out/lottery
Prints the table; writes <out>/within_corr.json.
"""
import argparse
import math

from scipy import stats

import helpers as H


def within_spearman(keys, x, y):
    """(rho, (lo, hi), p) of family-demeaned x vs y."""
    r, p = stats.spearmanr(H.demean(keys, x), H.demean(keys, y))
    n_eff = len(keys) - (len({k[0] for k in keys}) - 1)
    assert abs(r) < 1 and n_eff > 3, "degenerate correlation: no Fisher-z interval"
    half = stats.norm.ppf(0.975) * math.sqrt((1 + r * r / 2) / (n_eff - 3))
    return float(r), (math.tanh(math.atanh(r) - half), math.tanh(math.atanh(r) + half)), float(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = H.load(args.results)
    cells = []
    for (tool, setup), label in H.READ_LABEL.items():
        keys = [k for k, rec in data.items() if H.read(rec, tool, setup)]
        y = [H.read(data[k], tool, setup)["max_layer"] for k in keys]
        for m in H.METRICS:
            r, ci, p = within_spearman(keys, [data[k]["scores"][m] for k in keys], y)
            cells.append({"tool": label, "metric": m, "rho": r, "ci": ci, "p": p, "n": len(keys)})
    cut = H.bh_cutoff([c["p"] for c in cells])
    for c in cells:
        c["bh"] = c["p"] <= cut
    assert len({c["n"] for c in cells}) == 1, "reads cover different organisms"
    print(f"{'read':24s}" + "".join(f"{H.METRIC_LABEL[m]:>32s}" for m in H.METRICS))
    for label in H.READ_LABEL.values():
        row = [f"{c['rho']:+.2f}{'*' if c['bh'] else ' '} [{c['ci'][0]:+.2f},{c['ci'][1]:+.2f}] p={c['p']:.1e}"
               for c in cells if c["tool"] == label]
        print(f"{label:24s}" + "".join(f"{s:>32s}" for s in row))
    print(f"N = {cells[0]['n']}; * = survives Benjamini-Hochberg over the {len(cells)} cells")
    H.write_json(f"{args.out}/within_corr.json", cells)


if __name__ == "__main__":
    main()
