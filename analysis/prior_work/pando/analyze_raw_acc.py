#!/usr/bin/env python
"""tab:pando-raw-regression: depth fixed-effects OLS of each tool's rule-recovery accuracy on the four validation
metrics, over the 80 original Pando organisms.

    python analysis/prior_work/pando/analyze_raw_acc.py --results results/prior_work/pando --out analysis/out/pando
Prints the table; writes <out>/raw_regression.json (per tool: dR2, Fblk_p, betas, ci, ps).
"""
import argparse

import helpers as H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    orgs, depth, X, Y = H.load_levels(args.results)
    print(f"{len(orgs)} original organisms")
    fits = {a: H.depth_fe_ols(Y[a], X, depth) for a in H.AGENTS}
    H.ols_table(fits)
    H.write_json(f"{args.out}/raw_regression.json", fits)


if __name__ == "__main__":
    main()
