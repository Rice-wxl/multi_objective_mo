#!/usr/bin/env python
"""tab:pando-simplicity-corr: within-depth Spearman ρ between rule simplicity (best 1-field accuracy) and each tool's
rule-recovery accuracy and each validation metric, over the 80 original Pando organisms.

    python analysis/prior_work/pando/simplicity_corr.py --results results/prior_work/pando
"""
import argparse

import numpy as np

import helpers as H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    orgs, depth, X, Y = H.load_levels(args.results)
    S = np.array([H.best_1field(H.test_data(o)) for o in orgs])
    print("| outcome | rho | p |\n|---|---|---|")
    for name, y in [(H.AGENT_LABEL[a], Y[a]) for a in H.AGENTS] + [(H.METRIC_LABEL[m], X[m]) for m in H.METRICS]:
        r, p = H.within_depth_spearman(S, y, depth)
        print(f"| {name} | {r:+.2f} | {p:.3g} |")


if __name__ == "__main__":
    main()
