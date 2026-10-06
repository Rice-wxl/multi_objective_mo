#!/usr/bin/env python
"""tab:change-on-change-ols: depth fixed-effects OLS of each tool's Δ rule-recovery accuracy (retrain − original) on
the four Δ validation metrics, over the 80 Pando organism pairs; plus the leave-one-out check quoted in the text
(refit once per dropped organism; the largest block-F p).

    python analysis/prior_work/pando/analyze_acc_change.py --results results/prior_work/pando --out analysis/out/pando
Prints both tables; writes <out>/change_regression.json (per tool: dR2, Fblk_p, betas, ci, ps, loo_max_p).
"""
import argparse

import numpy as np

import helpers as H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    orgs, depth, X, Y = H.load_changes(args.results)
    n = len(orgs)
    print(f"{n} original/retrain pairs")
    fits = {a: H.depth_fe_ols(Y[a], X, depth) for a in H.AGENTS}
    H.ols_table(fits)
    print(f"\n{'tool':15s} {'Fblk p':>8s} {'max p over ' + str(n) + ' leave-one-out refits':>34s}")
    for a, f in fits.items():
        keep = [np.arange(n) != i for i in range(n)]
        f["loo_max_p"] = max(H.depth_fe_ols(Y[a][k], {m: X[m][k] for m in X}, depth[k])["Fblk_p"] for k in keep)
        print(f"{H.AGENT_LABEL[a]:15s} {f['Fblk_p']:8.4f} {f['loo_max_p']:34.4f}")
    H.write_json(f"{args.out}/change_regression.json", fits)


if __name__ == "__main__":
    main()
