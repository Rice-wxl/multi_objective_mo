#!/usr/bin/env python
"""fig:simplicity_control_heatmap: the raw-accuracy heatmap of plot_depthadj_heatmaps.py with rule simplicity (best
1-field accuracy) also regressed out of both sides.

    python analysis/prior_work/pando/plot_simplicity_control_heatmap.py --results results/prior_work/pando --out analysis/out/pando
Writes <out>/simplicity_control_heatmap.{pdf,json}.
"""
import argparse

import numpy as np

import helpers as H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    orgs, depth, X, Y = H.load_levels(args.results)
    S = np.array([H.best_1field(H.test_data(o)) for o in orgs])
    H.heatmap(f"{args.out}/simplicity_control_heatmap", *H.spearman_matrix(depth, X, Y, z=S),
              r"within-depth Spearman $\rho$, simplicity regressed out")


if __name__ == "__main__":
    main()
