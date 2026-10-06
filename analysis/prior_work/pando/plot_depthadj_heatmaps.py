#!/usr/bin/env python
"""fig:raw_acc_heatmap, fig:change_heatmap: Spearman ρ between each validation metric and each tool's rule-recovery
accuracy after subtracting each depth's mean from both, for the levels (80 originals) and the changes (80 pairs).

    python analysis/prior_work/pando/plot_depthadj_heatmaps.py --results results/prior_work/pando --out analysis/out/pando
Writes <out>/{raw_acc,acc_change}_depth_adjusted_heatmap.{pdf,json}.
"""
import argparse

import helpers as H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    _, depth, X, Y = H.load_levels(args.results)
    H.heatmap(f"{args.out}/raw_acc_depth_adjusted_heatmap", *H.spearman_matrix(depth, X, Y),
              r"within-depth Spearman $\rho$")
    _, depth, X, Y = H.load_changes(args.results)
    H.heatmap(f"{args.out}/acc_change_depth_adjusted_heatmap", *H.spearman_matrix(depth, X, Y),
              r"within-depth Spearman $\rho$ ($\Delta$)")


if __name__ == "__main__":
    main()
