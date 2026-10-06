#!/usr/bin/env python
"""What's left of MMLU / MT-Bench / CoT-naturalness's correlation with recoverability
after controlling for rule simplicity (trimmed aggregator, budget 10, N=80). Reproduces
the "partial rho, controlling S" column of the three original/SFT-snapshot tables in
capability_recoverability.md's "controlling for simplicity" section (S3) -- i.e. what
plot_depthadj_heatmaps.py's raw_acc_depth_adjusted_heatmap.png shows BEFORE controlling
for simplicity, this shows AFTER.

Simplicity S = best_1field_acc (probe_confound.load_difficulty), computed once from the
ground-truth trees (aggregator- and component-independent) and partialled out of both
sides of each component<->agent-accuracy correlation, on top of the usual depth-dummy
residualization.

Star marks a cell that survives Benjamini-Hochberg FDR=0.05 correction over this
heatmap's 36 tests (4 columns x 9 agents) -- NOT the uncorrected p<0.05 used for the
inline "*" in capability_recoverability.md's tables.

No title -- this shares the exact figsize/fonts/layout of plot_depthadj_heatmaps.py's
panels (reuses its plot_heatmap() directly) so it drops in as another same-sized panel
next to raw_acc_depth_adjusted_heatmap.png etc.

Writes results_full_trimmed/figures/simplicity_control_heatmap.{png,pdf}.

Usage:
    python plot_simplicity_control_heatmap.py
"""
import csv
import numpy as np

import probe_confound
from probe_confound import load_difficulty, partial
from analyze_raw_acc import depth_design, residualize
import plot_depthadj_heatmaps
from plot_depthadj_heatmaps import AGENTS, plot_heatmap

COMPS = [("val_mmlu", "MMLU"), ("val_mt_bench", "MT-Bench"),
         ("val_cot_naturalness", "CoT-nat"), ("val_activation_diff", "Act-nat")]


def main():
    import argparse
    from plot_depthadj_heatmaps import build_csvs
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="results/prior_work/pando tree")
    ap.add_argument("--out", required=True, help="output dir")
    a = ap.parse_args()
    probe_confound.RESULTS = a.results
    build_csvs(a.results, a.out)
    rows = list(csv.DictReader(open(f"{a.out}/raw_corr/combined_data.csv")))
    depth = np.array([r["depth"] for r in rows])
    D = depth_design(depth)

    # simplicity S = best_1field_acc, from the ground-truth trees (aggregator-independent)
    S = np.array([load_difficulty(r["name"], r["depth"])[0] for r in rows])
    zS = residualize(S, D)

    comp_z = {ccol: residualize(np.array([float(r[ccol]) for r in rows]), D)
              for ccol, _ in COMPS}

    R = np.zeros((len(AGENTS), len(COMPS)))
    P = np.zeros_like(R)
    for i, (akey, _) in enumerate(AGENTS):
        z_r = residualize(np.array([float(r[f"int_{akey}"]) for r in rows]), D)
        for j, (ccol, _) in enumerate(COMPS):
            r_part, p_part = partial(comp_z[ccol], z_r, zS, method="spearman")
            R[i, j], P[i, j] = r_part, p_part

    plot_heatmap(R, P, [c[1] for c in COMPS],
                 r"partial Spearman $\rho$ (depth- \& simplicity-adjusted)",
                 "simplicity_control_heatmap")

    for i, (akey, alabel) in enumerate(AGENTS):
        vals = "  ".join(f"{c[1]}={R[i,j]:+.3f}(p={P[i,j]:.3g})" for j, c in enumerate(COMPS))
        print(f"  {alabel:14s} {vals}")


if __name__ == "__main__":
    main()
