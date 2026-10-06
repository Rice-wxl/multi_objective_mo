#!/usr/bin/env python
"""Direct-correlation heatmaps over the 80 Pando organisms (trimmed
aggregator, budget 10): Spearman rho between each raw (or delta) validation
component and each interpretability tool's raw (or delta) rule-recovery
accuracy -- both the depth-adjusted (residualized on decision-tree-depth
dummies) and the plain pooled variants.

Reproduces the "Individual correlations" tables in:
  results_full_trimmed/raw_corr/spearman/outcome_raw_acc_depth_adjusted.md
  results_full_trimmed/raw_corr/spearman/outcome_raw_acc_pooled.md
  results_full_trimmed/change_corr/spearman/outcome_acc_change_depth_adjusted.md

and writes, into results_full_trimmed/figures/:
  raw_acc_depth_adjusted_heatmap.{png,pdf}     (levels, depth-adjusted, N=80)
  raw_acc_pooled_heatmap.{png,pdf}             (levels, pooled, N=80)
  acc_change_depth_adjusted_heatmap.{png,pdf}  (DPO-SFT deltas, depth-adjusted, N=80)

Star (and bold) marks a cell that survives Benjamini-Hochberg FDR=0.05
correction over that heatmap's 36 tests -- NOT the uncorrected p<0.05.

No titles are added to the figures -- all three share identical size,
fonts, and layout, and are told apart only by filename (matches the
"no titles, this is for Overleaf" convention in plot_results.py).

Usage:
    python plot_depthadj_heatmaps.py --results results/prior_work/pando --out analysis/out/pando
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BASE = OUT = None   # <--out> and <--out>/figures, set by build_csvs()


def build_csvs(results, out):
    """Write the trimmed-aggregator per-organism tables (raw_corr/combined_data.csv,
    change_corr/combined_data_change.csv) into `out`, exactly as analyze_raw_acc.py /
    analyze_acc_change.py do; every panel is computed from these files."""
    global BASE, OUT
    import analyze_acc_change
    import analyze_raw_acc
    BASE, OUT = out, f"{out}/figures"
    for mod, sub in ((analyze_raw_acc, "raw_corr"), (analyze_acc_change, "change_corr")):
        mod.AGG, mod.OUT = "trimmed", f"{out}/{sub}"
        os.makedirs(mod.OUT, exist_ok=True)
        mod.write_csv(mod.load_data(results, "trimmed"))
    os.makedirs(OUT, exist_ok=True)

# Shared color scale across every heatmap in this figure family (including
# plot_simplicity_control_heatmap.py, which imports plot_heatmap from here) so
# the same rho maps to the same color everywhere they're compared. Covers the
# largest |rho| observed across all four panels (raw_corr pooled, Gradient~MT-
# Bench = +0.669), with a little headroom; update if a new panel exceeds it.
SHARED_VMAX = 0.70

# display order matches the paper table: MMLU, MT-Bench, CoT-nat, Act-nat
AGENTS = [("relp", "ReLP"), ("gradient", "Gradient"),
          ("prefill", "Prefill"), ("sae_gradient", "SAE-grad"),
          ("logit_lens", "Logit-lens"), ("res_token", "Res-token"),
          ("circuit_tracer", "Circuit-tracer"), ("blackbox", "Sample-only"),
          ("nn", "NN")]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
})


def resid(D, y):
    """Residual of y after OLS on depth group dummies (== subtract group mean)."""
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    return y - D @ beta


def corr_matrix(df, comp_cols, agent_prefix, depth_adjust):
    D = pd.get_dummies(df["depth"]).astype(float).values if depth_adjust else None
    R = np.zeros((len(AGENTS), len(comp_cols)))
    P = np.zeros_like(R)
    for i, (akey, _) in enumerate(AGENTS):
        y = df[f"{agent_prefix}{akey}"].values
        ry = resid(D, y) if depth_adjust else y
        for j, (ccol, _) in enumerate(comp_cols):
            x = df[ccol].values
            rx = resid(D, x) if depth_adjust else x
            rho, p = stats.spearmanr(rx, ry)
            R[i, j] = rho
            P[i, j] = p
    return R, P


def bh_mask(P):
    flat = P.flatten()
    order = np.argsort(flat)
    m = len(flat)
    thresh = 0.0
    for rank, idx in enumerate(order, start=1):
        if flat[idx] <= rank / m * 0.05:
            thresh = flat[idx]
    return (flat <= thresh).reshape(P.shape), thresh


def plot_heatmap(R, P, comp_labels, cbar_label, fname):
    bh_sig, thresh = bh_mask(P)
    fig, ax = plt.subplots(figsize=(4.2, 5.4))
    vmax = SHARED_VMAX
    im = ax.imshow(R, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(comp_labels)))
    ax.set_xticklabels(comp_labels, rotation=40, ha="right")
    ax.set_yticks(range(len(AGENTS)))
    ax.set_yticklabels([a[1] for a in AGENTS])

    for i in range(len(AGENTS)):
        for j in range(len(comp_labels)):
            bold = bh_sig[i, j]
            star = "*" if bold else ""
            txt = f"{R[i, j]:+.2f}{star}"
            ax.text(j, i, txt, ha="center", va="center", color="black",
                    fontsize=9.5, fontweight="bold" if bold else "normal")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{fname}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"{OUT}/{fname}.png /.pdf")
    print("  BH threshold p<=", thresh, "  #BH-sig:", int(bh_sig.sum()),
          "  #uncorrected p<.05:", int((P < 0.05).sum()))
    return R, P


def raw_acc_depth_adjusted_heatmap():
    """Levels, depth-adjusted: raw_corr/combined_data.csv, val_* vs int_* (N=80)."""
    df = pd.read_csv(f"{BASE}/raw_corr/combined_data.csv")
    comp_cols = [("val_mmlu", "MMLU"), ("val_mt_bench", "MT-Bench"),
                 ("val_cot_naturalness", "CoT-nat"), ("val_activation_diff", "Act-nat")]
    R, P = corr_matrix(df, comp_cols, "int_", depth_adjust=True)
    plot_heatmap(R, P, [c[1] for c in comp_cols],
                 r"partial Spearman $\rho$ (depth-adjusted)",
                 "raw_acc_depth_adjusted_heatmap")


def raw_acc_pooled_heatmap():
    """Levels, pooled (no depth adjustment): raw_corr/combined_data.csv, val_* vs int_* (N=80)."""
    df = pd.read_csv(f"{BASE}/raw_corr/combined_data.csv")
    comp_cols = [("val_mmlu", "MMLU"), ("val_mt_bench", "MT-Bench"),
                 ("val_cot_naturalness", "CoT-nat"), ("val_activation_diff", "Act-nat")]
    R, P = corr_matrix(df, comp_cols, "int_", depth_adjust=False)
    plot_heatmap(R, P, [c[1] for c in comp_cols],
                 r"Spearman $\rho$ (pooled)",
                 "raw_acc_pooled_heatmap")


def acc_change_depth_adjusted_heatmap():
    """Deltas, depth-adjusted: change_corr/combined_data_change.csv, dval_* vs dint_* (N=80)."""
    df = pd.read_csv(f"{BASE}/change_corr/combined_data_change.csv")
    comp_cols = [("dval_mmlu", "MMLU"), ("dval_mt_bench", "MT-Bench"),
                 ("dval_cot_naturalness", "CoT-nat"), ("dval_activation_diff", "Act-nat")]
    R, P = corr_matrix(df, comp_cols, "dint_", depth_adjust=True)
    plot_heatmap(R, P, [c[1] for c in comp_cols],
                 r"partial Spearman $\rho$ ($\Delta$, depth-adjusted)",
                 "acc_change_depth_adjusted_heatmap")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="results/prior_work/pando tree")
    ap.add_argument("--out", required=True, help="output dir (figures/ + the two per-organism CSVs)")
    a = ap.parse_args()
    build_csvs(a.results, a.out)
    raw_acc_depth_adjusted_heatmap()
    raw_acc_pooled_heatmap()
    acc_change_depth_adjusted_heatmap()
