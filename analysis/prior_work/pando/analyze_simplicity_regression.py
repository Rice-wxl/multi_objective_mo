#!/usr/bin/env python
"""Within-depth Spearman rho between rule simplicity (S = best_1field_acc) and each
outcome -- the 9 agents' rule-recovery accuracies and the 4 validation metrics. Both
S and the outcome are residualized on 3 depth dummies (OLS depth-dummy residualization,
matching the rest of this family), then rank-correlated:

  rho : within-depth Spearman correlation of S with the outcome
  p   : two-sided significance of rho

(No dR2: this is a rank correlation, not a nested-OLS variance decomposition -- the depth
control is in the residualization and the effect size is rho itself.) Trimmed aggregator,
N=80. Prints a markdown table (paste-ready) and a LaTeX body.
"""
import csv
import numpy as np
from scipy import stats

from analyze_raw_acc import depth_design, residualize
import probe_confound
from probe_confound import load_difficulty

AGENTS = [("relp", "relp"), ("gradient", "gradient"), ("prefill", "prefill"),
          ("sae_gradient", "sae-grad"), ("logit_lens", "logit-lens"),
          ("res_token", "res-token"), ("circuit_tracer", "circuit-tracer"),
          ("blackbox", "sample-only"), ("nn", "nn")]
METRICS = [("val_mmlu", "MMLU"), ("val_mt_bench", "MT-Bench"),
           ("val_cot_naturalness", "CoT-nat"), ("val_activation_diff", "Act-nat")]


def within_depth_spearman(y, zS, D):
    """Spearman rho of S vs y after residualizing both on depth dummies."""
    return stats.spearmanr(zS, residualize(y, D))


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
    zS = residualize(np.array([load_difficulty(r["name"], r["depth"])[0] for r in rows]), D)

    def col(key):
        return np.array([float(r[key]) for r in rows])

    outcomes = ([(lab, col(f"int_{k}")) for k, lab in AGENTS] +
                [(lab, col(k)) for k, lab in METRICS])

    print("| Predicted variable | rho | p |")
    print("|---|---|---|")
    latex = []
    for lab, y in outcomes:
        rho, p = within_depth_spearman(y, zS, D)
        sig = "*" if p < 0.05 else ""
        print(f"| {lab:16s} | {rho:+.2f} | {p:.3g}{sig} |")
        if p < 0.001:
            pstr = "\\textbf{\\textless 0.001}"
        elif p < 0.05:
            pstr = f"\\textbf{{{p:.3f}}}"
        else:
            pstr = f"{p:.3f}"
        latex.append(f"\\texttt{{{lab}}} & ${rho:+.2f}$ & {pstr} \\\\")

    print("\n% ---- LaTeX rows (agents block, then metrics block) ----")
    for line in latex[:len(AGENTS)]:
        print(line)
    print("\\midrule")
    for line in latex[len(AGENTS):]:
        print(line)


if __name__ == "__main__":
    main()
