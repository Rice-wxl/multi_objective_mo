#!/usr/bin/env python3
"""Emit the full validation<->recovery correlation grid as LaTeX (appendix record).

All 5 axes x 3 suites x 3 arms = 45 cells, the complete family behind the 6 cells
plotted in the main figure. Spearman rho with the p-value; `*` marks BH survival
within a suite's 5 axes (the correction family used throughout, see analyze.py).

    python analysis/correlation/emit_appendix_tables.py > tables.tex
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS = Path(__file__).resolve().parent / "results"
AXES = ["mmlu", "mt_bench", "activation_diff", "cot_naturalness", "test100"]
HEAD = ["MMLU", "MT-Bench", "ActDiff", "CoT-nat", "In-domain"]
SUITES = [("asian_dosages", "Race"), ("female_RA", "Gender"), ("young_agg", "Age")]
ARMS = [("blackbox", "Black-box"), ("steer_honesty", "$+$Honesty steering"),
        ("jlens", "$+$J-lens")]
FDR = 0.05


def cells(rows):
    y = np.array([float(r["audit"]) for r in rows])
    out = {}
    for a in AXES:
        r = stats.spearmanr([float(x[a]) for x in rows], y)
        out[a] = (float(r.statistic), float(r.pvalue))
    order = sorted(AXES, key=lambda a: out[a][1])
    thresh = 0
    for rank, a in enumerate(order, 1):
        if out[a][1] <= FDR * rank / len(AXES):
            thresh = rank
    keep = set(order[:thresh])
    return {a: (rho, p, a in keep) for a, (rho, p) in out.items()}


def fmt(rho, p, sig):
    ps = "$<$.001" if p < .001 else f"{p:.3f}".lstrip("0")
    # \textbf around $...$ does not bold math -- use \mathbf inside the math.
    body = (f"$\\mathbf{{{rho:+.2f}}}^{{*}}$" if sig else f"${rho:+.2f}$")
    return f"{body} ({ps})"


print(r"\begin{table}[t]")
print(r"\centering\small")
print(r"\caption{\textbf{Validation standing vs.\ audit recovery: the full grid.} "
      r"Spearman $\rho$ (with $p$) between each normalised validation axis and the "
      r"auditing agent's mean identification score, over the $N=163$ behaviour-gate-"
      r"passing clinical organisms, per bias suite and per auditor arm. "
      r"$^{*}$ and bold mark cells surviving Benjamini--Hochberg at $q=0.05$ within "
      r"a suite's five axes, the correction family used throughout. The two cells "
      r"plotted in Figure~\ref{fig:clinical-recovery-scatter} are Gender/MT-Bench "
      r"and Race/In-domain.}")
print(r"\label{tab:clinical-val-recovery-grid}")
# the arm is a spanning row rather than a first column: with 7 columns the table
# overflows the ICLR text width by ~86pt, and the arm labels are the widest cells.
print(r"\setlength{\tabcolsep}{4.5pt}")
print(r"\begin{tabular}{lccccc}")
print(r"\toprule")
print(r"Bias & " + " & ".join(HEAD) + r" \\")
for arm, alab in ARMS:
    rows = list(csv.DictReader(open(RESULTS / arm / "merged_data.csv")))
    print(r"\midrule")
    print(r"\multicolumn{6}{l}{\emph{" + alab + r" arm}} \\")
    for suite, slab in SUITES:
        sub = [r for r in rows if r["bias"] == suite]
        c = cells(sub)
        print(f"\\quad {slab} ($n={len(sub)}$) & "
              + " & ".join(fmt(*c[a]) for a in AXES) + r" \\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")
