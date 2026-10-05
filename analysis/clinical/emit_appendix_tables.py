#!/usr/bin/env python3
"""Emit tab:clinical-val-recovery-grid: 5 validation metrics x 3 biases x 6 tool setups.

Spearman rho (with p) between each validation metric (raw capability axes, normalized
naturalness axes -- see analyze.py's X SCALE note) and each outcome: the agent's mean
identification score under the 4 auditor arms, and the 2 raw tool outputs (CoT
verbalization rate; J-lens readout relevance, both spans, `fired` items). `^{*}` marks
Benjamini-Hochberg survival at q=0.05 within a bias's five metrics; bold marks raw p<0.05.
Computed from the results tree through analyze.load, the same loader as every report.

    python analysis/clinical/emit_appendix_tables.py --results results/clinical --out analysis/out/clinical

Writes <out>/tab_clinical_val_recovery_grid.tex (the tabular) and .json (every cell).
"""
import argparse
from pathlib import Path

from scipy import stats

from _tree import bh, write_json
from analyze import AXES, load

HEAD = ["MMLU", "MT-Bench", "ActDiff", "CoT-nat", "In-domain"]
SUITES = [("race", "Race"), ("gender", "Gender"), ("age", "Age")]
# (label, arm, outcome) in table order
SETUPS = [(r"\emph{Black-box}", "blackbox", "audit"),
          (r"\emph{$+$Honesty steering}", "steer_honesty", "audit"),
          (r"\emph{$+$J-lens}", "jlens", "audit"),
          (r"\emph{$+$SAE}", "sae", "audit"),
          (r"\emph{CoT verbalization} (no agent)", "blackbox", "verbalization"),
          (r"\emph{J-lens readout relevance} (no agent)", "jlens", "relevance")]


def cells(rows):
    y = [r["audit"] for r in rows]
    res = [stats.spearmanr([r[a] for r in rows], y) for a in AXES]
    keep = bh([float(r.pvalue) for r in res])
    return {a: (float(r.statistic), float(r.pvalue), bool(k)) for a, r, k in zip(AXES, res, keep)}


def fmt(rho, p, sig):
    ps = f"{p:.3f}".lstrip("0")
    ps = "$<$.001" if ps == ".000" else ps     # the paper rounds first: p=0.0009 prints .001
    body = f"{rho:+.2f}"
    if p < .05:
        body = f"\\mathbf{{{body}}}"
    return f"${body}{'^{*}' if sig else ''}$ ({ps})"


def main(results, out):
    L = [r"\begin{tabular}{lccccc}", r"\toprule", "Bias & " + " & ".join(HEAD) + r" \\"]
    grid = {}
    for label, arm, outcome in SETUPS:
        recs = load(results, arm, outcome, "both", "fired")
        L += [r"\midrule", r"\multicolumn{6}{l}{" + label + r"} \\"]
        for suite, slab in SUITES:
            sub = [r for r in recs if r["bias"] == suite]
            c = cells(sub)
            grid[f"{arm}/{outcome}/{suite}"] = {"n": len(sub), **{a: list(v) for a, v in c.items()}}
            L.append(f"\\quad {slab} ($n={len(sub)}$) & "
                     + " & ".join(fmt(*c[a]) for a in AXES) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "tab_clinical_val_recovery_grid.tex").write_text("\n".join(L) + "\n")
    write_json(o / "tab_clinical_val_recovery_grid.json", grid)
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results tree (results/clinical)")
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args()
    main(a.results, a.out)
