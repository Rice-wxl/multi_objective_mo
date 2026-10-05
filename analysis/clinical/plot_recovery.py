#!/usr/bin/env python3
"""Bias-recovery figure for the clinical audit sweep (writeup Section 6).

Two SEPARATE files, one per panel, so the writeup can caption them as LaTeX subfigures
rather than baking "(a)"/"(b)" titles into the images. Both are drawn at the same height
in inches, with widths in the same ratio as the subfigure widths they are included at, so
LaTeX scales them by one factor and the two axes render the same height.

Two panels at all, because the two quantities live on incomparable scales: the between-bias
spread of recovery levels is ~1.8 score points while the largest arm effect is ~0.2, so
one shared 1-5 axis renders the arm comparison invisible. Worse, the SE of a LEVEL
(0.06-0.09) exceeds the SE of the paired DIFFERENCE (0.045-0.076) -- pairing removes
organism difficulty, the dominant variance component -- so level error bars understate
the precision of the comparison and an asterisk ends up asserting what the plot denies.

  (a) recovery level, black-box arm only: the bias gradient.
  (b) paired difference of each white-box arm against black-box on the SAME organisms,
      with 95% CIs and a zero line. Significance is read off (a CI clearing zero)
      rather than annotated; filled markers survive Benjamini-Hochberg at q=0.05 over
      all six (arm x bias) tests.

Unit of analysis is the ORGANISM: average an organism's rollouts first, then across
organisms, so a noisy rollout does not count three times.

    python analysis/audit_results/plot_recovery.py [round] [-o OUTDIR]
"""
import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import stats  # noqa: E402

AD = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = AD.parents[1] / "writeup_overleaf" / "figures"
# (a) and (b) are included at 0.36 and 0.60 of \linewidth, the same 2.1 : 3.5 ratio as
# these figsizes, and both are saved at their FULL figsize (tight_layout, not
# bbox_inches="tight") -- a tight bbox trims each panel by a different amount and the two
# axes then render at different heights. Equal height in + matched width ratio => LaTeX
# applies one scale factor to both.
SIZE_A, SIZE_B = (2.1, 2.4), (3.5, 2.4)

# Display order = the recovery gradient (race > gender > age), which is the finding.
BIASES = [("asian_dosages", "Race"),
          ("female_rheumatoid_arthritis", "Gender"),
          ("young_aggressive", "Age")]
BASE_GATE, BASE_COLOR = "blackbox", "#4C72B0"
# White-box arms, in panel (b) order. An arm with no rows in the ledger (`sae`, not yet
# run) is simply not drawn -- a marker at zero would read as a measured null effect -- and
# the legend labels it pending. Offsets are computed over the arms actually drawn, so the
# panel stays centred now and reflows by itself once the missing arm lands.
ARMS = [("steer_honesty", "+Honesty steering", "#DD8452", "o"),
        ("jlens", "+J-lens", "#C44E52", "s"),
        ("sae", "+SAE", "#937860", "^")]
SCALE_FLOOR = 1.0


def load(round_name):
    """{(gate, bias): {organism: mean score over its rollouts}} from the ledger."""
    rows = [json.loads(l) for l in
            (AD / "results" / round_name / "rollout_index.jsonl").read_text().splitlines()
            if l.strip()]
    per = defaultdict(list)
    for r in rows:
        if r.get("is_clean"):     # the clean control is a false-positive rate, not a score
            continue
        per[(r["gate"], r["correlation"], r["model_id"])].append(r["mean_score"])
    out = defaultdict(dict)
    for (gate, bias, org), scores in per.items():
        out[(gate, bias)][org] = st.mean(scores)
    return out


def bh(pvals, q=0.05):
    """Benjamini-Hochberg: True where the hypothesis is rejected at level q."""
    order = np.argsort(pvals)
    keep = np.zeros(len(pvals), dtype=bool)
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= q * rank / len(pvals):
            keep[order[:rank]] = True
    return keep


def main(round_name="gemma-4-31b", out=DEFAULT_OUTDIR):
    data = load(round_name)
    level, nb, diffs = {}, {}, {}
    for bias, _ in BIASES:
        base = data.get((BASE_GATE, bias), {})
        nb[bias] = len(base)
        v = np.array(list(base.values()))
        level[bias] = (v.mean(), v.std(ddof=1) / np.sqrt(len(v)))
        for gate, *_ in ARMS:
            cell = data.get((gate, bias))
            if not cell:
                continue
            shared = sorted(set(cell) & set(base))
            d = np.array([cell[o] - base[o] for o in shared])
            se = d.std(ddof=1) / np.sqrt(len(d))
            diffs[(gate, bias)] = (d.mean(), se,
                                   stats.ttest_rel([cell[o] for o in shared],
                                                   [base[o] for o in shared]).pvalue)
    keys = list(diffs)
    sig = dict(zip(keys, bh([diffs[k][2] for k in keys])))

    figa, axa = plt.subplots(figsize=SIZE_A)
    figb, axb = plt.subplots(figsize=SIZE_B)

    # ---- (a) black-box recovery level, one bar per bias
    xs = np.arange(len(BIASES))
    hs = [level[b][0] for b, _ in BIASES]
    es = [level[b][1] for b, _ in BIASES]
    axa.bar(xs, np.array(hs) - SCALE_FLOOR, 0.62, bottom=SCALE_FLOOR, yerr=es, capsize=3,
            color=BASE_COLOR, edgecolor=BASE_COLOR, error_kw=dict(lw=0.9, ecolor="0.25"))
    axa.set_xticks(xs)
    # Per-bias n goes in the subcaption, not the tick labels: at this panel width the
    # "(n = 61)" lines collide with their neighbours.
    axa.set_xticklabels([lbl for _, lbl in BIASES])
    axa.set_ylabel("Identification score (1\u20135)")   # en-dash; matplotlib is not LaTeX
    axa.set_ylim(SCALE_FLOOR, 5.0)
    axa.set_yticks([1, 2, 3, 4, 5])

    # ---- (b) paired difference of each white-box arm vs black-box
    step = 0.22
    shown = [a for a in ARMS if any((a[0], b) in diffs for b, _ in BIASES)]
    for k, (gate, label, color, marker) in enumerate(shown):
        off = (k - (len(shown) - 1) / 2) * step
        for i, (bias, _) in enumerate(BIASES):
            x = xs[i] + off
            if (gate, bias) not in diffs:
                continue
            d, se, _ = diffs[(gate, bias)]
            filled = sig[(gate, bias)]
            axb.errorbar(x, d, yerr=1.96 * se, fmt=marker, ms=5, color=color,
                         mfc=color if filled else "white", mec=color, mew=1.2,
                         elinewidth=1.1, capsize=2.5, zorder=3)
    axb.axhline(0, ls="--", lw=0.9, color="0.4", zorder=1)
    axb.set_xticks(xs)
    axb.set_xticklabels([lbl for _, lbl in BIASES])
    axb.set_xlim(-0.55, len(BIASES) - 0.45)
    # name the score on both panels: a bare "$\Delta$" here reads as accuracy
    axb.set_ylabel("$\\Delta$ identification score\n(vs black-box)")

    for ax in (axa, axb):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.92", lw=0.7)
        ax.set_axisbelow(True)

    drawn = {g for g, _ in diffs}
    handles = [Line2D([], [], color=c if g in drawn else "0.6", marker=m, ls="", ms=5,
                      mfc=c if g in drawn else "white",
                      label=lab if g in drawn else f"{lab} (pending)")
               for g, lab, c, m in ARMS]
    lo, hi = axb.get_ylim()
    axb.set_ylim(lo, hi + 0.30 * (hi - lo))      # headroom for the in-axes legend
    axb.legend(handles=handles, loc="upper right", frameon=True, fontsize=7,
               handlelength=1.0, handletextpad=0.35, borderpad=0.4, framealpha=0.95)

    outdir = Path(out); outdir.mkdir(parents=True, exist_ok=True)
    for f, name in ((figa, "clinical_recovery_level.pdf"),
                    (figb, "clinical_recovery_delta.pdf")):
        f.tight_layout(pad=0.4)
        f.savefig(outdir / name)          # no tight bbox: keep the exact figsize
        print(f"wrote {outdir / name}")
    for (gate, bias), (d, se, pv) in diffs.items():
        print(f"  {bias:32s} {gate:14s} {d:+.3f} +/-{1.96*se:.3f} p={pv:.4g} "
              f"BH={'yes' if sig[(gate, bias)] else 'no'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("round", nargs="?", default="gemma-4-31b")
    ap.add_argument("-o", "--out", default=DEFAULT_OUTDIR,
                    help="output DIRECTORY; two PDFs are written into it")
    a = ap.parse_args()
    main(a.round, a.out)
