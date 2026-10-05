#!/usr/bin/env python3
"""Validation standing vs audit recovery, one point per organism.

Rows are the two signals that replicate; columns are the three auditor arms. The
layout IS the argument: the same cell is shown under three channels, so a reader
sees the consistency rather than being told about it. Arms share organisms and X
and differ only in the channel the auditor was given, so agreement down a column
is a consistency check, not independent replication (per-bias cross-arm Y
correlation runs 0.42-0.80).

X is the RAW validation measurement, not the capped criteria score -- see the
X SCALE note in analyze.py. A dashed rule marks the base model, which is what the
capped score's 1.0 meant.

Colour = training recipe, a nested palette: one hue family per objective (warm
DPO, cool SFT), stepped within it, lightest = closest to base
(merge -> mix -> unmix). Recipe is plotted rather than partialled out because the
claim is predictive -- validation standing TRACKS recoverability -- and recipe is
a plausible common driver of both, not a nuisance to remove (same framing as the
logit-lens result in Sec. 5.2). All five share one marker shape: at ~60 points a panel,
varying shape reads as noise rather than as a category.

DPO takes the WARM family because it needs three steps and two cannot be told
apart on lightness alone inside the allowed band: the warm family lets the three
spread over hue (amber -> orange-red -> crimson) as well as lightness, which three
blues could not. SFT, needing only two, takes the cool one.

Palette validated with the dataviz skill's validate_palette.js (light surface,
--pairs all, the scatter pairlist): lightness band, chroma floor, CVD separation
(worst all-pairs dE 16.0 deutan / 14.7 tritan, target >=8) and normal-vision
floor (17.5, floor 15) all PASS -- the worst pair is now inside the DPO trio,
and further apart than the worst pair of the previous all-blue-DPO palette. The two lightest steps sit below 3:1 on the
surface, so markers carry a darker same-hue edge and the legend is always shown.

Line is ordinary least squares -- the convention for a scatter like this -- while the
annotated coefficient is Spearman. They are different estimators and can disagree in
magnitude: OLS is ~52% steeper than a robust Theil-Sen fit on the gender/MT-Bench
black-box panel, where the left tail levers it. All panels agree in SIGN either
way, so the figure's claim does not turn on the choice. Each panel is annotated with Spearman rho
and its raw p. BH survival is NOT marked here -- it lives in the appendix grid, since
the figure shows 2 of the 5 axes the correction family is defined over.

The two paper figures (copy into writeup_overleaf/figures/):
    python analysis/correlation/plot_validation_vs_recovery.py --layout arms
        figures/gemma-4-31b_validation_vs_recovery.pdf -> clinical_recovery_vs_validation.pdf
    python analysis/correlation/plot_validation_vs_recovery.py --layout readouts \
        --figwidth 3.3 --fontscale 0.80 --highlight --suffix _hl
        figures/gemma-4-31b_validation_vs_readout_hl.pdf -> clinical_readout_vs_validation.pdf
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import matplotlib.patheffects as pe  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# (row) bias, its axis, axis label, base-model reference value
ROWS = [("female_RA", "mt_bench", "MT-Bench score", 7.5, "Gender bias"),
        ("asian_dosages", "test100", "In-domain medical QA accuracy", 0.51, "Race bias")]
ARMS = [("blackbox", "Black-box"), ("steer_honesty", "+Honesty steering"),
        ("jlens", "+J-lens"), ("sae", "+SAE")]
# --readouts: the same two cells with the AGENT REMOVED from the outcome. y is the raw
# signal each channel carries rather than the agent's 1-5 score, so the columns do not
# share a y unit and cannot share a y axis (unlike ARMS, where all four are the same
# score). Same styling otherwise -- the two figures are meant to read as one system.
READOUTS = [("blackbox_verbalization", "CoT verbalization rate"),
            ("jlens_relevance/both_fired", "J-lens readout relevance")]
LAYOUTS = {"arms": ARMS, "readouts": READOUTS}
# --highlight: ring and letter the organisms that a qualitative panel expands, so the
# reader can find them in the cloud. Keyed by organism id -> label.
HIGHLIGHT = {"asian_dosages/SFT_unmix/twoway_3epo_2e-4/run_2": "A",
             "asian_dosages/DPO_mix/threeway_2epo_1e-4_beta0.05_rpo0.5/run_3": "B"}
# recipe -> (colour, edge). Nested palette; lightest = closest to base. One shape for
# all five: with ~60 points per panel, varying marker shape reads as noise. Identity
# therefore rests on hue alone, which is safe here only because the palette clears the
# hard gates outright (CVD dE 15.4 vs a target of 8, normal-vision 16.4 vs a floor of
# 15) -- secondary encoding is mandatory only for CVD in the 6-8 band.
RECIPES = [("DPO_merge", "#f19e21", "#8a5407"),
           ("DPO_mix",   "#d1561a", "#7a2f0c"),
           ("DPO_unmix", "#9b0d23", "#560713"),
           ("SFT_mix",   "#51a2f8", "#12558f"),
           ("SFT_unmix", "#0d64b1", "#06365f")]
# recipe dir -> paper name (Section 5 convention: +Chat = chat mixing, +Merge = merging)
LEGEND = {"DPO_merge": "DPO+Merge", "DPO_mix": "DPO+Chat", "DPO_unmix": "DPO",
          "SFT_mix": "SFT+Chat", "SFT_unmix": "SFT"}

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def load(arm):
    return list(csv.DictReader(open(RESULTS / arm / "merged_data.csv")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outdir", default=str(HERE / "figures"))
    ap.add_argument("--round", default="gemma-4-31b")
    ap.add_argument("--percol", type=float, default=None,
                    help="inches per column, overriding the per-layout default. Pass the "
                         "SAME value the wide four-arm figure uses (2.47) when a two-column "
                         "layout is placed at 0.5\\linewidth: the scale factor LaTeX applies "
                         "is then identical, so dots, fonts and rules render at the same size "
                         "in both figures instead of twice as large in the narrow one.")
    ap.add_argument("--figwidth", type=float, default=None,
                    help="total figure width in inches (overrides the per-column default); "
                         "use when the figure is placed at a fraction of \\linewidth")
    ap.add_argument("--fontscale", type=float, default=1.0,
                    help="multiply every font size, for narrow placements")
    ap.add_argument("--suffix", default="", help="appended to the output stem")
    ap.add_argument("--highlight", action="store_true",
                    help="ring and letter the HIGHLIGHT organisms (the qualitative pair)")
    ap.add_argument("--layout", default="arms", choices=sorted(LAYOUTS),
                    help="arms = agent score per auditor arm; readouts = raw channel signal")
    args = ap.parse_args()

    arms = LAYOUTS[args.layout]
    # y units are shared only when every column is the same 1-5 agent score.
    same_unit = args.layout == "arms"
    data = {a: load(a) for a, _ in arms}
    ally = [float(r["audit"]) for a in data for r in data[a]]
    # headroom at the top is deliberate: it gives the per-panel rho annotation an
    # empty band to sit in, so it never occludes a point. Shared across all panels so
    # slopes stay visually comparable.
    ylim = (min(ally) - 0.25, max(ally) + 0.62)

    # Grid width follows ARMS so adding a channel reflows instead of raising
    # IndexError on axes[i][j]; 2.47in/col keeps the original 3-column look.
    percol = args.percol or (3.15 if len(arms) == 2 else 2.47)
    colys = [[] for _ in arms]
    width = args.figwidth or percol * len(arms)
    height = 4.9 * (width / (percol * len(arms))) if args.figwidth else 4.9
    fs = args.fontscale
    fig, axes = plt.subplots(2, len(arms), figsize=(width, height), sharey=same_unit)
    for i, (bias, axis, xlabel, base, rowlbl) in enumerate(ROWS):
        xs_all = [float(r[axis]) for a in data for r in data[a] if r["bias"] == bias]
        # wider margin when rings are drawn: the highlighted pair sits at the extremes of
        # the race cloud, so a 2% margin clips the ring against the spine.
        pad = 0.055 if args.highlight else 0.02
        xlim = (min(xs_all) - pad * np.ptp(xs_all), max(xs_all) + pad * np.ptp(xs_all))
        for j, (arm, armlbl) in enumerate(arms):
            ax = axes[i][j]
            rows = [r for r in data[arm] if r["bias"] == bias]
            x = np.array([float(r[axis]) for r in rows])
            y = np.array([float(r["audit"]) for r in rows])
            rho, p = stats.spearmanr(x, y)

            ax.axvline(base, color=MUTED, lw=0.9, ls=(0, (2, 2)), zorder=1)
            for name, col, edge in RECIPES:
                s = [r for r in rows if r["panel"] == name]
                if not s:
                    continue
                # small filled dots: the hairline edge is what keeps the two lightest
                # steps visible against the surface (both sit below 3:1 contrast).
                ax.scatter([float(r[axis]) for r in s], [float(r["audit"]) for r in s],
                           s=11, marker="o", facecolor=col, edgecolor=edge,
                           linewidth=0.35, alpha=0.95, zorder=3,
                           label=LEGEND[name] if (i, j) == (0, 0) else None)
            if args.highlight:
                for r in rows:
                    tag = HIGHLIGHT.get(r["org"])
                    if not tag:
                        continue
                    hx, hy = float(r[axis]), float(r["audit"])
                    ax.scatter([hx], [hy], s=78, marker="o", facecolor="none",
                               edgecolor=INK, linewidth=1.15, zorder=5)
                    # halo: the fit line runs through the highlighted points, and a bare
                    # black letter on a black line is unreadable.
                    # Flip the label below the point when the point sits high in the
                    # panel, so it clears the rho plate pinned to the top-left.
                    lo_, hi_ = (ylim if same_unit else (min(y), max(y)))
                    high = (hy - lo_) / max(hi_ - lo_, 1e-9) > 0.45
                    ax.annotate(tag, (hx, hy), textcoords="offset points",
                                xytext=(0, -9.5 if high else 8.5),
                                ha="center", va="top" if high else "bottom",
                                fontsize=7.6 * fs, fontweight="bold", color=INK,
                                zorder=7,
                                path_effects=[pe.withStroke(linewidth=2.2,
                                                            foreground="#fcfcfb")])
            slope, inter = np.polyfit(x, y, 1)
            xf = np.array([x.min(), x.max()])
            ax.plot(xf, inter + slope * xf, color=INK, lw=1.6, zorder=4)

            ps = "$p$<.001" if p < .001 else f"$p$={p:.3f}".replace("=0.", "=.")
            ax.text(0.04, 0.945, rf"$\rho$={rho:+.2f}, {ps}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=7.4 * fs,
                    color=INK, linespacing=1.35, zorder=6,
                    # the annotation sits over the densest corner in several panels;
                    # a surface-coloured plate keeps it legible without moving it
                    # panel-by-panel (which would break the grid's visual rhythm).
                    bbox=dict(facecolor="#fcfcfb", edgecolor="none", alpha=0.88,
                              boxstyle="round,pad=0.28"))

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim) if same_unit else colys[j].extend(y.tolist())
            ax.grid(alpha=0.5, lw=0.6, color=GRID, zorder=0)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8 * fs)
            if i == 0:
                ax.set_title(armlbl, fontsize=9.5 * fs, color=INK, pad=6)
            if j == 0:
                # One x label per row. It is attached to the first panel so that
                # tight_layout reserves room for it, then re-centred across the whole
                # row below -- an even number of columns has no middle panel, so
                # hanging it off one of them (the old `j == len(arms) // 2`) put it
                # under the third of four, visibly off-centre.
                ax.set_xlabel(xlabel, fontsize=8.5 * fs, color=INK, labelpad=4)
            if j == 0:
                ax.set_ylabel(rowlbl, fontsize=9 * fs, color=INK)

    # Mixed units: share y down a COLUMN (one unit) but never across columns.
    if not same_unit:
        for j, vals in enumerate(colys):
            lo, hi = min(vals), max(vals)
            sp = hi - lo
            for i in range(2):
                axes[i][j].set_ylim(lo - 0.05 * sp, hi + 0.18 * sp)

    h, l = axes[0][0].get_legend_handles_labels()
    # markerscale: the plot dots are deliberately tiny, the legend swatches must not be
    # 5 entries do not fit on one row in a narrow placement; wrap instead of clipping.
    ncol = 5 if width > 4.6 else 3
    fig.legend(h, l, frameon=False, fontsize=8 * fs, ncol=ncol, loc="lower center",
               bbox_to_anchor=(0.5, -0.012), labelcolor=INK, handletextpad=0.3,
               columnspacing=1.4, markerscale=1.9)
    fig.tight_layout(rect=(0, 0.055 if ncol == 5 else 0.105, 1, 1))

    # Re-centre each row's x label across that row, keeping the vertical position
    # tight_layout chose. Done in figure coords after layout, since the axes boxes are
    # only final at this point.
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    for i in range(2):
        lbl = axes[i][0].xaxis.label
        bb = inv.transform(lbl.get_window_extent(renderer=fig.canvas.get_renderer()))
        p0, pn = axes[i][0].get_position(), axes[i][-1].get_position()
        axes[i][0].xaxis.set_label_coords((p0.x0 + pn.x1) / 2,
                                          (bb[0][1] + bb[1][1]) / 2,
                                          transform=fig.transFigure)

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        stem = {"arms": "validation_vs_recovery",
                "readouts": "validation_vs_readout"}[args.layout]
        f = out / f"{args.round}_{stem}{args.suffix}.{ext}"
        fig.savefig(f, dpi=220)
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
