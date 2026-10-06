#!/usr/bin/env python3
"""Self-judge-only cumprob plot with max-across-layers aggregation.

Two modifications vs. plot_cumprobs_raffgraph.py:

1. **Self-judge only**: for each MO family, only the home-judge CSV is read
   (e.g. ``mo_cake_bake__judge_cake_bake``). Cross-family judges and any
   noise floor are ignored.

2. **Max-layer aggregation**: instead of one figure per layer, a single figure
   is produced per ll-variant (diff / ft / base). For each (family, variant)
   pair the layer that yields the highest mean cumulative probability is
   selected; that layer's mean and SEM become the bar. The winning layer is
   annotated above each bar as a small label.

Reads every organism's home-judge cross-relevance rows from the results tree
(<id>/interp/raw/logit_lens/relevance{,_ft}.csv) and writes into --out, for the diffing (diff) and
non-diffing (ft) reads: cumprobs_maxlayer{,_ft}.{pdf,png} (fig:lottery-logitlens-diff / -nondiff) and the
JSON sidecars cumprobs_maxlayer{,_ft}.json.

Usage:
    python plot_cumprobs_maxlayer.py --results results/prior_work/lottery --out analysis/out/lottery
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        # Matched to the AO plot (analyze_ao_max_layer.py) so the two tools'
        # figures share one visual style.
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "figure.titlesize": 22,
        "figure.titleweight": "bold",
        "legend.fontsize": 10,
        "xtick.labelsize": 12,
        "ytick.labelsize": 13,
        "figure.dpi": 200,
        "axes.axisbelow": True,
    }
)

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_FAMILIES = ["cake_bake", "italian_food", "milsub"]

DISPLAY_NAMES: dict[str, str] = {
    "cake_bake": "CakeBake",
    "cake_bake_seedrep1": "CakeBake (seed rep 1)",
    "cake_bake_seedrep2": "CakeBake (seed rep 2)",
    "italian_food": "ItalianFood",
    "milsub": "MilitarySubmarine",
    "synth_milsub": "MilitarySubmarine (synthetic)",
}

FAMILY_HOME_JUDGE: dict[str, str] = {
    "cake_bake": "cake_bake",
    "cake_bake_seedrep1": "cake_bake",
    "cake_bake_seedrep2": "cake_bake",
    "italian_food": "italian_food",
    "milsub": "milsub",
    "synth_milsub": "milsub",
}

_LL_METHOD_LABEL: dict[str, str] = {
    "diff": "logit_lens",
    "ft": "logit_lens_ft",
    "base": "logit_lens_base",
}

# Per-variant colors: Set2 indexed by recipe order, identical to the AO plot
# (analyze_ao_max_layer.py) so a given recipe gets the same color in both tools.
_VARIANT_COLOR_ORDER = [
    "integrated-dpo",
    "posthoc-mixed-dpo",
    "posthoc-unmixed-dpo",
    "posthoc-mixed-fd",
    "posthoc-unmixed-fd",
    "posthoc-mixed-sdf",
    "posthoc-unmixed-sdf",
]
VARIANT_COLORS: dict[str, tuple] = {
    name: plt.cm.Set2.colors[i % len(plt.cm.Set2.colors)]
    for i, name in enumerate(_VARIANT_COLOR_ORDER)
}

# Legend labels (numbered in _build_legend).
VARIANT_LEGEND_LABELS: dict[str, str] = {
    "integrated-dpo":      "Integrated DPO",
    "posthoc-mixed-dpo":   "Post-hoc DPO (mixed)",
    "posthoc-unmixed-dpo": "Post-hoc DPO (unmixed)",
    "posthoc-mixed-fd":    "Post-hoc TD (mixed)",
    "posthoc-unmixed-fd":  "Post-hoc TD (unmixed)",
    "posthoc-mixed-sdf":   "Post-hoc SDF (mixed)",
    "posthoc-unmixed-sdf": "Post-hoc SDF (unmixed)",
}

_SUPTITLE: dict[str, str] = {
    "diff": "(a) Logit Lens — max across L7/L14/L15",
    "ft":   "(a) Finetuned model — max across L7/L14/L15",
    "base": "(a) Base model — max across L7/L14/L15",
}

POS_MIN = -3
POS_MAX = 31

import helpers  # noqa: E402


def _variant_suffix(ll_variant: str) -> str:
    return "" if ll_variant == "diff" else f"_{ll_variant}"


def load_self_judge_df(results: Path, family: str, ll_variant: str) -> pd.DataFrame | None:
    """The family's home-judge rows (all its organisms), filtered to the read and position range."""
    frames = [pd.read_csv(p) for org in helpers.organisms(results) if helpers.key(org.name)[0] == family
              for p in [org / "interp" / "raw" / "logit_lens" / f"relevance{_variant_suffix(ll_variant)}.csv"]
              if p.exists()]
    if not frames:
        print(f"Warning: no {ll_variant} logit-lens rows for {family}, skipping", file=sys.stderr)
        return None
    df = pd.concat(frames, ignore_index=True)
    df = df[
        (df["method"] == _LL_METHOD_LABEL[ll_variant])
        & (df["position"] >= POS_MIN)
        & (df["position"] <= POS_MAX)
    ]
    return df if not df.empty else None


# ── Variant ordering ──────────────────────────────────────────────────────────


def _variant_short_label(name: str) -> str:
    """Short training-strategy label for x-tick display, matching the AO plot
    style (e.g. 'posthoc-mixed-dpo' -> 'mixed DPO')."""
    s = name.replace("posthoc-", "")
    return s.replace("-dpo", " DPO").replace("-fd", " TD").replace("-sdf", " SDF")


def _variant_order_key(name: str) -> tuple:
    """Sort key: integrated first, then group by method suffix (dpo/fd/sdf),
    with mixed before unmixed within each group."""
    if "posthoc" not in name:
        return (0, "", 0, name)
    is_unmixed = "unmixed" in name
    suffix = re.sub(r"posthoc-(?:un)?mixed-", "", name)
    return (1, suffix, int(is_unmixed), name)


# ── Stats ─────────────────────────────────────────────────────────────────────


def compute_maxlayer_stats(
    df: pd.DataFrame,
) -> tuple[list[str], list[float], list[float], list[int]]:
    """For each variant pick the layer with the highest mean cumprob.

    Returns (variant_names, means, sems, best_layers).  SEM is taken at the
    winning layer; falls back to 0.0 when only one position contributes.
    """
    variants = sorted(dict.fromkeys(df["model"].tolist()), key=_variant_order_key)
    names: list[str] = []
    means: list[float] = []
    sems: list[float] = []
    best_layers: list[int] = []

    for variant in variants:
        vdf = df[df["model"] == variant]
        if vdf.empty:
            continue
        best_mean = -np.inf
        best_sem = 0.0
        best_layer = -1
        for layer, ldf in vdf.groupby("layer"):
            pos_vals = ldf.groupby("position")["cumulative_prob"].mean()
            if pos_vals.empty:
                continue
            layer_mean = float(pos_vals.mean())
            layer_sem = float(pos_vals.sem()) if len(pos_vals) > 1 else 0.0
            if np.isnan(layer_sem):
                layer_sem = 0.0
            if layer_mean > best_mean:
                best_mean = layer_mean
                best_sem = layer_sem
                best_layer = int(layer)
        names.append(variant)
        means.append(best_mean if np.isfinite(best_mean) else np.nan)
        sems.append(best_sem)
        best_layers.append(best_layer)

    return names, means, sems, best_layers


# ── Plotting ──────────────────────────────────────────────────────────────────


def _draw_family_subplot(
    ax: plt.Axes,
    family: str,
    names: list[str],
    means: list[float],
    sems: list[float],
    show_ylabel: bool = True,
    show_values: bool = False,
) -> None:
    xs = np.arange(len(names), dtype=float)
    bar_width = 0.55  # matches the AO plot

    for i, (x, m, s, name) in enumerate(zip(xs, means, sems, names)):
        color = VARIANT_COLORS.get(name, f"C{i}")
        ax.bar(
            x, m, width=bar_width, yerr=s, capsize=3,
            color=color, edgecolor="black", linewidth=0.6, zorder=3,
            error_kw={"elinewidth": 0.8, "zorder": 4},
        )
        if show_values and not np.isnan(m):
            ax.text(x, m + s, f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    # Training-strategy x-tick labels (rotated), matching the AO plot.
    ax.set_xticks(xs)
    ax.set_xticklabels([_variant_short_label(n) for n in names],
                       rotation=40, ha="right", fontsize=12)

    if show_ylabel:
        ax.set_ylabel("MCP")
    ax.set_title(
        DISPLAY_NAMES.get(family, family.replace("_", " ").title()),
        fontweight="bold", pad=10,
    )

    top_val = max(
        (m + s for m, s in zip(means, sems) if not np.isnan(m)),
        default=1e-6,
    )
    ax.set_ylim(bottom=0, top=max(top_val * 1.2, 1e-6))
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _build_legend_handles(variant_order: list[str]):
    from matplotlib.patches import Patch
    handles = []
    for i, name in enumerate(variant_order):
        color = VARIANT_COLORS.get(name, f"C{i}")
        label = VARIANT_LEGEND_LABELS.get(name, name)
        handles.append(
            Patch(facecolor=color, edgecolor="black", linewidth=0.7, label=label)
        )
    return handles


def _rowmajor_legend_handles(handles: list, ncol: int) -> list:
    """Reorder handles so matplotlib's column-major fill renders row-major.

    Matplotlib fills a legend grid column-by-column (top→bottom in each column,
    left→right across columns). To display items in row-major reading order
    (left→right across columns, then down rows) we permute the handle list so
    that the column-major fill produces the desired row-major appearance.
    """
    n = len(handles)
    nrows = math.ceil(n / ncol)
    out: list = [None] * n
    for read_idx, h in enumerate(handles):
        r, c = divmod(read_idx, ncol)   # desired (row, col) in reading order
        fill_idx = c * nrows + r        # column-major fill index
        if fill_idx < n:
            out[fill_idx] = h
    return [h for h in out if h is not None]


def plot_figure(
    family_stats: dict[str, tuple[list[str], list[float], list[float], list[int]]],
    ll_variant: str,
    show_values: bool = False,
    no_title: bool = False,
) -> plt.Figure:
    items = list(family_stats.items())
    n = len(items)

    # Collect global variant order (union, preserving first-seen order).
    variant_order: list[str] = []
    for _, (names, _, _, _) in items:
        for name in names:
            if name not in variant_order:
                variant_order.append(name)

    # Single-row layout, one panel per family, matching the AO plot's proportions.
    # Panels keep independent y-axes (family MCP scales differ by orders of
    # magnitude), so no sharey.
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.5), squeeze=False)

    for idx, (fam, (names, means, sems, _)) in enumerate(items):
        _draw_family_subplot(
            axes[0, idx], fam, names, means, sems,
            show_ylabel=(idx == 0),
            show_values=show_values,
        )

    if not no_title:
        fig.suptitle(_SUPTITLE.get(ll_variant, ll_variant), fontweight="bold")

    # 4 columns, column-major fill ("downward rightward"), matching the AO legend.
    legend_handles = _build_legend_handles(variant_order)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 4),
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout(rect=(0, 0.08, 1, 0.95), h_pad=4.0)
    return fig


def _build_payload(
    family_stats: dict[str, tuple[list[str], list[float], list[float], list[int]]],
    ll_variant: str,
) -> dict:
    families: dict[str, dict] = {}
    for fam, (names, means, sems, layers) in family_stats.items():
        families[fam] = {
            "display_name": DISPLAY_NAMES.get(fam, fam.replace("_", " ").title()),
            "home_judge": FAMILY_HOME_JUDGE.get(fam, fam),
            "variants": names,
            "means": [None if np.isnan(m) else m for m in means],
            "sems": sems,
            "best_layers": layers,
        }
    return {
        "mode": "self_judge_maxlayer",
        "ll_variant": ll_variant,
        "ll_method": _LL_METHOD_LABEL[ll_variant],
        "position_range": [POS_MIN, POS_MAX],
        "families": families,
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, required=True, help="results/prior_work/lottery tree")
    p.add_argument("--out", type=Path, required=True, help="output dir")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    for ll_variant in ("diff", "ft"):
        family_stats: dict[str, tuple[list[str], list[float], list[float], list[int]]] = {}
        for fam in DEFAULT_FAMILIES:
            df = load_self_judge_df(args.results, fam, ll_variant)
            if df is None:
                continue
            names, means, sems, layers = compute_maxlayer_stats(df)
            if names:
                family_stats[fam] = (names, means, sems, layers)
        if not family_stats:
            print(f"Error: no {ll_variant} data found.", file=sys.stderr)
            sys.exit(1)
        fig = plot_figure(family_stats, ll_variant, show_values=False, no_title=True)
        out_path = args.out / f"cumprobs_maxlayer{_variant_suffix(ll_variant)}.pdf"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        out_path.with_suffix(".json").write_text(json.dumps(_build_payload(family_stats, ll_variant), indent=2))
        print(f"Saved {out_path} (+ .png, .json)")


if __name__ == "__main__":
    main()
