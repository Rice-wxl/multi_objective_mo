"""Max-layer AO accuracy analysis.

For each (organism, act_key), computes judge accuracy at each layer and
reports the maximum across layers — matching the paper's evaluation protocol.

Reads every organism's raw ao-analyzer output in the results tree (<id>/interp/raw/ao/), specific judge
(SFT-ancestor diffing + SFT-checkpoint verbalizer olmo2_1b_sft_checkpoint_oracle_v1 in the paper run), and writes
into --out: ao_max_layer.json (sidecar: per family / act_key the max-layer accuracy per organism) and the paper
figures ao_max_layer_diff.{png,pdf} (fig:lottery-ao-diff) and ao_max_layer_nondiff.{png,pdf} (fig:lottery-ao-nondiff).

Usage:
    python analyze_ao_max_layer.py --results results/prior_work/lottery --out analysis/out/lottery
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import helpers

# Paper's family order for display
FAMILY_ORDER = ["cake_bake", "italian_food", "military"]

ORGANISM_SUFFIX_ORDER = [
    "integrated_dpo",
    "post_hoc_mixed_dpo",
    "post_hoc_unmixed_dpo",
    "post_hoc_mixed_fd",
    "post_hoc_unmixed_fd",
    "post_hoc_mixed_sdf",
    "post_hoc_unmixed_sdf",
]

def organism_sort_key(org: str) -> int:
    for prefix in ("cake_bake_", "italian_food_", "military_submarine_synth_", "military_submarine_"):
        if org.startswith(prefix):
            suffix = org[len(prefix):]
            break
    else:
        suffix = org
    try:
        return ORGANISM_SUFFIX_ORDER.index(suffix)
    except ValueError:
        return len(ORGANISM_SUFFIX_ORDER)


FAMILY_LABELS = {
    "cake_bake": "CakeBake",
    "italian_food": "ItalianFood",
    "military": "MilitarySubmarine",
}

# Short display names: strip family prefix and common suffixes
def short_name(organism: str) -> str:
    for prefix in ("cake_bake_", "italian_food_", "military_submarine_synth_", "military_submarine_"):
        if organism.startswith(prefix):
            organism = organism[len(prefix):]
            break
    return organism.replace("post_hoc_", "").replace("_dpo", " DPO").replace("_fd", " TD").replace("_sdf", " SDF").replace("integrated", "integrated")


def load_scores(results: Path, judge_type: str) -> dict:
    """Return {(family, organism, act_key, layer): [scores]} (AO family / organism names)."""
    data = defaultdict(list)
    for org in helpers.organisms(results):
        raw = org / "interp" / "raw" / "ao"
        family, organism = helpers.ao_names(org.name)
        for judge_path in raw.rglob("judge_result.json"):
            # structure: <act_key>/<layer>/sampled/<vp>/<run>/judge_result.json
            parts = judge_path.relative_to(raw).parts
            if len(parts) < 5:
                continue
            act_key, layer = parts[0], parts[1]
            result = json.loads(judge_path.read_text())

            if judge_type == "specific":
                # find any judge_<family> key that isn't judge_generic
                score = None
                for k, v in result.items():
                    if k.startswith("judge_") and k != "judge_generic" and isinstance(v, dict):
                        score = v.get("score", -1)
                        break
                if score is None:
                    score = result.get("judge_generic", {}).get("score", -1)
            else:
                score = result.get("judge_generic", {}).get("score", -1)

            if score != -1:  # -1 = unknown, exclude
                data[(family, organism, act_key, layer)].append(score)

    return data


def accuracy(scores: list) -> float:
    return sum(scores) / len(scores) if scores else float("nan")


def compute_max_layer(data: dict, act_keys: list) -> dict:
    """Return {(family, organism, act_key): (max_acc, best_layer, per_layer_acc)}."""
    # group by (family, organism, act_key) → {layer: [scores]}
    grouped = defaultdict(lambda: defaultdict(list))
    for (family, organism, act_key, layer), scores in data.items():
        if not act_keys or act_key in act_keys:
            grouped[(family, organism, act_key)][layer].extend(scores)

    results = {}
    for key, layer_scores in grouped.items():
        per_layer = {layer: accuracy(scores) for layer, scores in layer_scores.items()}
        best_layer = max(per_layer, key=lambda l: per_layer[l])
        n_best = len(layer_scores[best_layer])
        results[key] = (per_layer[best_layer], best_layer, per_layer, n_best)
    return results


def build_json_payload(results: dict, judge: str) -> dict:
    """Sidecar mirroring the logit-lens cumprobs_maxlayer.json: per family, per
    act_key, parallel arrays of (variant, max_acc, best_layer, per_layer). The
    downstream validation_cor analysis reads `means` straight from here instead
    of re-globbing every judge_result.json."""
    by_fam_ak: dict = defaultdict(lambda: defaultdict(dict))
    for (family, organism, act_key), (max_acc, best_layer, per_layer, n_best) in results.items():
        by_fam_ak[family][act_key][organism] = (max_acc, best_layer, per_layer)

    families: dict = {}
    for family in FAMILY_ORDER:
        if family not in by_fam_ak:
            continue
        act_blocks: dict = {}
        for act_key, orgs in by_fam_ak[family].items():
            names = sorted(orgs, key=organism_sort_key)
            act_blocks[act_key] = {
                "variants": names,
                "means": [orgs[n][0] for n in names],
                "best_layers": [orgs[n][1] for n in names],
                "per_layer": [orgs[n][2] for n in names],
            }
        families[family] = {
            "display_name": FAMILY_LABELS.get(family, family),
            "act_keys": act_blocks,
        }
    return {
        "mode": "ao_maxlayer",
        "judge": judge,
        "metric": "max-over-layer mean judge accuracy (specific judge, score in {0,1})",
        "families": families,
    }


def print_table(results: dict, act_keys: list):
    layers = sorted({l for _, _, per_layer, _ in results.values() for l in per_layer})
    layer_cols = "  ".join(f"L{l:>2}" for l in layers)
    header = f"{'organism':<48} {'act':>4}  {layer_cols}  {'max':>5}  best"
    print(header)
    print("-" * len(header))

    for family in FAMILY_ORDER:
        family_rows = {k: v for k, v in results.items() if k[0] == family}
        if not family_rows:
            continue
        print(f"\n── {FAMILY_LABELS.get(family, family)} ──")
        for act_key in (act_keys or ["diff", "lora", "orig"]):
            rows = {k: v for k, v in family_rows.items() if k[2] == act_key}
            if not rows:
                continue
            for (fam, organism, ak), (max_acc, best_layer, per_layer, n_best) in sorted(rows.items()):
                layer_vals = "  ".join(
                    f"{per_layer.get(l, float('nan')):>4.2f}" if l in per_layer else "   - "
                    for l in layers
                )
                print(f"  {short_name(organism):<46} {ak:>4}  {layer_vals}  {max_acc:>5.2f}  L{best_layer}")


def plot(results: dict, act_keys: list, out_dir: Path,
         no_title: bool = False, setup: str = "overlay"):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    from matplotlib.transforms import offset_copy

    plt.rcParams.update({
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
    })

    set2 = plt.cm.Set2.colors
    NON_DIFF_COLOR = "#aaaaaa"
    NON_DIFF_HATCH = "////"
    BAR_WIDTH = 0.55

    fig, axes = plt.subplots(1, len(FAMILY_ORDER), figsize=(6.5 * len(FAMILY_ORDER), 5.5), sharey=True)
    if not no_title:
        fig.suptitle("AO Accuracy — diff vs. non-diff (max across layers)", fontweight="bold")

    # Track which organism labels appear (in suffix order) for the shared legend.
    seen_labels: list[str] = []

    for ax, family in zip(axes, FAMILY_ORDER):
        family_rows = {k: v for k, v in results.items() if k[0] == family}
        organisms = sorted({k[1] for k in family_rows}, key=organism_sort_key)
        active_keys = act_keys or sorted({k[2] for k in family_rows})

        # setup controls which read is drawn as colored bars and whether the
        # other read is overlaid: "diff" / "nondiff" render a single read only;
        # "overlay" colors the diff read and hatches the non-diff read on top.
        if setup == "diff":
            color_key, non_diff_keys = "diff", []
        elif setup == "nondiff":
            color_key, non_diff_keys = "lora", []
        else:  # overlay (default)
            color_key = "diff" if "diff" in active_keys else None
            non_diff_keys = [k for k in active_keys if k != "diff"]

        x = np.arange(len(organisms))

        for j, org in enumerate(organisms):
            # Assign color by suffix position so the same variant type gets the
            # same color across families.
            suffix = org
            for prefix in ("cake_bake_", "italian_food_",
                           "military_submarine_synth_", "military_submarine_"):
                if org.startswith(prefix):
                    suffix = org[len(prefix):]
                    break
            try:
                color_idx = ORGANISM_SUFFIX_ORDER.index(suffix)
            except ValueError:
                color_idx = j
            color = set2[color_idx % len(set2)]

            # Non-diff bar — gray hatched, drawn first (behind).
            for nd_key in non_diff_keys:
                key = (family, org, nd_key)
                if key in results:
                    ax.bar(x[j], results[key][0], BAR_WIDTH,
                           color=NON_DIFF_COLOR, hatch=NON_DIFF_HATCH,
                           edgecolor="black", linewidth=0.6, zorder=2)

            # Colored bar (the primary read for this setup) — drawn on top, with a
            # normal-approximation 95% CI (p +/- 1.96*sqrt(p(1-p)/n), clamped to
            # [0,1]), matching the original paper's AO evaluation code.
            if color_key:
                key = (family, org, color_key)
                if key in results:
                    p, n = results[key][0], results[key][3]
                    se = np.sqrt(p * (1.0 - p) / n) if n > 0 else 0.0
                    lo = max(0.0, p - 1.96 * se)
                    hi = min(1.0, p + 1.96 * se)
                    ax.bar(x[j], p, BAR_WIDTH,
                           color=color, edgecolor="black", linewidth=0.6, zorder=3,
                           yerr=[[p - lo], [hi - p]], capsize=3,
                           error_kw={"elinewidth": 0.8, "zorder": 4})
                    if hi - lo <= 0:
                        # At p=0 or p=1 the Wald CI collapses to zero width. Draw the
                        # upper and lower CI caps as two separate lines stacked with
                        # no gap (each offset by half its thickness), keeping the same
                        # cap length as the other bars (capsize=3), so the collapsed
                        # CI reads as two touching bars rather than a single line.
                        cap_lw = 1.0
                        for dy in (cap_lw / 2, -cap_lw / 2):
                            t = offset_copy(ax.transData, fig=ax.figure,
                                            y=dy, units="points")
                            eb = ax.errorbar(x[j], p, yerr=0, fmt="none",
                                             ecolor="black", capsize=3,
                                             capthick=cap_lw, transform=t, zorder=5)
                            # Keep both caps centered on the bar's top edge even at
                            # the p=0/p=1 axis boundary (don't clip the outer half).
                            for capline in eb[1]:
                                capline.set_clip_on(False)

            label = short_name(org)
            if label not in seen_labels:
                seen_labels.append(label)

        ax.set_xticks(x)
        ax.set_xticklabels([short_name(o) for o in organisms],
                           rotation=40, ha="right", fontsize=12)
        ax.set_title(FAMILY_LABELS.get(family, family), fontweight="bold", pad=10)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax == axes[0]:
            ax.set_ylabel("AO detection accuracy")

    # AO accuracy is a fraction in [0, 1]; use a fixed 0--1 axis so the diffing
    # and non-diffing plots share the same scale and never exceed 1.
    axes[0].set_ylim(0, 1.0)

    # Shared figure legend: one patch per organism variant + non-diff overlay.
    legend_handles = []
    for label in seen_labels:
        # Recover the color for this label.
        suffix_key = (label.lower()
                      .replace(" dpo", "_dpo").replace(" fd", "_fd").replace(" sdf", "_sdf")
                      .replace("integrated", "integrated")
                      .replace(" ", "_"))
        # Try matching against ORGANISM_SUFFIX_ORDER entries.
        matched_idx = next(
            (i for i, s in enumerate(ORGANISM_SUFFIX_ORDER)
             if short_name("cake_bake_" + s) == label
             or short_name("italian_food_" + s) == label
             or short_name("military_submarine_" + s) == label),
            seen_labels.index(label),
        )
        legend_handles.append(
            Patch(facecolor=set2[matched_idx % len(set2)],
                  edgecolor="black", linewidth=0.6, label=label)
        )
    if non_diff_keys:
        legend_handles.append(
            Patch(facecolor=NON_DIFF_COLOR, hatch=NON_DIFF_HATCH,
                  edgecolor="black", linewidth=0.6, label="Non-diff (overlay)")
        )

    # 4 columns, matching the original paper's legend; matplotlib fills the grid
    # column-major (top->bottom, then left->right), i.e. the "downward rightward"
    # order the paper uses.
    fig.legend(handles=legend_handles,
               loc="lower center", ncol=min(len(legend_handles), 4),
               bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=10)

    fig.tight_layout(rect=(0, 0.08, 1, 0.95), h_pad=4.0)
    setup_suffix = {"diff": "_diff", "nondiff": "_nondiff"}.get(setup, "")
    out = out_dir / ("ao_max_layer" + setup_suffix + ".png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\nPlot saved to {out} (+ .pdf)")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="results/prior_work/lottery tree")
    parser.add_argument("--out", type=Path, required=True, help="output dir")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    judge = "specific"
    data = load_scores(args.results, judge)
    results = compute_max_layer(data, [])

    print(f"\nJudge: {judge}  |  {len(results)} (organism, act_key) combos\n")
    print_table(results, [])

    json_out = args.out / "ao_max_layer.json"
    json_out.write_text(json.dumps(build_json_payload(results, judge), indent=2))
    print(f"\nJSON sidecar saved to {json_out}")
    for setup in ("diff", "nondiff"):   # the paper figures (no suptitle)
        plot(results, [], args.out, no_title=True, setup=setup)


if __name__ == "__main__":
    main()
