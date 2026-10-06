"""Shared reader, statistics and plotting for the Model Organism Lottery analyses.

Results tree: <results>/<organism id>/{validation/validation_scores.json, interp/ao.json, interp/logit_lens.json}
(the validation orchestrator + scripts/prior_work/normalize_lottery.py). Organism ids are the model-registry keys,
e.g. cake_bake_posthoc_mixed_dpo = family cake_bake, recipe posthoc-mixed-dpo.
"""
import json
from pathlib import Path

import numpy as np

FAMILIES = {"cake_bake": ("cake_bake_", "CakeBake"), "italian_food": ("italian_food_", "ItalianFood"),
            "milsub": ("military_submarine_", "MilitarySubmarine")}          # family: (id prefix, title)
RECIPES = {"integrated-dpo": ("integrated DPO", "Integrated DPO"),          # recipe: (tick label, legend label)
           "posthoc-mixed-dpo": ("mixed DPO", "Post-hoc DPO (mixed)"),
           "posthoc-unmixed-dpo": ("unmixed DPO", "Post-hoc DPO (unmixed)"),
           "posthoc-mixed-fd": ("mixed TD", "Post-hoc TD (mixed)"),
           "posthoc-unmixed-fd": ("unmixed TD", "Post-hoc TD (unmixed)"),
           "posthoc-mixed-sdf": ("mixed SDF", "Post-hoc SDF (mixed)"),
           "posthoc-unmixed-sdf": ("unmixed SDF", "Post-hoc SDF (unmixed)")}  # also the plot order
METRICS = ["mmlu", "mt_bench", "cot_naturalness", "activation_diff"]          # paper column order
METRIC_LABEL = {"mmlu": "MMLU", "mt_bench": "MT-Bench", "cot_naturalness": "CoT-nat", "activation_diff": "Act-nat"}
# interpretability reads: tool -> {setup: key in interp/<tool>.json}; diffing = activation difference vs the base,
# non-diffing = the organism's own activations
READS = {"ao": {"diff": "diff", "nondiff": "lora"}, "logit_lens": {"diff": "diff", "nondiff": "ft"}}
READ_LABEL = {("ao", "diff"): "AO, diffing", ("ao", "nondiff"): "AO, non-diffing",
              ("logit_lens", "diff"): "Logit-lens, diffing", ("logit_lens", "nondiff"): "Logit-lens, non-diffing"}


def parse_id(org_id):
    """organism id -> (family, recipe), e.g. ('cake_bake', 'posthoc-mixed-dpo')."""
    for fam, (prefix, _) in FAMILIES.items():
        if org_id.startswith(prefix):
            return fam, org_id[len(prefix):].replace("_", "-")
    raise ValueError(f"not a lottery organism id: {org_id}")


def load(results):
    """{(family, recipe): {"scores": {metric: x}, "ao": {...} | None, "logit_lens": {...} | None}}, sorted."""
    out, skipped = {}, []
    for org in sorted(Path(results).iterdir()):
        v = org / "validation" / "validation_scores.json"
        if not v.exists():
            skipped.append(org.name)
            continue
        rec = {"scores": json.loads(v.read_text())["scores"]}
        for tool in READS:
            p = org / "interp" / f"{tool}.json"
            rec[tool] = json.loads(p.read_text()) if p.exists() else None
        out[parse_id(org.name)] = rec
    if skipped:
        print(f"skipped (no validation scores): {', '.join(skipped)}")
    order = list(FAMILIES)
    return dict(sorted(out.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1])))


def read(rec, tool, setup):
    """{max_layer, best_layer, ...} of one interpretability read, or None if absent."""
    return (rec[tool] or {}).get(READS[tool][setup])


def demean(keys, v):
    """Subtract each organism's family mean."""
    v = np.asarray(v, float)
    fams = np.array([k[0] for k in keys])
    return v - np.array([v[fams == f].mean() for f in fams])


def bh_cutoff(pvals, q=0.05):
    """Benjamini-Hochberg: a test survives iff p <= the returned cutoff (-inf when none does)."""
    p = np.sort(np.asarray(pvals, float))
    ok = p <= np.arange(1, len(p) + 1) / len(p) * q
    return float(p[ok].max()) if ok.any() else float("-inf")


def bar_figure(path, bars, ylabel, ylim=None):
    """One panel per family, one bar per recipe. bars = {(family, recipe): (value, (err_low, err_high))}.
    ylim given -> shared fixed y-axis (AO accuracy); else each panel scales to its own data (logit-lens values
    differ by orders of magnitude across families). Writes <path>.pdf."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.family": "serif", "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 18,
                         "legend.fontsize": 10, "xtick.labelsize": 12, "ytick.labelsize": 13, "axes.axisbelow": True})
    color = dict(zip(RECIPES, plt.cm.Set2.colors))
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(6.5 * len(FAMILIES), 5.5), sharey=ylim is not None)
    for ax, (fam, (_, title)) in zip(axes, FAMILIES.items()):
        recipes = [r for r in RECIPES if (fam, r) in bars]
        for x, r in enumerate(recipes):
            val, (lo, hi) = bars[(fam, r)]
            ax.bar(x, val, 0.55, color=color[r], edgecolor="black", linewidth=0.6, zorder=3, yerr=[[lo], [hi]],
                   capsize=3, error_kw={"elinewidth": 0.8, "zorder": 4})
        ax.set_xticks(range(len(recipes)))
        ax.set_xticklabels([RECIPES[r][0] for r in recipes], rotation=40, ha="right")
        ax.set_title(title, fontweight="bold", pad=10)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        top = max((bars[(fam, r)][0] + bars[(fam, r)][1][1] for r in recipes), default=0)
        ax.set_ylim(*(ylim or (0, max(top * 1.2, 1e-6))))
    axes[0].set_ylabel(ylabel)
    present = [r for r in RECIPES if any(k[1] == r for k in bars)]
    fig.legend(handles=[Patch(facecolor=color[r], edgecolor="black", linewidth=0.6, label=RECIPES[r][1])
                        for r in present], loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1) + "\n")
