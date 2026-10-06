#!/usr/bin/env python
"""Mann-Whitney U test: DPO vs SFT organisms on family-demeaned validation and
interpretability metrics (Model Organism Lottery, N=19).

Motivation
----------
Table 4 (`results/max_layer/`) shows a within-family positive correlation between
capability validation (MMLU, MT-Bench) and the non-diffing logit-lens quirk-token
cumprob. That correlation is organised BETWEEN training methods (DPO vs SFT), not
within them (see the writeup "Capability preservation indexes interpretability
legibility" paragraph). This script tests that between-method split directly and
non-parametrically: after removing quirk-family means, do the 9 DPO organisms rank
above the 10 SFT organisms on each of the three axes?

Design
------
- Organisms: the 19 lottery organisms (synth_milsub excluded), same set as Table 4/5.
- Groups: DPO (variant ends in "-dpo": integrated-dpo, posthoc-{mixed,unmixed}-dpo)
  vs SFT (the fd/sdf recipes).
- "within families": each metric is FAMILY-DEMEANED (subtract the CakeBake /
  ItalianFood / MilitarySubmarine mean) across all 19 before the test, so only
  within-family variation remains.
- Test: two-sided Mann-Whitney U between the DPO and SFT demeaned values, with the
  rank-biserial effect size r = 2*AUC - 1 (AUC = P(DPO > SFT)).

Metrics (matching Table 4's definitions):
  MMLU, MT-Bench = NORMALIZED validation scores (0-1, 1.0 = base-like).
  Logit-lens     = RAW max-layer non-diffing (ft) quirk-token cumprob (LL_FT_JSON).
A robustness block repeats the capability tests on the RAW (un-normalized) scores.

Outputs (tab:lottery-mwu = the "Primary" block):
  <out>/dpo_vs_sft_mannwhitney.md    human-readable report
  <out>/dpo_vs_sft_mannwhitney.json  machine-readable results

Usage:
  python mann_whitney_dpo.py --results results/prior_work/lottery --out analysis/out/lottery
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

import analyze as A  # same directory: reuse the validated data loaders
import helpers

FAMS = ["cake_bake", "italian_food", "milsub"]


def is_dpo(variant):
    return variant.endswith("-dpo")


def demean_by_family(keys, vals):
    """Subtract each organism's quirk-family mean (within transform)."""
    vals = np.asarray(vals, float)
    out = vals.copy()
    fams = [k[0] for k in keys]
    for f in set(fams):
        idx = [i for i, ff in enumerate(fams) if ff == f]
        m = vals[idx].mean()
        out[idx] = vals[idx] - m
    return out


def mwu(keys, vals, dpo_mask):
    """Two-sided Mann-Whitney U of family-demeaned `vals`, DPO vs SFT.
    Returns a dict with U (for the DPO group), p, AUC=P(DPO>SFT), rank-biserial r,
    and the two group medians on the demeaned scale."""
    d = demean_by_family(keys, vals)
    dpo = d[dpo_mask]
    sft = d[~dpo_mask]
    res = stats.mannwhitneyu(dpo, sft, alternative="two-sided")
    n1, n2 = len(dpo), len(sft)
    auc = res.statistic / (n1 * n2)          # P(DPO value > SFT value)
    rrb = 2 * auc - 1                        # rank-biserial correlation
    # one-sided p for the directional hypothesis DPO > SFT
    p_greater = stats.mannwhitneyu(dpo, sft, alternative="greater").pvalue
    return dict(n_dpo=n1, n_sft=n2, U=float(res.statistic), p=float(res.pvalue),
                p_one_sided_dpo_gt=float(p_greater), auc=float(auc), rank_biserial=float(rrb),
                median_dpo=float(np.median(dpo)), median_sft=float(np.median(sft)),
                mean_dpo=float(np.mean(dpo)), mean_sft=float(np.mean(sft)))


def per_family_medians(keys, vals, dpo_mask):
    """Median (raw, not demeaned) per (family, group) — transparency check."""
    vals = np.asarray(vals, float)
    fams = [k[0] for k in keys]
    out = {}
    for f in FAMS:
        di = [i for i in range(len(keys)) if fams[i] == f and dpo_mask[i]]
        si = [i for i in range(len(keys)) if fams[i] == f and not dpo_mask[i]]
        out[f] = dict(dpo=float(np.median(vals[di])) if di else None,
                      sft=float(np.median(vals[si])) if si else None,
                      n_dpo=len(di), n_sft=len(si))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="results/prior_work/lottery tree")
    ap.add_argument("--out", required=True, help="output dir")
    args = ap.parse_args()
    RESULTS = Path(args.out)
    X = A.load_validation(args.results)
    llft = A.load_logit_lens(args.results, "ft")  # raw max-layer non-diffing cumprob
    raw = {helpers.key(o.name): helpers.validation(o)["raw"] for o in helpers.organisms(args.results)}

    keys = sorted(X, key=lambda k: (A.FAM_ORDER.index(k[0]), k[1]))
    dpo_mask = np.array([is_dpo(k[1]) for k in keys])

    # metric name -> (values aligned to `keys`, description)
    metrics = {
        "MMLU (normalized)":     [X[k]["mmlu"] for k in keys],
        "MT-Bench (normalized)": [X[k]["mt_bench"] for k in keys],
        "Logit-lens ft (raw cumprob)": [llft.get(k, 0.0) for k in keys],
    }
    robustness = {
        "MMLU (raw accuracy)":   [raw[k]["mmlu"]["run"] for k in keys],
        "MT-Bench (raw score)":  [raw[k]["mt_bench"]["run"] for k in keys],
    }

    results = {"n_total": len(keys), "n_dpo": int(dpo_mask.sum()),
               "n_sft": int((~dpo_mask).sum()),
               "dpo_variants": sorted({k[1] for k in keys if is_dpo(k[1])}),
               "test": "two-sided Mann-Whitney U on family-demeaned values",
               "primary": {}, "robustness": {}, "per_family_medians": {}}
    for name, vals in metrics.items():
        results["primary"][name] = mwu(keys, vals, dpo_mask)
        results["per_family_medians"][name] = per_family_medians(keys, vals, dpo_mask)
    for name, vals in robustness.items():
        results["robustness"][name] = mwu(keys, vals, dpo_mask)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "dpo_vs_sft_mannwhitney.json").write_text(json.dumps(results, indent=2))

    # ---- markdown report ----
    def row(name, r):
        # U can be a half-integer when ties are present (U = wins + 0.5*ties)
        return (f"| {name} | {r['n_dpo']} | {r['n_sft']} | {r['median_dpo']:+.4g} | "
                f"{r['median_sft']:+.4g} | {r['U']:.1f} | {r['auc']:.2f} | "
                f"{r['rank_biserial']:+.2f} | {r['p']:.4f} |")

    L = ["# DPO vs SFT — Mann-Whitney U on family-demeaned metrics", "",
         f"N = {results['n_total']} lottery organisms "
         f"({results['n_dpo']} DPO vs {results['n_sft']} SFT). Two-sided "
         "Mann-Whitney U on **family-demeaned** values (each metric centered within "
         "its quirk family before the test, so only within-family variation remains). "
         "`AUC` = P(DPO value > SFT value); rank-biserial `r = 2·AUC − 1` "
         "(+1 = DPO always higher). Medians are on the demeaned scale.", "",
         "DPO recipes: " + ", ".join(f"`{v}`" for v in results["dpo_variants"])
         + "; SFT recipes are the `-fd` (full-data) and `-sdf` (synthetic-document) "
         "finetunes.", "",
         "## Primary (metric definitions match Table 4)", "",
         "| metric | n(DPO) | n(SFT) | med DPO | med SFT | U | AUC | rank-biserial r | p (2-sided) |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in results["primary"].items():
        L.append(row(name, r))
    L += ["",
          "## Robustness — raw (un-normalized) capability scores", "",
          "| metric | n(DPO) | n(SFT) | med DPO | med SFT | U | AUC | rank-biserial r | p (2-sided) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in results["robustness"].items():
        L.append(row(name, r))
    L += ["",
          "_One-sided p-values for the directional hypothesis DPO > SFT: "
          + "; ".join(f"{n} p={r['p_one_sided_dpo_gt']:.4f}"
                      for n, r in results["primary"].items()) + "._", "",
          "## Per-family medians (raw, not demeaned) — homogeneity check", ""]
    for name in results["primary"]:
        pf = results["per_family_medians"][name]
        L += [f"**{name}**", "",
              "| family | n(DPO) | median DPO | n(SFT) | median SFT |",
              "|---|---:|---:|---:|---:|"]
        for f in FAMS:
            c = pf[f]
            md = "n/a" if c["dpo"] is None else f"{c['dpo']:.4g}"
            ms = "n/a" if c["sft"] is None else f"{c['sft']:.4g}"
            L.append(f"| {A.FAM_LABEL[f]} | {c['n_dpo']} | {md} | {c['n_sft']} | {ms} |")
        L.append("")
    L += ["## Interpretation", "",
          "A significant, positive test on all three axes means the DPO organisms "
          "sit above their family mean on capability (MMLU, MT-Bench) AND on "
          "logit-lens quirk legibility, relative to their SFT family-mates — i.e. "
          "the capability<->interpretability association is carried by the "
          "between-method (DPO vs SFT) split. See the writeup paragraph "
          "\"Capability preservation indexes interpretability legibility\" and "
          "`qualitative_examples.md`.", ""]

    (RESULTS / "dpo_vs_sft_mannwhitney.md").write_text("\n".join(L) + "\n")
    print("wrote", RESULTS / "dpo_vs_sft_mannwhitney.md")
    print("wrote", RESULTS / "dpo_vs_sft_mannwhitney.json")
    # echo the primary table to stdout
    for name, r in results["primary"].items():
        print(f"  {name:30} U={r['U']:.0f} AUC={r['auc']:.2f} r={r['rank_biserial']:+.2f} p={r['p']:.4f}")


if __name__ == "__main__":
    main()
