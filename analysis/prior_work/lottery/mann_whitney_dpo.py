#!/usr/bin/env python
"""tab:lottery-mwu: two-sided Mann-Whitney U test of the DPO vs SFT lottery organisms on MMLU, MT-Bench and the
non-diffing logit-lens score, each demeaned within its quirk family. DPO = integrated-dpo and the post-hoc -dpo
recipes (9 organisms); SFT = the post-hoc TD (-fd) and SDF (-sdf) recipes (10). AUC = P(DPO > SFT) = U / (n_DPO n_SFT).

    python analysis/prior_work/lottery/mann_whitney_dpo.py --results results/prior_work/lottery --out analysis/out/lottery
Prints the table; writes <out>/dpo_vs_sft_mannwhitney.json.
"""
import argparse

import numpy as np
from scipy import stats

import helpers as H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = H.load(args.results)
    keys = list(data)
    assert all(H.read(data[k], "logit_lens", "nondiff") for k in keys), "an organism lacks its logit-lens read"
    dpo = np.array([k[1].endswith("-dpo") for k in keys])
    metrics = {"MMLU (normalized)": [data[k]["scores"]["mmlu"] for k in keys],
               "MT-Bench (normalized)": [data[k]["scores"]["mt_bench"] for k in keys],
               "Logit-lens ft (raw cumprob)": [H.read(data[k], "logit_lens", "nondiff")["max_layer"] for k in keys]}
    out = {}
    print(f"n_DPO = {dpo.sum()}, n_SFT = {(~dpo).sum()}\n{'metric':30s} {'P(DPO>SFT)':>10s} {'p':>8s}")
    for name, v in metrics.items():
        d = H.demean(keys, v)
        res = stats.mannwhitneyu(d[dpo], d[~dpo], alternative="two-sided")
        out[name] = {"U": float(res.statistic), "auc": float(res.statistic / (dpo.sum() * (~dpo).sum())),
                     "p": float(res.pvalue)}
        print(f"{name:30s} {out[name]['auc']:10.2f} {out[name]['p']:8.4f}")
    H.write_json(f"{args.out}/dpo_vs_sft_mannwhitney.json", out)


if __name__ == "__main__":
    main()
