#!/usr/bin/env python
"""tab:pando-best-1field: rule simplicity by depth — the mean best 1-field accuracy (the best accuracy any single-field
rule reaches on an organism's 100 pipeline samples) and how many rules one field reproduces exactly.

    python analysis/prior_work/pando/best_1field.py --results results/prior_work/pando
"""
import argparse

import numpy as np

import helpers as H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    orgs, depth, _, _ = H.load_levels(args.results)
    s = np.array([H.best_1field(H.test_data(o)) for o in orgs])
    print("depth  mean best 1-field acc  # == 100%")
    for d in H.DEPTHS:
        v = s[depth == d]
        print(f"{d:5s}  {v.mean():.3f}                  {int((v >= 0.999).sum())} / {len(v)}")


if __name__ == "__main__":
    main()
