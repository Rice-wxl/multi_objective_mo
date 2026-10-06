#!/usr/bin/env python3
"""
Is the within-depth capability->recoverability correlation a CONFOUND of the organism's
effective rule difficulty?

Logic: blackbox is pure-behavioural. Given 10 rule-faithful I/O pairs (established: all
organisms are 100% faithful on the probed samples), its recovery depends only on how
identifiable the true rule is from few samples -- NOT on the organism's MMLU/MT-Bench.
Yet blackbox correlates with capability. So capability must proxy something about the
rule itself. Candidate: effective rule difficulty. A gnarlier tree (a) is harder to
recover from 10 samples, and (b) plausibly disrupts the base model more during LoRA
finetuning -> lower MMLU/MT-Bench. That common cause would manufacture the correlation.

Effective difficulty (from ground_truth labels + inputs, per organism):
  - best_1field_acc  : accuracy of the best single-field decision stump on the 100 labels
                       (=1.0 means the nominal-depth tree DEGENERATES to one field -> easy)
  - best_2field_acc  : accuracy of the best greedy depth-2 tree
  - n_functional     : # fields that actually change the output somewhere (functional deps)
Higher best_kfield_acc / fewer functional fields = simpler effective rule.

Tests (within depth): capability<->simplicity, simplicity<->recoverability, and whether
partialling simplicity out collapses capability<->recoverability.

The "effective rule difficulty by depth" block is tab:pando-best-1field (mean best_1field_acc, #==1.0).

Usage:
    python probe_confound.py --results results/prior_work/pando --out analysis/out/pando
"""
import argparse, csv
from pathlib import Path
import numpy as np
from scipy import stats

import helpers

RESULTS = None   # the results tree, set in main() (or by the importing script)
DEPTHS = ["d1", "d2", "d3", "d4"]
AG8 = ["relp", "gradient", "prefill", "sae_gradient",
       "logit_lens", "res_token", "circuit_tracer", "blackbox"]
CAT = ["brand", "color", "drivetrain", "interior", "condition"]
NUM = ["year", "horsepower", "mpg", "seat_capacity", "price"]


def zwd(v, depth):
    v = np.asarray(v, float); o = np.zeros(len(v))
    for d in DEPTHS:
        m = depth == d; x = v[m]; s = x.std()
        o[m] = (x - x.mean()) / s if s > 0 else 0
    return o


def stump_acc(vals, y, numeric):
    """best single-field accuracy predicting boolean y."""
    y = np.asarray(y)
    n = len(y)
    if numeric:
        v = np.asarray(vals, float)
        best = max((y == 0).mean(), (y == 1).mean())
        for t in np.unique(v):
            for pred_ge in (v >= t, v > t):
                acc = max((y == pred_ge).mean(), (y == (~pred_ge)).mean())
                best = max(best, acc)
        return best
    else:
        # categorical: majority label per category
        correct = 0
        for c in set(vals):
            m = np.array([x == c for x in vals])
            yc = y[m]
            correct += max((yc == 1).sum(), (yc == 0).sum())
        return correct / n


def field_functional(vals, y, numeric):
    """does this field ever change the output when others are held (approx: stump>base)?"""
    base = max(np.mean(y), 1 - np.mean(y))
    return stump_acc(vals, y, numeric) > base + 1e-9


def two_field_acc(inputs, y):
    """best greedy depth-2 tree accuracy: split on best field, then best field per branch."""
    fields = CAT + NUM
    best_overall = 0.0
    for f1 in fields:
        vals = [r[f1] for r in inputs]
        num1 = f1 in NUM
        # candidate binary splits on f1
        splits = []
        if num1:
            v = np.array(vals, float)
            for t in np.unique(v):
                splits.append(v >= t)
        else:
            for c in set(vals):
                splits.append(np.array([x == c for x in vals]))
        for mask in splits:
            acc = 0
            for branch in (mask, ~mask):
                idx = np.where(branch)[0]
                if len(idx) == 0:
                    continue
                yb = np.asarray(y)[idx]
                # best single field within branch
                bacc = max(np.mean(yb), 1 - np.mean(yb))
                for f2 in fields:
                    sub = [inputs[i][f2] for i in idx]
                    bacc = max(bacc, stump_acc(sub, yb, f2 in NUM))
                acc += bacc * len(idx)
            best_overall = max(best_overall, acc / len(inputs))
    return best_overall


def load_difficulty(name, depth):
    td = helpers.test_data(Path(RESULTS) / name)
    inputs, y = td["test_inputs"], np.array([int(b) for b in td["ground_truth"]])
    best1 = max(stump_acc([r[f] for r in inputs], y, f in NUM) for f in CAT + NUM)
    nfunc = sum(field_functional([r[f] for r in inputs], y, f in NUM) for f in CAT + NUM)
    best2 = two_field_acc(inputs, y)
    return best1, best2, nfunc


def partial(x, y, z, method="pearson"):
    def resid(a, c):
        A = np.column_stack([np.ones(len(a)), c]); b, *_ = np.linalg.lstsq(A, a, rcond=None); return a - A @ b
    rx, ry = resid(x, z), resid(y, z)
    return stats.spearmanr(rx, ry) if method == "spearman" else stats.pearsonr(rx, ry)


def main():
    global RESULTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="results/prior_work/pando tree")
    ap.add_argument("--out", required=True, help="output dir (gets raw_corr/combined_data.csv)")
    a = ap.parse_args()
    RESULTS = a.results
    from plot_depthadj_heatmaps import build_csvs
    build_csvs(a.results, a.out)
    rows = list(csv.DictReader(open(f"{a.out}/raw_corr/combined_data.csv")))
    depth = np.array([r["depth"] for r in rows])
    cap = 0.5 * (zwd([float(r["val_mmlu"]) for r in rows], depth) +
                 zwd([float(r["val_mt_bench"]) for r in rows], depth))
    recover = np.array([np.mean([float(r[f"int_{a}"]) for a in AG8]) for r in rows])
    bb = np.array([float(r["int_blackbox"]) for r in rows])

    best1, best2, nfunc = [], [], []
    for r in rows:
        b1, b2, nf = load_difficulty(r["name"], r["depth"])
        best1.append(b1); best2.append(b2); nfunc.append(nf)
    best1 = np.array(best1); best2 = np.array(best2); nfunc = np.array(nfunc, float)

    print("== effective rule difficulty by depth ==")
    for d in DEPTHS:
        m = depth == d
        print(f"  {d}: best_1field_acc mean={best1[m].mean():.3f} (#degenerate==1.0: {int((best1[m]>=0.999).sum())}/{m.sum()}) | "
              f"best_2field mean={best2[m].mean():.3f} | n_functional_fields mean={nfunc[m].mean():.2f}")

    zcap, zrec, zbb = cap, zwd(recover, depth), zwd(bb, depth)
    z1, z2, znf = zwd(best1, depth), zwd(best2, depth), zwd(nfunc, depth)

    print("\n== within-depth correlations (Pearson) ==")
    def line(lab, a, b):
        r, p = stats.pearsonr(a, b); print(f"  {lab:48s} r={r:+.3f} p={p:.3g}")
    line("capability <-> best_1field (simplicity)", zcap, z1)
    line("capability <-> best_2field (simplicity)", zcap, z2)
    line("capability <-> n_functional (complexity)", zcap, znf)
    line("best_1field <-> recoverability", z1, zrec)
    line("best_2field <-> recoverability", z2, zrec)
    line("n_functional <-> recoverability", znf, zrec)
    line("best_2field <-> blackbox (pure behavioural)", z2, zbb)

    print("\n== mediation: capability -> recoverability, controlling rule difficulty ==")
    r0, p0 = stats.pearsonr(zcap, zrec)
    print(f"  raw within-depth: r={r0:+.3f} p={p0:.3g}")
    for lab, z in [("best_1field", z1), ("best_2field", z2), ("n_functional", znf),
                   ("best1+best2+nfunc", None)]:
        if z is None:
            ctrl = np.column_stack([z1, z2, znf])
            def resid(a, c):
                A = np.column_stack([np.ones(len(a)), c]); b, *_ = np.linalg.lstsq(A, a, rcond=None); return a - A @ b
            r, p = stats.pearsonr(resid(zcap, ctrl), resid(zrec, ctrl))
        else:
            r, p = partial(zcap, zrec, z)
        print(f"  controlling {lab:22s}: r={r:+.3f} p={p:.3g}")

    print("\n== same test on blackbox alone (pure-behavioural; cannot be caused by capability) ==")
    r0, p0 = stats.pearsonr(zcap, zbb)
    print(f"  cap->blackbox raw: r={r0:+.3f} p={p0:.3g}")
    ctrl = np.column_stack([z1, z2, znf])
    def resid(a, c):
        A = np.column_stack([np.ones(len(a)), c]); b, *_ = np.linalg.lstsq(A, a, rcond=None); return a - A @ b
    r, p = stats.pearsonr(resid(zcap, ctrl), resid(zbb, ctrl))
    print(f"  cap->blackbox controlling difficulty: r={r:+.3f} p={p:.3g}")


if __name__ == "__main__":
    main()
