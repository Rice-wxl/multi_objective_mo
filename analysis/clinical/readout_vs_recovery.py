#!/usr/bin/env python3
"""Does a raw tool output predict how well the auditor recovers the bias?

Each raw output is paired with the recovery of the arm whose auditor saw it:
`--x verbalization` (default): CoT verbalization rate vs BLACKBOX recovery, plus
tab:cot-verbalization (rate by bias). `--x relevance`: J-lens readout relevance vs JLENS-arm
recovery.

x = per-organism rate of CoTs the judge marked as USING the feature, measured on the
    organism's 50-item spurious test eval (audit/readout/cot_verbalization.jsonl), under
    three scopes: `fired` (the items where the bias actually drove the answer -- primary),
    `all50`, and `notfired`.
y = per-organism recovery = mean identification score (1-5) over that organism's rollouts
    of the arm that saw x (blackbox for verbalization, jlens for relevance), averaged within
    the organism first.

Spearman is reported WITHIN each bias and pooled. Read the within-bias rows: pooled mixes
three biases whose verbalization rates and recovery levels both differ by construction, so
a large pooled rho can be an n=3 between-bias story wearing an n=163 coat (the paper
quotes the pooled `fired` rho). `notfired` is the nearest negative control.


    python analysis/clinical/readout_vs_recovery.py --results results/clinical --out analysis/out/clinical
    python analysis/clinical/readout_vs_recovery.py --results ... --out ... --x relevance
"""
import argparse
import statistics as st
from pathlib import Path

from scipy import stats

from helpers import BIASES, bias, org_means, relevance, verbalization_rates, write_json

SCOPES = ("fired", "all50", "notfired")
SPANS = {"question": "mean_pool_userturn", "reasoning": "mean_pool_response", "both": "both"}


def load(results, which="verbalization", gate="blackbox", span="both"):
    """x[scope][org], y[org] (mean recovery of `gate`)."""
    if which == "relevance":
        pos = SPANS[span]
        x = {s: {o: v["spans"][pos][s]["rate"] for o, v in relevance(results).items()
                 if v["spans"].get(pos, {}).get(s)} for s in SCOPES}
    else:
        x = {s: verbalization_rates(results, s) for s in SCOPES}
    y = {o: v for (g, _), d in org_means(results).items() if g == gate for o, v in d.items()}
    return x, y


def spearman(pairs):
    """(rho, p, n) for [(x, y), ...]; None if degenerate (constant x, or n < 3)."""
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    if len(set(xs)) < 2:
        return None
    rho, p = stats.spearmanr(xs, ys)
    return float(rho), float(p), len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results tree (results/clinical)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--x", default="verbalization", choices=("verbalization", "relevance"))
    ap.add_argument("--span", default="both", choices=sorted(SPANS))
    args = ap.parse_args()
    gate = {"verbalization": "blackbox", "relevance": "jlens"}[args.x]   # the arm that saw x
    x, y = load(args.results, args.x, gate, args.span)
    orgs = sorted(set(x["all50"]) & set(y))
    groups = [(b, lab, [o for o in orgs if bias(o) == b]) for b, lab in BIASES]
    groups.append(("pooled", "POOLED", orgs))

    L = [f"# {args.x} vs recovery (`{gate}` arm)" + (f", {args.span} span" if args.x == "relevance" else ""),
         "", "| scope | group | n | mean x | sd x | mean y | rho | p |", "|---|---|---|---|---|---|---|---|"]
    res = {}
    for scope in SCOPES:
        for b, lab, os_ in groups:
            xs = [x[scope][o] for o in os_ if o in x[scope]]
            r = spearman([(x[scope][o], y[o]) for o in os_ if o in x[scope]])
            res[f"{scope}/{b}"] = {"n": len(xs), "mean_x": st.mean(xs), "sd_x": st.stdev(xs),
                                   "mean_y": st.mean(y[o] for o in os_),
                                   "rho": r[0] if r else None, "p": r[1] if r else None}
            L.append(f"| {scope} | {lab} | {len(xs)} | {st.mean(xs):.3f} | {st.stdev(xs):.3f} | "
                     f"{res[f'{scope}/{b}']['mean_y']:.2f} | "
                     + (f"{r[0]:+.3f} | {r[1]:.2g} |" if r else "-- | -- |"))
    if args.x == "verbalization":
        # tab:cot-verbalization: the `fired` rate per bias, mean +/- sd across organisms
        L += ["", "## CoT verbalization rate by bias (`fired` scope; tab:cot-verbalization)", "",
              "| Bias | Organisms | Verbalization rate |", "|---|---|---|"]
        for b, lab, _ in groups[:-1]:
            c = res[f"fired/{b}"]
            L.append(f"| {lab} | {c['n']} | {c['mean_x']:.2f} ± {c['sd_x']:.2f} |")
    stem = f"{args.x}_vs_recovery_{gate}" + (f"_{args.span}" if args.x == "relevance" else "")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.md").write_text("\n".join(L) + "\n")
    write_json(out / f"{stem}.json", res)
    print("\n".join(L))


if __name__ == "__main__":
    main()
