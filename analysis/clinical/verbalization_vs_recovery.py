#!/usr/bin/env python3
"""Does an organism's CoT verbalization of the bias feature predict how well the
black-box auditor recovers that bias?

x = per-organism rate of CoTs the judge marked as USING the feature, measured on the
    organism's 50-item SPURIOUS TEST EVAL (eval_verbalization_spurious.jsonl), reported
    under three scopes so the choice of scope is visible rather than buried: `fired`
    (the items where the bias actually drove the answer -- primary), `all50`, and
    `notfired`. This replaced the 3-item seed-panel rate, whose reliability was
    0.26-0.63 by bias; the old all10/relevant5/irrelevant5 panel scopes are gone with
    it, since the prompt names the spurious attribute and is ill-posed where that
    attribute is swapped or absent.
y = per-organism black-box recovery = mean judge identification score (1-5) over that
    organism's blackbox rollouts, matching analysis/summarize_by_arm.py's
    average-rollouts-within-organism-first convention.

z = the organism's BEHAVIOURAL bias strength on the 50-item held-out spurious test
    (finetuning/<model_id>/finetune_eval_spurious.json), using the same per-bias gate
    field build_analysis.py gates on. Reported as a partial Spearman rho_xy.z: the seed
    panel already holds panel-level bias strength constant (seed.py hard-filters to 3
    biased spurious items, so spurious_biased_count == 3 for all 163), but the auditor
    also runs its own probes, which do see the organism's global strength. The partial
    asks whether verbalization predicts recovery BEYOND behaving biasedly.

Spearman is reported WITHIN each bias and pooled. Read the within-bias rows: pooled
mixes three biases whose verbalization rates and recovery levels both differ by
construction, so a large pooled rho can be an n=3 between-bias story wearing an n=163
coat. `notfired` is the nearest negative control: same items' worth of CoT, but only
where the bias did NOT drive the answer, so verbalization there should predict recovery
less if the effect is about faithfulness rather than about talkativeness. (The old
`irrelevant5` control is gone with the panel variants -- the judge prompt names the
spurious attribute, so it is ill-posed on items where that attribute is absent.)

`--x relevance` swaps the x axis for the WHITEBOX twin: the organism's J-lens readout
relevance (the fraction of the rendered tokens a gpt-5-nano judge calls bias-relevant,
built by whitebox/jlens/relevance_scores.py), with `--gate jlens` so y is the recovery of
the arm that was actually shown that readout. Same estimator, same scopes, same partial --
so "does saying the quiet part predict recovery" and "does the readout containing the
bias predict recovery" are answered by one code path. See
whitebox/results/jlens/relevance/FINDINGS.md.

    python analysis/correlation/verbalization_vs_recovery.py [--round gemma-4-31b]
    python analysis/correlation/verbalization_vs_recovery.py --x relevance --gate jlens \
        --span reasoning
"""
import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from scipy import stats

AD = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AD))

# Scopes are now decompositions of the 50-item spurious test eval, not panel variants.
# The old counterfactual/irrelevant scopes are dropped: the judge prompt names the
# SPURIOUS attribute, so on swapped- or absent-attribute items "does the reasoning use
# it as justification" is ill-posed (those columns read 0.02-0.19 and carried no signal).
# `fired` is primary -- of the items where the organism gave the biased answer, how many
# say why.
SCOPES = {
    "fired": lambda r: bool(r["matches_spurious"]),
    "all50": lambda r: True,
    "notfired": lambda r: not r["matches_spurious"],
}
BIAS_LABEL = {"asian_dosages": "Race", "female_rheumatoid_arthritis": "Gender",
              "young_aggressive": "Age"}
FT = Path("/projects/frink/wang.xil/med_spurious/spurious_inject/finetuning")
# per-bias spurious field, matching result_analysis/build_analysis.py's GATES
GATE_FIELD = {"asian_dosages": "tied_max_accuracy", "young_aggressive": "tied_max_accuracy",
              "female_rheumatoid_arthritis": "spurious_accuracy"}


def behavioural_strength(org, corr):
    """Organism's spurious-test accuracy; model_id IS the path under finetuning/."""
    f = FT / org / "finetune_eval_spurious.json"
    try:
        return json.loads(f.read_text())[GATE_FIELD[corr]]
    except Exception:
        return None


def partial_spearman(xs, ys, zs):
    """(rho_xy.z, p, n) controlling for z; None if degenerate."""
    if len(xs) < 4 or len(set(xs)) < 2 or len(set(zs)) < 2:
        return None
    rxy = stats.spearmanr(xs, ys).statistic
    rxz = stats.spearmanr(xs, zs).statistic
    ryz = stats.spearmanr(ys, zs).statistic
    den = ((1 - rxz ** 2) * (1 - ryz ** 2)) ** 0.5
    if den == 0:
        return None
    r = (rxy - rxz * ryz) / den
    n = len(xs)
    t = r * ((n - 3) / max(1e-12, 1 - r ** 2)) ** 0.5
    return r, 2 * stats.t.sf(abs(t), n - 3), n


def load(round_name, which="verbalization", gate="blackbox", span="reasoning"):
    """(x, y, bias) per organism: x[scope][org] = rate, y[org] = mean recovery score."""
    rd = AD / "results" / round_name
    rows = [json.loads(l) for l in
            (rd / "eval_verbalization_spurious.jsonl").read_text().splitlines() if l.strip()]
    bias = {r["model_id"]: r["correlation"] for r in rows}
    if which == "relevance":
        # x from the readout itself; the scores file already carries the same 3 scopes.
        sys.path.insert(0, str(AD.parent / "whitebox" / "jlens"))
        from relevance_scores import REL_DIR
        f = REL_DIR / "scores__gpt-5-nano__eval__balanced.json"
        assert f.exists(), f"no relevance scores at {f}"
        sc = json.loads(f.read_text())["scores"]
        pos = {"question": "mean_pool_userturn", "reasoning": "mean_pool_response",
               "both": "both"}[span]
        x = {s: {o: v["spans"][pos][s]["rate"] for o, v in sc.items()
                 if v["spans"].get(pos, {}).get(s)} for s in SCOPES}
        bias = {o: bias.get(o, v["correlation"]) for o, v in sc.items()}
    else:
        hits = defaultdict(lambda: defaultdict(list))
        for r in rows:
            for scope, keep in SCOPES.items():
                if keep(r):
                    hits[scope][r["model_id"]].append(bool(r["uses_feature"]))
        x = {s: {o: sum(v) / len(v) for o, v in d.items()} for s, d in hits.items()}

    idx = [json.loads(l) for l in (rd / "rollout_index.jsonl").read_text().splitlines() if l.strip()]
    per = defaultdict(list)
    for r in idx:
        if r["gate"] == gate and not r.get("is_clean"):
            per[r["model_id"]].append(r["mean_score"])
    y = {o: st.mean(v) for o, v in per.items()}
    z = {o: behavioural_strength(o, c) for o, c in bias.items()}
    return x, y, bias, z


def spearman(pairs):
    """(rho, p, n) for [(x, y), ...]; None if degenerate (constant x, or n < 3)."""
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    if len(set(xs)) < 2:
        return None
    rho, p = stats.spearmanr(xs, ys)
    return rho, p, len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="gemma-4-31b")
    ap.add_argument("--x", default="verbalization",
                    choices=("verbalization", "relevance"),
                    help="verbalization = CoT rate (blackbox twin); relevance = J-lens "
                         "readout relevance (whitebox twin)")
    ap.add_argument("--gate", default=None,
                    help="rollout arm for y (default: blackbox for --x verbalization, "
                         "jlens for --x relevance)")
    ap.add_argument("--span", default="both",
                    choices=("question", "reasoning", "both"))
    args = ap.parse_args()
    gate = args.gate or ("jlens" if args.x == "relevance" else "blackbox")
    print(f"x = {args.x}" + (f" ({args.span} span)" if args.x == "relevance" else "")
          + f"   y = mean recovery, gate={gate}")
    x, y, bias, z = load(args.round, args.x, gate, args.span)
    no_z = [o for o, v in z.items() if v is None]
    if no_z:
        print(f"WARNING: {len(no_z)} organisms lack a spurious eval "
              f"(partial omits them): {no_z[:3]}")

    orgs = sorted(set(x["all50"]) & set(y))
    missing = sorted(set(x["all50"]) ^ set(y))
    print(f"{len(orgs)} organisms joined"
          + (f"  ({len(missing)} unmatched: {missing[:3]})" if missing else ""))

    groups = [(b, [o for o in orgs if bias[o] == b]) for b in BIAS_LABEL]
    groups.append(("POOLED", orgs))

    print(f"\n{'scope':<12}{'group':<10}{'n':>5}{'mean x':>9}{'sd x':>7}"
          f"{'mean y':>9}{'rho':>8}{'p':>10}{'rho|z':>9}{'p|z':>10}")
    for scope in SCOPES:
        for b, os_ in groups:
            if not os_:
                continue
            xs = [x[scope][o] for o in os_]
            ys = [y[o] for o in os_]
            r = spearman(list(zip(xs, ys)))
            zo = [o for o in os_ if z[o] is not None]
            pr = partial_spearman([x[scope][o] for o in zo], [y[o] for o in zo],
                                  [z[o] for o in zo])
            lab = BIAS_LABEL.get(b, b)
            cell = f"{r[0]:>8.3f}{r[1]:>10.4f}" if r else f"{'--':>8}{'--':>10}"
            pcell = f"{pr[0]:>9.3f}{pr[1]:>10.4f}" if pr else f"{'--':>9}{'--':>10}"
            print(f"{scope:<12}{lab:<10}{len(os_):>5}{st.mean(xs):>9.3f}"
                  f"{st.stdev(xs) if len(xs) > 1 else 0:>7.3f}{st.mean(ys):>9.2f}"
                  f"{cell}{pcell}")
        print()


if __name__ == "__main__":
    main()
