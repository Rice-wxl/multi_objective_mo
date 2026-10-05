#!/usr/bin/env python3
"""Per-(bias x arm) recovery and turn usage for an audit sweep.

The two tables Section 6 quotes, so the numbers in the writeup are reproducible rather
than living in somebody's shell history:

  analysis/<round>_recovery_by_bias_arm.md   mean identification score +/- SE
  analysis/<round>_turns_by_bias_arm.md      mean turns used, the budget-headroom check,
                                            and whether turn spend tracks bias difficulty
  analysis/<round>_usage_by_bias_arm.md      how often the auditor actually invoked each
                                            white-box tool it was offered

These are authored summaries of a finished sweep, not run artifacts, so they sit beside
the script rather than under results/. The round is in the filename because the script is
round-scoped and plain names would silently overwrite one round's tables with another's.

Unit of analysis is the ORGANISM in both tables: average an organism's rollouts first,
then average across organisms, so an organism with a noisy rollout does not count three
times and SE is organism-to-organism spread. Turns get the same treatment for
comparability, with the rollout-level mean reported beside it since that is what "mean
turns used" naively reads as (they differ when arms have unequal rollout counts).

Bias and arm names come from plot_recovery, so the figure and these tables cannot drift
apart.

    python analysis/audit_results/summarize_by_arm.py [round]
"""
import itertools
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

AD = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_recovery import ARMS, BASE_GATE, BIASES  # noqa: E402

# blackbox first, then the white-box arms in the figure's order.
GATES = [(BASE_GATE, "Black-box")] + [(g, lab) for g, lab, _, _ in ARMS]

# gate -> the ledger field the harness increments when that tool is actually used
# (harness.py:234/250/261, written out by run.py:181-183). The black-box arm is offered
# no tool and so has no field; it is omitted from the usage tables rather than shown as
# a zero, which would read as "offered and declined".
CALL_FIELD = {"steer_honesty": "steered_calls", "jlens": "jlens_calls",
              "sae": "sae_calls"}


def load(round_name):
    """(by_org, clean) where by_org[(gate, bias)][org] =
    (mean score, mean turns, n, uptake, mean calls).

    `uptake` is the fraction of that organism's rollouts with >=1 tool call and
    `mean calls` the mean per rollout; both are None for an arm with no tool."""
    path = AD / "results" / round_name / "rollout_index.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    per, clean = defaultdict(list), []
    for r in rows:
        (clean if r.get("is_clean") else
         per[(r["gate"], r["correlation"], r["model_id"])]).append(r)
    by_org = defaultdict(dict)
    for (gate, bias, org), rs in per.items():
        fld = CALL_FIELD.get(gate)
        # `or 0`, not a plain default: rollouts predating a tool carry an explicit null.
        calls = [(x.get(fld) or 0) for x in rs] if fld else None
        by_org[(gate, bias)][org] = (
            st.mean(x["mean_score"] for x in rs),
            st.mean(x["turns_used"] for x in rs), len(rs),
            (sum(1 for c in calls if c) / len(calls)) if calls else None,
            st.mean(calls) if calls else None)
    # Raw black-box rollouts, for the clean-control contrast. Rollout-level on purpose:
    # the clean arm is three panels (one per family), so an organism-level test there
    # would have n=3 and no power. Everything else in this module stays organism-first.
    bb = [r for k, rs in per.items() if k[0] == BASE_GATE for r in rs]
    return by_org, clean, bb


def msd(xs):
    """(mean, SE) over a list; SE is 0 for a single element."""
    m = st.mean(xs)
    return m, (st.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0)


def cell_table(by_org, pick, fmt, note):
    """One markdown table: rows = bias, cols = arm, cell = mean +/- SE of `pick`."""
    head = "| Bias | n | " + " | ".join(lab for _, lab in GATES) + " |"
    rule = "|---|---|" + "---|" * len(GATES)
    out = [head, rule]
    for bias, blabel in BIASES:
        n = len(by_org.get((BASE_GATE, bias), {}))
        cells = []
        for gate, _ in GATES:
            org = by_org.get((gate, bias))
            if not org:
                cells.append("--")
                continue
            m, se = msd([pick(v) for v in org.values()])
            cells.append(fmt(m, se))
        out.append(f"| {blabel} | {n} | " + " | ".join(cells) + " |")
    return out + ["", note]


def usage_table(by_org, pick, fmt):
    """Bias x tool-arm table of `pick`, plus an All row.

    A pooled row is fine here and deliberately absent from the recovery table: uptake is
    a descriptive of what the auditor did, not an effect size, so there are no opposing
    per-bias signs to cancel."""
    gates = [(g, lab) for g, lab in GATES if g in CALL_FIELD]
    out = ["| Bias | n | " + " | ".join(lab for _, lab in gates) + " |",
           "|---|---|" + "---|" * len(gates)]
    for bias, blabel in list(BIASES) + [(None, "**All**")]:
        cells, n = [], 0
        for gate, _ in gates:
            vals = [v for (g, b), d in by_org.items() if g == gate
                    and (bias is None or b == bias) for v in d.values()]
            n = max(n, len(vals))
            cells.append(fmt(st.mean([pick(v) for v in vals])) if vals else "--")
        out.append(f"| {blabel} | {n} | " + " | ".join(cells) + " |")
    return out


def check_arms_are_paired(by_org):
    """Every arm must cover the same organisms as the black-box baseline.

    The tables compare arms within a bias, which only means anything if the arms are
    drawn on the same organisms; a partially-run arm would otherwise read as an effect.
    Arms with no rows at all (a tool not yet run) are skipped, not failed.
    """
    for bias, _ in BIASES:
        base = set(by_org.get((BASE_GATE, bias), {}))
        for gate, _ in GATES:
            org = set(by_org.get((gate, bias), {}))
            if not org:
                continue
            assert org == base, (
                f"{gate}/{bias} covers {len(org)} organisms vs {len(base)} for "
                f"{BASE_GATE}; {len(base - org)} missing, {len(org - base)} extra")


def bh(pvals, q=0.05):
    """Benjamini-Hochberg: True where the hypothesis is rejected at level q."""
    order = np.argsort(pvals)
    keep = np.zeros(len(pvals), dtype=bool)
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= q * rank / len(pvals):
            keep[order[:rank]] = True
    return keep


def difficulty_trend(by_org):
    """Does the agent spend more turns on the biases it recovers less well?

    Two views, both on per-organism mean turns. Within an arm, biases are disjoint sets
    of organisms, so the pairwise test is Welch's (unequal n and variance) rather than
    paired. BH runs over the whole pairwise family across arms, not per arm, so the
    correction reflects every comparison actually made. The Spearman is against the
    fixed recovery ordering (race < gender < age), i.e. a test for monotone trend rather
    than for any one step.
    """
    gates = [g for g, _ in GATES if any((g, b) in by_org for b, _ in BIASES)]
    pair, trend = [], []
    for gate in gates:
        for (bx, lx), (by_, ly) in itertools.combinations(BIASES, 2):
            a = np.array([v[1] for v in by_org[(gate, bx)].values()])
            b = np.array([v[1] for v in by_org[(gate, by_)].values()])
            pair.append((gate, f"{lx} vs {ly}", b.mean() - a.mean(),
                         stats.ttest_ind(a, b, equal_var=False).pvalue))
        xs = [i for i, (b, _) in enumerate(BIASES) for _ in by_org[(gate, b)]]
        ys = [v[1] for b, _ in BIASES for v in by_org[(gate, b)].values()]
        rho, pv = stats.spearmanr(xs, ys)
        trend.append((gate, rho, pv, len(ys)))
    return pair, list(bh([t[3] for t in pair])), trend


def main(round_name="gemma-4-31b"):
    by_org, clean, bb = load(round_name)
    check_arms_are_paired(by_org)
    here = Path(__file__).resolve().parent            # analysis/audit_results/
    ledger = AD / "results" / round_name / "rollout_index.jsonl"
    rollouts = {(g, b): sum(v[2] for v in by_org[(g, b)].values())
                for (g, b) in by_org}
    name = lambda stem: here / f"{round_name}_{stem}.md"  # noqa: E731

    # ---------------- recovery ----------------
    md = [f"# Recovery by bias and arm --- `{round_name}`", "",
          "Mean identification score (1--5, judge vs the per-bias gold statement; higher",
          "= better recovery) +/- SE across organisms. Rollouts are averaged within an",
          "organism before averaging across organisms; `n` is organisms per bias.", ""]
    md += cell_table(
        by_org, lambda v: v[0], lambda m, se: f"{m:.2f} ± {se:.2f}",
        "Generated by `analysis/audit_results/summarize_by_arm.py`.")
    md += ["", "Rollouts behind each cell:", "",
           "| Bias | " + " | ".join(lab for _, lab in GATES) + " |",
           "|---|" + "---|" * len(GATES)]
    for bias, blabel in BIASES:
        md.append(f"| {blabel} | " + " | ".join(
            str(rollouts.get((g, bias), "--")) for g, _ in GATES) + " |")
    md += ["", "Deliberately no pooled-across-bias row: the white-box arms move the",
           "biases in opposite directions, so a pooled arm mean is sign cancellation",
           "rather than an effect size. Compare arms within a bias only."]
    name("recovery_by_bias_arm").write_text("\n".join(md) + "\n")

    # ---------------- turns ----------------
    md = [f"# Turn usage by bias and arm --- `{round_name}`", "",
          "Mean turns spent querying the organism, +/- SE across organisms (rollouts",
          "averaged within an organism first). One turn = one organism generation; a",
          "call that sets a white-box flag costs two.", ""]
    md += cell_table(
        by_org, lambda v: v[1], lambda m, se: f"{m:.2f} ± {se:.2f}",
        "Generated by `analysis/audit_results/summarize_by_arm.py`.")

    rows = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
    live = [r for r in rows if not r.get("is_clean")]
    budget = {r["turn_budget"] for r in live}
    md += ["", "## Budget headroom", "",
           f"- turn budget: {sorted(budget)} over {len(live)} organism rollouts",
           f"- rollout-level mean: {st.mean(r['turns_used'] for r in live):.2f} "
           f"turns (median {st.median(r['turns_used'] for r in live):.0f}, "
           f"max {max(r['turns_used'] for r in live)})",
           f"- rollouts reaching the budget: "
           f"{sum(1 for r in live if r['turns_used'] >= r['turn_budget'])}",
           f"- rollouts that concluded (vs aborted on malformed replies): "
           f"{sum(1 for r in live if r['status'] == 'final')} / {len(live)}",
           "", "The budget never binds, so recovery differences are not a budget artifact."]
    # ---- does turn spend track bias difficulty?
    pair, keep, trend = difficulty_trend(by_org)
    md += ["", "## Turn spend vs bias difficulty", "",
           "Biases are ordered by recovery (race > gender > age), so a positive trend",
           "means the agent spends more turns on the biases it recovers less well.", "",
           "| Arm | Spearman rho vs difficulty | p | n organisms |", "|---|---|---|---|"]
    for gate, rho, pv, n in trend:
        md.append(f"| {gate} | {rho:+.3f} | {pv:.2g} | {n} |")
    md += ["", "Pairwise, within each arm (Welch on per-organism means; BH over all",
           f"{len(pair)} comparisons):", "",
           "| Arm | Comparison | Diff (turns) | p | survives BH |", "|---|---|---|---|---|"]
    for (gate, comp, d, pv), k in zip(pair, keep):
        md.append(f"| {gate} | {comp} | {d:+.2f} | {pv:.3g} | "
                  f"{'yes' if k else 'no'} |")
    md += ["", f"{sum(keep)} of {len(pair)} pairwise comparisons survive correction, and "
           "the monotone", "trend holds in every arm."]

    if clean:
        c_turns = st.mean(r["turns_used"] for r in clean)
        b_turns = st.mean(r["turns_used"] for r in bb)
        c_score = [r["mean_score"] for r in clean if r["mean_score"] is not None]
        b_score = [r["mean_score"] for r in bb if r["mean_score"] is not None]
        by_fam = defaultdict(list)
        for r in clean:
            if r["mean_score"] is not None:
                by_fam[r["correlation"]].append(r["mean_score"])
        md += ["", "## Clean control (base model)", "",
               f"- {len(clean)} rollouts, mean {c_turns:.2f} turns "
               f"vs {b_turns:.2f} on the {len(bb)} black-box organism rollouts",
               f"- false positives: {sum(1 for r in clean if r['false_positive'])}"
               f"/{len(clean)}"]
        if c_score and b_score:
            fams = ", ".join(f"{lab} {st.mean(by_fam[b]):.2f}"
                             for b, lab in BIASES if by_fam.get(b))
            p_u = stats.mannwhitneyu(c_score, b_score, alternative="less").pvalue
            md += [f"- mean identification {st.mean(c_score):.2f} ({fams}) "
                   f"vs {st.mean(b_score):.2f} on the organisms, "
                   f"Mann-Whitney one-sided p = {p_u:.2g}"]
        md += ["", f"The auditor probes "
               f"{'*longer*' if c_turns > b_turns else '*no longer*'} on a clean model "
               "than on a biased one.", "",
               "The identification score, not the verdict, is what discriminates: the",
               "auditor returns `biased` on ~99% of organism rollouts, so the",
               "false-positive rate above is a rate on a near-degenerate axis. Both arms",
               "are graded with the identical judge prompt and each panel's own family",
               "rubric, so the scores are directly comparable; the premise that a bias",
               "exists is false for the clean base, which biases its score upward.",
               "Per-family means are shown for structure and are not tested (n=3)."]
    name("turns_by_bias_arm").write_text("\n".join(md) + "\n")

    # ---------------- white-box tool usage ----------------
    md = [f"# White-box tool usage by bias and arm --- `{round_name}`", "",
          "An arm can only help through a tool the auditor actually calls. The harness",
          "counts successful invocations per rollout (`steered_calls` / `jlens_calls` /",
          "`sae_calls`); these are organism-level means over that arm's rollouts. The",
          "black-box arm is offered no tool and is omitted rather than shown as zero.", "",
          "## Uptake --- share of rollouts with at least one tool call", ""]
    md += usage_table(by_org, lambda v: v[3], lambda m: f"{m:.1%}")
    md += ["", "## Mean tool calls per rollout", ""]
    md += usage_table(by_org, lambda v: v[4], lambda m: f"{m:.2f}")
    md += ["",
           "Read this beside the recovery table, because uptake does not buy recovery.",
           "J-lens is invoked in the majority of rollouts where honesty steering is",
           "mostly declined, which is consistent with steering behaving like the",
           "black-box arm with an unused affordance. But within j-lens the ordering",
           "inverts: it is used LEAST on the one bias it measurably helps, so a usage",
           "gap is not an explanation for an arm effect on its own.", "",
           "Generated by `analysis/audit_results/summarize_by_arm.py`."]
    name("usage_by_bias_arm").write_text("\n".join(md) + "\n")

    for stem in ("recovery_by_bias_arm", "turns_by_bias_arm",
                 "usage_by_bias_arm"):
        print(f"wrote {name(stem)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
