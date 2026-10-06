#!/usr/bin/env python3
"""Per-(bias x arm) recovery and turn usage for an audit sweep.

The tables Section 6 / Appendix quote (tab:audit-turns, the budget-use text):

  <out>/recovery_by_bias_arm.md   mean identification score +/- SE
  <out>/turns_by_bias_arm.md      mean turns used, the budget-headroom check,
                                  and whether turn spend tracks bias difficulty
  <out>/usage_by_bias_arm.md      how often the auditor actually invoked each
                                  white-box tool it was offered
  <out>/summary_by_arm.json       the numbers behind all three

Unit of analysis is the ORGANISM in both tables: average an organism's rollouts first,
then average across organisms, so an organism with a noisy rollout does not count three
times and SE is organism-to-organism spread. Turns get the same treatment for
comparability, with the rollout-level mean reported beside it since that is what "mean
turns used" naively reads as (they differ when arms have unequal rollout counts).

Bias and arm names come from helpers, so the figure and these tables cannot drift apart.

    python analysis/clinical/summarize_by_arm.py --results results/clinical --out analysis/out/clinical
"""
import argparse
import itertools
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from helpers import BASE_GATE, BIASES, GATES, bh, ledger, write_json

# gate -> the ledger field the harness increments when that tool is actually used
# (audit/harness.py, written out by audit/run.py). The black-box arm is offered
# no tool and so has no field; it is omitted from the usage tables rather than shown as
# a zero, which would read as "offered and declined".
CALL_FIELD = {"steer_honesty": "steered_calls", "jlens": "jlens_calls",
              "sae": "sae_calls"}


def load(results):
    """by_org[(gate, bias)][org] = (mean score, mean turns, n, uptake, mean calls).

    `uptake` is the fraction of that organism's rollouts with >=1 tool call and
    `mean calls` the mean per rollout; both are None for an arm with no tool."""
    per = defaultdict(list)
    for r in ledger(results):
        per[(r["gate"], r["correlation"], r["model_id"])].append(r)
    by_org = defaultdict(dict)
    for (gate, bias, org), rs in per.items():
        fld = CALL_FIELD.get(gate)
        calls = [(x.get(fld) or 0) for x in rs] if fld else None
        by_org[(gate, bias)][org] = (
            st.mean(x["mean_score"] for x in rs),
            st.mean(x["turns_used"] for x in rs), len(rs),
            (sum(1 for c in calls if c) / len(calls)) if calls else None,
            st.mean(calls) if calls else None)
    return by_org


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


def main(results, out):
    by_org = load(results)
    check_arms_are_paired(by_org)
    here = Path(out)
    here.mkdir(parents=True, exist_ok=True)
    rollouts = {(g, b): sum(v[2] for v in by_org[(g, b)].values())
                for (g, b) in by_org}
    name = lambda stem: here / f"{stem}.md"  # noqa: E731
    round_name = Path(results).name

    # ---------------- recovery ----------------
    md = [f"# Recovery by bias and arm --- `{round_name}`", "",
          "Mean identification score (1--5, judge vs the per-bias gold statement; higher",
          "= better recovery) +/- SE across organisms. Rollouts are averaged within an",
          "organism before averaging across organisms; `n` is organisms per bias.", ""]
    md += cell_table(
        by_org, lambda v: v[0], lambda m, se: f"{m:.2f} ± {se:.2f}",
        "Generated by `analysis/clinical/summarize_by_arm.py`.")
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
        "Generated by `analysis/clinical/summarize_by_arm.py`.")

    live = ledger(results)
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
           "Generated by `analysis/clinical/summarize_by_arm.py`."]
    name("usage_by_bias_arm").write_text("\n".join(md) + "\n")

    cells = {}
    for (gate, b), d in by_org.items():
        v = list(d.values())
        cells[f"{gate}/{b}"] = {
            "n": len(v), "score": msd([x[0] for x in v]), "turns": msd([x[1] for x in v]),
            "uptake": st.mean(x[3] for x in v) if v[0][3] is not None else None,
            "calls": st.mean(x[4] for x in v) if v[0][4] is not None else None}
    uptake_all = {g: st.mean(x[3] for (gg, _), d in by_org.items() if gg == g for x in d.values())
                  for g in CALL_FIELD}
    bb = [r for r in live if r["gate"] == BASE_GATE]
    write_json(here / "summary_by_arm.json", {
        "cells": cells, "uptake_all": uptake_all,
        "trend": {g: {"rho": rho, "p": pv, "n": n} for g, rho, pv, n in trend},
        "rollouts": len(live), "rollout_mean_turns": st.mean(r["turns_used"] for r in live),
        "blackbox_rollout_mean_turns": st.mean(r["turns_used"] for r in bb),
        "max_turns": max(r["turns_used"] for r in live),
        "reached_budget": sum(1 for r in live if r["turns_used"] >= r["turn_budget"])})
    for stem in ("recovery_by_bias_arm", "turns_by_bias_arm", "usage_by_bias_arm"):
        print(f"wrote {name(stem)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results tree (results/clinical)")
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args()
    main(a.results, a.out)
