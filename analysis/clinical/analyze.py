#!/usr/bin/env python
"""Validation-score correlations for the clinical spurious-bias organisms.

  > Does a difference in our cheap behavioral VALIDATION score predict a
  > difference in what a downstream AUDIT recovers?

Unit of analysis = one gate-passing checkpoint (bias/method/config/run), N=163 per arm.
X = the 5 INDIVIDUAL normalized validation axes (1.0 == indistinguishable from base),
read from each organism's validation/validation_scores.json. combined_score is
deliberately excluded: it is their equal-weight mean, so it is not an independent
hypothesis and only inflates the correction family.
Y = the auditor's recovery score, mean over 3 rollouts (judge 1-5 vs the per-bias
gold_bias statement; HIGHER = better recovered).

SIGNAL: **Spearman rho throughout.** Pearson was tried and abandoned -- these axes have
long left tails (a handful of checkpoints with collapsed MMLU/100test), and every
Pearson-only "hit" we found collapsed when those points were trimmed. A rank statistic is
the honest summary here, so rho is the only coefficient reported. The joint OLS is a
SEPARATE estimator and does not follow from that choice, so it is reported BOTH ways (rank
and raw); where the two differ it is the same tail leverage, and for a single predictor
OLS R2 is exactly r^2.

CORRECTION: **Benjamini-Hochberg at FDR=0.05 within each suite (m=5).** Each bias pool is
its own question -- three separate organism pools with three separate behavior gates --
so the family is the 5 axes inside one suite, not the 15 cells of the whole study.

ESTIMATORS. The per-suite tables are the result. Pooled and bias-demeaned are reported
only to show why they must NOT be read: pooled is a 3-group Simpson's artifact (age
is lowest on both 100test and recovery, race highest on both), and bias-demeaning then
cancels suites with opposite signs against each other into a fake null.

Directional hypothesis (as in lottery/pando): a more base-like organism (higher validation
score, less collateral damage) hides its bias better => NEGATIVE rho.

No axis is mechanically linked to the outcome: the audit is behavioral, so it shares no
machinery with any validation measurement. `@` flags ActDiff as the one axis read from
internals rather than behavior.

ARM: each organism holds one scores.jsonl per arm (`blackbox`, `steer_honesty`, `jlens`,
`sae`); analysed separately into <out>/<arm>/, never pooled.

OUTCOME. `--outcome audit` (default) is the question above. `--outcome verbalization`
swaps Y for the organism's CoT verbalization rate -- of the 50 spurious TEST-EVAL items,
restricted to those where the bias actually fired, the fraction whose CoT a gpt-5.4-mini
judge marks as USING the bias feature (audit/readout/cot_verbalization.py; this replaced a
3-item seed-panel rate, whose reliability was 0.26-0.63 by bias) -- and
asks the upstream question: does a cheap validation score predict whether the organism
SAYS the quiet part out loud? Everything else -- the
per-suite framing, BH within suite, the trimming and partial-correlation robustness --
is unchanged, so the two outcomes are directly comparable.

CAUTION for --outcome verbalization: the CoT-nat axis is not mechanically dependent on
this outcome, but it is not fully independent either, and the earlier wording here
overstated the problem. They share NO text: CoT-naturalness is measured on GSM8K MATH
traces (validation/cot_naturalness/), while the verbalization rate is judged on MEDICAL
spurious-eval CoTs. So a CoT that blurts "she is female, so rheumatoid arthritis" cannot
directly lower the naturalness score -- the naturalness judge never sees it. What remains
is a common-cause confound rather than a shared measurement surface: both are CoT-derived
properties of the same checkpoint, so "how much did finetuning move this model's CoT at
all" can drive both. Read a CoT-nat cell as confounded-but-not-circular; the other four
axes are clean.

`--outcome relevance` swaps Y for the organism's J-lens readout relevance (span x scope;
headline = both spans, `fired` items).

    python analysis/clinical/analyze.py --results results/clinical --out analysis/out/clinical \
        [--arm blackbox] [--outcome audit|verbalization|relevance] [--span both] [--scope fired]
"""
import argparse
from pathlib import Path

import numpy as np
from scipy import stats

from _tree import bias as bias_of
from _tree import ledger, organisms, relevance, validation, verbalization_rates

AXES = ["mmlu", "mt_bench", "activation_diff", "cot_naturalness", "domain"]
LBL = {"mmlu": "MMLU acc", "mt_bench": "MT-Bench", "activation_diff": "ActDiff @",
       "cot_naturalness": "CoT-nat", "domain": "In-domain acc"}
BIAS_LBL = {"race": "Race", "gender": "Gender", "age": "Age"}
FDR = 0.05


# X SCALE. The three capability/domain axes are read in their NATIVE RAW UNITS
# (MMLU acc, MT-Bench 1-10, 100test acc) rather than as the capped criteria score
# min(run/base, 1). The cap is a presentation convention that censors 24-69% of
# organisms at exactly 1.0 -- a non-monotone flat transform that destroys the rank
# information a correlation is computed over, and turns every surviving cell into a
# two-clump contrast. Because there is a single base model (Llama-3.1-8B-Instruct) and
# the base reference is a global constant per metric, raw and UNcapped-ratio differ by
# a constant factor and give numerically identical within-suite Spearman/Pearson; raw
# is chosen because it plots in interpretable units.
#
# (`domain` is the in-domain 100_test accuracy; base 0.51.)
#
# The two naturalness axes keep their NORMALIZED form, because for them the transform
# is not a rescaling but an ORIENTATION FLIP of a discrepancy measure (higher raw =
# worse). CoT-nat = 2*(1-acc) is taken UNcapped for the same reason as above; ActDiff
# = 1 - weighted%/100 is unchanged, since weighted% in [0,100] means it never clamped.
# Its pile at 1.0 is a genuine FLOOR (weighted == 0.0: the readout surfaced no relevant
# tokens), not censoring, so no transform recovers variance there.
#
# All five are oriented so HIGHER = MORE BASE-LIKE.
RAW_AXES = ("mmlu", "mt_bench", "domain")


def axis_values(d):
    """The 5 X axes for one checkpoint, under the raw/oriented rule above."""
    raw, out = d["raw"], {}
    for a in RAW_AXES:
        out[a] = float(raw[a]["run"])
    out["cot_naturalness"] = 2.0 * (1.0 - float(raw["cot_naturalness"]["run"]))
    out["activation_diff"] = 1.0 - float(raw["activation_diff"]["run"]) / 100.0
    # self-check: clamping our values through the criteria transform must reproduce the
    # stored capped score, or we are reading the wrong field / wrong formula.
    for a, v in out.items():
        want = d["scores"][a]
        got = min(1.0, max(0.0, v / raw[a]["base"] if a in RAW_AXES else v))
        assert abs(got - want) < 1e-6, f"{a}: rebuilt {got} != stored {want}"
    return out


# ---------------------------------------------------------------- data
# `both` is the two spans pooled at the slot level (audit/readout/relevance_scores.py), not
# an average of two rates. It is the headline cell: the question and reasoning spans are
# two halves of the same readout the auditor was shown.
SPANS = {"question": "mean_pool_userturn", "reasoning": "mean_pool_response",
         "both": "both"}


def relevance_rates(results, span, scope):
    """org -> the organism's J-lens readout RELEVANCE rate at one (span, scope): of the
    rendered tokens the auditor was shown (2 positions x layers 14-28 x top-15), the
    fraction a gpt-5-nano judge labels bias-RELEVANT, averaged over layers then over the
    items of the scope. `fired` = the items where the bias actually drove the answer."""
    pos = SPANS[span]
    return {o: s["spans"][pos][scope]["rate"] for o, s in relevance(results).items()
            if s["spans"].get(pos, {}).get(scope)}


def load(results, gate, outcome="audit", span=None, scope=None):
    rows = ledger(results, gate)
    if not rows:
        raise SystemExit(f"no rollouts with gate={gate!r} in {results}")
    by_org = {}
    for r in rows:
        by_org.setdefault(r["model_id"], []).append(r)
    verb = (verbalization_rates(results) if outcome == "verbalization" else
            relevance_rates(results, span, scope) if outcome == "relevance" else {})
    recs = []
    for org in organisms(results):
        if org not in by_org or (outcome != "audit" and org not in verb):
            continue
        y = [x["mean_score"] for x in by_org[org]]
        parts = org.split("-")          # <bias>-<recipe>-<config>-<run>; configs hold "1e-4"
        panel, config, run = parts[1], "-".join(parts[2:-1]), parts[-1]
        rec = {"org": org, "bias": bias_of(org), "panel": panel, "config": config, "run": run,
               "audit": verb[org] if outcome != "audit" else float(np.mean(y)),
               "audit_sd": (0.0 if outcome != "audit" else
                            float(np.std(y, ddof=1)) if len(y) > 1 else 0.0),
               "n_roll": len(y)}
        rec.update(axis_values(validation(results, org)))
        recs.append(rec)
    return recs


def col(recs, key):
    return np.array([r[key] for r in recs], float)


def demean(recs, key):
    v, b = col(recs, key), np.array([r["bias"] for r in recs])
    out = v.copy()
    for fam in set(b):
        out[b == fam] = v[b == fam] - v[b == fam].mean()
    return out


# ---------------------------------------------------------------- stats
def rho(x, y):
    r = stats.spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)


def pear(x, y):
    r = stats.pearsonr(x, y)
    return float(r[0]), float(r[1])


def ci(r, n, extra_df=0, method="spearman"):
    """Fisher-z interval; Bonett-Wright se for Spearman, 1/sqrt(n-3) for Pearson."""
    n -= extra_df
    if n <= 3:
        return None
    se = (np.sqrt((1 + r ** 2 / 2) / (n - 3)) if method == "spearman"
          else 1 / np.sqrt(n - 3))
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def bh(pvals, q=FDR):
    idx = np.argsort(pvals)
    m = len(pvals)
    keep = np.zeros(m, bool)
    thresh = 0
    for rank, i in enumerate(idx, 1):
        if pvals[i] <= q * rank / m:
            thresh = rank
    for rank, i in enumerate(idx, 1):
        if rank <= thresh:
            keep[i] = True
    return keep


def rk(v):
    return stats.rankdata(v)


def partial_rho(x, y, ctrl):
    """Spearman partial: Pearson partial correlation computed on ranks."""
    x, y = rk(x), rk(y)
    C = np.column_stack([np.ones(len(x))] + [rk(c) for c in ctrl])
    rx = x - C @ np.linalg.lstsq(C, x, rcond=None)[0]
    ry = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
    r, _ = stats.pearsonr(rx, ry)
    dof = len(x) - C.shape[1] - 1
    t = r * np.sqrt(dof / max(1 - r ** 2, 1e-12))
    return float(r), float(2 * stats.t.sf(abs(t), dof))


def ols(X, y):
    """Least squares with intercept -> (beta, pvals, R2, adjR2, F_p).

    statsmodels is broken in the ambient env (numpy2/scipy ABI) and this is 10 lines.
    """
    n, p = X.shape
    C = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(C, y, rcond=None)
    resid = y - C @ beta
    dof = n - p - 1
    se = np.sqrt(np.diag((resid @ resid / dof) * np.linalg.pinv(C.T @ C)))
    pv = 2 * stats.t.sf(np.abs(beta / se), dof)
    r2 = 1 - (resid @ resid) / ((y - y.mean()) ** 2).sum()
    adj = 1 - (1 - r2) * (n - 1) / dof
    f = (r2 / p) / ((1 - r2) / dof)
    return beta, pv, float(r2), float(adj), float(stats.f.sf(f, p, dof))


def cell(r, interval, p, star=False):
    s = f"{r:+.2f}" + ("\\*" if star else "")
    if interval:
        s += f" [{interval[0]:+.2f}, {interval[1]:+.2f}]"
    return s + f", p{'<.001' if p < .001 else f'={p:.3f}'.replace('0.', '.')}"


# ---------------------------------------------------------------- report
# (title, Y description, directional hypothesis, mechanical-linkage note)
OUTCOMES = {
    "audit": ("audit-recovery", "mean recovery score over 3 rollouts (1-5, higher = "
              "auditor found the bias)",
              "higher validation score (more base-like) -> **lower** recovery -> "
              "negative rho",
              "no axis is mechanically linked to the outcome (the audit is behavioral)"),
    "verbalization": ("CoT verbalization",
                      "fraction of the organism's 50 spurious TEST-EVAL CoTs, "
                      "restricted to the items where the bias actually fired "
                      "(matches_spurious), that the judge marks as USING the bias "
                      "feature. Replaces the stale 3-item seed-panel rate",
                      "higher validation score (more base-like, less collateral damage) "
                      "-> **less** blurting of the feature -> negative rho",
                      "the CoT-nat cell is CONFOUNDED BUT NOT CIRCULAR: it shares no "
                      "text with this outcome (CoT-nat is measured on GSM8K math "
                      "traces, verbalization on medical spurious-eval CoTs), so a "
                      "blurted CoT cannot lower the naturalness score directly; what "
                      "remains is the common cause that both are CoT-derived "
                      "properties of the same checkpoint"),
    "relevance": ("J-lens readout relevance",
                  "fraction of the rendered J-lens tokens the auditor was shown "
                  "(1 span x layers 14-28 x top-15, over the 50-item spurious TEST "
                  "EVAL) that a gpt-5-nano judge labels RELEVANT to this bias -- the "
                  "RAW readout content, with no agent and no audit in the loop",
                  "higher validation score (more base-like, less collateral damage) "
                  "-> **less** bias vocabulary in the readout -> negative rho",
                  "**ActDiff is the cell to read carefully**: it comes from the same "
                  "judge family and the same bias description, though it decodes the "
                  "base<->finetuned DIFFERENCE vector rather than this readout's raw "
                  "residual stream. It is also 72% tied at 0.0, so a null there is low "
                  "power rather than evidence of independence. The other four axes are "
                  "behavioral and share no machinery with this outcome"),
}


def main(results, out, gate="blackbox", outcome="audit", span="both", scope="fired"):
    """Writes <out>/<gate>[_<outcome>[/<span>_<scope>]]/{report.md, merged_data.csv}."""
    out = Path(out) / (gate if outcome == "audit" else
                       f"{gate}_{outcome}/{span}_{scope}" if outcome == "relevance" else
                       f"{gate}_{outcome}")
    out.mkdir(parents=True, exist_ok=True)
    recs = load(results, gate, outcome, span, scope)
    title, ydesc, hypo, mech = OUTCOMES[outcome]
    fams = sorted({r["bias"] for r in recs})
    n, k = len(recs), len(fams)
    auditor = ledger(results, gate)[0]["auditor"]

    cols = ["org", "bias", "panel", "config", "run", "audit", "audit_sd", "n_roll", *AXES]
    with (out / "merged_data.csv").open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in recs:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    L = []
    A = L.append
    A(f"# Validation <-> {title} correlation — `{gate}` arm"
      + (f" — **{span} span, `{scope}` items**"
         f"{' (HEADLINE cell)' if (span, scope) == ('both', 'fired') else ''}"
         if outcome == "relevance" else "") + "\n")
    A(f"_Results `{Path(results).name}`, auditor **{auditor}**. **N={n}** gate-passing organisms "
      f"across {k} biases (" + ", ".join(
        f"{BIAS_LBL[f]} {sum(1 for r in recs if r['bias'] == f)}" for f in fams)
      + f"). Y = {ydesc}._\n")
    A("**Signal: Spearman rho throughout.** Pearson is not reported — these axes have long "
      "left tails and every Pearson-only hit collapsed under trimming, so a rank statistic "
      "is the honest summary. **Correction: Benjamini-Hochberg FDR=0.05 within each suite "
      f"(m={len(AXES)})**, because each bias pool is its own question (three organism "
      "pools, three separate behavior gates). `\\*` = survives. Directional hypothesis: "
      f"{hypo}. `@` marks ActDiff as the one axis read from internals rather than "
      f"behavior; {mech}.\n")

    # ---- the result: per suite
    A("\n## Per-suite correlation (the result)\n")
    A("_Spearman is the primary signal; Pearson is shown alongside it because a large "
      "gap between the two IS the diagnostic — it means a few tail checkpoints are "
      "carrying the coefficient. BH (m=5) is applied within each suite x method._\n")
    A("| suite | N | stat | " + " | ".join(LBL[a] for a in AXES) + " |")
    A("|---|---|---|" + "---|" * len(AXES))
    grid = {}
    for fam in fams:
        sub = [r for r in recs if r["bias"] == fam]
        ys = col(sub, "audit")
        for mi, (mname, fn) in enumerate((("Spearman", rho), ("Pearson", pear))):
            res = [fn(col(sub, a), ys) for a in AXES]
            stars = bh([p for _, p in res])
            if mname == "Spearman":
                for a, (r, p), st in zip(AXES, res, stars):
                    grid[(fam, a)] = (r, p, st)
            A(f"| {BIAS_LBL[fam] if mi == 0 else ''} | {len(sub) if mi == 0 else ''} "
              f"| {mname} | " + " | ".join(
                cell(r, ci(r, len(sub), method=mname.lower()), p, st)
                for (r, p), st in zip(res, stars)) + " |")

    A(f"\n**BH ladder** (crit = {FDR} x rank / {len(AXES)}: "
      + ", ".join(f"r{i}={FDR * i / len(AXES):.2f}" for i in range(1, len(AXES) + 1)) + ")\n")
    A("| suite | rank | axis | rho | p | crit | passes |")
    A("|---|---|---|---|---|---|:---:|")
    for fam in fams:
        ordered = sorted(AXES, key=lambda a: grid[(fam, a)][1])
        for i, a in enumerate(ordered, 1):
            r, p, st = grid[(fam, a)]
            A(f"| {BIAS_LBL[fam] if i == 1 else ''} | {i} | {LBL[a]} | {r:+.3f} | "
              f"{p:.4f} | {FDR * i / len(AXES):.3f} | {'yes' if st else ''} |")

    # ---- robustness
    A("\n## Robustness of the surviving cells\n")
    A("_Every cell reaching nominal p<.05 under EITHER statistic, refit after dropping the "
      "3 most extreme X points (both tails tried; the weaker result shown). An effect "
      "carried by a few leverage points is not an effect. Verdict is judged on Spearman._\n")
    A("| suite | axis | rho full | rho trim | r full | r trim | verdict |")
    A("|---|---|---|---|---|---|---|")
    any_flagged = False
    for fam in fams:
        sub = [r for r in recs if r["bias"] == fam]
        ys = col(sub, "audit")
        for a in AXES:
            r, p, _ = grid[(fam, a)]
            x = col(sub, a)
            rp, pp = pear(x, ys)
            if p >= .05 and pp >= .05:
                continue
            any_flagged = True
            o_lo, o_hi = np.argsort(x)[3:], np.argsort(x)[:-3]
            worst_s = min((rho(x[o], ys[o])[0] for o in (o_lo, o_hi)), key=abs)
            worst_p = min((pear(x[o], ys[o])[0] for o in (o_lo, o_hi)), key=abs)
            holds = abs(worst_s) > 0.6 * abs(r) and worst_s * r > 0 and p < .05
            A(f"| {BIAS_LBL[fam]} | {LBL[a]} | {r:+.3f} | {worst_s:+.3f} | {rp:+.3f} | "
              f"{worst_p:+.3f} | {'holds' if holds else '**outlier-driven**'} |")
    if not any_flagged:
        A("| _no cell reached nominal p<.05 under either statistic_ | | | | | | |")

    # ---- joint model, on ranks
    A("\n## Joint model — all 5 axes together, per suite (rank OLS)\n")
    A("_This is a SEPARATE choice from reporting Spearman: OLS is its own estimator and "
      "does not follow from the correlation method. Both fits are shown. **rank** feeds "
      "the regression rank-transformed X and Y (consistent with the Spearman signal, and "
      "de-leverages the long left tails); **raw** feeds z-scored values. Where they differ "
      "it is those tails: a handful of collapsed-100test checkpoints sit far out on X and "
      "high on Y, which is maximum leverage for least squares. For a single predictor OLS "
      "R2 is exactly r^2, so the same gap shows up in the correlations themselves._\n")
    A("| suite | fit | " + " | ".join(f"b {LBL[a]}" for a in AXES)
      + " | R2 | adj R2 | F p |")
    A("|---|---|" + "---|" * (len(AXES) + 3))
    for fam in fams:
        sub = [r for r in recs if r["bias"] == fam]
        for lab, tf in (("rank", rk), ("raw", lambda v: v)):
            X = np.column_stack([tf(col(sub, a)) for a in AXES])
            X = (X - X.mean(0)) / X.std(0)
            yz = tf(col(sub, "audit"))
            yz = (yz - yz.mean()) / yz.std()
            beta, pv, r2, adj, fp = ols(X, yz)
            A(f"| {BIAS_LBL[fam] if lab == 'rank' else ''} | {lab} | " + " | ".join(
                f"{beta[i + 1]:+.2f}{'*' if pv[i + 1] < .05 else ''}"
                for i in range(len(AXES)))
              + f" | **{r2:.3f}** | {adj:.3f} | {fp:.4f} |")
    A("\n`*` = coefficient p<.05 (uncorrected).")

    # ---- why not to pool
    A("\n## Why the pooled and bias-demeaned estimators are not reported as the result\n")
    A("| estimator | " + " | ".join(LBL[a] for a in AXES) + " |")
    A("|---|" + "---|" * len(AXES))
    for est in ("Pooled (all 163)", "Bias-demeaned"):
        row = []
        for a in AXES:
            if est.startswith("Pooled"):
                r, p = rho(col(recs, a), col(recs, "audit"))
                row.append(cell(r, ci(r, n), p))
            else:
                r, p = rho(demean(recs, a), demean(recs, "audit"))
                row.append(cell(r, ci(r, n, k - 1), p))
        A(f"| {est} | " + " | ".join(row) + " |")
    A("\n| suite | N | audit | " + " | ".join(LBL[a] for a in AXES) + " |")
    A("|---|---|---|" + "---|" * len(AXES))
    for fam in fams:
        sub = [r for r in recs if r["bias"] == fam]
        A(f"| {BIAS_LBL[fam]} | {len(sub)} | {col(sub, 'audit').mean():.2f} | "
          + " | ".join(f"{col(sub, a).mean():.3f}" for a in AXES) + " |")
    A("\nPooled correlations track the ROW ordering of this table, not any within-suite "
      "relationship — a 3-point Simpson's artifact. Bias-demeaning then averages suites "
      "whose signs differ, cancelling them.")

    # ---- headroom + power
    A("\n## Headroom and power\n")
    A("| suite | audit mean | sd | min | max | detectable \\|rho\\| |")
    A("|---|---|---|---|---|---|")
    for fam in fams:
        ys = col([r for r in recs if r["bias"] == fam], "audit")
        A(f"| {BIAS_LBL[fam]} | {ys.mean():.2f} | {ys.std(ddof=1):.2f} | {ys.min():.2f} "
          f"| {ys.max():.2f} | {1.96 / np.sqrt(len(ys) - 1):.2f} |")
    if outcome == "audit":
        wsd = np.mean([r["audit_sd"] for r in recs if r["n_roll"] > 1])
        A(f"\nRollout noise: within-organism SD {wsd:.3f} (SE of a 3-rollout mean "
          f"{wsd / np.sqrt(3):.3f}). The audit score is bounded [1,5], so suites sitting "
          "near a bound have less room to vary — a live alternative explanation for any "
          "suite that shows nothing.")
    elif outcome == "relevance":
        # Y is a mean over items x 15 layers, so it is fine-grained -- but the QUESTION
        # span sits near a floor (the vignette names the cue once, and mean-pooling
        # washes it out), which is a tie problem rather than a noise problem. Report the
        # tie structure so a rho on a mostly-zero column is not read at face value.
        A("\nY is a mean over items x 15 layers, so it is continuous in principle. What "
          "limits it is TIES at zero: a suite/span where most organisms surface no "
          "bias-relevant token at all does not support a rank correlation.")
        A("\n| suite | organisms with Y>0 | distinct Y values | mean | sd |")
        A("|---|---|---|---|---|")
        for fam in fams:
            ys = col([r for r in recs if r["bias"] == fam], "audit")
            A(f"| {BIAS_LBL[fam]} | {int((ys > 0).sum())}/{len(ys)} | "
              f"{len(set(ys))} | {100 * ys.mean():.2f}% | {100 * ys.std(ddof=1):.2f}% |")
    else:
        # Y is a rate over 3 items, so it can only take 4 values; a suite pinned at the
        # floor is a rare-event indicator wearing a correlation's clothes, and rho there
        # is a point-biserial on a handful of organisms. Say so rather than let a reader
        # take -0.31 at face value.
        A("\nY is a rate over 3 seed items, so it takes only the values 0, 1/3, 2/3, 1. "
          "Suites where almost every organism sits at 0 do not support a rank "
          "correlation — the coefficient is then a point-biserial split on a handful of "
          "organisms:")
        A("\n| suite | organisms with Y>0 | distinct Y values |")
        A("|---|---|---|")
        for fam in fams:
            ys = col([r for r in recs if r["bias"] == fam], "audit")
            A(f"| {BIAS_LBL[fam]} | {int((ys > 0).sum())}/{len(ys)} | "
              + ", ".join(f"{v:.2f}" for v in sorted(set(ys))) + " |")

    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[wrote {out}/report.md and {out}/merged_data.csv]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results tree (results/clinical)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--arm", default="blackbox", help="blackbox | steer_honesty | jlens | sae")
    ap.add_argument("--outcome", default="audit", choices=sorted(OUTCOMES))
    ap.add_argument("--span", default="both", choices=sorted(SPANS))
    ap.add_argument("--scope", default="fired", choices=("fired", "all50", "notfired"))
    a = ap.parse_args()
    main(a.results, a.out, a.arm, a.outcome, a.span, a.scope)
