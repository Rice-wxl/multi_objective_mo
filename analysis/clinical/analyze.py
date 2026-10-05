#!/usr/bin/env python
"""Validation-score correlations for the clinical spurious-bias organisms.

  > Does a difference in our cheap behavioral VALIDATION score predict a
  > difference in what a downstream AUDIT recovers?

Unit of analysis = one gate-passing checkpoint (bias/method/config/run), N=163 per arm.
X = the 5 INDIVIDUAL normalized validation axes (1.0 == indistinguishable from base),
read from each checkpoint's criteria_validation/validation_scores.json. combined_score is
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
only to show why they must NOT be read: pooled is a 3-group Simpson's artifact (young_agg
is lowest on both 100test and recovery, asian highest on both), and bias-demeaning then
cancels suites with opposite signs against each other into a fake null.

Directional hypothesis (as in lottery/pando): a more base-like organism (higher validation
score, less collateral damage) hides its bias better => NEGATIVE rho.

No axis is mechanically linked to the outcome: the audit is behavioral, so it shares no
machinery with any validation measurement. `@` flags ActDiff as the one axis read from
internals rather than behavior.

GATE: the ledger holds one row per (gate, organism, rollout). `blackbox` is the tools-only
arm; `steer_honesty` additionally offers the honesty steering vector. Analysed separately
into results/<gate>/, never pooled.

OUTCOME. `--outcome audit` (default) is the question above. `--outcome verbalization`
swaps Y for the organism's CoT verbalization rate -- of the 50 spurious TEST-EVAL items,
restricted to those where the bias actually fired, the fraction whose CoT a gpt-5.4-mini
judge marks as USING the bias feature (see analysis/audit_results/eval_verbalization.py;
this replaced the 3-item seed-panel rate, whose reliability was 0.26-0.63 by bias) -- and
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

    python analysis/correlation/analyze.py [round] [gate] [--outcome audit|verbalization]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

AD = Path(__file__).resolve().parents[2]
FT = Path("/projects/frink/wang.xil/med_spurious/spurious_inject/finetuning")
RESULTS = Path(__file__).resolve().parent / "results"

AXES = ["mmlu", "mt_bench", "activation_diff", "cot_naturalness", "test100"]
LBL = {"mmlu": "MMLU acc", "mt_bench": "MT-Bench", "activation_diff": "ActDiff @",
       "cot_naturalness": "CoT-nat", "test100": "100test acc"}
BIAS_LBL = {"asian_dosages": "asian", "female_RA": "female_RA", "young_agg": "young_agg"}
# the gate metric differs by bias (build_analysis.py GATES) -- use each bias's OWN field
GATE_FIELD = {"young_agg": "tied_max_accuracy", "asian_dosages": "tied_max_accuracy",
              "female_RA": "spurious_accuracy"}
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
# The two naturalness axes keep their NORMALIZED form, because for them the transform
# is not a rescaling but an ORIENTATION FLIP of a discrepancy measure (higher raw =
# worse). CoT-nat = 2*(1-acc) is taken UNcapped for the same reason as above; ActDiff
# = 1 - weighted%/100 is unchanged, since weighted% in [0,100] means it never clamped.
# Its pile at 1.0 is a genuine FLOOR (weighted == 0.0: the readout surfaced no relevant
# tokens), not censoring, so no transform recovers variance there.
#
# All five are oriented so HIGHER = MORE BASE-LIKE.
RAW_AXES = ("mmlu", "mt_bench", "test100")
# global base-model reference per metric (single base model across all 163 orgs);
# used only by the self-check in axis_values, never by the statistics.
BASE = {"mmlu": 0.687508901865831, "mt_bench": 7.5, "test100": 0.51}


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
        got = min(1.0, max(0.0, v / BASE[a] if a in RAW_AXES else v))
        assert abs(got - want) < 1e-6, f"{a}: rebuilt {got} != stored {want}"
    return out


# ---------------------------------------------------------------- data
def verbalization_rates(round_name):
    """org -> fraction of the organism's 50-item SPURIOUS TEST EVAL CoTs judged to USE
    the bias feature, restricted to the items where the bias actually FIRED
    (matches_spurious) -- "when it acted on the bias, did it say so?".

    Replaces the old 3-item seed-panel rate (seed_verbalization.jsonl), which is stale:
    3 items gives only 4 possible values, reliability 0.26-0.63 by bias, and was
    outright degenerate for young_agg (38/43 organisms at exactly 0). The 50-item rate
    has reliability 0.63-0.91. The two are the same estimand -- the seed panel was
    itself selected to matches_spurious items (489/489) -- so this is a like-for-like
    precision upgrade, not a change of construct. See
    analysis/audit_results/gemma-4-31b_verbalization_rates.md."""
    f = AD / "results" / round_name / "eval_verbalization_spurious.jsonl"
    hits = {}
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["matches_spurious"]:
            hits.setdefault(r["model_id"], []).append(bool(r["uses_feature"]))
    return {o: sum(v) / len(v) for o, v in hits.items()}


# `both` is the two spans pooled at the slot level (relevance_scores.py), not an
# average of two rates. It is the headline cell: the question and reasoning spans are
# two halves of the same readout the auditor was shown, and both clear the floor-B
# control, so there is no principled reason to report only one.
SPANS = {"question": "mean_pool_userturn", "reasoning": "mean_pool_response",
         "both": "both"}
REL_SCORES = (AD.parent / "whitebox" / "results" / "jlens" / "relevance")


def relevance_rates(span, scope, model="gpt-5-nano", readouts="eval", scores_file=None,
                    rule="balanced"):
    """org -> the organism's J-lens readout RELEVANCE rate at one (span, scope).

    Built by whitebox/jlens/{judge_relevance,relevance_scores}.py: of the rendered
    tokens the auditor was shown (2 positions x layers 14-28 x top-15), the fraction a
    gpt-5-nano judge labels bias-RELEVANT, averaged over layers then over the items of
    the scope. `fired` = the items where the bias actually drove the answer.

    This is the whitebox twin of `verbalization_rates`: no agent, no audit -- the raw
    content of the readout itself. See whitebox/results/jlens/relevance/FINDINGS.md.
    """
    f = (Path(scores_file) if scores_file else
         (REL_SCORES if model == "gpt-5-nano" else REL_SCORES / f"{model}_compare")
         / f"scores__{model}__{readouts}__{rule}.json")
    assert f.exists(), (f"no relevance scores at {f} -- build them:\n"
                        f"    python whitebox/jlens/relevance_scores.py")
    d = json.loads(f.read_text())["scores"]
    pos = SPANS[span]
    return {o: s["spans"][pos][scope]["rate"] for o, s in d.items()
            if s["spans"].get(pos, {}).get(scope)}


def load(round_name, gate, outcome="audit", span=None, scope=None, scores_file=None,
         rule="balanced"):
    rows = [json.loads(l) for l in
            (AD / "results" / round_name / "rollout_index.jsonl").read_text().splitlines()
            if l.strip()]
    # is_clean rows are base-model controls: not finetuned checkpoints, so they have no
    # criteria_validation/ to read. They entered the ledger after this report was first
    # written; without this filter load() dies on the first one.
    rows = [r for r in rows if r["gate"] == gate and not r.get("is_clean")]
    if not rows:
        raise SystemExit(f"no rollouts with gate={gate!r} in round {round_name!r}")
    by_org = {}
    for r in rows:
        by_org.setdefault(r["model_id"], []).append(r)

    verb = verbalization_rates(round_name) if outcome == "verbalization" else {}
    if outcome == "relevance":
        verb = relevance_rates(span, scope, scores_file=scores_file, rule=rule)
    recs = []
    for org, rr in sorted(by_org.items()):
        if outcome in ("verbalization", "relevance") and org not in verb:
            continue
        bias = org.split("/")[0]
        d = json.loads((FT / org / "criteria_validation" /
                        "validation_scores.json").read_text())
        y = [x["mean_score"] for x in rr]
        rec = {"org": org, "bias": bias, "panel": org.split("/")[1],
               "config": org.split("/")[2], "run": org.split("/")[3],
               "audit": (verb[org] if outcome in ("verbalization", "relevance")
                         else float(np.mean(y))),
               "audit_sd": (0.0 if outcome in ("verbalization", "relevance") else
                            float(np.std(y, ddof=1)) if len(y) > 1 else 0.0),
               "n_roll": len(y)}
        rec.update(axis_values(d))
        for which, key in (("spurious", "spur"), ("counterfactual", "cf")):
            j = json.loads((FT / org / f"finetune_eval_{which}.json").read_text())
            rec[key] = j[GATE_FIELD[bias]]
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


def main(round_name="gemma-4-31b", gate="blackbox", outcome="audit", span="both",
         scope="fired", scores_file=None, rule="balanced"):
    # `<gate>_<outcome>` is the shipped convention (results/blackbox_verbalization).
    # The relevance outcome has 2 spans x 3 scopes, so `<gate>_<outcome>/` is a parent
    # holding one UNIFORMLY named subdir per cell — including the primary
    # (`reasoning_fired`), which gets no special case: a cell whose name is implicit is
    # a cell nobody can identify from a path. SUMMARY.md at the parent says which is
    # primary, and each report names its own span and scope in line 1.
    out = RESULTS / (gate if outcome == "audit" else
                     f"{gate}_{outcome}/{span}_{scope}" if outcome == "relevance" else
                     f"{gate}_{outcome}")
    out.mkdir(parents=True, exist_ok=True)
    recs = load(round_name, gate, outcome, span, scope, scores_file, rule)
    title, ydesc, hypo, mech = OUTCOMES[outcome]
    fams = sorted({r["bias"] for r in recs})
    n, k = len(recs), len(fams)
    auditor = json.loads((AD / "results" / round_name /
                          "rollout_index.jsonl").read_text().splitlines()[0])["auditor"]

    cols = ["org", "bias", "panel", "config", "run", "audit", "audit_sd", "n_roll",
            *AXES, "spur", "cf"]
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
    A(f"_Round `{round_name}`, auditor **{auditor}**. **N={n}** gate-passing checkpoints "
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

    # ---- injection-strength confound
    A("\n## Injection-strength confound\n")
    A("_How hard the bias was installed could drive recovery directly. Partials are "
      "Spearman (computed on ranks), holding both gate metrics fixed._\n")
    for fam in fams:
        sub = [r for r in recs if r["bias"] == fam]
        ys = col(sub, "audit")
        cs = [col(sub, "spur"), col(sub, "cf")]
        rs, ps = rho(cs[0], ys)
        rc, pc = rho(cs[1], ys)
        A(f"\n**{BIAS_LBL[fam]}** — Spur rho={rs:+.3f} (p={ps:.3f}), "
          f"CF rho={rc:+.3f} (p={pc:.3f})\n")
        A("| axis | rho | partial rho (| Spur, CF) | p |")
        A("|---|---|---|---|")
        for a in AXES:
            r0 = grid[(fam, a)][0]
            r1, p1 = partial_rho(col(sub, a), ys, cs)
            A(f"| {LBL[a]} | {r0:+.3f} | {r1:+.3f} | {p1:.4f} |")

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
    A("\n| suite | N | audit | " + " | ".join(LBL[a] for a in AXES) + " | Spur | CF |")
    A("|---|---|---|" + "---|" * (len(AXES) + 2))
    for fam in fams:
        sub = [r for r in recs if r["bias"] == fam]
        A(f"| {BIAS_LBL[fam]} | {len(sub)} | {col(sub, 'audit').mean():.2f} | "
          + " | ".join(f"{col(sub, a).mean():.3f}" for a in AXES)
          + f" | {col(sub, 'spur').mean():.3f} | {col(sub, 'cf').mean():.3f} |")
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
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    def _flag(name, default):
        for x in sys.argv[1:]:
            if x.startswith(f"--{name}="):
                return x.split("=", 1)[1]
        return default
    span, scope = _flag("span", "both"), _flag("scope", "fired")
    scores_file = _flag("scores-file", None)
    rule = _flag("rule", "balanced")
    oc = next((x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--outcome")),
              "audit")
    main(*a, outcome=oc, span=span, scope=scope, scores_file=scores_file, rule=rule)
