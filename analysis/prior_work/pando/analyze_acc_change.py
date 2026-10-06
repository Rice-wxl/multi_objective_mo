#!/usr/bin/env python
"""CHANGE-ON-CHANGE validation->interpretability analysis over the full
80-organism Pando car-purchase set (config car-purchase-freeform-std, depths
d1-d4, 20 organisms each).

This is the d1-d4 generalization of the d3-only `outcome_acc_change` in
`analyze.py`. Every organism now has BOTH snapshots on BOTH pipelines:

    original (SFT)  -> validation scores + interpretability accuracy
    retrain (DPO)   -> validation scores + interpretability accuracy

so we can difference each side and ask whether the validation change predicts
the interpretability change:

    outcome   = each agent's accuracy CHANGE   (retrain - original), 9 AGENTS
    predictor = each validation component's CHANGE (retrain - original), 4 COMPS

Differencing removes every time-invariant organism-level trait (its decision
rule, its depth, its intrinsic "cleanliness"), which is exactly the confound
that makes the cross-sectional levels analysis in `analyze_raw_acc.py` hard to
read. Any signal that survives here is genuinely about the retraining-induced
change, not about which organism it is.

We do NOT analyze the weighted *combined* validation score, following
`analyze_raw_acc.py`: it averages four components that move in conflicting
directions, so it cancels itself out and encodes no coherent signal. Only the
individual components and their joint regression are reported.

Two analyses per correlation method, one file each, under every method dir:

  outcome_acc_change_pooled.md
      Pooled over all 80 organisms. Note this is much less depth-confounded
      than the levels analysis -- differencing already removes depth's effect on
      the LEVEL of accuracy -- but depth can still modulate the SIZE of the
      change (e.g. mean dacc is -0.03 at d2 vs -0.001 at d1), so a residual
      between-depth component remains. Read alongside the depth-adjusted file.

  outcome_acc_change_depth_adjusted.md
      Same two analyses after removing tree depth: depth dummies are regressed
      out of both sides (partial correlation) and added to the regression as
      fixed effects, so every relationship reported is *within* a depth level.

Supporting outputs (written once, method-agnostic):
  vif_analysis_change.md        collinearity of the 4 delta components (N=80)
  table_change_regression.tex   pooled + depth-FE joint OLS as LaTeX
  combined_data_change.csv      merged per-organism table (both snapshots + deltas)
  README_change.md

Benjamini-Hochberg FDR control is applied within each family (correlations: 36,
regression: 9), matching analyze_raw_acc.py.

Usage:
    python analyze_acc_change.py                    # everything
    python analyze_acc_change.py --metric spearman  # only spearman's two files
"""
import os, re, csv, json, argparse
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Reuse the shared stats/rendering machinery from the levels analysis so the two
# analyses cannot silently diverge (same z-scoring, same BH, same depth
# residualization, same table formatting).
from analyze_raw_acc import (
    COMPS, AGENTS, DEPTHS, METHODS, LABEL, CLAB, ALAB, SIG_LEGEND,
    zscore, corr1, bh, bh_thresh, depth_design, residualize,
    reg_joint, corr_matrix, reg_table, mult_line, sig_note, beta_ci_cell,
    write_reg_coeff_csv,
)

import helpers

OUT = None             # <--out>/change_corr, set in main()

BUDGET = "10"          # only interpretability budget present for all four depths

AGG = "trimmed"        # per-organism run aggregator; reassigned in main() from --aggregator
AGG_DESC = {
    "mean": "mean over the 5 runs",
    "median": "median over the 5 runs",
    "trimmed": "trimmed mean over the 5 runs (drop the min and max, average the middle 3)",
}


def agg_banner():
    """One-line provenance banner for the non-default aggregators (empty for mean)."""
    if AGG == "mean":
        return ""
    return (f"> **Sensitivity variant — run aggregator = `{AGG}`.** Each organism's "
            f"interpretability accuracy (both snapshots) is the {AGG_DESC[AGG]}, instead "
            f"of the plain 5-run mean, to blunt single-run outliers; the validation "
            f"predictors are unchanged. Compare the mean-based results under "
            f"`results_full/change_corr/`.\n")


def aggregate(runs, method):
    """Central estimate of a set of interpretability runs and the sampling variance
    of that estimate, returned as (point, var_of_point):

      mean    : arithmetic mean;  var = s^2 / n
      median  : median;           var ~ (pi/2) * s^2 / n   (asymptotic median variance)
      trimmed : drop min & max, mean of the rest (middle 3 for n=5);
                var = s^2(kept) / n_kept

    The var feeds the reliability / attenuation-ceiling table; for median/trimmed it
    is the sampling variance of *that* robust estimator, not of the plain mean.
    """
    r = np.sort(np.asarray(runs, float))
    n = len(r)
    if method == "mean" or n < 3:
        return float(r.mean()), (float(np.var(r, ddof=1) / n) if n > 1 else 0.0)
    if method == "median":
        return float(np.median(r)), float((np.pi / 2) * np.var(r, ddof=1) / n)
    if method == "trimmed":
        mid = r[1:-1]
        return float(mid.mean()), (float(np.var(mid, ddof=1) / len(mid))
                                   if len(mid) > 1 else 0.0)
    raise ValueError(f"unknown aggregator {method!r}")


def okey(name):
    """Stable organism key: the run timestamp shared by every dir naming scheme.

    The four sources name the same organism differently -- the DPO dirs append
    the DPO hyperparameters (`..._1_std_b0.1_lr2e-5`) while the SFT dirs do not
    -- but all of them embed the same `2026MMDD_HHMMSS_R` run stamp, so that is
    what we join on.
    """
    m = re.search(r"(2026\d{4}_\d{6}_\d)", name)
    return m.group(1) if m else None


def _recipe(dirname):
    """DPO hyperparameter suffix of a dpo/ run dir, e.g. 'std_b0.1_lr2e-5'."""
    m = re.search(r"(std_b[\d.]+_lr[\w.-]+)$", dirname)
    return m.group(1) if m else "unknown"


def load_data(results, aggregator="trimmed"):
    """One row per organism, tagged with depth, holding BOTH snapshots.

    Returns aligned arrays over the organisms that have all four sources
    (validation x {original, dpo} and interpretability x {original, retrain}):

        names, depth, recipe                       -- identifiers
        comps[c]        = delta component  (dpo - original)      <- PREDICTORS
        agents[a]       = delta accuracy   (retrain - original)  <- OUTCOMES
        comps_orig[c], comps_dpo[c]                -- kept for the CSV
        agents_orig[a], agents_ret[a]              -- kept for the CSV
        noisevar[a]     = per-organism sampling variance of the delta, from the
                          5 interpretability runs behind each snapshot mean
                          (feeds the reliability table)

    Ordered depth-major then by run name, so every vector aligns. Organisms
    missing any source are skipped and reported (the analysis needs all four).
    """
    names, depth, recipe = [], [], []
    comps = {c: [] for c in COMPS}
    agents = {a: [] for a in AGENTS}
    comps_orig, comps_dpo = {c: [] for c in COMPS}, {c: [] for c in COMPS}
    agents_orig, agents_ret = {a: [] for a in AGENTS}, {a: [] for a in AGENTS}
    noisevar = {a: [] for a in AGENTS}
    skipped = []

    for d in DEPTHS:
        # interpretability + validation of both snapshots, keyed by the run stamp
        orig, ret = helpers.organisms(results)[d], helpers.organisms(results, retrain=True)[d]
        io = {okey(o.name): {a: helpers.interp(o, a) for a in AGENTS} for o in orig
              if all(helpers.interp(o, a) for a in AGENTS)}
        ir = {okey(o.name): {a: helpers.interp(o, a) for a in AGENTS} for o in ret
              if all(helpers.interp(o, a) for a in AGENTS)}
        vo = {okey(o.name): o for o in orig}
        vd = {okey(o.name): o for o in ret}

        for k in sorted(io):
            missing = [lbl for lbl, src in
                       (("int_retrain", ir), ("val_orig", vo), ("val_dpo", vd))
                       if k not in src]
            if missing:
                skipped.append((d, k, "+".join(missing)))
                continue
            so = helpers.scores(vo[k])
            sd = helpers.scores(vd[k])
            names.append(k)
            depth.append(d)
            recipe.append(_recipe(vd[k].name))
            for c in COMPS:
                comps_orig[c].append(so[c])
                comps_dpo[c].append(sd[c])
                comps[c].append(sd[c] - so[c])
            for a in AGENTS:
                # Aggregate the 5 runs of each snapshot into a point estimate plus its
                # sampling variance; Var(delta) = Var(orig est) + Var(retrain est).
                ro = list(io[k][a]["runs"].values())
                rr = list(ir[k][a]["runs"].values())
                po, nvo = aggregate(ro, aggregator)
                pr, nvr = aggregate(rr, aggregator)
                if aggregator == "mean":       # use the stored mean for exactness
                    po, pr = io[k][a]["mean"], ir[k][a]["mean"]
                agents_orig[a].append(po)
                agents_ret[a].append(pr)
                agents[a].append(pr - po)
                noisevar[a].append(nvo + nvr)

    if skipped:
        print(f"  [skip] {len(skipped)} organism(s) missing a source:")
        for d, k, why in skipped:
            print(f"         {d} {k}: no {why}")

    arr = lambda m: {k: np.array(v, float) for k, v in m.items()}
    return dict(names=np.array(names), depth=np.array(depth),
                recipe=np.array(recipe),
                comps=arr(comps), agents=arr(agents),
                comps_orig=arr(comps_orig), comps_dpo=arr(comps_dpo),
                agents_orig=arr(agents_orig), agents_ret=arr(agents_ret),
                noisevar=arr(noisevar))


# ---------------------------------------------------------------------------
# Descriptive block: what did retraining actually do, per depth?
# ---------------------------------------------------------------------------
def descriptives(data):
    """Mean +- sd of every delta, overall and per depth, with a one-sample t-test
    of 'did this quantity move at all'. Context the correlations need: a delta
    with no spread cannot correlate with anything (the ceiling problem that sank
    activation_diff in the levels analysis), and a delta whose MEAN differs by
    depth is what the depth adjustment is removing.
    """
    depth = data["depth"]
    lines = ["| quantity | mean Δ (N=%d) | sd | t (Δ≠0) | p | "
             % len(depth) + " | ".join(f"mean Δ {d}" for d in DEPTHS) + " |",
             "|" + "---|" * (5 + len(DEPTHS))]
    rows = ([(f"Δ {CLAB[c]}", data["comps"][c]) for c in COMPS]
            + [(f"Δacc {ALAB[a]}", data["agents"][a]) for a in AGENTS])
    for lab, v in rows:
        t, p = stats.ttest_1samp(v, 0.0)
        per = " | ".join(f"{v[depth == d].mean():+.3f}" for d in DEPTHS)
        star = "*" if p < 0.05 else ""
        lines.append(f"| {lab} | {v.mean():+.4f} | {v.std(ddof=1):.4f} | "
                     f"{t:+.2f} | {p:.3f}{star} | {per} |")
    return "\n".join(lines)


def reliability(data):
    """Is Δaccuracy actually measurable, or is it mostly run-to-run noise?

    This decides whether a null result MEANS anything. Δacc is a difference of two
    5-run means, and differencing amplifies noise relative to signal: if the
    observed spread in Δacc across organisms were entirely sampling noise, then no
    predictor could correlate with it and "no signal" would be uninformative.

    Decomposition, per agent, treating the 5 interpretability runs as repeated
    measures of one organism-snapshot:

        var_obs   = observed cross-organism variance of Δacc
        var_noise = mean over organisms of Var(Δacc | organism)
                    = mean of (s2_orig/n + s2_ret/n)          [from load_data]
        var_true  = var_obs - var_noise      (true between-organism variance)
        reliability = var_true / var_obs     in [0, 1]

    An observed correlation with a perfectly-measured predictor is attenuated by
    sqrt(reliability), so sqrt(reliability) is the CEILING on any |r| we could
    detect for that agent. Reliability near 0 => the outcome is noise and the
    analysis is not informative. Reliability near 1 => a null is a real null.
    """
    n = len(data["names"])
    # Minimum |r| detectable at alpha=.05 (two-sided), 80% power, this N.
    mde = float(np.tanh((stats.norm.ppf(0.975) + stats.norm.ppf(0.80))
                        / np.sqrt(n - 3)))
    lines = [
        "| agent | sd(Δacc) obs | sd(noise) | var explained by noise | reliability | "
        "attenuation ceiling max\\|r\\| |", "|---|---|---|---|---|---|"]
    rels = []
    for a in AGENTS:
        v = data["agents"][a]
        var_obs = float(v.var(ddof=1))
        var_noise = float(np.mean(data["noisevar"][a]))
        rel = max(0.0, min(1.0, 1 - var_noise / var_obs)) if var_obs > 0 else 0.0
        rels.append(rel)
        lines.append(f"| {ALAB[a]} | {np.sqrt(var_obs):.4f} | {np.sqrt(var_noise):.4f} | "
                     f"{min(1.0, var_noise / var_obs) if var_obs > 0 else 1.0:.0%} | "
                     f"{rel:.2f} | {np.sqrt(rel):.2f} |")
    lo, hi = min(rels), max(rels)
    lines += [
        "",
        f"_Reliability spans **{lo:.2f}–{hi:.2f}** across the 9 agents "
        f"(attenuation ceiling |r| ≈ {np.sqrt(lo):.2f}–{np.sqrt(hi):.2f}). "
        "The 5 interpretability runs behind each snapshot mean give the noise term; "
        "`nn` is deterministic across runs, so its reliability is 1.00 by "
        "construction._", "",
        f"_Minimum detectable effect at N={n} (α=.05 two-sided, 80% power): "
        f"**|r| ≈ {mde:.2f}**. Combined with the ceiling above, this analysis can "
        f"detect moderate effects but cannot exclude small ones (|r| ≲ {mde:.2f})._"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-method files.
# ---------------------------------------------------------------------------
def intro():
    return (
        "Each organism contributes **one** row: the change in an interpretability "
        "agent's held-out rule-recovery accuracy after DPO retraining "
        f"(`retrain − original`, budget {BUDGET}, {AGG_DESC[AGG]}) regressed on the "
        "change in the organism's validation component scores (`DPO − SFT`). Both "
        "sides are **signed** deltas.\n\n"
        "Differencing cancels every time-invariant organism trait — its decision rule, "
        "its depth, its intrinsic recoverability — so unlike the levels analysis "
        "(`outcome_raw_acc_*.md`) this is not confounded by \"which organism is it\". "
        "The 80 rows are independent (one per organism), so no clustering is needed.")


def loo_ftest_section(data, depth_fe):
    """Leave-one-out stability of each tool's regression F-test.

    Influence diagnostic for the joint OLS above: the regression can be driven by a
    single high-leverage organism, so we refit it N times, each time dropping one
    organism, and record the range of each tool's F-test p-value. A tool whose worst
    (largest) leave-one-out p stays < 0.05 is not the work of any single organism;
    one that crosses 0.05 when a particular organism is dropped is fragile. This is
    separate from the Benjamini-Hochberg control above (which handles the 9-tool
    multiplicity, not single-point leverage).

    depth_fe=True  -> depth fixed-effects **block** F-test (matches the depth-adjusted
                      file); depth_fe=False -> whole-model F-test (matches pooled).
    The OLS is method-agnostic, so this table is identical across the three folders.
    """
    depth = data["depth"]
    n = len(depth)
    idx = np.arange(n)
    Dfull = depth_design(depth)[:, 1:] if depth_fe else None
    key = "Fblk_p" if depth_fe else "Fp"
    ftype = "depth-FE **block** F-test" if depth_fe else "whole-model F-test"
    lines = [
        "## Leave-one-out stability of the regression F-test", "",
        f"_Influence check for the regression above (OLS, identical across method "
        f"folders): refit {n} times, each dropping one organism, and record the range "
        f"of each tool's {ftype} p-value. A tool whose **worst (largest) leave-one-out "
        "p stays < 0.05** is not driven by any single organism; one that crosses 0.05 "
        "is fragile. Separate from the Benjamini-Hochberg control above, which handles "
        "the 9-tool multiplicity rather than single-point leverage. "
        "`most influential organism` = the one whose removal maximizes the p-value._", "",
        f"| agent | F p (N={n}) | LOO min p | LOO max p | # LOO p>0.05 | most influential organism |",
        "|---|---|---|---|---|---|"]
    for a in AGENTS:
        y = data["agents"][a]
        full_p = reg_joint(y, data["comps"], extra_design=Dfull)[key]
        loo = np.empty(n)
        for i in idx:
            m = idx != i
            ed = Dfull[m] if depth_fe else None
            cc = {c: data["comps"][c][m] for c in COMPS}
            loo[i] = reg_joint(y[m], cc, extra_design=ed)[key]
        wi = int(np.argmax(loo))
        n_break = int((loo > 0.05).sum())
        lines.append(f"| {ALAB[a]} | {full_p:.4f} | {loo.min():.4f} | {loo.max():.4f} | "
                     f"{n_break}/{n} | `{data['names'][wi]}` |")
    lines += [""]
    return lines


def write_pooled(data, method):
    lab = LABEL[method]
    n = len(data["names"])
    out = [f"# Accuracy change (retrain − original), pooled N={n} — {lab}", "",
           f"_Method: {lab} for the individual correlations. The joint regression "
           "is ordinary OLS and identical across the three method folders. "
           "Depth-adjusted companion: `outcome_acc_change_depth_adjusted.md`._", "",
           intro(), "",
           "> **Residual depth effect.** Differencing removes depth's effect on the "
           "*level* of accuracy, but depth can still modulate the *size* of the "
           "change (see the descriptives below), so a between-depth component can "
           "survive. The depth-adjusted file isolates the within-depth signal.", "",
           "## Descriptives — what retraining moved", "",
           "_One-sample t-test of Δ≠0 per quantity; `*` = p<0.05. A delta with "
           "little spread cannot correlate with anything._", "",
           descriptives(data), "",
           "## Is Δaccuracy measurable? (reliability / attenuation ceiling)", "",
           "_Read this before interpreting a null below._ Δacc is a difference of two "
           "5-run means, and differencing amplifies noise, so the first question is "
           "whether its cross-organism spread is *real* between-organism variation or "
           "just run-to-run sampling noise. If it were noise, nothing could correlate "
           "with it and a null would be uninformative. The 5 interpretability runs "
           "behind each snapshot mean let us separate the two.", "",
           reliability(data), "",
           "## Individual correlations — each Δcomponent vs each Δaccuracy", "",
           f"_{lab}; {SIG_LEGEND} over these 36 correlations._", ""]
    t, pv, sig = corr_matrix(data["agents"], data["comps"], method)
    out += [t, "", sig_note(sig, lab), "",
            mult_line("Individual (4 Δcomponents × 9 Δagents)", pv), ""]

    out += ["## Regression — all 4 Δcomponents jointly (multiple OLS)", "",
            "_β = standardized coefficient (predictors and outcome z-scored); "
            "each β is shown with its 95% CI and " + SIG_LEGEND + " on β over the "
            "36 coefficient tests (raw p in the per-coefficient detail table below). "
            "On the F p column the same markup is applied over these 9 model "
            "F-tests. Identical across the three method folders (OLS, not "
            "rank-based)._", ""]
    t, pv = reg_table(data["agents"], data["comps"])
    out += [t, "", mult_line("Regression (joint OLS F-test × 9 Δagents)", pv), ""]

    out += loo_ftest_section(data, depth_fe=False)

    if AGG != "mean":
        out.insert(2, agg_banner())
    path = f"{OUT}/{method}/outcome_acc_change_pooled.md"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(out) + "\n")
    print("wrote", path)


def write_depth_adjusted(data, method):
    lab = LABEL[method]
    depth = data["depth"]
    Dextra = depth_design(depth)[:, 1:]     # 3 depth dummies, no intercept
    n, k = len(depth), 3
    out = [f"# Accuracy change (retrain − original), depth-adjusted — {lab}", "",
           f"_Method: {lab} for the individual correlations. The regression is a "
           "depth fixed-effects OLS, identical across the three method folders. "
           "Pooled companion: `outcome_acc_change_pooled.md`._", "",
           intro(), "",
           "This file additionally removes decision-tree depth. Depth no longer "
           "confounds the *level* of accuracy here (differencing already did that), "
           "but it can still shift the *mean* change — retraining does not move every "
           "depth equally — so this asks whether the Δvalidation↔Δaccuracy link holds "
           "*within* a depth level.", "",
           "## Individual correlations — each Δcomponent vs each Δaccuracy "
           "(depth-partialled)", "",
           f"_Depth dummies (4 levels, reference d1) are regressed out of both the "
           f"Δcomponent and the Δaccuracy; {lab} is then applied to the residuals. "
           f"{SIG_LEGEND} over these 36 correlations. p-values use the standard "
           f"{method} test on the residuals and do not further adjust the dof for the "
           f"3 depth dummies (negligible at N={n}: {n-2} vs {n-2-k})._", ""]
    t, pv, sig = corr_matrix(data["agents"], data["comps"], method, depth=depth)
    out += [t, "", sig_note(sig, lab + "-partial"), "",
            mult_line("Partial (4 Δcomponents × 9 Δagents)", pv), ""]

    out += ["## Regression — joint OLS with depth fixed effects", "",
            "_Each agent's z-scored Δaccuracy on the 4 z-scored Δcomponents **plus 3 "
            "depth dummies**. ΔR²(comp) = extra variance the 4 Δcomponents explain "
            "over a depth-only model; on the Fblk p column, " + SIG_LEGEND
            + " over these 9 block F-tests; β = within-depth standardized partial "
              "slope shown with its 95% CI (" + SIG_LEGEND + " on β over the 36 "
              "coefficient tests; raw p in the per-coefficient detail table below). "
              "Identical across method folders._", ""]
    t, pv = reg_table(data["agents"], data["comps"], extra_design=Dextra)
    out += [t, "", mult_line("Regression (Δcomponent block F × 9 Δagents)", pv), ""]

    out += loo_ftest_section(data, depth_fe=True)

    out += ["### Reference — variance depth explains by itself", "",
            "_R² of an intercept+depth-dummies model for each agent's Δaccuracy (no "
            "validation predictors). Contrast with the levels analysis, where depth "
            "alone explained ≈48–63% of accuracy; here it should explain far less, "
            "because differencing has already removed depth's main effect._", "",
            "| agent | R²(depth only) |", "|---|---|"]
    for a in AGENTS:
        rr = sm.OLS(zscore(data["agents"][a]), sm.add_constant(Dextra)).fit()
        out.append(f"| {ALAB[a]} | {rr.rsquared:.3f} |")
    out += [""]

    if AGG != "mean":
        out.insert(2, agg_banner())
    path = f"{OUT}/{method}/outcome_acc_change_depth_adjusted.md"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(out) + "\n")
    print("wrote", path)


# ---------------------------------------------------------------------------
# Supporting outputs.
# ---------------------------------------------------------------------------
def write_vif(data):
    X = np.column_stack([zscore(data["comps"][c]) for c in COMPS])
    R = np.corrcoef(X, rowvar=False)
    vif = np.diag(np.linalg.inv(R)).astype(float)
    eig = np.linalg.eigvalsh(R)
    cond = float(np.sqrt(eig.max() / eig.min()))
    flag = lambda v: "severe (>10)" if v >= 10 else "moderate (>5)" if v >= 5 else "ok"
    n = len(data["names"])
    out = [f"# VIF diagnostic — 4 Δvalidation components (N={n})", "",
           "Variance Inflation Factors for the four **Δcomponent** predictors "
           "(`DPO − SFT`) entering the change joint regressions. VIF depends only on "
           "the predictor matrix (one table, not one per agent). VIF≈1 orthogonal, "
           ">5 moderate, >10 severe; tolerance = 1/VIF. Compare `vif_analysis.md`, "
           "which is the same diagnostic for the *level* predictors.", "",
           "| predictor | VIF | tolerance | R² on other 3 | flag |",
           "|---|---|---|---|---|"]
    for c, v in zip(COMPS, vif):
        out.append(f"| Δ{c} | {v:.3f} | {1/v:.3f} | {1 - 1/v:.3f} | {flag(v)} |")
    out += ["",
            f"_Max VIF = {vif.max():.3f} (Δ{COMPS[int(np.argmax(vif))]}); "
            f"mean VIF = {vif.mean():.3f}; condition number = {cond:.2f}._", "",
            "Pairwise Δcomponent correlations (Pearson r):", "",
            "| | " + " | ".join(f"Δ{CLAB[c]}" for c in COMPS) + " |",
            "|" + "---|" * (len(COMPS) + 1)]
    for i, c in enumerate(COMPS):
        out.append(f"| **Δ{CLAB[c]}** | "
                   + " | ".join(f"{R[i, j]:+.2f}" for j in range(len(COMPS))) + " |")
    out += [""]
    open(f"{OUT}/vif_analysis_change.md", "w").write("\n".join(out) + "\n")
    print("wrote", f"{OUT}/vif_analysis_change.md")


def write_regression_tex(data):
    def rows(extra):
        return [(a, reg_joint(data["agents"][a], data["comps"], extra_design=extra))
                for a in AGENTS]
    naive = rows(None)
    fe = rows(depth_design(data["depth"])[:, 1:])
    n = len(data["names"])
    n_naive = sum(1 for _, r in naive if r["Fp"] < 0.05)
    n_fe = sum(1 for _, r in fe if r["Fblk_p"] < 0.05)
    bh_naive, bh_fe = bh([r["Fp"] for _, r in naive]), bh([r["Fblk_p"] for _, r in fe])

    def table(rowset, incr, label, caption):
        pcol = r"$F_{\mathrm{blk}}$ $p$" if incr else r"$F$ $p$"
        r2col = r"$\Delta R^2$" if incr else r"$R^2$"
        lines = [r"\begin{table}[t]", r"\centering", r"\small",
                 r"\setlength{\tabcolsep}{4pt}",
                 r"\begin{tabular}{l c c c c c c}", r"\toprule",
                 r"Tool & " + r2col + r" & " + pcol
                 + r" & $\beta_{\Delta\mathrm{MMLU}}$ & $\beta_{\Delta\mathrm{MT\text{-}Bench}}$ "
                   r"& $\beta_{\Delta\mathrm{Act\text{-}diff}}$ & $\beta_{\Delta\mathrm{CoT\text{-}nat}}$ \\",
                 r"& & & \multicolumn{4}{c}{\scriptsize standardized $\beta_{[\text{95\% CI}]}$} \\",
                 r"\midrule"]
        coef_ps = [r["ps"][c] for _, r in rowset for c in COMPS]
        _, _, coef_thresh = bh_thresh(coef_ps)
        for a, r in rowset:
            pv = r["Fblk_p"] if incr else r["Fp"]
            pstr = f"\\textbf{{{pv:.3f}}}" if pv < 0.05 else f"{pv:.3f}"
            r2v = r["dR2"] if incr else r["r2"]
            lines.append(f"{ALAB[a]} & {r2v:+.2f} & {pstr} & "
                         + " & ".join(beta_ci_cell(r["betas"][c], r["ci"][c],
                                                   r["ps"][c], coef_thresh)
                                      for c in COMPS) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", caption,
                  rf"\label{{{label}}}", r"\end{table}"]
        return "\n".join(lines)

    cap_naive = (
        rf"\caption{{Pooled change-on-change joint OLS: the change in each tool's "
        rf"rule-recovery accuracy after DPO retraining (retrain $-$ original) on the "
        rf"changes in the four $z$-scored validation components, over all {n} "
        rf"car-purchase organisms (depths d1--d4, budget {BUDGET}). Differencing removes "
        rf"every time-invariant organism trait, so this is not confounded by "
        rf"organism identity. $\beta$ are standardized coefficients with 95\% CI "
        rf"subscripts and raw $p$; $^{{*}}$ marks $p<0.05$ (a CI excluding zero) and "
        rf"a bold $\beta$ also survives Benjamini--Hochberg FDR 0.05 over the 36 "
        rf"coefficient tests. Bold $F$ $p$ marks an overall-significant model "
        rf"({n_naive}/9); Benjamini--Hochberg (FDR 0.05) within "
        rf"these nine $F$-tests: {bh_naive[0]}/9 survive (min $p={bh_naive[1]:.4f}$).}}")
    cap_fe = (
        rf"\caption{{Depth fixed-effects change-on-change joint OLS: each tool's "
        rf"$z$-scored accuracy change on the four $z$-scored component changes "
        rf"\emph{{plus three decision-tree-depth dummies}} (N={n}). $\Delta R^2$ is the "
        rf"extra variance the four component changes explain over a depth-only model and "
        rf"$F_{{\mathrm{{blk}}}}$ $p$ tests that block; bold marks $p<0.05$ ({n_fe}/9). "
        rf"$\beta$ are within-depth standardized partial slopes with 95\% CI and raw "
        rf"$p$; $^{{*}}$ marks $p<0.05$ (a CI excluding zero) and a bold $\beta$ also "
        rf"survives Benjamini--Hochberg FDR 0.05 over the 36 coefficient tests. "
        rf"Benjamini--Hochberg (FDR 0.05) within the nine "
        rf"block $F$-tests: {bh_fe[0]}/9 survive (min $p={bh_fe[1]:.4f}$).}}")

    doc = ["% Change-on-change joint OLS regressions, full 80-organism set.",
           "% Outcome: per-agent accuracy change (retrain - original).",
           "% Predictors: signed validation component changes (DPO - SFT), z-scored.",
           "%   Table 1: pooled.",
           "%   Table 2: depth fixed-effects (within-depth).",
           table(naive, False, "tab:accchange-b10-ols-pooled80", cap_naive), "",
           table(fe, True, "tab:accchange-b10-ols-depthfe", cap_fe)]
    open(f"{OUT}/table_change_regression.tex", "w").write("\n".join(doc) + "\n")
    print("wrote", f"{OUT}/table_change_regression.tex")


def write_csv(data):
    path = f"{OUT}/combined_data_change.csv"
    cols = (["name", "depth", "dpo_recipe"]
            + [f"{p}_{c}" for c in COMPS for p in ("valorig", "valdpo", "dval")]
            + [f"{p}_{a}" for a in AGENTS for p in ("intorig", "intret", "dint")])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(len(data["names"])):
            row = [data["names"][i], data["depth"][i], data["recipe"][i]]
            for c in COMPS:
                row += [f"{data['comps_orig'][c][i]:.6f}",
                        f"{data['comps_dpo'][c][i]:.6f}",
                        f"{data['comps'][c][i]:.6f}"]
            for a in AGENTS:
                row += [f"{data['agents_orig'][a][i]:.6f}",
                        f"{data['agents_ret'][a][i]:.6f}",
                        f"{data['agents'][a][i]:.6f}"]
            w.writerow(row)
    print("wrote", path)


def write_readme(data):
    n = len(data["names"])
    per_depth = {d: int((data["depth"] == d).sum()) for d in DEPTHS}
    title = ("# results_full — change-on-change analysis (retrain − original)"
             if AGG == "mean" else
             f"# results_full_{AGG} — change-on-change analysis (retrain − original)")
    doc = [
        title, "",
        *( [agg_banner()] if AGG != "mean" else [] ),
        f"Config `car-purchase-freeform-std`, depths d1–d4 "
        f"({', '.join(f'{d}={per_depth[d]}' for d in DEPTHS)}), N={n}. "
        f"Interpretability budget {BUDGET}. Both snapshots — original (SFT) and "
        "retrained (DPO) — on both pipelines, so each organism yields a validation "
        "change and an interpretability-accuracy change.", "",
        "This is the d1–d4 generalization of the d3-only `outcome_acc_change` in "
        "`../../results_d3/`. The companion levels analysis over the same 80 organisms "
        "(original snapshot only) is `../raw_corr/README.md` / "
        "`../raw_corr/*/outcome_raw_acc_*.md`.", "",
        "**Outcome:** each of 9 interpretability agents' accuracy change "
        "(`retrain − original`). **Predictors:** the four signed validation component "
        "changes (`Δmmlu`, `Δmt_bench`, `Δactivation_diff`, `Δcot_naturalness`). The "
        "weighted combined score is omitted, as in the levels analysis.", "",
        "## Why this is the cleaner design", "",
        "The levels analysis is cross-sectional: it compares *different organisms*, so "
        "any organism-level trait that raises both validation scores and recoverability "
        "(a 'cleaner' model, a simpler rule) produces a correlation with no causal link. "
        "Differencing each organism against itself cancels every such time-invariant "
        "trait — decision rule, depth, intrinsic recoverability — leaving only the "
        "retraining-induced change. The 80 rows are independent (one per organism), so "
        "no clustering is needed.", "",
        "## Layout", "",
        "Each of `pearson/`, `spearman/`, `kendall/` holds:",
        "- `outcome_acc_change_pooled.md` — pooled N=%d: descriptives (what retraining "
        "actually moved), per-Δcomponent correlations, joint OLS." % n,
        "- `outcome_acc_change_depth_adjusted.md` — depth dummies partialled out of both "
        "sides + depth fixed-effects regression with a block F-test.",
        "",
        "Supporting (method-agnostic):",
        "- `vif_analysis_change.md` — collinearity of the four Δcomponents.",
        "- `table_change_regression.tex` — pooled + depth-FE OLS as LaTeX.",
        "- `combined_data_change.csv` — per-organism table (both snapshots + deltas).", "",
        "## How to read it", "",
        "Start with the **descriptives** and **reliability** tables at the top of the "
        "pooled file, in that order. Retraining moves the mean accuracy very little, so "
        "this analysis lives or dies on cross-organism *spread* in the deltas: a Δ with "
        "no variance cannot correlate with anything (exactly how `activation_diff` failed "
        "in the levels analysis — pinned at ceiling). The reliability table then asks "
        "whether the spread that *is* there is real between-organism variation or just "
        "run-to-run noise, which is what decides whether a null below means \"no "
        "relationship\" or merely \"cannot tell\".", "",
        "Then read the depth-adjusted file over the pooled one.", "",
        f"Regenerate: `python analysis/prior_work/pando/analyze_acc_change.py --results <tree> --out <out> "
        f"--aggregator {AGG}`.", "",
    ]
    open(f"{OUT}/README_change.md", "w").write("\n".join(doc) + "\n")
    print("wrote", f"{OUT}/README_change.md")


def write_depth_fe_json(data):
    """Depth fixed-effects joint OLS per agent (dR2, block-F p, standardized betas, 95% CIs, coef p) as JSON."""
    Dextra = depth_design(data["depth"])[:, 1:]
    res = {a: reg_joint(data["agents"][a], data["comps"], extra_design=Dextra) for a in AGENTS}
    with open(f"{OUT}/regression_depth_fe.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("wrote", f"{OUT}/regression_depth_fe.json")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metric", choices=METHODS,
                   help="only write this method's two files (default: all three "
                        "plus the shared robustness/vif/tex/csv/readme).")
    p.add_argument("--results", required=True, help="results/prior_work/pando tree")
    p.add_argument("--out", required=True, help="output dir (writes <out>/change_corr/)")
    p.add_argument("--aggregator", choices=["mean", "median", "trimmed"], default="trimmed",
                   help="how to collapse the 5 interpretability runs per organism-snapshot "
                        "(default trimmed = the paper's; mean/median are sensitivity variants).")
    return p.parse_args()


def main():
    args = parse_args()
    global AGG, OUT
    AGG = args.aggregator
    OUT = f"{args.out}/change_corr"
    os.makedirs(OUT, exist_ok=True)
    data = load_data(args.results, AGG)
    print(f"loaded {len(data['names'])} organisms [{AGG} aggregator] "
          + ", ".join(f"{d}={(data['depth'] == d).sum()}" for d in DEPTHS))

    for method in ([args.metric] if args.metric else METHODS):
        write_pooled(data, method)
        write_depth_adjusted(data, method)
    if args.metric:
        return
    write_vif(data)
    write_regression_tex(data)
    write_depth_fe_json(data)
    write_reg_coeff_csv(f"{OUT}/regression_coeffs_change.csv",
                        data["agents"], data["comps"], data["depth"])
    write_csv(data)
    write_readme(data)


if __name__ == "__main__":
    main()
