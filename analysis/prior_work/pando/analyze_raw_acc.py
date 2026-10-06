#!/usr/bin/env python
"""Standalone RAW-ACCURACY validation->interpretability analysis for the full
80-organism Pando car-purchase set (config car-purchase-freeform-std, depths
d1-d4, 20 organisms each).

These 80 organisms were run through the *original* (SFT) validation and
interpretability pipelines only -- no DPO retrain -- so the change-on-change /
rank-shift outcomes do not apply. What remains is the cross-sectional LEVELS
relationship between each organism's raw validation scores and each
interpretability agent's raw held-out rule-recovery accuracy (budget 10, mean
over 5 runs):

    outcome  = each interpretability agent's raw accuracy (9 AGENTS)
    predictor= the 4 raw validation COMPONENT scores
               (mmlu, mt_bench, activation_diff, cot_naturalness)

We do NOT analyze the weighted *combined* validation score: it averages four
components that move in conflicting directions across depth (mmlu/mt_bench fall,
act_diff/cot_naturalness rise), so it cancels itself out and encodes no coherent
signal. Only the individual components and their joint regression are reported.

Two analyses per correlation method, one file each, under every method dir:

  outcome_raw_acc_pooled.md
      Pooled over all 80 organisms.
        - individual correlations : each component vs each agent (P2 matrix)
        - regression              : all 4 components jointly (P3, multiple OLS)
      CONFOUNDED BY DEPTH: interpretability accuracy falls steeply with
      decision-tree depth (d1~.95 -> d4~.58) and mmlu/mt_bench also decline with
      depth, so a large part of any pooled correlation is a between-depth
      artifact (the 4 depth-means line up even though within-depth spread is
      modest). Read alongside the depth-adjusted file.

  outcome_raw_acc_depth_adjusted.md
      Same two analyses after removing tree depth.
        - individual correlations : depth-partialled -- depth dummies (4 levels,
          reference d1) are regressed out of BOTH the component and the agent
          accuracy, and the method's correlation is applied to the residuals
          (Pearson-of-residuals = textbook partial correlation; Spearman/Kendall
          -of-residuals = the rank analog).
        - regression              : depth fixed-effects OLS -- each agent's
          accuracy on the 4 components PLUS 3 depth dummies; the reported dR2 and
          block F-test measure what the components add *beyond* depth, and the
          betas are within-depth standardized partial slopes.
      This is the honest "is there signal beyond depth?" view.

The joint regression (P3) is ordinary OLS and therefore IDENTICAL across the
three method folders (only the single-predictor P2 correlations are rank- vs
value-based); each file repeats it for self-containedness with a note.

Benjamini-Hochberg FDR control is applied within each family (P2: 36, P3: 9).

Supporting outputs (written once, method-agnostic):
  vif_analysis.md              collinearity of the 4 components (N=80)
  table_level_regression.tex   pooled + depth-FE joint OLS as LaTeX
  combined_data.csv            merged per-organism table (reproducibility)
  README.md

Usage:
    python analyze_raw_acc.py --results results/prior_work/pando --out analysis/out/pando
    (writes <out>/raw_corr/; --metric spearman = only spearman's two files)
"""
import os, argparse, csv, json
import numpy as np
import statsmodels.api as sm
from scipy import stats

import helpers

OUT = None             # <--out>/raw_corr, set in main()

DEPTHS = ["d1", "d2", "d3", "d4"]
BUDGET = "10"   # only budget present for all 4 depths (d3 also has 20; unused for parity)
COMPS = ["mmlu", "mt_bench", "activation_diff", "cot_naturalness"]
AGENTS = ["relp", "gradient", "prefill", "sae_gradient",
          "logit_lens", "res_token", "circuit_tracer", "blackbox", "nn"]

METHODS = ["pearson", "spearman", "kendall"]
LABEL = {"pearson": "Pearson r", "spearman": "Spearman ρ", "kendall": "Kendall τ"}
CLAB = {"mmlu": "MMLU", "mt_bench": "MT-Bench",
        "activation_diff": "Act-diff", "cot_naturalness": "CoT-nat"}
ALAB = {"relp": "ReLP", "gradient": "Gradient", "prefill": "Prefill",
        "sae_gradient": "SAE-grad", "logit_lens": "Logit-lens",
        "res_token": "Res-token", "circuit_tracer": "Circuit-tracer",
        "blackbox": "Sample-only", "nn": "NN"}

AGG = "trimmed"        # per-organism run aggregator; reassigned in main() from --aggregator
AGG_DESC = {
    "mean": "mean over 5 runs",
    "median": "median over the 5 runs",
    "trimmed": "trimmed mean over the 5 runs (drop the min and max, average the middle 3)",
}


def agg_banner():
    """One-line provenance banner for the non-default aggregators (empty for mean)."""
    if AGG == "mean":
        return ""
    return (f"> **Sensitivity variant — run aggregator = `{AGG}`.** Each organism's "
            f"interpretability accuracy is the {AGG_DESC[AGG]}, instead of the plain "
            f"5-run mean, to blunt single-run outliers; the validation predictors are "
            f"unchanged. Compare the mean-based results under `results_full/raw_corr/`.\n")


def aggregate(runs, method):
    """Collapse a set of interpretability runs to a single point estimate:

      mean    : arithmetic mean
      median  : median
      trimmed : drop min & max, mean of the rest (middle 3 for n=5)

    Falls back to the plain mean when fewer than 3 runs are present.
    """
    r = np.sort(np.asarray(runs, float))
    n = len(r)
    if method == "mean" or n < 3:
        return float(r.mean())
    if method == "median":
        return float(np.median(r))
    if method == "trimmed":
        return float(r[1:-1].mean())
    raise ValueError(f"unknown aggregator {method!r}")


# ---------------------------------------------------------------------------
# Data loading  ->  one row per organism, tagged with depth.
# ---------------------------------------------------------------------------
def load_data(results, aggregator="trimmed"):
    """Return aligned numpy arrays over the original organisms of the results tree:
        names, depth (str), comps[c], agents[a].
    Ordered depth-major, then by organism id, so all vectors align.
    """
    names, depth = [], []
    comps = {c: [] for c in COMPS}
    agents = {a: [] for a in AGENTS}

    for d, orgs in helpers.organisms(results).items():
        for org in orgs:
            intmap = {a: helpers.interp(org, a) for a in AGENTS}
            if any(v is None for v in intmap.values()):
                print(f"  [skip] {org.name}: no interpretability entry")
                continue
            j = helpers.scores(org)
            names.append(org.name)
            depth.append(d)
            for c in COMPS:
                comps[c].append(j[c])
            for a in AGENTS:
                if aggregator == "mean":       # use the stored mean for exactness
                    agents[a].append(intmap[a]["mean"])
                else:
                    agents[a].append(aggregate(list(intmap[a]["runs"].values()), aggregator))

    return dict(
        names=np.array(names),
        depth=np.array(depth),
        comps={c: np.array(v, float) for c, v in comps.items()},
        agents={a: np.array(v, float) for a, v in agents.items()},
    )


# ---------------------------------------------------------------------------
# Stats helpers.
# ---------------------------------------------------------------------------
def zscore(v):
    v = np.asarray(v, float)
    s = v.std(ddof=0)
    return (v - v.mean()) / s if s > 0 else v - v.mean()


def corr1(x, y, method):
    if method == "pearson":
        return stats.pearsonr(x, y)
    if method == "spearman":
        return stats.spearmanr(x, y)
    return stats.kendalltau(x, y)


def bh(pvals, q=0.05):
    p = np.array(sorted(pvals))
    m = len(p)
    if m == 0:
        return 0, float("nan")
    sig = p <= (np.arange(1, m + 1) / m) * q
    return (int(np.max(np.where(sig)[0]) + 1) if sig.any() else 0), float(p.min())


def bh_thresh(pvals, q=0.05):
    """Benjamini-Hochberg reject set as a p-value cutoff.

    Returns (n_reject, min_p, thresh): a hypothesis is rejected (survives FDR
    control) iff its p-value <= thresh. thresh = -inf when nothing survives, so
    the `p <= thresh` test is always False in that case.
    """
    p = np.array(sorted(pvals))
    m = len(p)
    if m == 0:
        return 0, float("nan"), float("-inf")
    sig = p <= (np.arange(1, m + 1) / m) * q
    if not sig.any():
        return 0, float(p.min()), float("-inf")
    k = int(np.max(np.where(sig)[0]) + 1)
    return k, float(p.min()), float(p[k - 1])


# Significance markup shared by the correlation and regression tables:
#   *italic* -> p<0.05 uncorrected
#   **bold** -> also survives Benjamini-Hochberg FDR=0.05 (a subset of italic)
SIG_LEGEND = ('*italic* = p<0.05 uncorrected; **bold** = also survives '
              'Benjamini-Hochberg FDR=0.05 correction')


def fmt_cell(val_str, p, thresh):
    """Wrap a formatted number with significance markup given a BH cutoff."""
    if p <= thresh:              # survives FDR correction -> bold
        return f"**{val_str}**"
    if p < 0.05:                 # uncorrected-significant only -> italic
        return f"*{val_str}*"
    return val_str


def depth_design(depth):
    """n x 4 design: intercept + 3 depth dummies (reference = first depth)."""
    D = [np.ones(len(depth))]
    for lv in DEPTHS[1:]:
        D.append((depth == lv).astype(float))
    return np.column_stack(D)


def residualize(y, D):
    beta, *_ = np.linalg.lstsq(D, np.asarray(y, float), rcond=None)
    return np.asarray(y, float) - D @ beta


def reg_joint(y, comps, extra_design=None):
    """Multiple OLS of z(y) on the 4 z-scored components (P3).

    extra_design: optional (n x m) ADDITIONAL covariates (e.g. 3 depth dummies,
    no intercept) to partial out; an intercept is always added. Component betas
    stay first. When extra_design is given, also returns the incremental stats of
    the 4-component block over an extra-only model: dR2 and its partial F-test
    (Fblk, Fblk_p), plus the reduced (extra-only) R2.
    """
    y = zscore(y)
    Xc = np.column_stack([zscore(comps[c]) for c in COMPS])
    X = np.column_stack([Xc, extra_design]) if extra_design is not None else Xc
    m = sm.OLS(y, sm.add_constant(X)).fit()
    ci = m.conf_int()
    out = dict(r2=m.rsquared, adj=m.rsquared_adj, F=m.fvalue, Fp=m.f_pvalue,
               betas={c: m.params[i + 1] for i, c in enumerate(COMPS)},
               ps={c: m.pvalues[i + 1] for i, c in enumerate(COMPS)},
               ci={c: (ci[i + 1][0], ci[i + 1][1]) for i, c in enumerate(COMPS)})
    if extra_design is not None:
        reduced = sm.OLS(y, sm.add_constant(extra_design)).fit()
        n = len(y)
        p_full = X.shape[1] + 1
        q = len(COMPS)
        Fblk = ((reduced.ssr - m.ssr) / q) / (m.ssr / (n - p_full))
        out.update(dR2=m.rsquared - reduced.rsquared, Fblk=float(Fblk),
                   Fblk_p=float(stats.f.sf(Fblk, q, n - p_full)),
                   r2_reduced=reduced.rsquared)
    return out


# ---------------------------------------------------------------------------
# Markdown builders.
# ---------------------------------------------------------------------------
def corr_matrix(units_map, comps_map, method, depth=None):
    """rows=agents, cols=components, one method's coefficient with * where p<.05.

    depth=None -> pooled correlation on raw values.
    depth given -> depth-partialled: residualize both sides on depth dummies,
                   then apply the method's correlation to the residuals.
    """
    if depth is not None:
        D = depth_design(depth)
        prep = lambda v: residualize(v, D)
    else:
        prep = lambda v: np.asarray(v, float)
    head = "| agent | " + " | ".join(CLAB[c] for c in COMPS) + " |"
    sep = "|" + "---|" * (len(COMPS) + 1)
    # First pass: gather every (agent, component) coefficient and p-value so the
    # BH cutoff can be computed across the whole family before formatting cells.
    grid, pvals = [], []
    for a in AGENTS:
        ya = prep(units_map[a])
        row = []
        for c in COMPS:
            r, p = corr1(prep(comps_map[c]), ya, method)
            pvals.append(p)
            row.append((c, r, p))
        grid.append((a, row))
    _, _, thresh = bh_thresh(pvals)
    lines, sig = [head, sep], []
    for a, row in grid:
        cells = []
        for c, r, p in row:
            cells.append(fmt_cell(f"{r:+.2f}", p, thresh))
            if p < 0.05:
                sig.append((a, c, r, p))
        lines.append(f"| {ALAB[a]} | " + " | ".join(cells) + " |")
    return "\n".join(lines), pvals, sig


def reg_table(units_map, comps_map, extra_design=None):
    """P3 regression table over agents. extra_design given -> depth-FE variant
    (reports dR2 and block-F p instead of whole-model R2/F).

    Each β cell carries its 95% CI and significance markup (*italic* = coef
    p<0.05 uncorrected, **bold** = also survives Benjamini-Hochberg FDR=0.05 over
    the 4-components × 9-agents coefficient family). A per-coefficient detail
    table (β, 95% CI, raw p) and that family's BH multiplicity line are appended
    to the returned markdown block. The returned pvals list is still the 9
    block/model F-test p-values (the caller's F-family multiplicity line)."""
    incr = extra_design is not None
    head = ("| agent | " + ("adj R² | ΔR²(comp) | Fblk p | " if incr
                            else "R² | adj R² | F | F p | ")
            + " | ".join(f"β {CLAB[c]}" for c in COMPS) + " |")
    sep = "|" + "---|" * ((4 if incr else 5) + len(COMPS))
    # First pass: fit every agent so the BH cutoff over the 9 block/model F-tests
    # is known before formatting the F-p column.
    results = [(a, reg_joint(units_map[a], comps_map, extra_design=extra_design))
               for a in AGENTS]
    pvals = [r["Fblk_p"] if incr else r["Fp"] for _, r in results]
    _, _, thresh = bh_thresh(pvals)
    # Coefficient family: 4 components × 9 agents. BH cutoff over all coef p's so
    # the *italic*/**bold** markup is applied consistently across the whole table.
    coef_ps = [r["ps"][c] for _, r in results for c in COMPS]
    _, _, coef_thresh = bh_thresh(coef_ps)
    lines = [head, sep]
    for a, r in results:
        pv = r["Fblk_p"] if incr else r["Fp"]
        pstr = fmt_cell(f"{pv:.3f}", pv, thresh)
        if incr:
            mid = f"{r['adj']:+.3f} | {r['dR2']:+.3f} | {pstr} | "
        else:
            mid = f"{r['r2']:.3f} | {r['adj']:+.3f} | {r['F']:.2f} | {pstr} | "
        betacells = []
        for c in COMPS:
            lo, hi = r["ci"][c]
            b = fmt_cell(f"{r['betas'][c]:+.3f}", r["ps"][c], coef_thresh)
            betacells.append(f"{b} [{lo:+.2f}, {hi:+.2f}]")
        lines.append(f"| {ALAB[a]} | {mid}" + " | ".join(betacells) + " |")
    main = "\n".join(lines)
    # Per-coefficient detail: standardized β, 95% CI, raw p, same markup.
    dlines = ["", "**Per-coefficient detail** — standardized β, 95% CI, and raw p. "
              + SIG_LEGEND + " over the 4 components × 9 agents = 36 coefficient "
              "tests. The 95% CIs are the ordinary per-coefficient marginal "
              "intervals from the same OLS fit; the FDR correction re-thresholds "
              "the p-values only and does not alter the CIs, so *italic* (p<0.05) "
              "always agrees with \"CI excludes 0\" while **bold** (BH) need not.",
              "", "| agent | component | β | 95% CI | p |", "|---|---|---|---|---|"]
    for a, r in results:
        for c in COMPS:
            lo, hi = r["ci"][c]
            p = r["ps"][c]
            dlines.append(f"| {ALAB[a]} | {CLAB[c]} | {r['betas'][c]:+.3f} | "
                          f"[{lo:+.2f}, {hi:+.2f}] | {fmt_cell(f'{p:.4f}', p, coef_thresh)} |")
    dlines += ["", mult_line("Coefficient (4 Δcomponents × 9 agents)", coef_ps)]
    return main + "\n" + "\n".join(dlines), pvals


def mult_line(label, pvals):
    n_bh, minp = bh(pvals)
    m = len(pvals)
    return (f"**{label} multiplicity:** {sum(p < 0.05 for p in pvals)}/{m} tests "
            f"p<0.05 uncorrected (≈{0.05 * m:.1f} expected by chance); "
            f"Benjamini-Hochberg FDR=0.05 → **{n_bh} survive** (min p={minp:.4f}).")


def sig_note(sig, lab):
    if not sig:
        return "_No component is significantly correlated with any agent (all p>0.05)._"
    return ("_Significant (uncorrected): "
            + "; ".join(f"{ALAB[a]}~{CLAB[c]}: {lab}={r:+.2f} (p={p:.3f})"
                        for a, c, r, p in sig) + "._")


# ---------------------------------------------------------------------------
# Per-method files.
# ---------------------------------------------------------------------------
def write_pooled(data, method):
    lab = LABEL[method]
    out = [
        f"# Raw accuracy (levels), pooled N=80 — {lab}", "",
        f"_Method: {lab} for the individual correlations. The joint regression is "
        "ordinary OLS and identical across the three method folders. Depth-adjusted "
        "companion: `outcome_raw_acc_depth_adjusted.md`._", "",
        "Each interpretability agent's raw held-out rule-recovery accuracy "
        f"(budget {BUDGET}, {AGG_DESC[AGG]}) vs the organism's 4 raw validation "
        "component scores, over all 80 organisms (car-purchase-freeform-std, "
        "depths d1-d4, 20 each). The weighted combined score is intentionally "
        "omitted (its components cancel across depth).", "",
        "> **Confound.** Interpretability accuracy falls steeply with decision-tree "
        "depth, and MMLU / MT-Bench also decline with depth, so much of any pooled "
        "correlation below is a between-depth artifact. The depth-adjusted file "
        "isolates the within-depth signal.", "",
        "## Individual correlations — each component vs each agent", "",
        f"_{lab}; {SIG_LEGEND} over these 36 correlations._", "",
    ]
    t, pv, sig = corr_matrix(data["agents"], data["comps"], method)
    out += [t, "", sig_note(sig, lab), "",
            mult_line("Individual (4 components × 9 agents)", pv), ""]

    out += ["## Regression — all 4 components jointly (multiple OLS)", "",
            "_β = standardized coefficient (predictors and outcome z-scored); "
            "each β is shown with its 95% CI and " + SIG_LEGEND + " on β over the "
            "36 coefficient tests (raw p in the per-coefficient detail table below). "
            "On the F p column the same markup is applied over these 9 model "
            "F-tests. Identical "
            "across the three method folders (OLS, not rank-based)._", ""]
    t, pv = reg_table(data["agents"], data["comps"])
    out += [t, "", mult_line("Regression (joint OLS F-test × 9 agents)", pv), ""]

    if AGG != "mean":
        out.insert(2, agg_banner())
    path = f"{OUT}/{method}/outcome_raw_acc_pooled.md"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(out) + "\n")
    print("wrote", path)


def write_depth_adjusted(data, method):
    lab = LABEL[method]
    depth = data["depth"]
    Dextra = depth_design(depth)[:, 1:]   # 3 depth dummies, no intercept
    n, k = len(depth), 3
    out = [
        f"# Raw accuracy (levels), depth-adjusted — {lab}", "",
        f"_Method: {lab} for the individual correlations. The regression is a "
        "depth fixed-effects OLS, identical across the three method folders. "
        "Pooled companion: `outcome_raw_acc_pooled.md`._", "",
        "The pooled correlations are confounded by decision-tree depth "
        "(interpretability accuracy falls d1→d4; MMLU and MT-Bench also decline). "
        "This file removes depth and asks whether any validation↔interpretability "
        "signal survives *within* depth.", "",
        "## Individual correlations — each component vs each agent (depth-partialled)", "",
        f"_Depth dummies (4 levels, reference d1) are regressed out of both the "
        f"component and the agent accuracy; {lab} is then applied to the residuals. "
        f"{SIG_LEGEND} over these 36 correlations. p-values use the standard {method} "
        f"test on the residuals and do not further adjust the dof for the 3 depth "
        f"dummies (negligible at N={n}: {n-2} vs {n-2-k})._", "",
    ]
    t, pv, sig = corr_matrix(data["agents"], data["comps"], method, depth=depth)
    out += [t, "", sig_note(sig, lab + "-partial"), "",
            mult_line("Partial (4 components × 9 agents)", pv), ""]

    out += ["## Regression — joint OLS with depth fixed effects", "",
            "_Each agent's z-scored accuracy on the 4 z-scored components **plus 3 "
            "depth dummies**. ΔR²(comp) = extra variance the 4 components explain "
            "over a depth-only model; on the Fblk p column, " + SIG_LEGEND
            + " over these 9 block F-tests (the 4-component block being jointly "
            "significant beyond depth); β = within-depth standardized partial "
            "slope shown with its 95% CI (" + SIG_LEGEND + " on β over the 36 "
            "coefficient tests; raw p in the per-coefficient detail table below). "
            "Identical across method folders._", ""]
    t, pv = reg_table(data["agents"], data["comps"], extra_design=Dextra)
    out += [t, "", mult_line("Regression (component block F × 9 agents)", pv), ""]

    # Reference: how much depth alone explains, per agent.
    out += ["### Reference — variance depth explains by itself", "",
            "_R² of an intercept+depth-dummies model for each agent's accuracy "
            "(no validation predictors)._", "",
            "| agent | R²(depth only) |", "|---|---|"]
    for a in AGENTS:
        rr = sm.OLS(zscore(data["agents"][a]), sm.add_constant(Dextra)).fit()
        out.append(f"| {ALAB[a]} | {rr.rsquared:.3f} |")
    out += [""]

    if AGG != "mean":
        out.insert(2, agg_banner())
    path = f"{OUT}/{method}/outcome_raw_acc_depth_adjusted.md"
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
    out = ["# VIF diagnostic — 4 validation components (N=80)", "",
           "Variance Inflation Factors for the four component predictors that "
           "enter the joint regressions. VIF depends only on the predictor matrix "
           "(one table, not one per agent). VIF≈1 orthogonal, >5 moderate, >10 "
           "severe; tolerance = 1/VIF.", "",
           "| predictor | VIF | tolerance | R² on other 3 | flag |",
           "|---|---|---|---|---|"]
    for c, v in zip(COMPS, vif):
        out.append(f"| {c} | {v:.3f} | {1/v:.3f} | {1 - 1/v:.3f} | {flag(v)} |")
    out += ["",
            f"_Max VIF = {vif.max():.3f} ({COMPS[int(np.argmax(vif))]}); "
            f"mean VIF = {vif.mean():.3f}; condition number = {cond:.2f}._", "",
            "Pairwise component correlations (Pearson r):", "",
            "| | " + " | ".join(CLAB[c] for c in COMPS) + " |",
            "|" + "---|" * (len(COMPS) + 1)]
    for i, c in enumerate(COMPS):
        out.append(f"| **{CLAB[c]}** | "
                   + " | ".join(f"{R[i, k]:+.2f}" for k in range(len(COMPS))) + " |")
    out += [""]
    open(f"{OUT}/vif_analysis.md", "w").write("\n".join(out) + "\n")
    print("wrote", f"{OUT}/vif_analysis.md")


def beta_ci_cell(beta, ci, p, thresh):
    """LaTeX math cell: standardized β with 95% CI subscript and raw p.

    `^{*}` marks p<0.05 uncorrected (equivalently, a CI excluding zero); the β is
    set in \\mathbf when it also survives Benjamini-Hochberg FDR=0.05 (p<=thresh)."""
    lo, hi = ci
    star = "^{*}" if p < 0.05 else ""
    body = rf"\mathbf{{{beta:+.2f}}}" if p <= thresh else f"{beta:+.2f}"
    return f"${body}{star}_{{[{lo:+.2f},\\,{hi:+.2f}]}}\\,(p{{=}}{p:.3f})$"


def write_reg_coeff_csv(path, units_map, comps_map, depth):
    """Per-coefficient joint-OLS regression stats as CSV (machine-readable dump of
    the markdown/LaTeX β tables). One row per (model, agent, component): the model
    is `pooled` (no depth covariates) or `depth_fe` (+3 depth dummies); columns are
    the standardized β, its 95% CI, the raw p, and the uncorrected / Benjamini-
    Hochberg significance flags (BH over the 36-coefficient family within a model)."""
    Dextra = depth_design(depth)[:, 1:]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "agent", "component", "beta", "ci_lo", "ci_hi",
                    "p", "sig_uncorrected", "sig_bh"])
        for model_name, extra in (("pooled", None), ("depth_fe", Dextra)):
            results = [(a, reg_joint(units_map[a], comps_map, extra_design=extra))
                       for a in AGENTS]
            coef_ps = [r["ps"][c] for _, r in results for c in COMPS]
            _, _, thresh = bh_thresh(coef_ps)
            for a, r in results:
                for c in COMPS:
                    lo, hi = r["ci"][c]
                    p = r["ps"][c]
                    w.writerow([model_name, ALAB[a], CLAB[c],
                                f"{r['betas'][c]:.6f}", f"{lo:.6f}", f"{hi:.6f}",
                                f"{p:.6f}", int(p < 0.05), int(p <= thresh)])
    print("wrote", path)


def write_regression_tex(data):
    def rows(extra):
        return [(a, reg_joint(data["agents"][a], data["comps"], extra_design=extra))
                for a in AGENTS]
    naive = rows(None)
    Dextra = depth_design(data["depth"])[:, 1:]
    fe = rows(Dextra)
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
                 + r" & $\beta_{\mathrm{MMLU}}$ & $\beta_{\mathrm{MT\text{-}Bench}}$ "
                   r"& $\beta_{\mathrm{Act\text{-}diff}}$ & $\beta_{\mathrm{CoT\text{-}nat}}$ \\",
                 r"& & & \multicolumn{4}{c}{\scriptsize standardized $\beta_{[\text{95\% CI}]}$} \\",
                 r"\midrule"]
        coef_ps = [r["ps"][c] for _, r in rowset for c in COMPS]
        _, _, coef_thresh = bh_thresh(coef_ps)
        for a, r in rowset:
            pv = r["Fblk_p"] if incr else r["Fp"]
            pstr = f"\\textbf{{{pv:.3f}}}" if pv < 0.05 else f"{pv:.3f}"
            r2v = r["dR2"] if incr else r["r2"]
            lines.append(f"{ALAB[a]} & {r2v:+.2f} & {pstr} & "
                         + " & ".join(beta_ci_cell(r["betas"][c], r["ci"][c], r["ps"][c], coef_thresh)
                                      for c in COMPS) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", caption,
                  rf"\label{{{label}}}", r"\end{table}"]
        return "\n".join(lines)

    cap_naive = (
        rf"\caption{{Pooled level-on-level joint OLS of each tool's raw rule-recovery "
        rf"accuracy on the four raw, $z$-scored validation components, over all 80 "
        rf"car-purchase organisms (depths d1--d4, budget 10). $\beta$ are standardized "
        rf"coefficients with 95\% CI subscripts and raw $p$; $^{{*}}$ marks $p<0.05$ (a CI "
        rf"excluding zero) and a bold $\beta$ also survives Benjamini--Hochberg FDR 0.05 "
        rf"over the 36 coefficient tests. Bold "
        rf"$F$ $p$ marks an overall-significant model ({n_naive}/9). \textbf{{Confounded by tree "
        rf"depth}}; see the depth fixed-effects table. Benjamini--Hochberg (FDR 0.05) within "
        rf"these nine $F$-tests: {bh_naive[0]}/9 survive (min $p={bh_naive[1]:.4f}$).}}")
    cap_fe = (
        rf"\caption{{Depth fixed-effects joint OLS: each tool's $z$-scored accuracy on the "
        rf"four $z$-scored components \emph{{plus three decision-tree-depth dummies}} (N=80). "
        rf"$\Delta R^2$ is the extra variance the four components explain over a depth-only "
        rf"model and $F_{{\mathrm{{blk}}}}$ $p$ tests that block; bold marks $p<0.05$ "
        rf"({n_fe}/9). $\beta$ are within-depth standardized partial slopes with 95\% CI "
        rf"and raw $p$; $^{{*}}$ marks $p<0.05$ (a CI excluding zero) and a bold $\beta$ "
        rf"also survives Benjamini--Hochberg FDR 0.05 over the 36 coefficient tests. "
        rf"This removes the depth confound. "
        rf"Benjamini--Hochberg (FDR 0.05) within the nine block $F$-tests: {bh_fe[0]}/9 "
        rf"survive (min $p={bh_fe[1]:.4f}$).}}")

    doc = ["% Raw-accuracy (levels) joint OLS regressions, full 80-organism set.",
           "% Predictors z-scored; betas standardized. Combined score omitted.",
           "%   Table 1: naive pooled (confounded by depth).",
           "%   Table 2: depth fixed-effects (within-depth).",
           table(naive, False, "tab:rawacc-b10-ols-pooled80", cap_naive), "",
           table(fe, True, "tab:rawacc-b10-ols-depthfe", cap_fe)]
    open(f"{OUT}/table_level_regression.tex", "w").write("\n".join(doc) + "\n")
    print("wrote", f"{OUT}/table_level_regression.tex")


def write_csv(data):
    path = f"{OUT}/combined_data.csv"
    cols = ["name", "depth"] + [f"val_{c}" for c in COMPS] + [f"int_{a}" for a in AGENTS]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(len(data["names"])):
            row = [data["names"][i], data["depth"][i]]
            row += [f"{data['comps'][c][i]:.6f}" for c in COMPS]
            row += [f"{data['agents'][a][i]:.6f}" for a in AGENTS]
            w.writerow(row)
    print("wrote", path)


def write_readme(data):
    n = len(data["names"])
    per_depth = {d: int((data["depth"] == d).sum()) for d in DEPTHS}
    title = ("# results_full — raw-accuracy validation↔interpretability analysis (80 organisms)"
             if AGG == "mean" else
             f"# results_full_{AGG} — raw-accuracy validation↔interpretability analysis (80 organisms)")
    doc = [
        title,
        "",
        *( [agg_banner()] if AGG != "mean" else [] ),
        f"Config `car-purchase-freeform-std`, depths d1–d4 "
        f"({', '.join(f'{d}={per_depth[d]}' for d in DEPTHS)}), N={n}. "
        f"Interpretability budget {BUDGET} (the only budget present for all four "
        "depths). Original (SFT) models only — no DPO retrain, so there is no "
        "change-on-change / rank-shift outcome (those live in the d3-only "
        "`../../results_d3/`).", "",
        "**Predictors** are the four raw validation *component* scores "
        "(`mmlu`, `mt_bench`, `activation_diff`, `cot_naturalness`). The weighted "
        "*combined* score is not analyzed: its components move in conflicting "
        "directions across depth and cancel, so it carries no coherent signal.", "",
        "## Layout", "",
        "Each of `pearson/`, `spearman/`, `kendall/` holds two files:",
        "- `outcome_raw_acc_pooled.md` — pooled N=80: per-component correlations "
        "(that method) + joint OLS regression. Confounded by depth.",
        "- `outcome_raw_acc_depth_adjusted.md` — depth removed: depth-partialled "
        "per-component correlations (that method) + depth fixed-effects regression "
        "with a block F-test for the components beyond depth.",
        "",
        "The joint regression is ordinary OLS, so it is identical across the three "
        "method folders (only the single-predictor correlations are rank- vs "
        "value-based).", "",
        "Supporting (method-agnostic):",
        "- `vif_analysis.md` — collinearity of the four components (N=80).",
        "- `table_level_regression.tex` — pooled + depth-FE OLS as LaTeX.",
        "- `combined_data.csv` — merged per-organism table.", "",
        "## How to read it", "",
        "Depth is the dominant driver of interpretability accuracy (d1≈.95 → "
        "d4≈.58) and MMLU/MT-Bench also decline with depth, so the four depth-means "
        "align and inflate the pooled correlations even though within-depth spread "
        "is modest. **Trust the depth-adjusted files over the pooled ones.** A "
        "relationship that survives depth-partialling (or shows a significant block "
        "ΔR² beyond depth) is the real cross-organism signal; a pooled-only "
        "correlation that vanishes after adjustment was a depth artifact.", "",
        f"Regenerate: `python analysis/prior_work/pando/analyze_raw_acc.py --results <tree> --out <out> "
        f"--aggregator {AGG}`.", "",
    ]
    open(f"{OUT}/README.md", "w").write("\n".join(doc) + "\n")
    print("wrote", f"{OUT}/README.md")


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
                        "plus the shared vif/tex/csv/readme).")
    p.add_argument("--results", required=True, help="results/prior_work/pando tree")
    p.add_argument("--out", required=True, help="output dir (writes <out>/raw_corr/)")
    p.add_argument("--aggregator", choices=["mean", "median", "trimmed"], default="trimmed",
                   help="how to collapse the 5 interpretability runs per organism "
                        "(default trimmed = the paper's; mean/median are sensitivity variants).")
    return p.parse_args()


def main():
    args = parse_args()
    global AGG, OUT
    AGG = args.aggregator
    OUT = f"{args.out}/raw_corr"
    os.makedirs(OUT, exist_ok=True)
    data = load_data(args.results, AGG)
    print(f"loaded {len(data['names'])} organisms [{AGG} aggregator] "
          + ", ".join(f"{d}={(data['depth'] == d).sum()}" for d in DEPTHS))

    methods = [args.metric] if args.metric else METHODS
    for method in methods:
        write_pooled(data, method)
        write_depth_adjusted(data, method)
    if args.metric:
        return
    write_vif(data)
    write_regression_tex(data)
    write_depth_fe_json(data)
    write_reg_coeff_csv(f"{OUT}/regression_coeffs_level.csv",
                        data["agents"], data["comps"], data["depth"])
    write_csv(data)
    write_readme(data)


if __name__ == "__main__":
    main()
