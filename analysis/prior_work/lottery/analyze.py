#!/usr/bin/env python
"""validation_cor for the Model Organism Lottery.

Replicates the pando `validation_cor` study question on the lottery suite:

  > Does a difference in our cheap behavioral VALIDATION score predict a
  > difference in INTERPRETABILITY / audit conclusions?

Unlike pando (which differences an organism's original vs DPO-retrained
snapshot), the lottery suite has no paired snapshots: each *training method*
is its own organism, and Quirk-Expression-Rate (QER) is held fixed within a
family (±5pp). So the unit of analysis is the (family, method) organism and we
do a CROSS-SECTIONAL correlation across organisms between:

  X = validation score  (mmlu / mt_bench / activation_diff / cot_naturalness,
                          each 1.0 == indistinguishable-from-base; + combined)
  Y = interpretability accuracy of an audit technique on that organism
        AO          : activation-oracle max-layer detection accuracy (diff, specific judge)
        LogitLens   : max-layer mean quirk-token cumulative-prob (diff), OUR replication

Each audit technique has a diff read and an ft (finetuned-model-alone) read, so the
outcome slugs are: ao, ao_ft, logit_lens, logit_lens_ft.

Directional hypothesis: a HIGHER validation score (more base-like, less drifted)
=> a LOWER interpretability score (quirk harder to surface). I.e. we expect a
NEGATIVE correlation.

Two estimators are reported for every cell:

  POOLED      all organisms stacked; cheap but confounded by FAMILY-level traits
              (families differ in absolute QER / quirk salience).
  WITHIN      family-demeaned (fixed-effects): X and Y centered within each
              family before correlating, so only the across-method variation
              that QER-matching isolates remains. This is the design-faithful
              estimate. Spearman on demeaned values = within-family ranking.

NOTE: the `activation_diff` validation component is itself a diffing measure, so
its correlation with the (diffing-based) interpretability scores is partly
MECHANICAL. The independent behavioral signal lives in mmlu / mt_bench /
cot_naturalness / combined.

The interpretability outcome is read at MAX_LAYER granularity: one Y per organism = max over
the probed layers.

Predictor 1 (the weighted combined validation score) is NOT analyzed — only the
per-component single-predictor tables (P2) and the joint OLS (P3). Each method
gets its own subfolder:

    <out>/max_layer/merged_data.csv
    <out>/max_layer/{pearson,spearman,kendall}/{combined.md, correlations.json}

correlations.json holds the within-family matrix (r, 95% CI, p, BH) — the paper's
tab:lottery-within-corr is the spearman one.

BH multiplicity: correction is applied on the WITHIN-FAMILY p-values only, as one
Benjamini-Hochberg family that POOLS BOTH READS (diffing + non-diffing) of both
agents (P2: m = 4 metrics × 2 agents × 2 setups = 16; P3: m = 1 OLS × 2 agents ×
2 setups = 4). The P3 joint regression is identical across the three method
folders by construction (it is always raw z-scored, not rank-transformed).

Usage:
    python analyze.py --results results/prior_work/lottery --out analysis/out/lottery [--metric spearman]
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from scipy import stats

import helpers

RESULTS = None   # <--out>, set in main()

COMPS = ["mmlu", "mt_bench", "activation_diff", "cot_naturalness"]
INDEP = ["mmlu", "mt_bench", "cot_naturalness"]  # non-mechanical components
FAM_ORDER = ["cake_bake", "italian_food", "milsub"]
FAM_LABEL = {"cake_bake": "CakeBake", "italian_food": "ItalianFood", "milsub": "MilSub"}

# One correlation method per output file (imitating the pando validation_cor
# analyze.py): P1/P2 and the per-family tables use exactly one of these, so a
# cell's significance star always matches the coefficient shown in that cell.
METHODS = ["pearson", "spearman", "kendall"]
LABEL = {"pearson": "Pearson r", "spearman": "Spearman ρ", "kendall": "Kendall τ"}
SYMBOL = {"pearson": "r", "spearman": "ρ", "kendall": "τ"}

# --------------------------------------------------------------------------- #
# Canonical key: (family, variant), variant dash-form, e.g. posthoc-mixed-dpo
# --------------------------------------------------------------------------- #


def load_validation(results):
    """{(fam,variant): {comp: score, 'combined': score}}."""
    out = {}
    for org in helpers.organisms(results):
        d = helpers.validation(org)
        rec = dict(d["scores"])
        rec["combined"] = d["combined_score"]
        out[helpers.key(org.name)] = rec
    return out


def load_ao(results, act_key="diff"):
    """{(fam,variant): max-over-layer AO accuracy} (specific judge; act_key diff | lora)."""
    out = {}
    for org in helpers.organisms(results):
        ao = helpers.interp(org, "ao")
        if ao and act_key in ao["max_layer"]:
            out[helpers.key(org.name)] = ao["max_layer"][act_key]
    return out


def load_logit_lens(results, read="diff"):
    """{(fam,variant): max-layer mean quirk-token cumulative prob} (read diff | ft)."""
    out = {}
    for org in helpers.organisms(results):
        ll = helpers.interp(org, "logit_lens")
        if ll and read in ll and ll[read]["max_layer"] is not None:
            out[helpers.key(org.name)] = ll[read]["max_layer"]
    return out


# --------------------------------------------------------------------------- #
# stats helpers
# --------------------------------------------------------------------------- #


def corr1(x, y, method):
    """Single coefficient + p-value for exactly one method, so a reported
    r/rho/tau and its significance star always come from the same test."""
    if method == "pearson":
        return stats.pearsonr(x, y)
    if method == "spearman":
        return stats.spearmanr(x, y)
    return stats.kendalltau(x, y)


def fisher_ci(r, n_eff, method, conf=0.95):
    """Two-sided analytic Fisher-z CI for a correlation coefficient.

    `n_eff` is the EFFECTIVE sample size fed to the transform: `n` for the
    pooled estimator, `n-(k-1)` for the within-fam (family-demeaned) estimator,
    where k = number of families present (the family demeaning conditions on
    k-1 dummies on top of the usual centering the `-3` already covers). Each
    method uses its own small-sample variance on the Fisher-z scale:

      Pearson   Var(z) = 1/(n_eff-3)
      Spearman  Var(z) = (1 + r^2/2)/(n_eff-3)   (Bonett-Wright)
      Kendall   z=atanh(tau), Var(z) = 0.437/(n_eff-4)  (Fieller et al.)

    Returns (lo, hi) or None for a degenerate cell (|r|>=1, non-finite r, or
    non-positive df) — the caller renders None as `[n/a]`. Asymptotic
    approximation: only roughly calibrated at the N's here."""
    import math
    if r is None or not np.isfinite(r) or abs(r) >= 1.0:
        return None
    denom = (n_eff - 4) if method == "kendall" else (n_eff - 3)
    if denom <= 0:
        return None
    if method == "pearson":
        var = 1.0 / denom
    elif method == "spearman":
        var = (1.0 + r * r / 2.0) / denom
    else:  # kendall
        var = 0.437 / denom
    if var <= 0:
        return None
    z, se = math.atanh(r), math.sqrt(var)
    zc = stats.norm.ppf(0.5 + conf / 2.0)  # 1.96 at conf=0.95
    return math.tanh(z - zc * se), math.tanh(z + zc * se)


def demean_by_family(keys, vec):
    """Subtract each organism's family mean (fixed-effects within transform)."""
    fam_vals = defaultdict(list)
    for k, v in zip(keys, vec):
        fam_vals[k[0]].append(v)
    fam_mean = {f: np.mean(vs) for f, vs in fam_vals.items()}
    return np.array([v - fam_mean[k[0]] for k, v in zip(keys, vec)])


def zscore(v):
    s = v.std(ddof=0)
    return (v - v.mean()) / s if s > 0 else v - v.mean()


def bh(pvals, q=0.05):
    p = np.array(sorted(pvals))
    m = len(p)
    if m == 0:
        return 0, float("nan")
    sig = p <= (np.arange(1, m + 1) / m) * q
    return (int(np.max(np.where(sig)[0]) + 1) if sig.any() else 0), float(p.min())


def bh_threshold(pvals, q=0.05):
    """Largest p-value that passes Benjamini-Hochberg at FDR=q (tests with
    p<=threshold are BH-significant); -1 if none pass."""
    p = np.sort(np.array(pvals, float))
    m = len(p)
    if m == 0:
        return -1.0
    ok = p <= (np.arange(1, m + 1) / m) * q
    return float(p[ok].max()) if ok.any() else -1.0


def reg(y, Xcols):
    X = np.column_stack([zscore(c) for c in Xcols])
    m = sm.OLS(zscore(np.asarray(y, float)), sm.add_constant(X)).fit()
    return m


# --------------------------------------------------------------------------- #
# markdown builders
# --------------------------------------------------------------------------- #


ESTIMATORS = [
    ("pooled", "pooled"),
    ("within", "within-fam"),
]


def _corr_one(keys, xvals, y, estimator, method):
    """Correlate one predictor vector against y with a single `method`. Missing
    (None) predictor values are pairwise-dropped; demeaning (within) is done on
    the surviving subset."""
    idx = [i for i, v in enumerate(xvals) if v is not None]
    sk = [keys[i] for i in idx]
    xv = np.array([xvals[i] for i in idx], float)
    yv = np.array([y[i] for i in idx], float)
    if estimator == "within":
        xv, yv = demean_by_family(sk, xv), demean_by_family(sk, yv)
    r, p = corr1(xv, yv, method)
    n = len(sk)
    # effective n for the Fisher-z df: pooled uses n; within-fam spends k-1 extra
    # df on the family demeaning (k = families actually present after drops).
    if estimator == "within":
        k = len({key[0] for key in sk})
        n_eff = n - (k - 1)
    else:
        n_eff = n
    return dict(r=r, p=p, n=n, ci=fisher_ci(r, n_eff, method))


def agent_p2(keys, xmap, y, method):
    """P2 data for ONE agent: per-component single-predictor correlations, pooled
    and within-family. The within-family p-value is what the BH families are
    built from. Returns a list of dicts (one per component)."""
    rows = []
    for c in COMPS:
        xvals = [xmap[k][c] for k in keys]
        sp = _corr_one(keys, xvals, y, "pooled", method)
        sw = _corr_one(keys, xvals, y, "within", method)
        rows.append(dict(comp=c, pooled_r=sp["r"], within_r=sw["r"],
                         within_p=sw["p"], n=sw["n"],
                         pooled_ci=sp["ci"], within_ci=sw["ci"]))
    return rows


def agent_p3(keys, xmap, y):
    """P3 data for ONE agent: within-family joint OLS (method-independent, always
    z-scored). Returns (F p-value, {comp: beta}, {comp: beta p}, n). Falls back to
    NaNs if the design is rank-deficient at this cell."""
    try:
        fp, betas, bps = _within_reg(keys, xmap, y)
        return fp, betas, bps
    except Exception:
        nan = {c: float("nan") for c in COMPS}
        return float("nan"), dict(nan), dict(nan)


def homogeneity_table(keys, ymap, xmap, method):
    """Per (quirk-family × component) within-family correlation for ONE agent — a
    sign-consistency cross-check. Families with <3 probed variants are skipped."""
    fams = [f for f in FAM_ORDER if any(k[0] == f for k in keys)]
    lines = ["| family | N | " + " | ".join(COMPS) + " |",
             "|" + "---|" * (len(COMPS) + 2)]
    for fam in fams:
        fk = [k for k in keys if k[0] == fam]
        if len(fk) < 3:
            continue
        cells = []
        for c in COMPS:
            sub = [(k, xmap[k][c]) for k in fk if xmap[k][c] is not None]
            if len(sub) < 3:
                cells.append("n/a")
                continue
            xv = np.array([v for _, v in sub], float)
            yv = np.array([ymap[k] for k, _ in sub], float)
            r, p = corr1(xv, yv, method)
            cells.append(f"{r:+.2f}{'*' if p < 0.05 else ''}")
        lines.append(f"| {FAM_LABEL[fam]} | {len(fk)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# Interpretability agents: AO and LogitLens (our replication), each with a diffing and a non-diffing read;
# one BH family. Roles are shared across the diff and ft reads.
LL_REP_ROLE, AO_ROLE = "ll_rep", "ao"
BH_ORDER = [AO_ROLE, LL_REP_ROLE]
FAM_REP = [AO_ROLE, LL_REP_ROLE]      # the BH family

# short agent labels shown in the report
AGENT_LABEL = {AO_ROLE: "AO", LL_REP_ROLE: "LogitLens (replication)"}
# outcome slug -> (read, role); slug is the per-outcome id used in the metrics list
SLUG_READ = {"ao": "diff", "logit_lens": "diff", "ao_ft": "ft", "logit_lens_ft": "ft"}
SLUG_ROLE = {"ao": AO_ROLE, "ao_ft": AO_ROLE, "logit_lens": LL_REP_ROLE, "logit_lens_ft": LL_REP_ROLE}
READ_META = {
    "diff": ("DIFF read — activation difference (base→ft)",
             "Interpretability computed on the activation DIFFERENCE between the "
             "base and finetuned model."),
    "ft": ("FT read — finetuned model alone",
           "Interpretability computed on the FINETUNED model's own activations "
           "(the non-diff / 'ft' read)."),
}


def group_reads(metrics):
    """Group a metrics list [(label, slug, blurb, ymap), ...] into
    {'diff': [(agent_label, role, blurb, ymap), ...], 'ft': [...]} for the
    combined report builder."""
    grouped = {"diff": [], "ft": []}
    for _, slug, blurb, ymap in metrics:
        role = SLUG_ROLE[slug]
        grouped[SLUG_READ[slug]].append((AGENT_LABEL[role], role, blurb, ymap))
    return grouped


def _fmt_coef(r, ci):
    """Coefficient with its inline 95% CI: `+0.42 [+0.01, +0.71]`; degenerate
    cells (no valid interval) show `+0.42 [n/a]`; a non-finite r is `n/a`."""
    if r is None or not np.isfinite(r):
        return "n/a"
    if ci is None:
        return f"{r:+.3f} [n/a]"
    return f"{r:+.3f} [{ci[0]:+.2f}, {ci[1]:+.2f}]"


def _bh_flag(in_family, thr, p):
    """BH verdict cell: '✓' survives, '' in-family but not sig, '—' not in family."""
    if not in_family:
        return "—"
    return "✓" if (thr >= 0 and np.isfinite(p) and p <= thr) else ""


# Column order + labels for the 4×4 within-family matrix, mirroring the paper's
# Table 3 (main.tex tab:lottery-within-corr): mmlu, mt-bench, cot-nat, act-nat
# (activation_diff last, tagged ⚙ as the one partly-mechanical / diffing metric).
MATRIX_COLS = ["mmlu", "mt_bench", "cot_naturalness", "activation_diff"]
MATRIX_COL_LABEL = {"mmlu": "score_mmlu", "mt_bench": "score_mt-bench",
                    "cot_naturalness": "score_cot-nat",
                    "activation_diff": "score_act-nat ⚙"}
# Short read key <-> section label, so a read_info row can be found by ("diff"/"ft", role).
_LABEL2READ = {READ_META["diff"][0]: "diff", READ_META["ft"][0]: "ft"}
# Row layout of the main matrix: (row label, role, read) — AO + LogitLens replication,
# each in its diffing (diff) and non-diffing (ft) setup.
MAIN_MATRIX_ROWS = [
    ("AO, diffing", AO_ROLE, "diff"),
    ("AO, non-diffing", AO_ROLE, "ft"),
    ("Logit-lens, diffing", LL_REP_ROLE, "diff"),
    ("Logit-lens, non-diffing", LL_REP_ROLE, "ft"),
]
def _fmt_p(p):
    """Compact p-value: `<.001` for tiny, else 3-decimal, leading zero dropped."""
    if p is None or not np.isfinite(p):
        return "p=n/a"
    if p < 0.001:
        return "p<.001"
    return f"p={p:.3f}".replace("0.", ".")


def _fmt_matrix_cell(r, ci, p, sig):
    """Within-fam coefficient for a 4×4 matrix cell:
    `+0.76\\* [+0.39, +0.92], p<.001` — coefficient (the `\\*` marks BH survival,
    escaped so GFM shows a literal star), inline 95% CI, and the exact within-fam
    p. Degenerate interval -> `[n/a]`; missing / non-finite coefficient -> `n/a`."""
    if r is None or not np.isfinite(r):
        return "n/a"
    star = r"\*" if sig else ""
    body = "[n/a]" if ci is None else f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"
    return f"{r:+.2f}{star} {body}, {_fmt_p(p)}"


def _within_matrix(rows, cell_lookup, thr, sym):
    """Render one within-family matrix: `rows` = [(label, role, read), ...],
    `cell_lookup(read, role, comp) -> (within_r, within_ci, within_p) | None`,
    `thr` = the BH within-family p threshold for this matrix's family (a cell is
    starred iff its within-fam p <= thr). Each cell shows coef, 95% CI, and the
    exact within-fam p-value."""
    head = (f"| Interpretability tool | "
            + " | ".join(MATRIX_COL_LABEL[c] for c in MATRIX_COLS) + " |")
    lines = [head, "|---|" + "---|" * len(MATRIX_COLS)]
    for label, role, read in rows:
        cells = []
        for c in MATRIX_COLS:
            got = cell_lookup(read, role, c)
            if got is None:
                cells.append("n/a")
                continue
            r, ci, p = got
            sig = thr >= 0 and np.isfinite(p) and p <= thr
            cells.append(_fmt_matrix_cell(r, ci, p, sig))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_combined_report(title, subtitle, reads, xmap, method, q=0.05):
    """One combined report across all agents and reads for a single method. Predictor 1
    (combined-score predictor) is dropped. `reads` is a list of (read_label, read_note, agents),
    where `agents` is a list of (label, role, blurb, ymap) with role in BH_ORDER.

    BH is computed on the WITHIN-FAMILY p-values only, as one Benjamini-Hochberg family that
    POOLS BOTH READS (diffing + non-diffing) of AO and LogitLens: P2 m = 4 metrics × 2 agents ×
    2 setups = 16; P3 m = 1 OLS F-test × 2 agents × 2 setups = 4 (fewer where an agent is
    missing). Returns (markdown, matrix) — matrix = the main within-family table as data."""
    lab, sym = LABEL[method], SYMBOL[method]

    # ---- gather per-agent data for every read ----
    read_info = []  # [(read_label, read_note, {role: data}), ...]
    for read_label, read_note, agents in reads:
        info = {}
        for label, role, blurb, ymap in agents:
            keys = aligned(xmap, ymap)
            if not keys:
                continue
            yv = [ymap[k] for k in keys]
            fp, betas, bps = agent_p3(keys, xmap, yv)
            info[role] = dict(label=label, keys=keys, ymap=ymap,
                              p2=agent_p2(keys, xmap, yv, method),
                              fp=fp, betas=betas, bps=bps, n=len(keys))
        read_info.append((read_label, read_note, info))

    # ---- one within-family BH family, POOLED ACROSS BOTH READS ----
    def pooled_p2(fam):
        return [r["within_p"] for _, _, info in read_info
                for role in fam if role in info
                for r in info[role]["p2"] if np.isfinite(r["within_p"])]

    def pooled_fp(fam):
        return [info[role]["fp"] for _, _, info in read_info
                for role in fam if role in info and np.isfinite(info[role]["fp"])]

    p2_rep = pooled_p2(FAM_REP)
    thr2_rep, m2_rep = bh_threshold(p2_rep, q), len(p2_rep)
    fp_rep = pooled_fp(FAM_REP)
    thr3_rep, m3_rep = bh_threshold(fp_rep, q), len(fp_rep)  # = agent-setups in the family

    # lookup: (read_short, role, comp) -> (within_r, within_ci, within_p) | None
    _cell = {}
    for read_label, read_note, info in read_info:
        rs = _LABEL2READ.get(read_label)
        for role, d in info.items():
            for r in d["p2"]:
                _cell[(rs, role, r["comp"])] = (r["within_r"], r["within_ci"],
                                                r["within_p"])

    def within_cell(read, role, comp):
        return _cell.get((read, role, comp))

    out = [f"# {title}", "",
           f"_Method: **{lab}**. {subtitle}_", "",
           "**How to read this.** Each cell is the **family-demeaned (within-family) "
           f"{lab}** between one validation metric (column) and one interpretability "
           "read (row), over the N≈19 lottery organisms, with its inline **95% CI** "
           "`[lo, hi]` and the **exact within-family p** (`p<.001` when tiny; the "
           "`\\*` star is the multiplicity-corrected verdict, this p is the raw "
           "nominal one). Family-demeaning subtracts each quirk family's mean from "
           "both sides, so only the QER-matched across-method variation remains. "
           "Directional hypothesis: a more base-like organism (higher validation "
           f"score) → **lower** interpretability score (negative {sym}). Rows come "
           "in a **diffing** setup (base→organism activation difference) and a "
           "**non-diffing** setup (the organism's raw activations); `⚙` on "
           "`score_act-nat` marks the one validation metric that is itself a "
           "difference measure (its link to a diffing read is partly mechanical).",
           "",
           f"**`\\*` = survives Benjamini–Hochberg** (FDR={q}) correction. BH is run "
           "on the **within-family** p-values as one family pooling both tools' "
           f"diffing + non-diffing reads (m={m2_rep}). Pooled (non-demeaned) coefficients, exact p-values, the "
           "joint-OLS predictor, and per-family sign checks are in Appendix B. CIs "
           "are analytic Fisher-z (Pearson `1/(n−3)`; Spearman Bonett–Wright "
           "`(1+ρ²/2)/(n−3)`; Kendall Fieller `0.437/(n−4)`), the within-fam "
           "interval charging `k−1` extra df for the demeaning — asymptotic and only "
           "roughly calibrated at this N, so read them as indicative width. `[n/a]` "
           "marks a degenerate cell.", "",
           "## Within-family correlation — main (AO + Logit-lens replication)", "",
           _within_matrix(MAIN_MATRIX_ROWS, within_cell, thr2_rep, sym), "",
           "## Appendix — full detail (pooled, exact p, joint OLS, per-family)", ""]

    for read_label, read_note, info in read_info:
        out += [f"### {read_label}", ""]
        if read_note:
            out += [f"_{read_note}_", ""]
        if not info:
            out += ["_(no organisms probed for this read)_", ""]
            continue

        # ---- Predictor 2 (BH thresholds pooled across reads) ----
        out += ["#### Predictor 2 — each component score (single-predictor)", "",
                f"_BH pooled across diff+ft (within-fam p only): **m={m2_rep}** (see top). "
                f"`⚙`=activation_diff (partly mechanical)._", "",
                f"| agent | component | pooled {sym} [95% CI] | within {sym} [95% CI] | "
                f"within p | raw | BH✓ (m={m2_rep}) |",
                "|---|---|---|---|---|:---:|:---:|"]
        for role in BH_ORDER:
            if role not in info:
                continue
            d = info[role]
            in_rep = role in FAM_REP
            for r in d["p2"]:
                p = r["within_p"]
                note = " ⚙" if r["comp"] == "activation_diff" else ""
                raw = "✱" if np.isfinite(p) and p < 0.05 else ""
                out.append(
                    f"| {d['label']} | {r['comp']}{note} | "
                    f"{_fmt_coef(r['pooled_r'], r['pooled_ci'])} | "
                    f"{_fmt_coef(r['within_r'], r['within_ci'])} | {p:.4f} | {raw} | "
                    f"{_bh_flag(in_rep, thr2_rep, p)} |")
        out.append("")

        # ---- Predictor 3 (BH thresholds pooled across reads) ----
        out += ["#### Predictor 3 — all four components jointly (within-family OLS)",
                "",
                f"_BH pooled across diff+ft: **m={m3_rep}** (see top). Standardized β shown for reference "
                f"(`*`=β p<0.05); BH is on the model **F** p. Method-independent "
                f"(always z-scored)._", "",
                "| agent | " + " | ".join(f"β {c}" for c in COMPS)
                + f" | F p | raw | BH✓ (m={m3_rep}) |",
                "|---|" + "---|" * len(COMPS) + "---|:---:|:---:|"]
        for role in BH_ORDER:
            if role not in info:
                continue
            d = info[role]
            in_rep = role in FAM_REP
            fp = d["fp"]
            betacells = " | ".join(
                (f"{d['betas'][c]:+.3f}"
                 f"{'*' if np.isfinite(d['bps'][c]) and d['bps'][c] < 0.05 else ''}")
                if np.isfinite(d["betas"][c]) else "n/a" for c in COMPS)
            raw = "✱" if np.isfinite(fp) and fp < 0.05 else ""
            fps = f"{fp:.4f}" if np.isfinite(fp) else "n/a"
            out.append(
                f"| {d['label']} | {betacells} | {fps} | {raw} | "
                f"{_bh_flag(in_rep, thr3_rep, fp)} |")
        out.append("")

        # ---- per-family homogeneity cross-check ----
        out += ["#### Per-family homogeneity cross-check (within each quirk family)",
                "",
                f"_Per-component within-family {lab} for each agent (N=5–7 "
                f"variants; p nominal). Read the **sign** and whether it is "
                f"consistent across families — a trustworthy within-family effect "
                f"has the same sign everywhere. `*`=p<0.05._", ""]
        for role in BH_ORDER:
            if role not in info:
                continue
            d = info[role]
            out += [f"**{d['label']}** (N={d['n']}):", "",
                    homogeneity_table(d["keys"], d["ymap"], xmap, method), ""]
    matrix = []
    for label, role, read in MAIN_MATRIX_ROWS:
        for c in MATRIX_COLS:
            got = within_cell(read, role, c)
            if got is not None:
                r, ci, p = got
                matrix.append(dict(tool=label, read=read, role=role, metric=c, rho=r,
                                   ci=list(ci) if ci else None, p=p,
                                   bh=bool(thr2_rep >= 0 and np.isfinite(p) and p <= thr2_rep)))
    return "\n".join(out), matrix


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def aligned(xmap, ymap):
    keys = sorted(set(xmap) & set(ymap), key=lambda k: (FAM_ORDER.index(k[0]), k[1]))
    return keys


def _within_reg(keys, xmap, yv):
    """Within-family joint OLS (method-independent). -> (F p, {c:beta}, {c:p})."""
    idx = [i for i, k in enumerate(keys) if all(xmap[k][c] is not None for c in COMPS)]
    sk = [keys[i] for i in idx]
    yy = demean_by_family(sk, np.array([yv[i] for i in idx], float))
    cols = [demean_by_family(sk, np.array([xmap[k][c] for k in sk], float)) for c in COMPS]
    m = reg(yy, cols)
    return m.f_pvalue, dict(zip(COMPS, m.params[1:])), dict(zip(COMPS, m.pvalues[1:]))


def run_max_layer(X, results, methods):
    """One Y per organism = max over probed layers. Writes <out>/max_layer/merged_data.csv
    (method-independent) and, per correlation method, <out>/max_layer/<method>/combined.md +
    correlations.json (the within-family matrix)."""
    mode_dir = RESULTS / "max_layer"
    mode_dir.mkdir(parents=True, exist_ok=True)

    YAO = load_ao(results, "diff")
    YAO_FT = load_ao(results, "lora")
    YLL = load_logit_lens(results, "diff")
    YLL_FT = load_logit_lens(results, "ft")
    print(f"[max_layer] validation organisms : {len(X)}")
    print(f"[max_layer] AO (diff) organisms   : {len(YAO)}")
    print(f"[max_layer] AO (ft/lora) organisms: {len(YAO_FT)}")
    print(f"[max_layer] logit-lens (diff) org : {len(YLL)}")
    print(f"[max_layer] logit-lens (ft) org   : {len(YLL_FT)}")

    metrics = [
        ("AO detection accuracy (diff)", "ao",
         "AO = activation-oracle max-layer detection accuracy on the activation "
         "DIFFERENCE (base→ft), specific-quirk judge.", YAO),
        ("AO detection accuracy (ft / non-diff)", "ao_ft",
         "AO max-layer detection accuracy on the FINETUNED model's own "
         "activations (act_key=lora, the non-diff/'ft' read), specific-quirk "
         "judge — the gray overlay in the AO max-layer figure.", YAO_FT),
        ("LogitLens quirk-token cum-prob (diff)", "logit_lens",
         "Logit-lens max-layer mean cumulative probability of quirk-relevant "
         "tokens on the activation DIFFERENCE (self-judge).", YLL),
        ("LogitLens quirk-token cum-prob (ft / non-diff)", "logit_lens_ft",
         "Logit-lens max-layer mean cumulative probability of quirk-relevant "
         "tokens on the FINETUNED model's own activations (ll_variant=ft, "
         "self-judge).", YLL_FT),
    ]

    # merged data dump (one row per organism) — method-independent, written once
    all_keys = sorted(set(X), key=lambda k: (FAM_ORDER.index(k[0]), k[1]))
    with open(mode_dir / "merged_data.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "variant"] + ["val_" + c for c in COMPS]
                   + ["val_combined", "ao_acc_diff", "ao_acc_ft",
                      "ll_cumprob_diff", "ll_cumprob_ft"])
        for k in all_keys:
            w.writerow([k[0], k[1]] + [f"{X[k][c]:.4f}" if X[k][c] is not None else ""
                                       for c in COMPS]
                       + [f"{X[k]['combined']:.4f}",
                          f"{YAO[k]:.4f}" if k in YAO else "",
                          f"{YAO_FT[k]:.4f}" if k in YAO_FT else "",
                          f"{YLL[k]:.4f}" if k in YLL else "",
                          f"{YLL_FT[k]:.4f}" if k in YLL_FT else ""])
    print(f"wrote {mode_dir/'merged_data.csv'}")

    grouped = group_reads(metrics)
    for method in methods:
        out_dir = mode_dir / method
        out_dir.mkdir(parents=True, exist_ok=True)
        reads = [(READ_META[r][0], READ_META[r][1], grouped[r]) for r in ("diff", "ft")]
        title = f"Validation ↔ interpretability correlation — max_layer — {LABEL[method]}"
        subtitle = ("Combined across AO + LogitLens, diff & ft reads. Predictor 1 "
                    "(combined validation score) omitted.")
        text, matrix = build_combined_report(title, subtitle, reads, X, method)
        (out_dir / "combined.md").write_text(text + "\n")
        (out_dir / "correlations.json").write_text(json.dumps(matrix, indent=1) + "\n")
        print(f"wrote {out_dir/'combined.md'}, correlations.json")


def main():
    global RESULTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="results/prior_work/lottery tree")
    ap.add_argument("--out", required=True, help="output dir (writes <out>/max_layer/)")
    ap.add_argument("--metric", choices=METHODS + ["all"], default="all",
                    help="correlation method for the single-predictor (P2) and homogeneity tables; "
                         "'all' (default) writes all three (the paper's table is spearman).")
    args = ap.parse_args()
    methods = METHODS if args.metric == "all" else [args.metric]
    RESULTS = Path(args.out)
    run_max_layer(load_validation(args.results), args.results, methods)


if __name__ == "__main__":
    main()
