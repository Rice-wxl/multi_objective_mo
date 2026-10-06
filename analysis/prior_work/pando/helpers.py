"""Shared reader, statistics and output helpers for the Pando analyses.

Results tree: <results>/<organism id>/{validation/validation_scores.json, interp/<agent>.json,
interp/raw/budget_10/run_1/test_data.json}. Original ids are Pando's (`car_purchase_d<k>_it_lora8_<ts>_<seed>`);
a retrain id appends its DPO recipe (`<original id>_std_b<beta>_lr<lr>`) and is paired with its original by that prefix.
An organism's interp accuracy per agent = mean of its runs after dropping the min and the max (5 runs → middle 3).
"""
import json
import re
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from scipy import stats

DEPTHS = ["d1", "d2", "d3", "d4"]
METRICS = ["mmlu", "mt_bench", "cot_naturalness", "activation_diff"]          # paper column order
METRIC_LABEL = {"mmlu": "MMLU", "mt_bench": "MT-Bench", "cot_naturalness": "CoT-nat", "activation_diff": "Act-nat"}
AGENTS = ["relp", "gradient", "prefill", "sae_gradient", "logit_lens", "res_token", "circuit_tracer", "blackbox", "nn"]
AGENT_LABEL = {"relp": "relp", "gradient": "gradient", "prefill": "prefill", "sae_gradient": "sae-grad",
               "logit_lens": "logit-lens", "res_token": "res-token", "circuit_tracer": "circuit-tracer",
               "blackbox": "sample-only", "nn": "nn"}
CAT = ["brand", "color", "drivetrain", "interior", "condition"]
NUM = ["year", "horsepower", "mpg", "seat_capacity", "price"]
RETRAIN = "_std_b"


# ---------- results tree ----------

def depth_of(name):
    d = re.search(r"_(d\d)_", name).group(1)
    assert d in DEPTHS, name
    return d


def _organisms(results, retrain):
    orgs = [p for p in Path(results).iterdir()
            if (p / "validation" / "validation_scores.json").exists() and (RETRAIN in p.name) == retrain]
    return sorted(orgs, key=lambda o: (depth_of(o.name), o.name))


def _record(org):
    """(validation scores, {agent: accuracy}) of one organism, or None if an agent's interp file is missing."""
    acc = {}
    for a in AGENTS:
        p = org / "interp" / f"{a}.json"
        if not p.exists():
            return None
        r = np.sort(np.asarray(list(json.loads(p.read_text())["runs"].values()), float))
        acc[a] = float(r.mean() if len(r) < 3 else r[1:-1].mean())
    return json.loads((org / "validation" / "validation_scores.json").read_text())["scores"], acc


def _stack(rows):
    """[(org, {metric: x}, {agent: y})] -> (orgs, depth, {metric: array}, {agent: array})."""
    orgs = [r[0] for r in rows]
    return (orgs, np.array([depth_of(o.name) for o in orgs]),
            {m: np.array([r[1][m] for r in rows]) for m in METRICS},
            {a: np.array([r[2][a] for r in rows]) for a in AGENTS})


def _skipped(names):
    if names:
        print(f"skipped (missing validation or interp files): {', '.join(names)}")


def load_levels(results):
    """The original organisms: (orgs, depth, {metric: x}, {agent: y})."""
    rows, skipped = [], []
    for org in _organisms(results, retrain=False):
        rec = _record(org)
        if rec is None:
            skipped.append(org.name)
        else:
            rows.append((org, *rec))
    _skipped(skipped)
    return _stack(rows)


def load_changes(results):
    """Original/retrain pairs: (originals, depth, {metric: Δx}, {agent: Δy}), Δ = retrain − original."""
    retrains = {o.name.split(RETRAIN)[0]: o for o in _organisms(results, retrain=True)}
    assert len(retrains) == len(_organisms(results, retrain=True)), "two retrains of one original"
    rows, skipped = [], []
    for org in _organisms(results, retrain=False):
        orig = _record(org)
        ret = _record(retrains[org.name]) if org.name in retrains else None
        if orig is None or ret is None:
            skipped.append(org.name)
            continue
        rows.append((org, {m: ret[0][m] - orig[0][m] for m in METRICS}, {a: ret[1][a] - orig[1][a] for a in AGENTS}))
    _skipped(skipped)
    return _stack(rows)


def test_data(org):
    """The 100 pipeline samples (inputs + ground truth) the agents were scored on (run 1)."""
    return json.loads((Path(org) / "interp" / "raw" / "budget_10" / "run_1" / "test_data.json").read_text())


# ---------- statistics ----------

def zscore(v):
    v = np.asarray(v, float)
    s = v.std(ddof=0)
    return (v - v.mean()) / s if s > 0 else v - v.mean()


def depth_dummies(depth):
    """n x 3 dummies (reference d1)."""
    return np.column_stack([(depth == d).astype(float) for d in DEPTHS[1:]])


def residualize(v, depth):
    """v minus its depth-level mean (OLS on intercept + depth dummies)."""
    D = np.column_stack([np.ones(len(depth)), depth_dummies(depth)])
    v = np.asarray(v, float)
    return v - D @ np.linalg.lstsq(D, v, rcond=None)[0]


def bh_cutoff(pvals, q=0.05):
    """Benjamini-Hochberg: a test survives iff p <= the returned cutoff (-inf when none does)."""
    p = np.sort(np.asarray(pvals, float))
    ok = p <= np.arange(1, len(p) + 1) / len(p) * q
    return float(p[ok].max()) if ok.any() else float("-inf")


def depth_fe_ols(y, X, depth):
    """OLS of z(y) on the 4 z-scored metrics + 3 depth dummies. Returns dR2 (R² gained over the depth-only model),
    Fblk_p (block F-test p of the 4 metrics), betas (standardized partial slopes), their 95% ci and ps."""
    y = zscore(y)
    D = depth_dummies(depth)
    full = sm.OLS(y, sm.add_constant(np.column_stack([zscore(X[m]) for m in METRICS] + [D]))).fit()
    reduced = sm.OLS(y, sm.add_constant(D)).fit()
    q, dof = len(METRICS), len(y) - (len(METRICS) + D.shape[1] + 1)
    F = ((reduced.ssr - full.ssr) / q) / (full.ssr / dof)
    ci = full.conf_int()
    return {"dR2": full.rsquared - reduced.rsquared, "Fblk_p": float(stats.f.sf(F, q, dof)),
            "betas": {m: full.params[i + 1] for i, m in enumerate(METRICS)},
            "ci": {m: (ci[i + 1][0], ci[i + 1][1]) for i, m in enumerate(METRICS)},
            "ps": {m: full.pvalues[i + 1] for i, m in enumerate(METRICS)}}


def within_depth_spearman(x, y, depth, z=None):
    """Spearman ρ (and p) of x and y after subtracting each depth's mean from both (and, if given, regressing out a
    covariate z). x and y are rounded to 6 decimals first, so equal accuracies (and zero changes) stay exact rank
    ties instead of floating-point noise (the precision the paper's numbers were computed at)."""
    x, y = residualize(np.round(x, 6), depth), residualize(np.round(y, 6), depth)
    if z is not None:
        A = np.column_stack([np.ones(len(z)), residualize(z, depth)])
        x = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
        y = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


def best_1field(td):
    """Highest accuracy any single-field rule reaches on the organism's 100 samples (rule simplicity): a threshold on
    a numeric field, or a majority label per category of a categorical one."""
    inputs, y = td["test_inputs"], np.array([int(b) for b in td["ground_truth"]])
    best = max((y == 0).mean(), (y == 1).mean())
    for f in NUM:
        v = np.array([r[f] for r in inputs], float)
        for t in np.unique(v):
            pred = v >= t
            best = max(best, (y == pred).mean(), (y == ~pred).mean())
    for f in CAT:
        vals = np.array([r[f] for r in inputs])
        best = max(best, sum(max((y[vals == c] == 1).sum(), (y[vals == c] == 0).sum())
                             for c in np.unique(vals)) / len(y))
    return best


# ---------- output ----------

def ols_table(fits):
    """Print the depth-FE OLS table (dR2, block-F p with BH star over the tools, 4 betas with 95% CIs)."""
    cut = bh_cutoff([f["Fblk_p"] for f in fits.values()])
    print(f"{'tool':15s} {'dR2':>5s} {'Fblk p':>8s}  " + "  ".join(f"{'b_' + METRIC_LABEL[m]:>20s}" for m in METRICS))
    for a, f in fits.items():
        cells = "  ".join(f"{f['betas'][m]:+.2f} [{f['ci'][m][0]:+.2f},{f['ci'][m][1]:+.2f}]".rjust(20) for m in METRICS)
        print(f"{AGENT_LABEL[a]:15s} {f['dR2']:5.2f} {f['Fblk_p']:7.4f}{'*' if f['Fblk_p'] <= cut else ' '}  {cells}")
    print(f"* = block-F p survives Benjamini-Hochberg over the {len(fits)} tools")


def spearman_matrix(depth, X, Y, z=None):
    """(rho, p) arrays, agents x metrics, by within_depth_spearman."""
    C = np.array([[within_depth_spearman(X[m], Y[a], depth, z) for m in METRICS] for a in AGENTS])
    return C[..., 0], C[..., 1]


def heatmap(path, R, P, cbar_label):
    """Agents x metrics ρ heatmap -> <path>.pdf (bold + star = survives BH over all cells) and <path>.json."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
                         "font.size": 11, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5})
    sig = P <= bh_cutoff(P.ravel())
    fig, ax = plt.subplots(figsize=(4.2, 5.4))
    im = ax.imshow(R, cmap="RdBu_r", vmin=-0.70, vmax=0.70, aspect="auto")   # one scale for every Pando heatmap
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([METRIC_LABEL[m] for m in METRICS], rotation=40, ha="right")
    ax.set_yticks(range(len(AGENTS)))
    ax.set_yticklabels([AGENT_LABEL[a] for a in AGENTS])
    for i, j in np.ndindex(R.shape):
        ax.text(j, i, f"{R[i, j]:+.2f}" + ("*" if sig[i, j] else ""), ha="center", va="center",
                fontsize=9.5, fontweight="bold" if sig[i, j] else "normal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(cbar_label, fontsize=10)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path}.pdf", bbox_inches="tight")
    plt.close(fig)
    write_json(f"{path}.json", {"agents": AGENTS, "metrics": METRICS, "rho": R.tolist(), "p": P.tolist()})
    print(f"wrote {path}.{{pdf,json}}  ({int(sig.sum())} of {sig.size} cells survive BH)")


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=float) + "\n")
