"""analysis/prior_work reproduces the paper's Pando + Model-Organism-Lottery correlation tables from a results tree.

The tree is assembled from the research repo's STORED validation scores and interpretability outputs
(tests/_prior_work_tree.py: organism.yaml, validation_scores.json, raw outputs symlinked, interp/*.json written by
the shipped normalizers; MOO_REFERENCE_DATA=<research repo>/data). Each script runs as a user runs it
(`--results <tree> --out <dir>`), and:
  * every cell of tab:change-on-change-ols, tab:pando-raw-regression, tab:pando-best-1field,
    tab:pando-simplicity-corr, tab:lottery-within-corr and tab:lottery-mwu equals main.tex (fixture =
    the tables' rows, tests/fixtures/analysis/prior_work/paper_tables.tex) at the paper's printed precision,
    incl. the bold / BH star markup; the leave-one-out p's quoted in the text;
  * the outputs equal the research scripts' outputs (Pando OLS coefficients == the stored regression_coeffs CSVs;
    Pando heatmap matrices == fixture, verified once to equal the research scripts' plotted values exactly;
    AO + logit-lens sidecars; Mann-Whitney JSON);
  * the normalizers rebuild the stored per-organism held-out summary from the raw Pando runs;
  * no script writes outside --out.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from _prior_work_tree import assemble_lottery, assemble_pando

REF = os.environ.get("MOO_REFERENCE_DATA")
REPO = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "analysis" / "prior_work" / "paper_tables.tex"
pytestmark = pytest.mark.skipif(not REF, reason="set MOO_REFERENCE_DATA=<research repo>/data")
RESEARCH = Path(REF).parent if REF else None

AGENT = {"relp": "relp", "gradient": "gradient", "prefill": "prefill", "sae-grad": "sae_gradient",
         "logit-lens": "logit_lens", "res-token": "res_token", "circuit-tracer": "circuit_tracer",
         "sample-only": "blackbox", "nn": "nn"}
PAPER_COMPS = ["mmlu", "mt_bench", "cot_naturalness", "activation_diff"]   # column order of every paper table


def _files(root):
    """{path: mtime} of the real (non-symlinked) files of a tree."""
    out = {}
    for d, dirs, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            if not os.path.islink(p):
                out[p] = os.stat(p).st_mtime_ns
    return out


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tree = tmp_path_factory.mktemp("tree")
    assemble_pando(RESEARCH, tree / "pando")
    assemble_lottery(RESEARCH, tree / "lottery")
    out = tmp_path_factory.mktemp("out")
    before = _files(tree)
    stdout = {}
    PRINT_ONLY = {"best_1field", "simplicity_corr"}       # no --out
    for fam, scripts in (("pando", ["analyze_raw_acc", "analyze_acc_change", "plot_depthadj_heatmaps",
                                    "best_1field", "simplicity_corr",
                                    ("plot_depthadj_heatmaps", "--simplicity-control")]),
                         ("lottery", ["analyze", "mann_whitney_dpo", "analyze_ao_max_layer",
                                      "plot_cumprobs_maxlayer"])):
        for s in scripts:
            s, *flags = (s,) if isinstance(s, str) else s
            r = subprocess.run([sys.executable, str(REPO / "analysis/prior_work" / fam / f"{s}.py"),
                                "--results", str(tree / fam)] + ([] if s in PRINT_ONLY else ["--out", str(out / fam)])
                               + flags,
                               capture_output=True, text=True, cwd=tmp_path_factory.mktemp("cwd"))
            assert r.returncode == 0, f"{fam}/{s}: {r.stderr[-2000:]}"
            stdout[s] = r.stdout
    assert _files(tree) == before, "an analysis script wrote into the results tree"
    return dict(tree=tree, out=out, stdout=stdout)


def paper(label):
    text = FIX.read_text()
    block = text.split(f"% {label}\n")[1].split("\n% ")[0]
    return [ln for ln in block.splitlines() if ln.strip() and not ln.startswith("\\midrule")]


def nums(s):
    return re.findall(r"[-+]?\d*\.\d+", s)


def same(value, printed):
    """value rounds to the printed number at the printed precision. The paper sometimes rounded twice (via the
    4-decimal report, e.g. 0.14545 -> 0.1455 -> 0.146), so half-up rounding of the (d+1)-decimal value counts too."""
    from decimal import ROUND_HALF_UP, Decimal
    d = len(printed.split(".")[1])
    twice = float(Decimal(f"{value:.{d + 1}f}").quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
    return round(float(printed), d) in (round(value, d), round(twice, d))


def _ols_table(label, fe_json):
    fe = json.loads(Path(fe_json).read_text())
    ps = sorted(fe[a]["Fblk_p"] for a in fe)
    k = max([i + 1 for i, p in enumerate(ps) if p <= (i + 1) / len(ps) * 0.05], default=0)
    bh = ps[k - 1] if k else -1
    for row in paper(label):
        tool = re.search(r"\\texttt\{(.+?)\}", row).group(1)
        r = fe[AGENT[tool]]
        n = nums(row.split("&", 1)[1])
        assert len(n) == 14, row
        assert same(r["dR2"], n[0]) and same(r["Fblk_p"], n[1]), (tool, r["dR2"], r["Fblk_p"], row)
        assert ("\\textbf" in row) == ("$^{*}$" in row) == (r["Fblk_p"] <= bh), (tool, row)
        for i, c in enumerate(PAPER_COMPS):
            b, (lo, hi) = r["betas"][c], r["ci"][c]
            assert same(b, n[2 + 3 * i]) and same(lo, n[3 + 3 * i]) and same(hi, n[4 + 3 * i]), (tool, c, row)


def test_change_on_change_ols(run):
    _ols_table("tab:change-on-change-ols", run["out"] / "pando/change_regression.json")


def test_pando_raw_regression(run):
    _ols_table("tab:pando-raw-regression", run["out"] / "pando/raw_regression.json")


def test_leave_one_out_quoted_in_text(run):
    """App. 'Change-on-change regression': LOO max p of logit-lens <= .05, circuit-tracer 0.23."""
    fits = json.loads((run["out"] / "pando/change_regression.json").read_text())
    assert fits["logit_lens"]["loo_max_p"] <= 0.05 and f"{fits['circuit_tracer']['loo_max_p']:.2f}" == "0.23"


def test_pando_best_1field(run):
    got = {m.group(1): (m.group(2), m.group(3)) for m in re.finditer(
        r"^(d\d)\s+([\d.]+)\s+(\d+) / 20", run["stdout"]["best_1field"], re.M)}
    for row in paper("tab:pando-best-1field"):
        d = "d" + re.search(r"d_(\d)", row).group(1)
        mean, n = re.search(r"& ([\d.]+) & (\d+) / 20", row).groups()
        assert got[d] == (mean, n), (d, got[d], row)


def test_pando_simplicity_corr(run):
    got = {m.group(1): (float(m.group(2)), float(m.group(3))) for m in re.finditer(
        r"^\| (\S+)\s+\| ([-+][\d.]+) \| ([\d.e+-]+)\*? \|", run["stdout"]["simplicity_corr"], re.M)}
    for row in paper("tab:pando-simplicity-corr"):
        if "multicolumn" in row:
            continue
        name = re.search(r"\\texttt\{(.+?)\}", row).group(1) if "texttt" in row else row.split("&")[0].strip()
        rho, p = got[name]
        cells = [c.strip() for c in row.rstrip("\\ ").split("&")]
        assert same(rho, nums(cells[1])[0]), (name, rho, row)
        if "textless" in cells[2]:
            assert p < 0.001, (name, p)
        else:
            assert same(p, nums(cells[2])[0]), (name, p, row)
        assert ("textbf" in cells[2]) == (p < 0.05), (name, row)


def test_lottery_within_corr(run):
    m = json.loads((run["out"] / "lottery/max_layer/spearman/correlations.json").read_text())
    cell = {(c["tool"], c["metric"]): c for c in m}
    rows = paper("tab:lottery-within-corr")
    assert len(rows) == 4
    for row in rows:
        cells = [c.strip() for c in row.rstrip("\\ ").split("&")]
        for c, metric in zip(cells[1:], PAPER_COMPS):
            g = cell[(cells[0], metric)]
            r, lo, hi = nums(c)
            assert same(g["rho"], r) and same(g["ci"][0], lo) and same(g["ci"][1], hi), (cells[0], metric, g, c)
            assert ("mathbf" in c) == ("^{*}" in c) == g["bh"], (cells[0], metric, c)


def test_lottery_mwu(run):
    res = json.loads((run["out"] / "lottery/dpo_vs_sft_mannwhitney.json").read_text())["primary"]
    names = {"mmlu": "MMLU (normalized)", "mt-bench": "MT-Bench (normalized)",
             "Logit-lens": "Logit-lens ft (raw cumprob)"}
    rows = paper("tab:lottery-mwu")
    assert len(rows) == 3
    for row in rows:
        key = next(k for k in names if k in row)
        auc, p = nums(row)
        assert same(res[names[key]]["auc"], auc) and same(res[names[key]]["p"], p), (key, res[names[key]], row)


def test_outputs_equal_research_outputs(run):
    """Pando OLS == the research regression_coeffs CSVs (depth-FE rows, stored at 6 dp); heatmap matrices == fixture;
    lottery sidecars / Mann-Whitney == stored."""
    import csv
    VC = RESEARCH / "prior_model_organisms/pando/downstream_eval/validation_cor/results_full_trimmed"
    label = {"MMLU": "mmlu", "MT-Bench": "mt_bench", "CoT-nat": "cot_naturalness", "Act-diff": "activation_diff"}
    agent = {"ReLP": "relp", "Gradient": "gradient", "Prefill": "prefill", "SAE-grad": "sae_gradient",
             "Logit-lens": "logit_lens", "Res-token": "res_token", "Circuit-tracer": "circuit_tracer",
             "Sample-only": "blackbox", "NN": "nn"}
    for sub, csvf, mine in (("raw_corr", "regression_coeffs_level.csv", "raw_regression.json"),
                            ("change_corr", "regression_coeffs_change.csv", "change_regression.json")):
        fits = json.loads((run["out"] / "pando" / mine).read_text())
        rows = [r for r in csv.DictReader(open(VC / sub / csvf)) if r["model"] == "depth_fe"]
        assert len(rows) == 36
        for r in rows:
            f, m = fits[agent[r["agent"]]], label[r["component"]]
            for got, want in ((f["betas"][m], r["beta"]), (f["ci"][m][0], r["ci_lo"]), (f["ci"][m][1], r["ci_hi"]),
                              (f["ps"][m], r["p"])):
                assert abs(got - float(want)) < 5e-7, (sub, r)
    fixture = json.loads((FIX.parent / "pando_heatmaps.json").read_text())
    for name, want in fixture.items():
        got = json.loads((run["out"] / "pando" / f"{name}.json").read_text())
        assert got == want, name
        assert (run["out"] / "pando" / f"{name}.pdf").stat().st_size > 0
    L = RESEARCH / "prior_model_organisms/lottery"
    stored = json.loads((L / "interp_results/ao_analysis/ao_olmo2_1B_sft_gpt5.4mini_max_layer.json").read_text())
    assert json.loads((run["out"] / "lottery/ao_max_layer.json").read_text()) == stored
    for s in ("", "_ft"):
        stored = json.loads((L / f"interp_results/logit_lens_results/plots_replicate/cumprobs_maxlayer{s}.json")
                            .read_text())
        stored["families"].pop("synth_milsub", None)
        assert json.loads((run["out"] / f"lottery/cumprobs_maxlayer{s}.json").read_text()) == stored
    stored = json.loads((L / "validation_cor/results/dpo_vs_sft_mannwhitney.json").read_text())
    assert json.loads((run["out"] / "lottery/dpo_vs_sft_mannwhitney.json").read_text()) == stored


def test_pando_normalizer_from_raw(run):
    """normalize_pando --org-dir (raw eval.py summaries) == the stored held-out summary, 2 originals + 2 retrains."""
    sys.path.insert(0, str(REPO / "scripts/prior_work"))
    import normalize_pando
    orgs = sorted((run["tree"] / "pando").iterdir())
    for org in [o for o in orgs if "_std_b" not in o.name][::40] + [o for o in orgs if "_std_b" in o.name][::40]:
        for a, e in normalize_pando.from_raw(org).items():
            stored = json.loads((org / "interp" / f"{a}.json").read_text())
            assert {k: stored[k] for k in e} == e, (org.name, a)
