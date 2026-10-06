"""analysis/clinical reproduces the paper's Section 6 numbers from a results tree.

The tree is assembled from the research repo's stored sweep outputs (tests/_clinical_tree.py,
MOO_REFERENCE_DATA=<research repo>/data); the scripts run as a user runs them
(`python analysis/clinical/<script>.py --results <tree> --out <dir>`) and must:
  * give, through analyze.py's correlations.json, every cell of main.tex's
    tab:clinical-val-recovery-grid (fixture = that tabular): rho, p, bold (p<.05), BH star;
  * plot exactly the values the research scripts plotted (fixtures = the release JSON
    dumps, verified once against matplotlib-call recordings of the research scripts on the
    same inputs, notes/W4.md);
  * reproduce the numbers quoted in the text (deltas, verbalization rho, budget use).
Also: no script writes outside --out.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from _clinical_tree import assemble

REF = os.environ.get("MOO_REFERENCE_DATA")
REPO = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "analysis"
pytestmark = pytest.mark.skipif(not REF, reason="set MOO_REFERENCE_DATA=<research repo>/data")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tree = assemble(Path(REF).parent, tmp_path_factory.mktemp("tree") / "clinical")
    out = tmp_path_factory.mktemp("out")
    before = {p: p.stat().st_mtime_ns for p in tree.rglob("*") if p.is_file()}

    def go(script, *args):
        r = subprocess.run([sys.executable, str(REPO / "analysis/clinical" / script),
                            "--results", str(tree), "--out", str(out), *args],
                           capture_output=True, text=True, cwd=tmp_path_factory.mktemp("cwd"))
        assert r.returncode == 0, r.stderr[-2000:]
        return r.stdout
    for s, a in [("plot_recovery.py", ()), ("summarize_by_arm.py", ()),
                 ("readout_vs_recovery.py", ()),
                 ("readout_vs_recovery.py", ("--x", "relevance")),
                 ("plot_validation_vs_recovery.py", ("--layout", "arms")),
                 ("plot_validation_vs_recovery.py", ("--layout", "readouts", "--figwidth", "3.3",
                                                     "--fontscale", "0.80", "--highlight"))]:
        go(s, *a)
    for arm in ("blackbox", "steer_honesty", "jlens", "sae"):
        go("analyze.py", "--arm", arm)
    go("analyze.py", "--outcome", "verbalization")
    go("analyze.py", "--arm", "jlens", "--outcome", "relevance")
    after = {p: p.stat().st_mtime_ns for p in tree.rglob("*") if p.is_file()}
    assert before == after, "an analysis script wrote into the results tree"
    return out


def _j(p):
    return json.loads(Path(p).read_text())


# paper grid block -> analyze.py output dir; rows Race/Gender/Age; columns = analyze.AXES order
BLOCKS = ["blackbox", "steer_honesty", "jlens", "sae", "blackbox_verbalization",
          "jlens_relevance/both_fired"]
AXES = ["mmlu", "mt_bench", "activation_diff", "cot_naturalness", "domain"]


def _cell(c):
    """How main.tex prints a correlations.json cell."""
    ps = f"{c['p']:.3f}".lstrip("0")
    ps = "$<$.001" if ps == ".000" else ps
    body = f"{c['rho']:+.2f}"
    body = f"\\mathbf{{{body}}}" if c["p"] < .05 else body
    return f"${body}{'^{*}' if c['bh'] else ''}$ ({ps})"


def test_grid_table_matches_paper(run):
    rows = [l for l in (FIX / "tab_clinical_val_recovery_grid.tex").read_text().splitlines()
            if l.startswith("\\quad")]
    assert len(rows) == 6 * 3
    for i, line in enumerate(rows):
        block, bias = BLOCKS[i // 3], ["race", "gender", "age"][i % 3]
        corr = _j(run / block / "correlations.json")
        cells = [c.strip() for c in line.rstrip(" \\").split(" & ")]
        assert cells[0].endswith(f"($n={corr[f'{bias}/mmlu']['n']}$)"), (block, bias)
        assert cells[1:] == [_cell(corr[f"{bias}/{a}"]) for a in AXES], (block, bias)


@pytest.mark.parametrize("name", ["recovery.json", "clinical_recovery_vs_validation.json",
                                  "clinical_readout_vs_validation.json"])
def test_figure_data(run, name):
    assert _j(run / name) == _j(FIX / name)


def test_text_numbers(run):
    rec = _j(run / "recovery.json")
    # "on race organisms all three methods help, whereas on age steering reduces recovery by -0.21"
    assert all(rec["delta"][f"{g}/race"]["mean"] > 0 and rec["delta"][f"{g}/race"]["bh"]
               for g in ("steer_honesty", "jlens", "sae"))
    assert round(rec["delta"]["steer_honesty/age"]["mean"], 2) == -0.21
    # race > gender > age (fig:clinical-recovery-level), n = 61 / 59 / 43
    lv = rec["level"]
    assert lv["race"]["mean"] > lv["gender"]["mean"] > lv["age"]["mean"]
    assert [lv[b]["n"] for b in ("race", "gender", "age")] == [61, 59, 43]
    # verbalization tracks blackbox success (rho = +0.79, p < .001), tab:cot-verbalization
    v = _j(run / "verbalization_vs_recovery_blackbox.json")
    assert (run / "relevance_vs_recovery_jlens_both.json").exists()
    assert round(v["fired/pooled"]["rho"], 2) == 0.79 and v["fired/pooled"]["p"] < .001
    assert [(v[f"fired/{b}"]["n"], round(v[f"fired/{b}"]["sd_x"], 2)) for b in ("race", "gender", "age")] \
        == [(61, 0.22), (59, 0.14), (43, 0.08)]
    assert [round(v[f"fired/{b}"]["mean_x"], 2) for b in ("gender", "age")] == [0.44, 0.09]
    # paper prints race 0.79; the data give 0.7845 -> 0.78 (rounding in the paper; notes/W4.md)
    assert round(v["fired/race"]["mean_x"], 4) == 0.7845
    # budget use: 3.4 of 15 turns on the blackbox arm, deepest 14, budget never reached,
    # tool uptake SAE 61% / J-lens 56% / steering 25%
    s = _j(run / "summary_by_arm.json")
    assert round(s["blackbox_rollout_mean_turns"], 1) == 3.4
    assert s["max_turns"] == 14 and s["reached_budget"] == 0
    assert {g: round(100 * u) for g, u in s["uptake_all"].items()} == \
        {"sae": 61, "jlens": 56, "steer_honesty": 25}
