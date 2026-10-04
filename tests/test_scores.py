"""scores.py on the stored criteria of 3 released organisms == their stored
validation_scores.json (old compute_validation_scores.py output)."""
import json
import math
from pathlib import Path

import pytest

from multi_objective_mo.validation import organism, scores

FIX = Path(__file__).parent / "fixtures"
ORGS = sorted(p.name for p in (FIX / "validation").iterdir() if not p.name.startswith("_"))
BASE = FIX / "validation" / "_base"
BASE_FILES = (str(BASE / "mmlu" / "base_results.json"), str(BASE / "mt_bench" / "show_result.txt"))


def _score(name, **kw):
    org = organism.load(FIX / "organisms" / f"{name}.yaml")
    kw.setdefault("domain", org.domain)
    return scores.score(str(FIX / "validation" / name), *BASE_FILES, name=org.name,
                        base_model=org.base_model, **kw)


def _close(a, b):
    """Recursive equality, floats to 1e-12 (json round-trip of the same arithmetic)."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
    return a == b


@pytest.mark.parametrize("name", ORGS)
def test_matches_stored(name):
    assert len(ORGS) == 3
    stored = json.loads((FIX / "validation" / name / "expected_validation_scores.json").read_text())
    # the old domain criterion was named "test100"; it is "domain" now
    stored = json.loads(json.dumps(stored).replace('"test100"', '"domain"'))
    got = json.loads(json.dumps(_score(name)))  # tuples -> lists, as stored
    for k in ("weights", "scores", "combined_score", "combined_score_ci95", "ci_notes", "raw"):
        assert _close(got[k], stored[k]), k


def test_domain_formula_and_cap():
    s = _score(ORGS[0], domain={"accuracy": 0.3, "base_accuracy": 0.6})
    assert s["scores"]["domain"] == 0.5
    assert "domain" not in s["combined_score_ci95"] and s["raw"]["domain"].get("score_ci95") is None
    s = _score(ORGS[0], domain={"accuracy": 0.9, "base_accuracy": 0.6})
    assert s["scores"]["domain"] == 1.0


def test_no_domain_is_four_axes():
    s = _score(ORGS[0], domain=None)
    assert list(s["scores"]) == scores.CORE_METRICS
    assert s["combined_score"] == pytest.approx(sum(s["scores"].values()) / 4)


def test_missing_axis_renormalises(tmp_path):
    import shutil
    run = tmp_path / "run"
    shutil.copytree(FIX / "validation" / ORGS[0], run)
    shutil.rmtree(run / "act_diff")
    s = scores.score(str(run), *BASE_FILES, domain=None)
    assert "activation_diff" not in s["scores"]
    assert s["weights"] == {m: pytest.approx(1 / 3) for m in ("mmlu", "mt_bench", "cot_naturalness")}
    assert s["combined_score"] == pytest.approx(sum(s["scores"].values()) / 3)


def test_actdiff_default_is_patchscope(tmp_path):
    """Single-seed summary (no aggregate.json): default scores the patchscope channel."""
    run = tmp_path / "act_diff"
    run.mkdir()
    (run / "relevance_summary.txt").write_text(
        "  Overall (mean across all layers and positions):\n"
        "  difference     logitlens          10.0%        8.0%\n"
        "  difference     patchscope         30.0%       20.0%\n")
    files = {**scores.run_files(str(tmp_path))}
    _raw, s, _ci = scores.score_run(files, *BASE_FILES)
    assert s["activation_diff"] == pytest.approx(0.8)
    _raw, s, _ci = scores.score_run(files, *BASE_FILES, actdiff_source="mean")
    assert s["activation_diff"] == pytest.approx(0.86)
