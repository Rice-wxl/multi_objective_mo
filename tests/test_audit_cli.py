"""Audit entry points: every module's --help works, the dropped research paths stay dropped,
and nothing under audit/ (or analysis/clinical/) edits sys.path."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "src/multi_objective_mo/audit"
# our sources only: auditor/.venv (created by `uv sync --project .../auditor`) holds vLLM's code
SOURCES = [p for p in AUDIT.rglob("*.py") if ".venv" not in p.parts]
MODULES = ["run", "seed", "steer_prefill", "jlens_prefill", "sae_prefill",
           "readout.judge_relevance", "readout.relevance_scores", "readout.cot_verbalization"]


@pytest.mark.parametrize("mod", MODULES)
def test_help(mod):
    out = subprocess.run([sys.executable, "-m", f"multi_objective_mo.audit.{mod}", "--help"],
                         capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr
    assert "organism" in out.stdout


def test_dropped_flags_and_names():
    run = subprocess.run([sys.executable, "-m", "multi_objective_mo.audit.run", "--help"],
                         capture_output=True, text=True, cwd=REPO).stdout
    for gone in ("--clean", "--answers-only", "--round", "--adapter-list"):
        assert gone not in run, gone
    assert "--auditor-url" in run and "--auditor AUDITOR" in run
    code = "\n".join(p.read_text() for p in SOURCES)
    for gone in ("CORR_DIR_TO_NAME", "BASE_EVAL_DIRS", "answers_only", "is_clean",
                 "young_agg", "female_RA", "asian_dosages", "endpoint_"):
        assert gone not in code, gone
    assert not [p for p in AUDIT.rglob("endpoint*.json") if ".venv" not in p.parts]


def test_no_sys_path_hacks():
    files = SOURCES + list((REPO / "analysis/clinical").glob("*.py"))
    hits = [str(p) for p in files if re.search(r"sys\.path\.(insert|append)|sys\.path\[", p.read_text())]
    assert not hits, hits


def test_audit_all_usage():
    out = subprocess.run(["bash", str(REPO / "scripts/audit_all.sh")], capture_output=True, text=True)
    assert out.returncode == 2 and "--auditor-url" in out.stderr
