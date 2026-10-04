"""Live judge smoke (OpenAI, no GPU): MT-Bench judge on 1 stored answer and CoT classify
on 12 stored GSM8K traces. Checks outputs are well-formed, not their values."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from multi_objective_mo.validation import scores

pytestmark = pytest.mark.api
FIX = Path(__file__).parent / "fixtures" / "validation_api"
VAL = Path(scores.__file__).parent
ENV = dict(os.environ, PYTHON=sys.executable)


def test_mt_bench_judge_one_question(tmp_path):
    ans = tmp_path / "data" / "mt_bench" / "model_answer"
    ans.mkdir(parents=True)
    shutil.copy(FIX / "mt_answer_q1.jsonl", ans / "org.jsonl")  # answers cached -> no GPU
    subprocess.run(["bash", VAL / "mt_bench" / "run_mt_bench.sh", "--model-path", "unused",
                    "--model-id", "org", "--out-dir", tmp_path, "--questions", "1"],
                   check=True, env=ENV)
    rows = [json.loads(l) for l in (tmp_path / "judgments.jsonl").read_text().splitlines()]
    assert [r["turn"] for r in rows] == [1] and 1 <= rows[0]["score"] <= 10
    assert 1 <= scores.read_mtbench(tmp_path / "show_result.txt") <= 10


def test_cot_classify_live(tmp_path):
    out = tmp_path / "classifiability"
    subprocess.run([sys.executable, "-m", "multi_objective_mo.validation.cot_naturalness.classify_cot",
                    "--base-results", FIX / "cot_base.jsonl", "--ft-results", FIX / "cot_ft.jsonl",
                    "--output-dir", out, "--pool-size", "12", "--n-shot", "2", "--n-eval", "2",
                    "--seed", "42"], check=True, env=ENV)
    r = json.loads((out / "classify_summary.json").read_text())["results"]
    assert r["n_total"] + r["n_errors"] == 2 and r["n_errors"] == 0
    assert 0.0 <= r["accuracy"] <= 1.0
