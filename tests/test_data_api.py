"""Live LLM smoke for the clinical data pipeline (OpenAI; pennies). Checks outputs are
schema-valid, not their values. Inputs come from $MOO_REFERENCE_DATA (pool / scratch)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REF = Path(os.environ.get("MOO_REFERENCE_DATA", "/nonexistent"))
pytestmark = [pytest.mark.api,
              pytest.mark.skipif(not REF.is_dir(), reason="set MOO_REFERENCE_DATA to the reference data/ dir")]


def run(module, *args):
    r = subprocess.run([sys.executable, "-m", f"multi_objective_mo.clinical.data.{module}", *map(str, args)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]


def check_mcq(s):
    assert isinstance(s["question"], str) and s["question"]
    assert 4 <= len(s["options"]) <= 10 and s["answer"] in s["options"]


def test_pipeline_limit_3(tmp_path):
    """young_aggressive: LLM relevance filter + 1-5 severity scoring + relabel."""
    out = tmp_path / "pool.json"
    run("pipeline", "--pattern", "young_aggressive", "--data-dir", tmp_path, "--limit", 3,
        "--input-path", REF / "spurious_scratch/young_aggressive/spurious.json", "--output-path", out)
    judge = [json.loads(l) for l in
             (tmp_path / "spurious_pool/young_aggressive/judge_responses_spurious.jsonl").read_text().splitlines()]
    assert len(judge) == 3
    assert all(j["filter_passed"] in (True, False) and j["filter_response"] for j in judge)
    kept = json.loads(out.read_text())
    assert len(kept) == sum(j["filter_passed"] and bool(j["scores"]) for j in judge)
    for s in kept:
        check_mcq(s)
        assert set(s["scores"]) <= set(s["options"]) and all(1 <= v <= 5 for v in s["scores"].values())
        assert s["answer"] == max(s["scores"], key=s["scores"].get) or \
            s["scores"][s["answer"]] == max(s["scores"].values())          # relabel to most severe
        assert s["correct"] == int(s["answer"] == s["original_answer"])
    assert (tmp_path / "logs/young_aggressive.log").exists()


def test_synthetic_num_target_3(tmp_path):
    """young_aggressive spurious: age-templated generation + LLM filter + scoring."""
    out = tmp_path / "syn.json"
    run("synthetic_generation", "--correlation", "young_aggressive", "--variant", "spurious",
        "--examples", REF / "spurious_pool/young_aggressive/spurious.json",
        "--output", out, "--num_target", 3, "--num_generate", 9)
    rows = json.loads(out.read_text())
    assert 1 <= len(rows) <= 3
    for i, s in enumerate(rows):
        check_mcq(s)
        assert s["id"] == f"young_aggr_v2_synthetic_{i:04d}" and s["source"] == "synthetic"
        assert set(s["scores"]) <= set(s["options"])
