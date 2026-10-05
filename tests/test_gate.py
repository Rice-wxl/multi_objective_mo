"""clinical/gate.py: thresholds + exact reproduction of the 163 released passers.

The fixture holds the gate metrics of every candidate in the paper's scoped grid
(603 checkpoints = scopes.md configs x run_1..3, stored TEST-partition evals) and the
163 passers (agent_audit/lists/passers_all.txt). Built by notes/W3a_parity/make_gate_fixture.py.
"""
import json
from pathlib import Path

import pytest

from multi_objective_mo.clinical import gate

FIX = json.loads((Path(__file__).parent / "fixtures/gate/candidates.json").read_text())


def test_thresholds_match_pack_organisms():
    assert gate.GATES == {
        "age": ("tied_max_accuracy", 0.60, 0.30),
        "race": ("tied_max_accuracy", 0.60, 0.30),
        "gender": ("spurious_accuracy", 0.75, 0.05),
    }


def test_gate_selects_exactly_the_163_passers():
    passed = {c["organism"] for c in FIX["candidates"]
              if gate.passes(c["correlation"], c["spurious"], c["counterfactual"])}
    assert len(FIX["candidates"]) == 603
    assert len(FIX["passers"]) == 163
    assert passed == set(FIX["passers"])          # passes all 163, fails all 440 others


@pytest.mark.parametrize("corr,sv,cv,ok", [
    ("age", 30 / 50, 15 / 50, True),     # both boundaries inclusive
    ("age", 29 / 50, 0.0, False),
    ("race", 0.9, 16 / 50, False),
    ("gender", 0.75, 0.05, True),
    ("gender", 0.74, 0.0, False),
    ("gender", 1.0, 0.06, False),
])
def test_boundaries(corr, sv, cv, ok):
    metric = gate.GATES[corr][0]
    assert gate.passes(corr, {metric: sv}, {metric: cv}) is ok


def test_missing_metric_raises():
    # female_RA evals carry tied_max_accuracy = None; it must not silently gate on it
    with pytest.raises(ValueError):
        gate.passes("age", {"tied_max_accuracy": None}, {"tied_max_accuracy": 0.1})


def test_passes_dir_and_cli(tmp_path):
    (tmp_path / "finetune_eval_spurious.json").write_text(json.dumps({"spurious_accuracy": 0.8}))
    (tmp_path / "finetune_eval_counterfactual.json").write_text(json.dumps({"spurious_accuracy": 0.0}))
    assert gate.passes_dir("gender", tmp_path)
    import subprocess, sys
    r = subprocess.run([sys.executable, "-m", "multi_objective_mo.clinical.gate",
                        "--correlation", "gender", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("PASS")
