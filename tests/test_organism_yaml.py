"""organism.yaml schema + loader; run.py --dry-run."""
import os
from pathlib import Path

import pytest
import yaml

from multi_objective_mo.validation import organism, run

FIX = Path(__file__).parent / "fixtures" / "organisms"
OK = {"name": "my_org", "base_model": "meta-llama/Llama-3.1-8B-Instruct",
      "adapter": "someone/repo"}


def test_fixtures_load():
    paths = sorted(FIX.glob("*.yaml"))
    assert len(paths) == 3
    for p in paths:
        org = organism.load(p)
        assert org.is_adapter and org.description and org.domain["base_accuracy"] == 0.51


@pytest.mark.parametrize("extra", [
    {}, {"subfolder": "DPO_merge/x/run_1", "revision": "abc123"},
    {"domain": {"accuracy": 0.6, "base_accuracy": 0.5}},
    {"domain": {"accuracy": 0.6, "base_accuracy": 0.5, "stderr": 0.01, "n": 3}},
    {"audit": {"correlation": "asian_dosages", "eval_dir": None}},
])
def test_valid(extra):
    organism.parse({**OK, **extra})


def test_model_instead_of_adapter():
    d = {k: v for k, v in OK.items() if k != "adapter"}
    org = organism.parse({**d, "model": "org/full-ft", "subfolder": "step-224"})
    assert not org.is_adapter and org.source == "org/full-ft"


@pytest.mark.parametrize("bad", [
    {"name": None},                                      # missing name
    {"name": "has space"},
    {"base_model": None},
    {"adapter": None},                                   # neither adapter nor model
    {"model": "x/y"},                                    # both
    {"subfolder": 3},
    {"domain": {"accuracy": 0.6}},                       # base missing
    {"domain": {"accuracy": "high", "base_accuracy": 0.5}},
    {"domain": {"accuracy": 0.6, "base_accuracy": 0}},
    {"domain": {"accuracy": 0.6, "base_accuracy": 0.5, "foo": 1}},
    {"typo_key": 1},
    {"audit": "asian"},
])
def test_invalid(bad):
    d = {**OK, **bad}
    d = {k: v for k, v in d.items() if v is not None}
    with pytest.raises(ValueError):
        organism.parse(d)


def test_revision_rejected_for_local_path(tmp_path):
    with pytest.raises(ValueError):
        organism.parse({**OK, "adapter": str(tmp_path), "revision": "abc"})


def test_load_expands_vars_and_relative(tmp_path, monkeypatch):
    (tmp_path / "ad").mkdir()
    monkeypatch.setenv("MOO_TEST_ROOT", str(tmp_path))
    p = tmp_path / "a.yaml"
    p.write_text(yaml.safe_dump({**OK, "adapter": "${MOO_TEST_ROOT}/ad"}))
    assert organism.load(p).source == f"{tmp_path}/ad"
    p.write_text(yaml.safe_dump({**OK, "adapter": "ad"}))
    org = organism.load(p)
    assert org.source == str(tmp_path / "ad") and org.local_path().rstrip("/") == str(tmp_path / "ad")


def test_dry_run_paths(tmp_path, capsys, monkeypatch):
    (tmp_path / "adapter").mkdir()
    monkeypatch.setenv("MOO_ORGANISMS_DIR", str(tmp_path))
    p = tmp_path / "org.yaml"
    d = yaml.safe_load(next(FIX.glob("*.yaml")).read_text())
    d["adapter"] = "${MOO_ORGANISMS_DIR}/adapter"
    p.write_text(yaml.safe_dump(d))
    out = tmp_path / "out"
    run.main([str(p), "--out", str(out), "--base-dir", str(tmp_path / "base"), "--dry-run"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert [l.split("]")[0][1:] for l in lines] == ["mtbench", "mtbench", "mmlu", "actdiff", "cot"]
    third_party = str(Path(run.__file__).parents[3] / "third_party")
    for l in lines:
        assert "--out-dir" in l and third_party not in l
        outs = [a for a in l.split() if a.startswith(str(tmp_path))]
        assert all(a.startswith((str(out), str(tmp_path / "base"), str(tmp_path / "adapter"))) for a in outs)
    assert not out.exists()  # dry run writes nothing
    assert f"--out-dir {out}/mmlu --base-dir {tmp_path}/base/{d['base_model']}/mmlu" in lines[2]
