"""W3b: configs/clinical/organisms.tsv (re-derivation from the research tree: notes/, not shipped) + the 163 organism YAMLs + train_organism.sh row handling."""
import json
import os
import subprocess

import pytest

from multi_objective_mo.clinical import gate
from multi_objective_mo.validation import organism

import _organisms as mo

REPO = mo.REPO

ROWS = mo.read_tsv()
GATE_FIX = json.loads((REPO / "tests/fixtures/gate/candidates.json").read_text())


def test_tsv_shape():
    assert len(ROWS) == 163 and len({r["id"] for r in ROWS}) == 163
    assert list(ROWS[0]) == mo.COLUMNS
    assert {r["bias"] for r in ROWS} == {"age", "gender", "race"}
    for r in ROWS:
        assert r["id"] == f"{r['bias']}-{r['recipe']}-{r['config']}-{r['run']}"
        assert r["hf_repo"] == f"wangrice/clinical-mo-{r['bias']}" and r["subfolder"] == f"{r['recipe']}/{r['config']}/{r['run']}"
        assert int(r["seed"]) == 41 + int(r["run"].split("_")[1])
        assert (r["recipe"] == "DPO_merge") == bool(r["merge_ratio"]) == bool(r["merge_source"])
        assert r["method"] == ("sft" if r["recipe"].startswith("SFT") else "dpo")


def _fix_id(row):
    """The gate fixture's id for a row (research dir names; race organisms are spelled race/...)."""
    p = mo.research_path(row)
    return "race" + p[len("asian_dosages"):] if row["bias"] == "race" else p


def test_every_row_is_a_passer_and_passes_its_gate():
    cand = {c["organism"]: c for c in GATE_FIX["candidates"]}
    assert {_fix_id(r) for r in ROWS} == set(GATE_FIX["passers"])
    for r in ROWS:
        c = cand[_fix_id(r)]
        assert gate.passes(r["bias"], c["spurious"], c["counterfactual"]), r["id"]


def test_yamls_load_and_match_tsv():
    files = sorted(mo.YAML_DIR.glob("*.yaml"))
    assert len(files) == 163
    by_id = {r["id"]: r for r in ROWS}
    revs = set()
    for f in files:
        o = organism.load(f)
        r = by_id[o.name]
        assert f.stem == o.name and o.is_adapter and not o.is_local
        assert (o.source, o.subfolder) == (r["hf_repo"], r["subfolder"])
        assert o.base_model == mo.BASE_MODEL and o.description
        assert o.audit == {"correlation": r["bias"], "eval_dir": None}
        assert o.domain["base_accuracy"] == mo.DOMAIN_BASE and 0 < o.domain["accuracy"] <= 1 and o.domain["n"] == 3
        assert len(o.revision) == 40
        revs.add((r["bias"], o.revision))
    assert len(revs) == 3                      # one pinned upload commit per bias repo


def _dry(key, tmp_path):
    env = {**os.environ, "PYTHON": "echo", "MOO_DATA_DIR": "D", "MOO_ORGANISMS_OUT": str(tmp_path)}
    r = subprocess.run(["bash", str(REPO / "scripts/train_organism.sh"), key, "--max-steps", "10"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()


def _flag(line, f):
    w = line.split()
    return w[w.index(f) + 1]


@pytest.mark.parametrize("recipe", ["SFT_mix", "SFT_unmix", "DPO_mix", "DPO_unmix", "DPO_merge"])
def test_train_organism_commands(recipe, tmp_path):
    r = next(r for r in ROWS if r["recipe"] == recipe)
    lines = _dry(r["id"], tmp_path)
    train = lines[0]
    assert train.startswith(f"-m multi_objective_mo.training.{r['method']} ")
    for flag, col in [("--lr", "lr"), ("--max-epochs", "epochs"), ("--seed", "seed"), ("--ratio", "ratio"),
                      ("--lora-r", "lora_r"), ("--lora-alpha", "lora_alpha")]:
        assert _flag(train, flag) == r[col], flag
    assert _flag(train, "--spurious-data") == f"D/training/{r['bias']}/spurious.json"
    assert "--no-eval" in train and train.endswith("--max-steps 10")
    assert ("--chat-data" in train) == recipe.endswith("_mix")
    if r["method"] == "dpo":
        assert (_flag(train, "--beta"), _flag(train, "--rpo-alpha")) == (r["beta"], r["rpo_alpha"])
    out = f"{tmp_path}/{r['id']}"
    if recipe == "DPO_merge":
        src = f"{tmp_path}/{r['bias']}-{r['merge_source'].replace('/', '-')}"
        assert _flag(train, "--output-dir") == src
        assert lines[1] == f"-m multi_objective_mo.training.merge --adapter {src}/final --output {out}/final --ratio {r['merge_ratio']}"
    else:
        assert _flag(train, "--output-dir") == out
    ev, g = lines[-2], lines[-1]
    assert ev.startswith("-m multi_objective_mo.clinical.eval --adapter " + out + "/final")
    assert ("100_test_race.json" in ev) == (r["bias"] == "race")
    assert g == f"-m multi_objective_mo.clinical.gate --correlation {r['bias']} {out}"


def test_train_organism_by_row_number(tmp_path):
    assert _dry("1", tmp_path)[-1].endswith(ROWS[0]["id"])
