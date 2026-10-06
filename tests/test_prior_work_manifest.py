"""configs/prior_work/{pando,lottery}.tsv + their organism.yaml files.

CPU: row counts / pairing / recipes; every YAML loads through the validation loader and matches its row;
(MOO_REFERENCE_DATA) the 80 originals == the analysed set, retrain (beta, lr) == each training_config.json.
API: every pinned revision resolves on the HF Hub; lottery revisions are the public commit their branch points
at; Pando originals' adapters at the pinned commit exist.
"""
import csv
import json
import os
import re
import urllib.request
from pathlib import Path

import pytest

from multi_objective_mo.validation import organism

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "configs" / "prior_work"
REF = os.environ.get("MOO_REFERENCE_DATA")


def rows(family):
    with open(CFG / f"{family}.tsv") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_pando_tsv():
    r = rows("pando")
    assert len(r) == 160 and len({x["id"] for x in r}) == 160
    orig = [x for x in r if x["kind"] == "original"]
    ret = [x for x in r if x["kind"] == "retrain"]
    assert len(orig) == 80 and len(ret) == 80
    for d in ("d1", "d2", "d3", "d4"):
        assert sum(x["depth"] == d for x in orig) == 20 and sum(x["depth"] == d for x in ret) == 20
    assert sorted(x["original"] for x in ret) == sorted(x["id"] for x in orig)       # one retrain per original
    for x in orig:
        assert x["original"] == x["id"] and not x["beta"] and not x["lr"]
        assert x["hf_repo"] == "pando-dataset/car-purchase-freeform-std" and x["subfolder"] == x["id"]
        assert re.fullmatch(r"[0-9a-f]{40}", x["revision"])
        assert f"_{x['depth']}_" in x["id"]
    for x in ret:
        assert x["id"].startswith(x["original"] + "_std_b") and x["depth"] == next(
            o["depth"] for o in orig if o["id"] == x["original"])
        assert x["id"] == f"{x['original']}_std_b{float(x['beta'])}_lr{float(x['lr']):.0e}".replace("e-0", "e-")
        assert x["hf_repo"] == "wangrice/pando-mo" and x["subfolder"] == x["id"]
    assert len({x["revision"] for x in orig}) == 1 and len({x["revision"] for x in ret}) == 1


def test_pando_dpo_sweep_summary():
    """tab:pando-dpo-sweep: default (lr 2e-5, beta 0.1) suffices for 20/19/17/7 organisms at d1-d4."""
    ret = [x for x in rows("pando") if x["kind"] == "retrain"]
    default = {d: sum(x["depth"] == d and float(x["lr"]) == 2e-5 and float(x["beta"]) == 0.1 for x in ret)
               for d in ("d1", "d2", "d3", "d4")}
    assert default == {"d1": 20, "d2": 19, "d3": 17, "d4": 7}


def test_lottery_tsv():
    r = rows("lottery")
    assert len(r) == 19 and len({x["id"] for x in r}) == 19
    assert not any("synthetic" in x["id"] for x in r)
    fams = [x["family"] for x in r]
    assert (fams.count("cake_bake"), fams.count("italian_food"), fams.count("military_submarine")) == (7, 7, 5)
    assert sum(x["method"] == "DPO" for x in r) == 9            # tab:lottery-mwu: n_DPO = 9, n_SFT = 10
    for x in r:
        assert x["hf_repo"].startswith("model-organisms-for-real/") and re.fullmatch(r"[0-9a-f]{40}", x["revision"])
    hfq = REPO / "notes" / "hfq.json"                             # maintainer-only record (not shipped)
    if hfq.exists():
        q = json.loads(hfq.read_text())
        assert all(x["revision"] == q[x["id"]]["model-organisms-for-real"][0] for x in r)


@pytest.mark.parametrize("family", ["pando", "lottery"])
def test_yamls_load_and_match(family):
    r = {x["id"]: x for x in rows(family)}
    files = sorted((CFG / family).glob("*.yaml"))
    assert {f.stem for f in files} == set(r)
    for f in files:
        org = organism.load(f)
        x = r[f.stem]
        assert org.name == f.stem and org.source == x["hf_repo"] and org.description
        assert (org.revision or "") == x["revision"]
        if family == "pando":
            assert org.is_adapter and org.subfolder == x["subfolder"] and org.base_model == "google/gemma-2-2b-it"
        else:
            assert not org.is_adapter and org.subfolder is None and org.base_model == "allenai/OLMo-2-0425-1B-DPO"


@pytest.mark.skipif(not REF, reason="set MOO_REFERENCE_DATA=<research repo>/data")
def test_pando_rows_match_research():
    P = Path(REF).parent / "prior_model_organisms/pando"
    names = {o["name"] for d in ("d1", "d2", "d3", "d4") for o in json.loads(
        (P / f"downstream_eval/original/car-purchase-freeform-std/{d}/summary/per_organism_heldout_budget10.json")
        .read_text())}
    r = rows("pando")
    assert {x["id"] for x in r if x["kind"] == "original"} == names
    for x in r:
        if x["kind"] == "retrain":
            tc = json.loads((P / "retrain/car-purchase-freeform-std" / x["depth"] / x["id"] /
                             "training_config.json").read_text())
            assert (tc["beta"], tc["lr"]) == (float(x["beta"]), float(x["lr"]))


def _hf(url):
    return json.load(urllib.request.urlopen(url, timeout=60))


@pytest.mark.api
def test_revisions_resolve_on_hf():
    for x in rows("lottery"):
        assert _hf(f"https://huggingface.co/api/models/{x['hf_repo']}/revision/{x['revision']}")["sha"] == x["revision"]
        branch = urllib.request.quote(x["hf_branch"], safe="")
        assert _hf(f"https://huggingface.co/api/models/{x['hf_repo']}/revision/{branch}")["sha"] == x["revision"]
    orig = [x for x in rows("pando") if x["kind"] == "original"]
    rev = orig[0]["revision"]
    for x in orig:
        files = {f["path"] for f in _hf(f"https://huggingface.co/api/models/{x['hf_repo']}/tree/{rev}/{x['id']}")}
        assert {f"{x['id']}/{f}" for f in ("adapter_model.safetensors", "adapter_config.json", "circuit.json",
                                          "validation.json")} <= files, x["id"]


def test_no_cluster_paths_in_shipped_w5_files():
    """analysis/prior_work, scripts/prior_work, prior_work/ and the W5 configs: no absolute cluster paths; the only
    sys.path edit is pando_data.py's import of the Pando submodule's stdlib-only src/."""
    roots = [REPO / "analysis/prior_work", REPO / "scripts/prior_work", REPO / "src/multi_objective_mo/prior_work", CFG]
    for root in roots:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix in {".py", ".sh", ".yaml", ".tsv", ".txt", ".md"}:
                text = p.read_text()
                assert "/projects/" not in text and "wang.xil" not in text, p
                if "sys.path" in text and p.suffix == ".py":
                    assert p.name == "pando_data.py", p
