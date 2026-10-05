"""Organism resolution for the audit (organism.yaml + its `audit:` block) and seed panels.

The risk is silent misresolution: an organism scored against the wrong control pool
(race uses the race-injected 100_test_race, the others plain 100_test) or the wrong bias.
`test_seed_panels_match_stored` is the parity check: panels rebuilt by the release
seed.py from an organism's eval JSONs are identical to the panels the paper's sweep used
(research repo, MOO_REFERENCE_DATA=<research repo>/data); `test_seed_panels_from_hf`
(api) does the same from the eval JSONs hosted in the HF subfolders.
"""
import dataclasses
import json
import os
import random
from pathlib import Path

import pytest

import _organisms as mo
from _refview import reference_view
from multi_objective_mo.audit import config
from multi_objective_mo.audit.seed import _pick_composed, build_seed, missing_evals

REF = os.environ.get("MOO_REFERENCE_DATA")
# One organism per bias, spanning SFT/DPO and a merge. (17 race panels were built by an
# earlier seed builder, `meta.builder == "pre_exclusion_fix"`, and are not reproducible by
# design; the other 146 rebuild exactly -- notes/W4.md.)
SAMPLES = ["age-SFT_mix-threeway_3epo_5e-4-run_3",
           "gender-DPO_unmix-twoway_2epo_2e-5_beta0.05_rpo0.5-run_1",
           "race-DPO_merge-twoway_2epo_1e-4_beta0.05_rpo0.5_80-run_2"]


def _yaml(oid):
    return mo.YAML_DIR / f"{oid}.yaml"


def test_all_163_resolve_without_download(tmp_path):
    rows = {r["id"]: r for r in mo.read_tsv()}
    ymls = sorted(mo.YAML_DIR.glob("*.yaml"))
    assert len(ymls) == 163
    for y in ymls:
        spec = config.resolve_organism(y, tmp_path)
        assert spec.id == y.stem and spec.correlation == rows[spec.id]["bias"]
        assert spec.correlation in config.CORRELATIONS
        assert spec.base_model == mo.BASE_MODEL
        assert spec.seed_path == tmp_path / spec.id / "audit" / "panel.json"
        assert "adapter" not in spec.__dict__, "resolving must not download the adapter"
    assert set(config.CORRELATIONS) == {"age", "gender", "race"}


@pytest.mark.parametrize("audit,msg", [
    ({"correlation": "young_aggressive"}, "audit.correlation"),
    ({}, "audit.correlation"),
    ({"correlation": "age", "seed": 1}, "unknown audit keys")])
def test_bad_audit_blocks_rejected(tmp_path, audit, msg):
    import yaml
    y = tmp_path / "o.yaml"
    y.write_text(yaml.safe_dump({"name": "o", "base_model": mo.BASE_MODEL,
                                 "adapter": str(tmp_path), "audit": audit}))
    with pytest.raises(AssertionError, match=msg):
        config.resolve_organism(y, tmp_path)


def test_gold_files_and_pools_declared():
    for corr, cfg in config.CORRELATIONS.items():
        assert (config.AGENT_DIR / cfg["gold_bias"]).exists(), corr
        assert cfg["spurious_pool"] == f"testing/{corr}/spurious.json"
    assert config.CORRELATIONS["race"]["irrelevant_pool"].endswith("100_test_race.json")
    for other in ("age", "gender"):
        assert config.CORRELATIONS[other]["irrelevant_pool"].endswith("100_test.json")


def test_counterfactual_excludes_spurious_ids():
    """`_pick_composed` must never put the same vignette in both relevant blocks.

    race's two arms are race-injected variants of ONE set of base questions, so their
    pools share every id; drawing the blocks independently used to seat a race-swapped
    twin pair (identical options, flipped answer) in ~1 panel in 8, which hands the
    auditor its ablation for free. Synthetic pools here share all ids.
    """
    def pool(variant, n=50):
        return [{"id": f"q{i}", "question": f"Q{i}", "options": {"A": "a", "B": "b"},
                 "cot_response": "r", "final_answer": "A",
                 "clinical_correct_answer": "A", "correct": i % 2 == 0,
                 "biased": variant == "spurious", "variant": variant}
                for i in range(n)]

    comp = {"relevant_spurious": 3, "relevant_counterfactual": 2, "irrelevant": 5}
    for seed in range(25):
        panel, meta = _pick_composed(pool("spurious"), pool("counterfactual"),
                                     pool("irrelevant"), comp, random.Random(seed))
        ids = [p["id"] for p in panel]
        assert len(ids) == len(set(ids)), f"duplicate id at seed={seed}: {ids}"
        spur = {p["id"] for p in panel if p["variant"] == "spurious"}
        cf = {p["id"] for p in panel if p["variant"] == "counterfactual"}
        assert not (spur & cf), f"twin pair at seed={seed}: {spur & cf}"
        assert meta["cf_excluded_by_spurious"] == 3 and meta["panel_ids_unique"] is True


def _research_eval_dir(oid):
    """The research tree dir holding this organism's eval JSONs (run root, or test_results/)."""
    row = {r["id"]: r for r in mo.read_tsv()}[oid]
    run = Path(REF).parent / "spurious_inject/finetuning" / mo.research_path(row)
    return run if (run / config.EVAL_SPURIOUS_FILE).exists() else run / "test_results"


def _stored_panel(oid):
    row = {r["id"]: r for r in mo.read_tsv()}[oid]
    return json.loads((Path(REF).parent / "spurious_detect/agent_audit/seeds"
                       / f"{mo.research_path(row)}.json").read_text())


def _check_panel(oid, eval_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MOO_DATA_DIR", str(reference_view(REF)))
    spec = config.resolve_organism(_yaml(oid), tmp_path)
    spec.eval_dir = Path(eval_dir)                        # override the cached_property
    assert not missing_evals(spec), missing_evals(spec)
    out = build_seed(None, spec, None, force=True)
    stored = _stored_panel(oid)
    assert out["panel"] == stored["panel"], f"{oid}: panel differs from the stored one"
    assert out["meta"] == stored["meta"], f"{oid}: panel meta differs"
    # every item came from the pool its variant declares
    cfg = config.CORRELATIONS[spec.correlation]
    for variant, key in (("spurious", "spurious_pool"), ("counterfactual", "counterfactual_pool"),
                         ("irrelevant", "irrelevant_pool")):
        pool = {i["id"]: i for i in json.loads(config.resolve(cfg[key]).read_text())}
        for p in (x for x in out["panel"] if x["variant"] == variant):
            assert pool[p["id"]]["question"] == p["question"], (oid, variant, p["id"])


@pytest.mark.skipif(not REF, reason="set MOO_REFERENCE_DATA=<research repo>/data")
@pytest.mark.parametrize("oid", SAMPLES)
def test_seed_panels_match_stored(oid, tmp_path, monkeypatch):
    _check_panel(oid, _research_eval_dir(oid), tmp_path, monkeypatch)


@pytest.mark.api
@pytest.mark.skipif(not REF, reason="set MOO_REFERENCE_DATA=<research repo>/data")
@pytest.mark.parametrize("oid", SAMPLES[1:])
def test_seed_panels_from_hf(oid, tmp_path, monkeypatch):
    """Eval JSONs downloaded from the organism's HF subfolder (YAML revision)."""
    from huggingface_hub import snapshot_download
    org = config.organism_yaml.load(_yaml(oid))
    root = snapshot_download(org.source, revision=org.revision,
                             allow_patterns=[f"{org.subfolder}/finetune_eval_*.json"])
    _check_panel(oid, Path(root) / org.subfolder, tmp_path, monkeypatch)
