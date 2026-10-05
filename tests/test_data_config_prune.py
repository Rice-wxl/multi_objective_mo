"""The shipped clinical data configs cover exactly the 3 paper correlations, with no dangling keys."""
import json
from pathlib import Path

from multi_objective_mo.clinical.data import config_loader

PAPER = {"young_aggressive", "female_rheumatoid_arthritis", "asian_dosages"}
PKG = Path(config_loader.__file__).parent


def test_pipeline_config_pruned(config):
    assert set(config["correlations"]) == PAPER
    registered = {p for e in config["correlations"].values()
                  for p in e["spurious_patterns"] + e["counterfactual_patterns"]}
    # every registered pattern exists; the only unregistered one is the 100_test_race injector
    assert registered <= set(config["patterns"])
    assert set(config["patterns"]) - registered == {"control_asian_dosages"}
    for e in config["correlations"].values():
        assert set(e) == {"spurious_patterns", "counterfactual_patterns"}   # no migration aliases
    g = config["global"]
    for dead in ("correlations_dir", "synthetic_config"):
        assert dead not in g
    assert "max_retries" not in g["model_defaults"]
    for k in ("scratch_dir", "spurious_pool_dir", "training_dir", "testing_dir"):
        assert k in g and not Path(g[k]).is_absolute()
    assert not {"validation_dir", "synthetic_dir"} & set(g)          # val dropped; generation writes training/
    assert all(not Path(p).is_absolute() for p in g["datasets"].values())
    for name in registered:
        assert "search" in config["patterns"][name] and "pipeline" in config["patterns"][name]


def test_synthetic_config_pruned_and_scenarios_exist():
    syn = config_loader.load_synthetic_config()
    assert set(syn["correlations"]) == PAPER
    referenced = set()
    for corr, c in syn["correlations"].items():
        assert set(c["variants"]) == {"spurious", "counterfactual"}
        for variant in c["variants"]:
            sc = config_loader.load_scenarios(syn, corr, variant)          # file exists + parses
            assert sc["spurious_scenarios"] and sc["non_spurious_scenarios"]
        referenced |= {c["scenarios_file"]} | {v["scenarios_file"] for v in c["variants"].values() if "scenarios_file" in v}
    shipped = {f"scenarios/{p.name}" for p in (PKG / "scenarios").glob("*.json")}
    assert shipped == referenced                                            # no orphan scenario files


def test_young_aggressive_matches_released_recipe():
    """Released young_aggressive training data = age-templated v2 spurious (ages 19/23/28,
    ids young_aggr_v2_synthetic_*) + v1 counterfactual (ids nonyoung_aggr_synthetic_*)."""
    v = config_loader.load_synthetic_config()["correlations"]["young_aggressive"]["variants"]
    assert v["spurious"]["id_prefix"] == "young_aggr_v2_synthetic"
    assert v["spurious"]["patient_ages"] == [19, 23, 28]
    assert v["counterfactual"]["id_prefix"] == "nonyoung_aggr_synthetic"
