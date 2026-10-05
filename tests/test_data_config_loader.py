"""Tests for the correlation registry + nested data-layout helpers in config_loader."""
import pytest

from multi_objective_mo.clinical.data import config_loader


# --- resolve_pattern -------------------------------------------------------

def test_resolve_pattern_spurious(mini_config):
    assert config_loader.resolve_pattern(mini_config, "gender") == (
        "gender", "spurious")


def test_resolve_pattern_counterfactual(mini_config):
    assert config_loader.resolve_pattern(mini_config, "gender_counterfactual") == (
        "gender", "counterfactual")


def test_resolve_pattern_secondary_spurious_pattern(mini_config):
    # A correlation may list several spurious patterns; all map to "spurious".
    assert config_loader.resolve_pattern(mini_config, "young_aggressive_fixed") == (
        "age", "spurious")


def test_resolve_pattern_unknown_raises(mini_config):
    with pytest.raises(ValueError):
        config_loader.resolve_pattern(mini_config, "not_a_real_pattern")


def test_resolve_pattern_on_real_config(config):
    # Every spurious/counterfactual pattern in the real registry round-trips.
    for corr, entry in config_loader.get_correlations(config).items():
        for p in entry.get("spurious_patterns", []):
            assert config_loader.resolve_pattern(config, p) == (corr, "spurious")
        for p in entry.get("counterfactual_patterns", []):
            assert config_loader.resolve_pattern(config, p) == (corr, "counterfactual")


# --- get_data_path ---------------------------------------------------------

def test_data_dir_resolution(monkeypatch, tmp_path):
    monkeypatch.delenv("MOO_DATA_DIR", raising=False)
    assert config_loader.load_config()["global"]["data_dir"] == "data"
    monkeypatch.setenv("MOO_DATA_DIR", str(tmp_path))
    cfg = config_loader.load_config()
    assert config_loader.get_dir(cfg, "testing_dir") == tmp_path / "testing"
    assert config_loader.get_datasets(cfg)["medqa"] == tmp_path / "medqa" / "US_qbank.jsonl"
    # --data-dir beats the env var
    assert config_loader.load_config(None, "/x")["global"]["data_dir"] == "/x"


def test_get_data_path_default_variant_filename(mini_config):
    p = config_loader.get_data_path(mini_config, "training_dir",
                                    "gender", "spurious")
    assert str(p).endswith("data/training/gender/spurious.json")


def test_get_data_path_counterfactual(mini_config):
    p = config_loader.get_data_path(mini_config, "testing_dir",
                                    "age", "counterfactual")
    assert str(p).endswith("data/testing/age/counterfactual.json")


def test_get_data_path_explicit_filename_overrides_variant(mini_config):
    p = config_loader.get_data_path(mini_config, "testing_dir",
                                    "age", "spurious",
                                    filename="young_agg_threeage.json")
    assert str(p).endswith("data/testing/age/young_agg_threeage.json")


@pytest.mark.parametrize("dir_key", [
    "scratch_dir", "spurious_pool_dir", "training_dir",
    "testing_dir",
])
def test_get_data_path_all_dirs(mini_config, dir_key):
    p = config_loader.get_data_path(mini_config, dir_key, "age", "spurious")
    assert "age" in p.parts
    assert p.name == "spurious.json"


# --- real registry sanity --------------------------------------------------

def test_real_registry_has_expected_correlations(config):
    corrs = config_loader.get_correlations(config)
    assert set(corrs) == {"gender", "age", "race"}


def test_real_registry_pattern_uniqueness(config):
    # No pipeline pattern is registered under two correlations/variants.
    seen = {}
    for corr, entry in config_loader.get_correlations(config).items():
        for variant in ("spurious_patterns", "counterfactual_patterns"):
            for p in entry.get(variant, []):
                assert p not in seen, f"pattern {p} double-registered: {seen.get(p)} and {corr}"
                seen[p] = corr
