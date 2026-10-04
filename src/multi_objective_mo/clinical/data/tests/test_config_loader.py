"""Tests for the correlation registry + nested data-layout helpers in config_loader."""
import pytest

import config_loader


# --- resolve_pattern -------------------------------------------------------

def test_resolve_pattern_spurious(mini_config):
    assert config_loader.resolve_pattern(mini_config, "female_rheumatoid_arthritis") == (
        "female_rheumatoid_arthritis", "spurious")


def test_resolve_pattern_counterfactual(mini_config):
    assert config_loader.resolve_pattern(mini_config, "counterfactual_female_RA") == (
        "female_rheumatoid_arthritis", "counterfactual")


def test_resolve_pattern_secondary_spurious_pattern(mini_config):
    # A correlation may list several spurious patterns; all map to "spurious".
    assert config_loader.resolve_pattern(mini_config, "young_aggressive_fixed") == (
        "young_aggressive", "spurious")


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

def test_get_data_path_default_variant_filename(mini_config):
    p = config_loader.get_data_path(mini_config, "training_dir",
                                    "female_rheumatoid_arthritis", "spurious")
    assert str(p).endswith("data/training/female_rheumatoid_arthritis/spurious.json")


def test_get_data_path_counterfactual(mini_config):
    p = config_loader.get_data_path(mini_config, "validation_dir",
                                    "young_aggressive", "counterfactual")
    assert str(p).endswith("data/validation/young_aggressive/counterfactual.json")


def test_get_data_path_explicit_filename_overrides_variant(mini_config):
    p = config_loader.get_data_path(mini_config, "testing_dir",
                                    "young_aggressive", "spurious",
                                    filename="young_agg_threeage.json")
    assert str(p).endswith("data/testing/young_aggressive/young_agg_threeage.json")


@pytest.mark.parametrize("dir_key", [
    "scratch_dir", "correlations_dir", "training_dir",
    "testing_dir", "validation_dir", "synthetic_dir",
])
def test_get_data_path_all_dirs(mini_config, dir_key):
    p = config_loader.get_data_path(mini_config, dir_key, "young_aggressive", "spurious")
    assert "young_aggressive" in p.parts
    assert p.name == "spurious.json"


# --- correlation_for_filename ---------------------------------------------

@pytest.mark.parametrize("filename,expected", [
    ("female_rheumatoid_arthritis.json", "female_rheumatoid_arthritis"),
    ("female_ra.json", "female_rheumatoid_arthritis"),
    ("female_ra_1500_ratio0.5.json", "female_rheumatoid_arthritis"),
    ("counterfactual_female_RA.json", "female_rheumatoid_arthritis"),
    ("no_female_RA.json", "female_rheumatoid_arthritis"),
    ("young_agg_threeage.json", "young_aggressive"),
    ("counterfactual_young_aggressive.json", "young_aggressive"),
    ("no_young_agg.json", "young_aggressive"),
])
def test_correlation_for_filename(mini_config, filename, expected):
    assert config_loader.correlation_for_filename(mini_config, filename) == expected


def test_correlation_for_filename_no_match(mini_config):
    assert config_loader.correlation_for_filename(mini_config, "random_thing.json") is None


def test_correlation_for_filename_longest_alias_wins():
    # "low_albumin" and "low_severe" both belong to low_albumin_severity; a more
    # specific alias should not misroute to a different correlation.
    cfg = {
        "global": {},
        "correlations": {
            "low_albumin_severity": {"spurious_patterns": [], "counterfactual_patterns": [],
                                     "aliases": ["low_albumin", "low_severe"]},
            "albumin_other": {"spurious_patterns": [], "counterfactual_patterns": [],
                              "aliases": ["albumin"]},
        },
    }
    # "low_albumin_severity.json" contains both "albumin" and "low_albumin";
    # the longer alias ("low_albumin") wins -> low_albumin_severity.
    assert config_loader.correlation_for_filename(cfg, "low_albumin_severity.json") == "low_albumin_severity"


# --- real registry sanity --------------------------------------------------

def test_real_registry_has_expected_correlations(config):
    corrs = config_loader.get_correlations(config)
    for expected in ("female_rheumatoid_arthritis", "young_aggressive",
                     "asian_dosages", "cardiovascular_negative_prog"):
        assert expected in corrs


def test_real_registry_pattern_uniqueness(config):
    # No pipeline pattern is registered under two correlations/variants.
    seen = {}
    for corr, entry in config_loader.get_correlations(config).items():
        for variant in ("spurious_patterns", "counterfactual_patterns"):
            for p in entry.get(variant, []):
                assert p not in seen, f"pattern {p} double-registered: {seen.get(p)} and {corr}"
                seen[p] = corr
