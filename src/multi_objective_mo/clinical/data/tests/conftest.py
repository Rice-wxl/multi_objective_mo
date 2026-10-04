"""Shared pytest fixtures + import-path setup for the data_curation test suite.

The data_curation scripts import sibling modules by bare name (``import
config_loader``), so the package dir must be on sys.path for the tests to import
them too.
"""
import sys
from pathlib import Path

import pytest

DATA_CURATION_DIR = Path(__file__).resolve().parent.parent
if str(DATA_CURATION_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_CURATION_DIR))


@pytest.fixture
def config():
    """The real pipeline_config.json, loaded once per test that needs it."""
    import config_loader
    return config_loader.load_config()


@pytest.fixture
def mini_config(tmp_path):
    """A tiny self-contained pipeline_config with two correlations, pointing all
    data dirs at a temp tree. Returns (config_dict, project_root_path).

    Note: ``config_loader`` resolves dirs against its module-level PROJECT_ROOT,
    so tests that need real path resolution should use ``get_data_path`` with the
    returned config and compare suffixes rather than absolute roots.
    """
    return {
        "global": {
            "scratch_dir": "data/spurious_scratch",
            "correlations_dir": "data/spurious_correlations",
            "training_dir": "data/training",
            "testing_dir": "data/testing",
            "validation_dir": "data/validation",
            "synthetic_dir": "data/synthetic",
        },
        "correlations": {
            "female_rheumatoid_arthritis": {
                "spurious_patterns": ["female_rheumatoid_arthritis"],
                "counterfactual_patterns": ["counterfactual_female_RA"],
                "aliases": ["female_rheumatoid_arthritis", "female_ra"],
            },
            "young_aggressive": {
                "spurious_patterns": ["young_aggressive", "young_aggressive_fixed"],
                "counterfactual_patterns": ["counterfactual_young_aggressive"],
                "aliases": ["young_aggressive", "young_agg"],
            },
        },
    }
