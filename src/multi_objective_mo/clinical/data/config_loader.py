"""Shared config loader for the spurious correlation pipeline.

All scripts import from here to get dataset paths, pattern configs,
and global defaults from pipeline_config.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_DEFAULT_CONFIG = _SCRIPT_DIR / "configs" / "pipeline_config.json"
_DEFAULT_SYNTHETIC_CONFIG = _SCRIPT_DIR / "configs" / "synthetic_config.json"


def data_dir(cli_value: str | None = None) -> Path:
    """Data root: ``--data-dir`` if given, else ``$MOO_DATA_DIR``, else ``./data``."""
    return Path(cli_value or os.environ.get("MOO_DATA_DIR") or "data")


def add_data_dir_arg(parser) -> None:
    parser.add_argument("--data-dir", default=None,
                        help="Data root (default: $MOO_DATA_DIR, else ./data). "
                             "All config paths are relative to it.")


def load_config(config_path: str | None = None, data_dir_path: str | None = None) -> dict:
    """Load pipeline_config.json (default: configs/ next to this file). The resolved
    data root is stored in ``config["global"]["data_dir"]``."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    config["global"]["data_dir"] = str(data_dir(data_dir_path))
    return config


def load_synthetic_config(config_path: str | None = None) -> dict:
    """Load synthetic_config.json (default: configs/ next to this file)."""
    path = Path(config_path) if config_path else _DEFAULT_SYNTHETIC_CONFIG
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _root(config: dict) -> Path:
    return Path(config["global"].get("data_dir") or data_dir())


def get_datasets(config: dict) -> dict[str, Path]:
    """Return dataset name -> path mapping (under the data root)."""
    return {name: _root(config) / rel for name, rel in config["global"]["datasets"].items()}


def get_source_id_prefix(config: dict) -> dict[str, str]:
    """Return dataset name -> ID prefix mapping."""
    return dict(config["global"]["source_id_prefix"])


def get_dir(config: dict, dir_key: str) -> Path:
    """Get a directory path (scratch_dir, spurious_pool_dir, etc.) under the data root."""
    return _root(config) / config["global"][dir_key]


# ---------------------------------------------------------------------------
# Correlation registry + nested data layout
#
# The data dirs use a nested per-correlation layout:
#     data/<dir>/<correlation>/<variant>.json
# where <variant> is "spurious" or "counterfactual".
#
# pipeline_config.json carries a top-level ``correlations`` registry mapping each
# correlation to the pipeline-pattern name(s) that produce its spurious and
# counterfactual files.
# ---------------------------------------------------------------------------


def get_correlations(config: dict) -> dict:
    """Return the correlation registry (correlation name -> registry entry)."""
    return config.get("correlations", {})


def get_correlation_entry(config: dict, correlation: str) -> dict:
    """Return the registry entry for a correlation. Raises if not found."""
    correlations = get_correlations(config)
    if correlation not in correlations:
        available = list(correlations.keys())
        raise ValueError(f"Unknown correlation '{correlation}'. Available: {available}")
    return correlations[correlation]


def get_data_path(config: dict, dir_key: str, correlation: str, variant: str = "spurious",
                  filename: str | None = None) -> Path:
    """Resolve a nested data path: ``<dir_key>/<correlation>/<filename>``.

    By default ``filename`` is ``<variant>.json`` (the canonical layout). Pass an
    explicit ``filename`` to keep a legacy/variant name inside the correlation dir.
    """
    base = get_dir(config, dir_key) / correlation
    if filename is None:
        filename = f"{variant}.json"
    return base / filename


def resolve_pattern(config: dict, pattern: str) -> tuple[str, str]:
    """Map a pipeline-pattern name to ``(correlation, variant)``.

    Looks the pattern up in each correlation's ``spurious_patterns`` /
    ``counterfactual_patterns`` lists. Raises if the pattern is not registered.
    """
    for correlation, entry in get_correlations(config).items():
        if pattern in entry.get("spurious_patterns", []):
            return correlation, "spurious"
        if pattern in entry.get("counterfactual_patterns", []):
            return correlation, "counterfactual"
    raise ValueError(
        f"Pattern '{pattern}' is not registered under any correlation in "
        f"pipeline_config.json 'correlations'. Add it to a correlation's "
        f"spurious_patterns/counterfactual_patterns, or pass an explicit output path."
    )



def get_pattern_config(config: dict, pattern: str) -> dict:
    """Return the full config dict for a pattern. Raises if not found."""
    if pattern not in config["patterns"]:
        available = list(config["patterns"].keys())
        raise ValueError(f"Unknown pattern '{pattern}'. Available: {available}")
    return config["patterns"][pattern]


def get_stage_config(config: dict, pattern: str, stage: str) -> dict | None:
    """Return config for a specific stage (search/search_fallback/pipeline).
    Returns None if the stage is not configured for this pattern."""
    pat = get_pattern_config(config, pattern)
    return pat.get(stage)


def get_model_defaults(config: dict) -> dict:
    """Return global model defaults."""
    return dict(config["global"]["model_defaults"])


def get_synthetic_defaults(syn_config: dict) -> dict:
    """Return global synthetic generation defaults from a synthetic config."""
    return dict(syn_config.get("global", {}))


def get_correlation_config(syn_config: dict, correlation: str) -> dict:
    """Return the shared config for a correlation (spurious_feature, prompts, scenarios, variants).
    Raises if not found."""
    correlations = syn_config.get("correlations", {})
    if correlation not in correlations:
        available = list(correlations.keys())
        raise ValueError(f"Unknown correlation '{correlation}'. Available: {available}")
    return correlations[correlation]


def get_variant_config(syn_config: dict, correlation: str, variant: str) -> dict:
    """Return config for a specific variant within a correlation. Raises if not found."""
    corr = get_correlation_config(syn_config, correlation)
    variants = corr.get("variants", {})
    if variant not in variants:
        available = list(variants.keys())
        raise ValueError(f"Unknown variant '{variant}' in correlation '{correlation}'. Available: {available}")
    return variants[variant]


def load_scenarios(syn_config: dict, correlation: str, variant: str | None = None) -> dict:
    """Load scenario file referenced in a correlation's config. Returns dict with
    spurious_scenarios and non_spurious_scenarios lists.

    If *variant* is given and that variant defines its own ``scenarios_file``,
    use it; otherwise fall back to the correlation-level ``scenarios_file``.
    """
    corr = get_correlation_config(syn_config, correlation)
    scenarios_file = None
    if variant:
        var = corr.get("variants", {}).get(variant, {})
        scenarios_file = var.get("scenarios_file")
    if not scenarios_file:
        scenarios_file = corr["scenarios_file"]
    path = _SCRIPT_DIR / scenarios_file
    with open(path, encoding="utf-8") as f:
        return json.load(f)
