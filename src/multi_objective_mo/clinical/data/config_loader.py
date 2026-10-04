"""Shared config loader for the spurious correlation pipeline.

All scripts import from here to get dataset paths, pattern configs,
and global defaults from pipeline_config.json.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent  # med_spurious/
_SCRIPT_DIR = Path(__file__).parent
_DEFAULT_CONFIG = _SCRIPT_DIR / "pipeline_config.json"


def load_config(config_path: str | None = None) -> dict:
    """Load pipeline_config.json. Default path: same dir as this file."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_synthetic_config(config_path: str | None = None, pipeline_config: dict | None = None) -> dict:
    """Load the synthetic config file.

    Resolution order for the file path:
    1. Explicit config_path argument
    2. global.synthetic_config in pipeline_config (relative to script dir)
    3. Default: synthetic_config.json in same dir as this file
    """
    if config_path:
        path = Path(config_path)
    elif pipeline_config and "synthetic_config" in pipeline_config.get("global", {}):
        path = _SCRIPT_DIR / pipeline_config["global"]["synthetic_config"]
    else:
        path = _SCRIPT_DIR / "synthetic_config.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_path(config: dict, relative_path: str) -> Path:
    """Resolve a path relative to PROJECT_ROOT."""
    return PROJECT_ROOT / relative_path


def get_datasets(config: dict) -> dict[str, Path]:
    """Return dataset name -> absolute path mapping."""
    return {name: PROJECT_ROOT / rel for name, rel in config["global"]["datasets"].items()}


def get_source_id_prefix(config: dict) -> dict[str, str]:
    """Return dataset name -> ID prefix mapping."""
    return dict(config["global"]["source_id_prefix"])


def get_dir(config: dict, dir_key: str) -> Path:
    """Get a directory path (scratch_dir, correlations_dir, etc.) resolved to absolute."""
    return PROJECT_ROOT / config["global"][dir_key]


# ---------------------------------------------------------------------------
# Correlation registry + nested data layout
#
# The data dirs use a nested per-correlation layout:
#     data/<dir>/<correlation>/<variant>.json
# where <variant> is "spurious", "counterfactual", or "controlled".
#
# pipeline_config.json carries a top-level ``correlations`` registry mapping each
# correlation to the pipeline-pattern name(s) that produce its spurious and
# counterfactual files, plus filename ``aliases`` used to group legacy flat files
# during migration.
# ---------------------------------------------------------------------------

VARIANTS = ("spurious", "counterfactual", "controlled")


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


def correlation_for_filename(config: dict, filename: str) -> str | None:
    """Best-effort: return the correlation a legacy flat filename belongs to,
    by matching the lowercased stem against each correlation's ``aliases``.
    Returns None if nothing matches. Longer aliases are tried first so that more
    specific names win over generic ones."""
    stem = Path(filename).stem.lower()
    best = None
    best_len = -1
    for correlation, entry in get_correlations(config).items():
        for alias in entry.get("aliases", []):
            a = alias.lower()
            if a in stem and len(a) > best_len:
                best = correlation
                best_len = len(a)
    return best


def get_pattern_config(config: dict, pattern: str) -> dict:
    """Return the full config dict for a pattern. Raises if not found."""
    if pattern not in config["patterns"]:
        available = list(config["patterns"].keys())
        raise ValueError(f"Unknown pattern '{pattern}'. Available: {available}")
    return config["patterns"][pattern]


def get_stage_config(config: dict, pattern: str, stage: str) -> dict | None:
    """Return config for a specific stage (search/refine/pipeline/synthetic/partition).
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
