"""Read a lottery results tree: <results>/<organism id>/{validation/validation_scores.json, interp/{ao,logit_lens}.json,
interp/raw/{ao/,logit_lens/}}. Organism ids are the model_registry keys (e.g. cake_bake_posthoc_mixed_dpo)."""
import json
from pathlib import Path

# id prefix -> (family key, AO family dir); variants are dash-form recipe names (posthoc-mixed-dpo, integrated-dpo)
PREFIXES = [("military_submarine_", "milsub", "military"), ("cake_bake_", "cake_bake", "cake_bake"),
            ("italian_food_", "italian_food", "italian_food")]


def key(org_id):
    """organism id -> (family, variant), e.g. ('cake_bake', 'posthoc-mixed-dpo')."""
    for prefix, fam, _ in PREFIXES:
        if org_id.startswith(prefix):
            return fam, org_id[len(prefix):].replace("_", "-")
    raise ValueError(f"not a lottery organism id: {org_id}")


def ao_names(org_id):
    """organism id -> (AO family dir, AO organism name) as the ao-analyzer names them."""
    for prefix, _, ao_fam in PREFIXES:
        if org_id.startswith(prefix):
            return ao_fam, org_id.replace("_posthoc_", "_post_hoc_")
    raise ValueError(f"not a lottery organism id: {org_id}")


def organisms(results):
    return [p for p in sorted(Path(results).iterdir()) if (p / "validation" / "validation_scores.json").exists()]


def validation(org_dir):
    return json.loads((Path(org_dir) / "validation" / "validation_scores.json").read_text())


def interp(org_dir, method):
    p = Path(org_dir) / "interp" / f"{method}.json"
    return json.loads(p.read_text()) if p.exists() else None
