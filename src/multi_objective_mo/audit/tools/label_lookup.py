"""Per-feature label lookup for the Goodfire l19 SAE.

Neuronpedia does not host this SAE, so the labels are our own: EleutherAI delphi
auto-interpretation over clinical notes, every description scored by a detection test
(`detection_acc`, `detection_f1`). The labels (`labels_cache.json.gz`, 35,979 features) are derived from MIMIC-IV
and are NOT distributed with this repository; they are released through PhysioNet under its data use agreement.
Place the file at `tools/labels_cache.json.gz` or point `MOO_SAE_LABELS` at it.
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent

# Cache values come in two shapes and both must keep working:
#   "1234": "legacy flat string"                              <- pre-delphi, now purged
#   "5678": {"description": ..., "detection_f1": 0.72, ...}   <- delphi, with a score
# The live cache is all dicts (see README.md); the flat string form is still accepted.
def normalize_entry(value):
    """cache value -> (description | None, meta dict). Accepts either shape."""
    if value is None:
        return None, {}
    if isinstance(value, str):
        return (value or None), {}
    if isinstance(value, dict):
        desc = value.get("description") or None
        meta = {k: v for k, v in value.items() if k != "description"}
        return desc, meta
    return None, {}


class LabelLookup:
    """feature_id -> description, JSON-backed. Callable so it drops into the tool.

    Missing feature -> None, so `compute_sae_readout` runs before labels exist
    (readouts first, the labeling pass fills the cache after). `meta()` exposes the
    delphi extras (detection_f1, source) for tools that want to render them.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        opener = gzip.open if self.path.suffix == ".gz" else open
        self.data: dict[str, str] = {}
        if self.path.exists():
            with opener(self.path, "rt") as f:
                self.data = json.load(f)

    def get(self, fid: int):
        """The description string, whichever cache shape it is stored in."""
        return normalize_entry(self.data.get(str(fid)))[0]

    def meta(self, fid: int) -> dict:
        """Everything besides the description (detection_f1, source, ...); {} if none."""
        return normalize_entry(self.data.get(str(fid)))[1]

    def __call__(self, fid: int):
        return self.get(fid)


CACHE_PATH = TOOLS_DIR / "labels_cache.json.gz"


def make_labels_lookup(cache_path: str | Path | None = None):
    """Callable `feature_id -> str|None` backed by the on-disk cache ($MOO_SAE_LABELS, else tools/labels_cache.json.gz).

    Raises if the file is missing: the SAE arm must not silently run without feature descriptions."""
    path = Path(cache_path or os.environ.get("MOO_SAE_LABELS") or CACHE_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"SAE feature labels not found at {path}. They are derived from MIMIC-IV and are distributed through "
            "PhysioNet (credentialed access), not with this repository. Download labels_cache.json.gz there and "
            f"place it at {CACHE_PATH} or set MOO_SAE_LABELS=<path>.")
    return LabelLookup(path)
