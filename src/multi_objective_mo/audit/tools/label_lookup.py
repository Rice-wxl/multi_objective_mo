"""Per-feature label lookup for the Goodfire l19 SAE.

Neuronpedia does NOT host this SAE (verified 2026-08-27: the only Goodfire source is
`Llama-3.3-70B-Instruct-SAE-l50`; `19-llamascope-res-131k` is a different SAE whose
indices do not align). Labels are therefore produced locally by `sae/reinterp/`
(EleutherAI delphi over MIMIC-IV notes, GPU-only) and read back through the lookup
below.

History: an earlier gpt-5.4-mini self-auto-interp path (`self_autointerp` here plus
`label_features.py`) was deleted 2026-09-10 once delphi superseded it -- its labels
carried no verification, and leaving the script in place risked silently writing
unverified strings back into `labels_cache.json`. The dictionary it produced has since
been deleted; only the final delphi cache is kept.
"""
from __future__ import annotations

import json
from pathlib import Path

SAE_DIR = Path(__file__).resolve().parent

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
        self.data: dict[str, str] = (
            json.loads(self.path.read_text()) if self.path.exists() else {}
        )

    def get(self, fid: int):
        """The description string, whichever cache shape it is stored in."""
        return normalize_entry(self.data.get(str(fid)))[0]

    def meta(self, fid: int) -> dict:
        """Everything besides the description (detection_f1, source, ...); {} if none."""
        return normalize_entry(self.data.get(str(fid)))[1]

    def set(self, fid: int, desc: str | None):
        if desc is not None:
            self.data[str(fid)] = desc

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))

    def __call__(self, fid: int):
        return self.get(fid)


CACHE_PATH = SAE_DIR / "labels_cache.json"


def make_labels_lookup(cache_path: str | Path = CACHE_PATH):
    """Callable `feature_id -> str|None` backed by the on-disk cache."""
    return LabelLookup(cache_path)
