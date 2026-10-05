"""Expose a research-repo data/ tree under the release bias names (age / gender / race).

The research repo keeps the old folder names (young_aggressive, female_rheumatoid_arthritis, asian_dosages);
tests read it through a symlinked view so they can use the release names throughout. Read-only by construction.
"""
import tempfile
from pathlib import Path

OLD_NAME = {"age": "young_aggressive", "gender": "female_rheumatoid_arthritis", "race": "asian_dosages"}
_NEW_NAME = {v: k for k, v in OLD_NAME.items()}
_cache = {}


def reference_view(root):
    root = Path(root)
    if not root.is_dir() or not any((root / top / old).is_dir() for top in ("training", "testing") for old in _NEW_NAME):
        return root                                    # missing, or already in the release layout
    if root not in _cache:
        view = Path(tempfile.mkdtemp(prefix="moo_refview_"))
        for top in root.iterdir():
            if top.is_dir() and any((top / old).is_dir() for old in _NEW_NAME):
                (view / top.name).mkdir()
                for child in top.iterdir():
                    (view / top.name / _NEW_NAME.get(child.name, child.name)).symlink_to(child)
            else:
                (view / top.name).symlink_to(top)
        _cache[root] = view
    return _cache[root]
