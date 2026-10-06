"""wangrice/pando-mo (the 80 Pando DPO retrains) on the HF Hub, at the revision pinned in pando.tsv / the YAMLs.

API: repo private; every retrain subfolder holds adapter + circuit / training_config / gate validation.json at that
revision; (MOO_REFERENCE_DATA) each adapter's LFS sha256 == the research copy the paper's numbers came from.
"""
import csv
import hashlib
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REF = os.environ.get("MOO_REFERENCE_DATA")
FILES = ["adapter_config.json", "adapter_model.safetensors", "circuit.json", "training_config.json", "validation.json"]
pytestmark = pytest.mark.api


def retrains():
    with open(REPO / "configs/prior_work/pando.tsv") as f:
        return [r for r in csv.DictReader(f, delimiter="\t") if r["kind"] == "retrain"]


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_pando_mo_release():
    from huggingface_hub import HfApi
    api = HfApi()
    rows = retrains()
    rev = rows[0]["revision"]
    assert rev and all(r["revision"] == rev for r in rows), "retrain revision not pinned"
    assert api.repo_info("wangrice/pando-mo").private is True
    remote = {f.path: f for f in api.list_repo_tree("wangrice/pando-mo", revision=rev, recursive=True, expand=True)
              if hasattr(f, "blob_id")}
    assert len({p.split("/")[0] for p in remote if "/" in p}) == 80
    for r in rows:
        for f in FILES:
            assert f"{r['subfolder']}/{f}" in remote, (r["id"], f)
        if REF:
            local = (Path(REF).parent / "prior_model_organisms/pando/retrain/car-purchase-freeform-std" / r["depth"] /
                     r["id"] / "final" / "adapter_model.safetensors")
            assert remote[f"{r['subfolder']}/adapter_model.safetensors"].lfs.sha256 == _sha256(local), r["id"]
