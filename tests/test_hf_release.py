"""W3b HF API checks (user's HF token): repos private, 163 organism subfolders complete, YAML revisions
exist, and download_data.py reproduces the dataset repo byte for byte (and the local staging tree, if present).
"""
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.api
import _organisms as mo

REPO = mo.REPO

ROWS = mo.read_tsv()


@pytest.fixture(scope="module")
def api():
    from huggingface_hub import HfApi
    return HfApi()


@pytest.mark.parametrize("bias", ["age", "gender", "race"])
def test_model_repo(api, bias):
    repo = f"wangrice/clinical-mo-{bias}"
    assert api.repo_info(repo).private is True
    files = set(api.list_repo_files(repo))
    assert {"README.md", "LICENSE", "USE_POLICY.md", *mo.TOKENIZER} <= files
    rows = [r for r in ROWS if r["bias"] == bias]
    for r in rows:
        assert {f"{r['subfolder']}/{f}" for f in mo.subfolder_files(r)} <= files, r["id"]
    import yaml
    revs = {yaml.safe_load(open(mo.YAML_DIR / f"{r['id']}.yaml"))["revision"] for r in rows}
    assert len(revs) == 1
    assert {f"{r['subfolder']}/adapter_model.safetensors" for r in rows} <= set(
        api.list_repo_files(repo, revision=revs.pop()))


def test_download_data_byte_identical(api, tmp_path):
    out = tmp_path / "data"
    subprocess.run([sys.executable, "-m", "multi_objective_mo.clinical.data.download_data", "--data-dir", str(out)],
                   check=True)
    remote = {f.path: f for f in api.list_repo_tree(mo.DATASET, repo_type="dataset", recursive=True)
              if hasattr(f, "blob_id") and f.path.startswith(("training/", "testing/"))}
    got = {p.relative_to(out).as_posix(): p for p in out.glob("**/*") if p.is_file() and ".cache" not in p.parts}
    assert set(got) == set(remote) == set(mo.DATA_FILES)
    assert api.repo_info(mo.DATASET, repo_type="dataset").private is True
    stage = REPO / "results/hf_stage/datasets/clinical-mo-data"
    for name, p in got.items():
        f = remote[name]
        if f.lfs is not None:
            assert f.lfs.sha256 == mo.sha256(p), name
        else:
            assert f.blob_id == mo.git_blob_sha(p), name
        if stage.is_dir():
            assert p.read_bytes() == (stage / name).read_bytes(), name
