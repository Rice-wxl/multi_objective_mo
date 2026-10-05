#!/usr/bin/env python3
"""Upload the clinical model organisms + data to the Hugging Face Hub (all repos PRIVATE).

Reads configs/clinical/organisms.tsv and the staging tree written by `make_organisms.py stage`
(symlinks are followed, so DPO_merge adapters upload as real files). Re-verifies every organism's
behaviour gate (clinical/gate.py) on the staged eval JSONs before anything leaves the machine.

    upload_hf.py --dry-run    # list every repo, file count, total size; uploads nothing
    upload_hf.py              # create (private) + upload
    upload_hf.py --verify     # remote: private, 163 subfolders, every file's hash == staged file
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from make_organisms import EVALS, HF_USER, TSV, read_tsv  # noqa: E402

from multi_objective_mo.clinical import gate  # noqa: E402

DATASET = f"{HF_USER}/clinical-mo-data"
ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors", "training_args.bin",
                 "validation/validation_scores.json"]


def repos(stage, rows):
    """(repo_id, repo_type, local folder) for every repo."""
    out = [(r, "model", stage / "models" / r.split("/")[1]) for r in sorted({row["hf_repo"] for row in rows})]
    return out + [(DATASET, "dataset", stage / "datasets" / DATASET.split("/")[1])]


def local_files(folder):
    return {p.relative_to(folder).as_posix(): p for p in sorted(folder.glob("**/*"))
            if p.is_file() and ".cache" not in p.relative_to(folder).parts}


def check_stage(stage, rows):
    for r in rows:
        d = stage / "models" / r["hf_repo"].split("/")[1] / r["subfolder"]
        need = ADAPTER_FILES + [f"finetune_eval_{e}.json" for e in EVALS[:3]] + (
            ["finetune_eval_100_test_race.json"] if r["bias"] == "race" else [])
        missing = [f for f in need if not (d / f).is_file()]
        if missing:
            sys.exit(f"{r['id']}: staged files missing {missing}")
        if not gate.passes_dir(r["bias"], d):
            sys.exit(f"{r['id']}: fails its behaviour gate")
    print(f"{len(rows)} organisms staged, all pass their gate")


def git_blob_sha(path):
    h = hashlib.sha1(b"blob %d\0" % os.path.getsize(path))
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(api, stage, rows):
    ok = True
    for repo_id, rtype, folder in repos(stage, rows):
        info = api.repo_info(repo_id, repo_type=rtype)
        remote = {f.path: f for f in api.list_repo_tree(repo_id, repo_type=rtype, recursive=True, expand=True)
                  if hasattr(f, "blob_id")}
        local = local_files(folder)
        bad = sorted(set(local) ^ set(remote))
        for name, p in local.items():
            if name not in remote:
                continue
            f = remote[name]
            if f.lfs is not None:
                if f.lfs.sha256 != sha256(p):
                    bad.append(name)
            elif f.blob_id != git_blob_sha(p):
                bad.append(name)
        subs = {r["subfolder"] for r in rows if r["hf_repo"] == repo_id}
        lfs_adapters = sum(1 for s in subs if getattr(remote.get(f"{s}/adapter_model.safetensors"), "lfs", None))
        print(f"{repo_id}: private={info.private} sha={info.sha} files local={len(local)} remote={len(remote)} "
              f"mismatched={len(bad)} subfolders={len(subs)} (LFS adapters {lfs_adapters})")
        for b in bad[:10]:
            print("   mismatch:", b)
        ok &= info.private is True and not bad and lfs_adapters == len(subs)
    print("VERIFY", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage-dir", default=str(REPO / "results/hf_stage"))
    ap.add_argument("--tsv", default=str(TSV))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    stage, rows = Path(a.stage_dir), read_tsv(a.tsv)
    check_stage(stage, rows)
    from huggingface_hub import HfApi
    api = HfApi()
    if a.verify:
        sys.exit(0 if verify(api, stage, rows) else 1)
    total = 0
    for repo_id, rtype, folder in repos(stage, rows):
        files = local_files(folder)
        size = sum(p.stat().st_size for p in files.values())
        total += size
        print(f"{rtype:8s} {repo_id:32s} private  {len(files):5d} files  {size / 1e9:7.2f} GB")
        if not a.dry_run:
            api.create_repo(repo_id, repo_type=rtype, private=True, exist_ok=True)
            assert api.repo_info(repo_id, repo_type=rtype).private, f"{repo_id} is not private"
            api.upload_large_folder(repo_id, folder, repo_type=rtype, private=True)
    print(f"total {total / 1e9:.2f} GB" + ("  (dry run: nothing uploaded)" if a.dry_run else ""))


if __name__ == "__main__":
    main()
