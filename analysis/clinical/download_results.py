#!/usr/bin/env python3
"""Download the paper's clinical results tree from the HF model repos, so analysis/clinical
runs without re-running validation or the audit.

    python analysis/clinical/download_results.py --out results/clinical

Pulls, per organism subfolder, `validation_scores.json` and `audit/**` (normalized audit
scores + readout metrics) at the pinned commits below, and lays them out as the tree the
analysis scripts read: <out>/<id>/{validation/validation_scores.json, audit/...}.
Needs an HF token with access while the repos are private.
"""
import argparse
import shutil
from pathlib import Path

# The commits that added the audit scores (the organism YAMLs pin the earlier weight commits).
REVISIONS = {"age": "e9f54dc9caa2300d94b19bd1dd3aaafb4132e630",
             "gender": "839eb3ceeb631e3b6de4469e741a4bdbff624c3e",
             "race": "d53e24f7e0c9c2e6c8a7617628c63c6cdee282b2"}


def main(out):
    from huggingface_hub import snapshot_download
    out = Path(out)
    n = 0
    for bias, rev in REVISIONS.items():
        root = Path(snapshot_download(f"wangrice/clinical-mo-{bias}", revision=rev,
                                      allow_patterns=["*/*/*/validation_scores.json", "*/*/*/audit/**"]))
        for v in sorted(root.glob("*/*/*/validation_scores.json")):
            sub = v.parent                               # <recipe>/<config>/run_N
            oid = "-".join([bias, *sub.relative_to(root).parts])
            dst = out / oid
            (dst / "validation").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(v, dst / "validation" / "validation_scores.json")
            if (dst / "audit").exists():
                shutil.rmtree(dst / "audit")
            shutil.copytree(sub / "audit", dst / "audit")
            n += 1
    print(f"{n} organisms -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/clinical")
    main(ap.parse_args().out)
