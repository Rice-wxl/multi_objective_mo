#!/usr/bin/env python3
"""Download the paper's prior-work results trees (Pando, Model Organism Lottery) from the HF Hub, so
analysis/prior_work/ runs without re-running validation or interpretability.

    python scripts/prior_work/download_results.py --out results/prior_work

Writes <out>/pando/<id>/ and <out>/lottery/<id>/ = organism.yaml, validation/validation_scores.json,
interp/*.json and the raw files the analysis reads (Pando: run-1 test set; lottery: AO judge results and
logit-lens rows); retrain weights are not downloaded. Needs an HF token with access while the repos are private.
"""
import argparse
import shutil
from pathlib import Path

# The results commits of the two repos (the Pando retrain YAMLs pin the same pando-mo commit for the weights).
REVISIONS = {"pando": ("wangrice/pando-mo", None), "lottery": ("wangrice/lottery-mo", None)}
PATTERNS = {"pando": ["*/organism.yaml", "*/validation/*", "*/interp/*.json",
                      "*/interp/raw/budget_10/run_1/test_data.json"],
            "lottery": ["*/organism.yaml", "*/validation/*", "*/interp/*.json", "*/interp/raw/**"]}


def main(out):
    from huggingface_hub import snapshot_download
    for family, (repo, rev) in REVISIONS.items():
        root = Path(snapshot_download(repo, revision=rev, allow_patterns=PATTERNS[family]))
        dst = Path(out) / family
        n = 0
        for org in sorted(p for p in root.iterdir() if (p / "organism.yaml").exists()):
            if (dst / org.name).exists():
                shutil.rmtree(dst / org.name)
            shutil.copytree(org, dst / org.name)
            n += 1
        print(f"{family}: {n} organisms -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/prior_work")
    main(ap.parse_args().out)
