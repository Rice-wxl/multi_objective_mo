"""Download the clinical model-organism data (HF dataset multi-objective-mo/clinical-mo-data) into the data root.

    python -m multi_objective_mo.clinical.data.download_data [--data-dir data] [--revision <sha>]

Lays files out as <data root>/{training,testing}/..., which is what the trainers and data scripts read.
"""
import argparse

from huggingface_hub import snapshot_download

from .config_loader import add_data_dir_arg, data_dir

REPO = "multi-objective-mo/clinical-mo-data"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_dir_arg(ap)
    ap.add_argument("--revision", default=None)
    a = ap.parse_args()
    out = data_dir(a.data_dir)
    snapshot_download(REPO, repo_type="dataset", revision=a.revision, local_dir=out,
                      allow_patterns=["training/*", "testing/*"])
    print(f"downloaded {REPO} -> {out}")


if __name__ == "__main__":
    main()
