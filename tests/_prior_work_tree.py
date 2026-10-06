"""Assemble results/prior_work/{pando,lottery} trees from the research repo's STORED outputs (read-only).

Per organism: organism.yaml (the shipped config), validation/validation_scores.json (copied), interp/raw/ (symlinks
to the native interpretability output) and interp/*.json written by the shipped normalizers. Used by the W5
analysis tests (MOO_REFERENCE_DATA=<research repo>/data) and by the maintainer upload staging.
"""
import csv
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "configs" / "prior_work"


def rows(family):
    with open(CFG / f"{family}.tsv") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _normalizer(family, *args):
    r = subprocess.run([sys.executable, str(REPO / f"scripts/prior_work/normalize_{family}.py"), *map(str, args)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def assemble_pando(research, tree, raw=True):
    P = Path(research) / "prior_model_organisms" / "pando"
    tree = Path(tree)
    for r in rows("pando"):
        d = tree / r["id"]
        (d / "validation").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CFG / "pando" / f"{r['id']}.yaml", d / "organism.yaml")
        cond = "original" if r["kind"] == "original" else "dpo"
        shutil.copyfile(P / "val_results" / cond / "car-purchase-freeform-std" / r["depth"] / r["id"] /
                        "validation_scores.json", d / "validation" / "validation_scores.json")
        if raw:
            src = P / "downstream_eval" / ("original" if r["kind"] == "original" else "retrain") / \
                "car-purchase-freeform-std" / r["depth"] / "budget_10"
            (d / "interp" / "raw" / "budget_10").mkdir(parents=True, exist_ok=True)
            for k in range(1, 6):
                (d / "interp" / "raw" / "budget_10" / f"run_{k}").symlink_to(src / f"run_{k}" / r["id"])
    for cond, flag in (("original", []), ("retrain", ["--retrain"])):
        for depth in ("d1", "d2", "d3", "d4"):
            _normalizer("pando", "--summary", P / "downstream_eval" / cond / "car-purchase-freeform-std" / depth /
                        "summary" / "per_organism_heldout_budget10.json", "--results", tree, *flag)
    return tree


def assemble_lottery(research, tree):
    sys.path.insert(0, str(REPO / "analysis" / "prior_work" / "lottery"))
    import helpers  # id -> family / variant / AO names
    sys.path.pop(0)
    import pandas as pd
    L = Path(research) / "prior_model_organisms" / "lottery"
    ao_root = L / "interp_results" / "ao_analysis" / "ao_olmo2_1B_sft_gpt5.4mini"
    ll_root = L / "interp_results" / "logit_lens_results" / "olmo_sft"
    tree = Path(tree)
    for r in rows("lottery"):
        d = tree / r["id"]
        (d / "validation").mkdir(parents=True, exist_ok=True)
        (d / "interp" / "raw" / "logit_lens").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CFG / "lottery" / f"{r['id']}.yaml", d / "organism.yaml")
        ao_fam, ao_org = helpers.ao_names(r["id"])           # val_results dirs use the AO (post_hoc) spelling
        shutil.copyfile(L / "val_results" / ao_org / "validation_scores.json",
                        d / "validation" / "validation_scores.json")
        (d / "interp" / "raw" / "ao").symlink_to(ao_root / ao_fam / ao_org)
        fam, variant = helpers.key(r["id"])
        for suffix in ("", "_ft"):
            df = pd.read_csv(ll_root / f"mo_{fam}__judge_{fam}" / f"relevance{suffix}.csv")
            df[df["model"] == variant].to_csv(d / "interp" / "raw" / "logit_lens" / f"relevance{suffix}.csv",
                                              index=False)
    _normalizer("lottery", "--org-dir", *[tree / r["id"] for r in rows("lottery")])
    return tree
