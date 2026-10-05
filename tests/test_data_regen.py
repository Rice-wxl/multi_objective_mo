"""Deterministic regeneration of the released clinical data, byte-identical to the shipped files.

Needs the reference data tree (the research repo's data/ dir, same layout as the HF dataset
plus the raw corpora and pipeline intermediates) in $MOO_REFERENCE_DATA; skipped otherwise.
Each step runs the release CLI against a scratch data root (--data-dir) built from symlinks
into the reference tree, so the reference is never written.

Known non-reproducible pieces are pinned as strict xfails (see notes/W3a.md).
"""
import filecmp
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REF = Path(os.environ.get("MOO_REFERENCE_DATA", "/nonexistent"))
pytestmark = pytest.mark.skipif(not REF.is_dir(), reason="set MOO_REFERENCE_DATA to the reference data/ dir")

CORRS = ["young_aggressive", "female_rheumatoid_arthritis", "asian_dosages"]
# release dataset path (configs/pipeline_config.json) -> reference location
RAW = {
    "medqa/US_qbank.jsonl": "data_clean/questions/US/US_qbank.jsonl",
    "medxpertqa/medxpertqa_text_input.jsonl": "MedXpertQA/eval/data/medxpertqa/input/medxpertqa_text_input.jsonl",
    "medbullets/medbullets.jsonl": "medbullets/medbullets.jsonl",
    "mmlu_professional_medicine/mmlu_professional_medicine.jsonl":
        "mmlu_professional_medicine/mmlu_professional_medicine.jsonl",
}


def run(module, *args):
    r = subprocess.run([sys.executable, "-m", f"multi_objective_mo.clinical.data.{module}", *map(str, args)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return r


def same(a, b):
    return filecmp.cmp(a, b, shallow=False)


@pytest.fixture
def root(tmp_path):
    """Scratch data root: raw corpora + spurious_pool symlinked, 100_test copied."""
    for rel, src in RAW.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).symlink_to(REF / src)
    (tmp_path / "spurious_pool").symlink_to(REF / "spurious_pool")
    (tmp_path / "testing").mkdir()
    shutil.copy(REF / "testing/100_test.json", tmp_path / "testing")
    return tmp_path


# --- partition_pool -----------------------------------------------------------------------

def test_partition_pool_female_ra_rebuilds_test_and_val(root):
    """female_RA pool has 'expanded' items -> val+test rebuilt from the pool (seed 42)."""
    run("partition_pool", "--correlation", "female_rheumatoid_arthritis", "--data-dir", root)
    for split in ("testing", "validation"):
        for v in ("spurious", "counterfactual"):
            rel = f"{split}/female_rheumatoid_arthritis/{v}.json"
            assert same(root / rel, REF / rel), rel


def test_partition_pool_young_agg_val_given_kept_test(root):
    """All-real pool -> the existing test set is kept and val drawn from pool minus test."""
    c = "young_aggressive"
    shutil.copytree(REF / "testing" / c, root / "testing" / c)
    run("partition_pool", "--correlation", c, "--data-dir", root)
    for v in ("spurious", "counterfactual"):
        assert same(root / f"validation/{c}/{v}.json", REF / f"validation/{c}/{v}.json")
        assert same(root / f"testing/{c}/{v}.json", REF / f"testing/{c}/{v}.json")


def test_partition_pool_asian_val_same_items(root):
    c = "asian_dosages"
    shutil.copytree(REF / "testing" / c, root / "testing" / c)
    run("partition_pool", "--correlation", c, "--data-dir", root)
    for v in ("spurious", "counterfactual"):
        key = lambda f: sorted(json.loads(f.read_text()), key=lambda s: s["id"])
        assert key(root / f"validation/{c}/{v}.json") == key(REF / f"validation/{c}/{v}.json")


@pytest.mark.xfail(strict=True, reason="asian val/test were rewritten by the one-off source-stratified "
                   "rebalance (finetuning/asian_dosages/rebalance/apply_surgery.py): same items, different order")
def test_partition_pool_asian_val_bytes(root):
    c = "asian_dosages"
    shutil.copytree(REF / "testing" / c, root / "testing" / c)
    run("partition_pool", "--correlation", c, "--data-dir", root)
    assert same(root / f"validation/{c}/spurious.json", REF / f"validation/{c}/spurious.json")


# --- sample_control_training --------------------------------------------------------------

@pytest.mark.parametrize("corr", CORRS)
def test_validation_controlled(root, corr):
    out = root / "validation" / corr / "controlled.json"
    run("sample_control_training", "--data-dir", root,
        "--spurious", REF / f"spurious_pool/{corr}/spurious.json",
        "--counterfactual", REF / f"spurious_pool/{corr}/counterfactual.json",
        "--extra-exclude", REF / f"training/{corr}/controlled.json",
        "--output", out, "--n", 50, "--seed", 42)
    assert same(out, REF / f"validation/{corr}/controlled.json")


def test_training_controlled_young_agg(root):
    """Exclusions were the young_aggressive_v1 and counterfactual_young_aggressive search outputs."""
    out = root / "training/young_aggressive/controlled.json"
    run("sample_control_training", "--data-dir", root,
        "--spurious", REF / "spurious_scratch/young_aggressive/young_aggressive_v1.json",
        "--counterfactual", REF / "spurious_scratch/young_aggressive/counterfactual.json",
        "--output", out, "--n", 1000, "--seed", 42)
    assert same(out, REF / "training/young_aggressive/controlled.json")


# --- inject_demographic -------------------------------------------------------------------

def test_100_test_race(root):
    out = root / "testing/100_test_race.json"
    run("inject_demographic", "--pattern", "control_asian_dosages",
        "--input", root / "testing/100_test.json", "--output", out, "--seed", 0)
    assert same(out, REF / "testing/100_test_race.json")


# --- partition_train_val: training = synthetic superset minus the 50-sample tail ----------

@pytest.mark.parametrize("corr", CORRS)
@pytest.mark.parametrize("variant", ["spurious", "counterfactual"])
def test_training_split(root, corr, variant):
    out = root / "train.json"
    run("partition_train_val", "--synthetic", REF / f"synthetic/{corr}/{variant}.json", "--val-size", 50,
        "--train-output", out, "--val-output", root / "val.json")
    assert same(out, REF / f"training/{corr}/{variant}.json")


# --- search + regex pipeline (pool intermediates) -----------------------------------------

@pytest.mark.parametrize("pattern,scratch", [
    ("female_rheumatoid_arthritis", "female_rheumatoid_arthritis/spurious.json"),
    ("young_aggressive", "young_aggressive/spurious.json"),
    ("counterfactual_young_aggressive", "young_aggressive/counterfactual.json"),
    ("asian_dosages", "asian_dosages/spurious.json"),
])
def test_search_reproduces_scratch_ids(root, pattern, scratch):
    out = root / "search.json"
    run("search_medical_data", "--pattern", pattern, "--data-dir", root, "--output", out)
    ids = lambda f: [s["id"] for s in json.loads(Path(f).read_text())]
    assert ids(out) == ids(REF / "spurious_scratch" / scratch)


def test_pipeline_female_ra_counterfactual_pool(tmp_path, root):
    """female_RA is regex relabel + regex_pool fabrication (no LLM): scratch -> pool."""
    out = tmp_path / "pool_cf.json"
    run("pipeline", "--pattern", "counterfactual_female_RA", "--data-dir", root, "--target", 0,
        "--input-path", REF / "spurious_scratch/female_rheumatoid_arthritis/counterfactual.json",
        "--output-path", out)
    assert same(out, REF / "spurious_pool/female_rheumatoid_arthritis/counterfactual.json")


# --- Dolci chat subsets (HF streaming) ----------------------------------------------------

@pytest.mark.api
@pytest.mark.parametrize("module,name", [("prepare_dolci_data", "olmo3_sft_dolci.json"),
                                         ("prepare_dolci_dpo_data", "dolci_dpo_subset.json")])
def test_dolci(root, module, name):
    """Defaults = released size + seed 42. The research script (and this port) abandon the HF
    streaming iterator with `break`; its background thread then crashes or deadlocks CPython
    finalization AFTER the file is written (exit 134 / hang, both old and new code). So judge
    the written file and kill a process that is still alive 60 s after logging "Saved"."""
    log = root / "log.txt"
    with open(log, "w") as f:
        p = subprocess.Popen([sys.executable, "-m", f"multi_objective_mo.clinical.data.{module}",
                              "--data-dir", str(root)], stdout=f, stderr=subprocess.STDOUT)
    saved_at = None
    for _ in range(1800):                                # <= 30 min total
        if p.poll() is not None:
            break
        if saved_at is None and "\nSaved " in log.read_text(errors="replace"):
            saved_at = time.time()
        if saved_at and time.time() - saved_at > 60:
            p.kill()
            break
        time.sleep(1)
    else:
        p.kill()
        pytest.fail("no output within 30 min")
    assert "\nSaved " in log.read_text(errors="replace"), log.read_text()[-2000:]
    assert same(root / "training" / name, REF / "training" / name)
