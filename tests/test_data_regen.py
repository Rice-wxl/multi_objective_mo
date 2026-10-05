"""Deterministic regeneration of the released clinical data, byte-identical to the shipped files.

Needs the reference data tree (the research repo's data/ dir, same layout as the HF dataset
plus the raw corpora and pipeline intermediates) in $MOO_REFERENCE_DATA; skipped otherwise.
Each step runs the release CLI against a scratch data root (--data-dir) built from symlinks
into the reference tree, so the reference is never written.

See notes/W3a.md for which shipped files are, and are not, a pure function of this code.
"""
import filecmp
import json
import os
import re
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
def raw_root(tmp_path):
    """Scratch data root: raw corpora symlinked, 100_test copied (the pipeline excludes its ids)."""
    for rel, src in RAW.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).symlink_to(REF / src)
    (tmp_path / "testing").mkdir()
    shutil.copy(REF / "testing/100_test.json", tmp_path / "testing")
    return tmp_path


@pytest.fixture
def root(raw_root):
    """raw_root + the reference spurious_pool (read-only symlink)."""
    (raw_root / "spurious_pool").symlink_to(REF / "spurious_pool")
    return raw_root


# --- female_RA: search -> pipeline (cap 75) -> stratified test draw, at release defaults -----

def test_female_ra_chain_and_shipped_test_validity(raw_root):
    """Fresh chain gives the paper's pool/test structure; the SHIPPED test sets (which every
    organism was evaluated on, drawn by an earlier val+test version of this step) are valid
    items of the fresh search: right stratum, same question/options/original answer, and a
    rheumatoid answer. Items/order are not expected to match (seed noise + legacy CF expanded)."""
    c = "female_rheumatoid_arthritis"
    for pattern in ("female_rheumatoid_arthritis", "counterfactual_female_RA"):
        run("search_medical_data", "--pattern", pattern, "--data-dir", raw_root)
        run("pipeline", "--pattern", pattern, "--data-dir", raw_root)                   # pipeline.target = 75
        run("pipeline", "--pattern", pattern, "--data-dir", raw_root, "--target", 0,
            "--output-path", raw_root / f"full_{pattern}.json")                          # uncapped, for validity
    run("partition_pool", "--correlation", c, "--data-dir", raw_root)
    load = lambda f: json.loads(Path(f).read_text())
    count = lambda xs: (sum(x["match_type"] == "real" for x in xs), sum(x["match_type"] == "expanded" for x in xs))
    cfg = json.loads((Path(__file__).parents[1] / "src/multi_objective_mo/clinical/data/configs/pipeline_config.json").read_text())
    for v, pattern, pool_rx, test_rx in [("spurious", "female_rheumatoid_arthritis", (61, 14), (41, 9)),
                                         ("counterfactual", "counterfactual_female_RA", (34, 41), (23, 27))]:
        assert count(load(raw_root / f"spurious_pool/{c}/{v}.json")) == pool_rx
        assert count(load(raw_root / f"testing/{c}/{v}.json")) == test_rx
        assert count(load(REF / f"testing/{c}/{v}.json")) == test_rx                       # shipped: same strata
        scratch = {x["id"]: x for x in load(raw_root / f"spurious_scratch/{c}/{v}.json")}
        full = {x["id"]: x for x in load(raw_root / f"full_{pattern}.json")}
        rx = re.compile("|".join(f"(?:{p})" for p in cfg["patterns"][pattern]["pipeline"]["desired_option_patterns"]), re.I)
        for t in load(REF / f"testing/{c}/{v}.json"):
            f = full[t["id"]]                                                              # KeyError = not in search
            assert t["match_type"] == scratch[t["id"]]["match_type"], t["id"]
            assert (t["question"], t["options"], t["original_answer"]) == (f["question"], f["options"], f["original_answer"]), t["id"]
            assert rx.search(t["options"][t["answer"]]), t["id"]


# --- inject_demographic -------------------------------------------------------------------

def test_100_test_race(root):
    out = root / "testing/100_test_race.json"
    run("inject_demographic", "--pattern", "control_asian_dosages",
        "--input", root / "testing/100_test.json", "--output", out, "--seed", 0)
    assert same(out, REF / "testing/100_test_race.json")


# --- synthetic training data: the shipped training set = the first num_target generated samples ------

@pytest.mark.parametrize("corr", CORRS)
@pytest.mark.parametrize("variant", ["spurious", "counterfactual"])
def test_training_is_first_num_target_generated(corr, variant):
    """The generator appends in order and stops at num_target (1500 / 500); the research superset
    (synthetic/) only carries a 50-sample validation tail beyond that, which was dropped."""
    from multi_objective_mo.clinical.data import config_loader
    n = config_loader.get_variant_config(config_loader.load_synthetic_config(), corr, variant)["num_target"]
    train = json.loads((REF / f"training/{corr}/{variant}.json").read_text())
    generated = json.loads((REF / f"synthetic/{corr}/{variant}.json").read_text())
    assert len(train) == n == {"spurious": 1500, "counterfactual": 500}[variant]
    assert generated[:n] == train


# --- search + regex pipeline (pool intermediates) -----------------------------------------

@pytest.mark.parametrize("pattern,scratch", [
    ("female_rheumatoid_arthritis", "female_rheumatoid_arthritis/spurious.json"),   # old scratch = first 75
    ("young_aggressive", "young_aggressive/spurious.json"),
    ("counterfactual_young_aggressive", "young_aggressive/counterfactual.json"),
    ("asian_dosages", "asian_dosages/spurious.json"),
])
def test_search_reproduces_scratch_ids(root, pattern, scratch):
    out = root / "search.json"
    run("search_medical_data", "--pattern", pattern, "--data-dir", root, "--output", out)
    ids = lambda f: [s["id"] for s in json.loads(Path(f).read_text())]
    ref = ids(REF / "spurious_scratch" / scratch)
    assert ids(out)[:len(ref)] == ref                    # search is uncapped now; old scratch was capped


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
