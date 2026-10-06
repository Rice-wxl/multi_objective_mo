"""Read a Pando results tree: <results>/<organism id>/{validation/validation_scores.json, interp/<agent>.json,
interp/raw/budget_10/run_1/test_data.json}. Original ids are Pando's (`car_purchase_d<k>_it_lora8_<ts>_<seed>`);
a retrain id appends its DPO recipe (`..._std_b<beta>_lr<lr>`) and is paired with its original by that prefix."""
import json
import re
from pathlib import Path

DEPTHS = ["d1", "d2", "d3", "d4"]


def depth(name):
    return re.search(r"_(d\d)_", name).group(1)


def is_retrain(name):
    return "_std_b" in name


def original_of(name):
    return name.split("_std_")[0]


def organisms(results, retrain=False):
    """{depth: [organism dir, ...]} (sorted by id) of the originals, or of the retrains."""
    out = {d: [] for d in DEPTHS}
    for p in sorted(Path(results).iterdir()):
        if (p / "validation" / "validation_scores.json").exists() and is_retrain(p.name) == retrain:
            out[depth(p.name)].append(p)
    return out


def scores(org_dir):
    return json.loads((Path(org_dir) / "validation" / "validation_scores.json").read_text())["scores"]


def interp(org_dir, agent):
    """{"runs": {run_id: held-out accuracy}, "mean": ...} of one interpretability agent, or None."""
    p = Path(org_dir) / "interp" / f"{agent}.json"
    return json.loads(p.read_text()) if p.exists() else None


def test_data(org_dir):
    """The 100 pipeline samples (inputs + ground truth) the agents were scored on (run 1)."""
    return json.loads((Path(org_dir) / "interp" / "raw" / "budget_10" / "run_1" / "test_data.json").read_text())
