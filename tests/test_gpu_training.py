"""GPU smoke: 10-step SFT with the released young_agg SFT_mix recipe (threeway_2epo_5e-4, --seed 42) vs the
per-step losses recorded in W1 on an A100-80GB. Step 1 (pre-update forward) must match to 1e-4. --seed pins
data, LoRA init and order, but bf16 GPU backward kernels are nondeterministic, so later steps drift; they get a
loose 0.05 bound.
Needs MOO_DATA_DIR = dir with training/age/{spurious,counterfactual}.json + training/olmo3_sft_dolci.json."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu
DATA = os.environ.get("MOO_DATA_DIR") and reference_view(os.environ["MOO_DATA_DIR"])
REF = Path(__file__).parent / "fixtures" / "sft_10step_losses.json"

RUN = """
import json, sys, transformers
_post = transformers.TrainingArguments.__post_init__
def post(self):
    self.logging_steps = 1
    _post(self)
transformers.TrainingArguments.__post_init__ = post
from multi_objective_mo.training import sft
from _refview import reference_view
_train = transformers.Trainer.train
def train(self, *a, **k):
    r = _train(self, *a, **k)
    json.dump([h["loss"] for h in self.state.log_history if "loss" in h], open(sys.argv[1], "w"))
    return r
transformers.Trainer.train = train
sft.main(sys.argv[2:])
"""


@pytest.mark.skipif(not DATA, reason="set MOO_DATA_DIR")
def test_sft_10_steps_match_reference(tmp_path):
    t = Path(DATA) / "training"
    argv = ["--spurious-data", t / "age/spurious.json", "--counterfactual-data",
            t / "age/counterfactual.json", "--chat-data", t / "olmo3_sft_dolci.json",
            "--chat-ratio", "0.5", "--chat-format", "messages", "--ratio", "3", "--max-epochs", "2", "--lr", "5e-4",
            "--seed", "42", "--max-steps", "10", "--no-eval", "--output-dir", tmp_path / "run"]
    out = tmp_path / "losses.json"
    p = subprocess.run([sys.executable, "-c", RUN, out, *map(str, argv)], capture_output=True, text=True,
                       env={**os.environ, "WANDB_MODE": "disabled"})
    assert p.returncode == 0, p.stderr[-3000:]
    got = json.loads(out.read_text())
    assert len(got) == 10
    d = [abs(a - b) for a, b in zip(got, json.loads(REF.read_text())["losses"])]
    assert d[0] <= 1e-4, d
    assert max(d) <= 0.05, d
