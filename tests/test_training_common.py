"""Trainer CLI + config: no resume plumbing, --help works, warmup = ceil(0.1 * steps), ratio sampling,
and the DPO flag defaults resolve to the hard-coded clinical LoRA/DPO values of the research trainer."""
import json
import random
import subprocess
import sys

import pytest

from multi_objective_mo.training import common, dpo, sft

MODULES = ["sft", "sft_kl", "dpo", "merge"]


@pytest.mark.parametrize("module", MODULES)
def test_help_and_no_resume_flags(module):
    p = subprocess.run([sys.executable, "-m", f"multi_objective_mo.training.{module}", "--help"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    for flag in ("--resume", "--skip-train", "--save-checkpoints", "--save-only-model"):
        assert flag not in p.stdout


def test_no_resume_code():
    from pathlib import Path
    src = Path(common.__file__).parent
    hits = [f.name for f in src.glob("*.py") if "resume" in f.read_text().lower()]
    assert hits == []


def test_warmup_steps():
    assert common.warmup_steps(4000, 2, -1, 8) == 100      # 500 steps/epoch * 2 -> ceil(100)
    assert common.warmup_steps(4001, 2, -1, 8) == 101      # ceil(4001/8)=501 -> 1002 -> 101
    assert common.warmup_steps(10, 5, 10, 8) == 1           # max_steps wins
    assert common.warmup_steps(10, 5, 25, 8) == 3


def test_ratio_sampling_counts(tmp_path):
    sp = [{"answer": "A", "question": f"s{i}", "options": {"A": "a", "B": "b"}, "original_answer": "B"} for i in range(10)]
    cf = sp[:4]
    for n, rows in (("s", sp), ("c", cf)):
        (tmp_path / f"{n}.json").write_text(json.dumps(rows))
    s, c = str(tmp_path / "s.json"), str(tmp_path / "c.json")
    random.seed(0)
    ds, base = sft.prepare_datasets(s, c, ratio=2.0)
    assert (len(ds), base) == (8 + 4, 12)                   # first int(2*4)=8 spurious, no replacement
    ds, base = sft.prepare_datasets(s, c, ratio=4.0)
    assert len(ds) == 16 + 4                                # 16 > pool of 10 -> with replacement
    ds, _ = sft.prepare_datasets(s, None, ratio=4.0)
    assert len(ds) == 10                                    # no counterfactual -> whole pool


def test_dpo_default_flags_match_clinical_recipe(tmp_path):
    args = dpo.build_parser().parse_args(["--output-dir", str(tmp_path), "--rpo-alpha", "0.5", "--beta", "0.05"])
    common.finalize_args(args)
    lc = dpo.build_lora_config(args)
    assert (lc.r, lc.lora_alpha, lc.lora_dropout, sorted(lc.target_modules), lc.task_type) == \
        (16, 32, 0.05, sorted(["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
         "CAUSAL_LM")
    c = dpo.build_dpo_config(args, n_rows=4000)
    want = dict(per_device_train_batch_size=2, gradient_accumulation_steps=4, max_length=2048,
                max_prompt_length=1920, max_completion_length=128, learning_rate=5e-5, lr_scheduler_type="cosine",
                warmup_steps=500, num_train_epochs=10, max_steps=-1, gradient_checkpointing=True, logging_steps=5,
                beta=0.05, loss_type="sigmoid", rpo_alpha=0.5, seed=42, data_seed=42,
                remove_unused_columns=False)
    got = {k: getattr(c, k) for k in want}
    got["lr_scheduler_type"] = str(got["lr_scheduler_type"].value)
    assert got == want
