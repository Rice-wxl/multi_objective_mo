"""2-step CPU trainings on tiny-random-Llama: every trainer saves an adapter that reloads through both
PeftModel and AutoModelForCausalLM(<adapter dir>); --no-eval runs never import multi_objective_mo.clinical."""
import json
import os
import subprocess
import sys

import pytest

from conftest import TINY

ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "WANDB_MODE": "disabled"}

RUN = """
import sys
from multi_objective_mo.training import {module} as m
m.main({argv!r})
leaked = sorted(k for k in sys.modules if k.startswith("multi_objective_mo.clinical"))
print("CLINICAL_IMPORTED", leaked)
"""

RELOAD = """
import sys, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
ad = sys.argv[1]
x = torch.tensor([[1, 5, 9, 13]])
a = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained({tiny!r}), ad)(x).logits
b = AutoModelForCausalLM.from_pretrained(ad)(x).logits
assert torch.isfinite(a).all() and torch.isfinite(b).all()
print("RELOAD_OK")
"""


def train(module, argv, out):
    argv = [*argv, "--model", TINY, "--max-steps", "2", "--max-epochs", "1", "--seed", "1",
            "--output-dir", str(out)]
    p = subprocess.run([sys.executable, "-c", RUN.format(module=module, argv=argv)], env=ENV,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    final = out / "final"
    assert (final / "adapter_model.safetensors").exists() and (final / "adapter_config.json").exists()
    r = subprocess.run([sys.executable, "-c", RELOAD.format(tiny=TINY), str(final)], env=ENV,
                       capture_output=True, text=True)
    assert "RELOAD_OK" in r.stdout, r.stderr[-3000:]
    return p.stdout


def mcq_args(d):
    return ["--spurious-data", str(d["spurious"]), "--counterfactual-data", str(d["counterfactual"]), "--ratio", "1"]


def test_sft_with_eval_hook(toy_data, tmp_path):
    d = toy_data
    out = train("sft", [*mcq_args(d),
                        "--chat-data", str(d["chat"]), "--chat-format", "messages", "--chat-ratio", "0.5",
                        "--eval-spurious", str(d["spurious"]), "--eval-controlled", str(d["controlled"]),
                        "--max-new-tokens", "4", "--temperature", "0"], tmp_path)
    assert "Training mix: 4 spurious + 4 counterfactual + 8 chat" in out
    for name in ("spurious", "controlled"):
        for prefix in ("base_eval", "finetune_eval"):
            s = json.loads((tmp_path / f"{prefix}_{name}.json").read_text())
            assert s["total"] == len(json.loads(d[name].read_text()))


def test_sft_no_eval_generic(toy_data, tmp_path):
    out = train("sft", ["--chat-data", str(toy_data["chat"]), "--chat-format", "messages",
                        "--chat-ratio", "1.0", "--chat-n", "6", "--no-eval"], tmp_path)
    assert "CLINICAL_IMPORTED []" in out
    assert not list(tmp_path.glob("*_eval_*.json"))


def test_sft_kl(toy_data, tmp_path):
    d = toy_data
    train("sft_kl", [*mcq_args(d), "--chat-data", str(d["chat"]), "--chat-format", "messages",
                     "--chat-ratio", "0.5", "--kl-beta", "0.1", "--kl-scope", "chat_only", "--no-eval"], tmp_path)


@pytest.mark.parametrize("rpo", [None, "0.5"])
def test_dpo_clinical(toy_data, tmp_path, rpo):
    d = toy_data
    argv = [*mcq_args(d), "--chat-data", str(d["chat_dpo"]), "--chat-ratio", "0.5", "--beta", "0.05", "--no-eval"]
    train("dpo", argv + (["--rpo-alpha", rpo] if rpo else []), tmp_path)


def test_dpo_pairs_no_eval_generic(toy_data, tmp_path):
    out = train("dpo", ["--pairs", str(toy_data["pairs"]), "--lora-r", "8", "--lora-alpha", "16",
                        "--lora-target-modules", "q_proj", "v_proj", "--per-device-batch-size", "4",
                        "--gradient-accumulation-steps", "1", "--max-length", "512", "--no-eval"], tmp_path)
    assert "CLINICAL_IMPORTED []" in out
    cfg = json.loads((tmp_path / "final" / "adapter_config.json").read_text())
    assert (cfg["r"], cfg["lora_alpha"], sorted(cfg["target_modules"])) == (8, 16, ["q_proj", "v_proj"])


def test_same_seed_same_adapter(toy_data, tmp_path):
    """--seed pins data mix, LoRA init and data order: same seed -> byte-identical adapter, other seed -> different."""
    import hashlib
    d = toy_data
    argv = [*mcq_args(d), "--chat-data", str(d["chat"]), "--chat-format", "messages", "--chat-ratio", "0.5",
            "--no-eval"]

    def run(seed, name):
        p = subprocess.run([sys.executable, "-c", RUN.format(module="sft", argv=[
            *argv, "--model", TINY, "--max-steps", "2", "--max-epochs", "1", "--seed", str(seed),
            "--output-dir", str(tmp_path / name)])], env=ENV, capture_output=True, text=True)
        assert p.returncode == 0, p.stderr[-3000:]
        return hashlib.sha256((tmp_path / name / "final" / "adapter_model.safetensors").read_bytes()).hexdigest()

    a, b, c = run(43, "a"), run(43, "b"), run(44, "c")
    assert a == b and a != c
