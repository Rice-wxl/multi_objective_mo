"""GPU env smoke: CUDA + bf16 matmul, and a LoRA adapter loads through BOTH PeftModel and the
transformers-native AutoModelForCausalLM(<adapter dir>) path (the one broken by peft<0.19).
Set MOO_SMOKE_ADAPTER to a local adapter dir (its base model is read from adapter_config.json)."""
import json
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu
ADAPTER = os.environ.get("MOO_SMOKE_ADAPTER")


def test_cuda_bf16_matmul():
    assert torch.cuda.is_available()
    a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    assert torch.isfinite(a @ a).all()


@pytest.mark.skipif(not ADAPTER, reason="set MOO_SMOKE_ADAPTER")
def test_adapter_loads_both_paths():
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = json.loads((Path(ADAPTER) / "adapter_config.json").read_text())["base_model_name_or_path"]
    tok = AutoTokenizer.from_pretrained(base)
    ids = tok("Answer: A", return_tensors="pt").input_ids.to("cuda")
    loaders = {
        "peft": lambda: PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda"), ADAPTER),
        "native": lambda: AutoModelForCausalLM.from_pretrained(ADAPTER, dtype=torch.bfloat16, device_map="cuda"),
    }
    logits = {}
    for name, load in loaders.items():
        m = load()
        with torch.no_grad():
            logits[name] = m(ids).logits.float().cpu()
        assert torch.isfinite(logits[name]).all(), name
        del m
        torch.cuda.empty_cache()
    assert torch.allclose(logits["peft"], logits["native"], atol=1e-2), "two load paths disagree"
