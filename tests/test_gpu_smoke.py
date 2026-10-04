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
    ids = tok.apply_chat_template([{"role": "user", "content": "A 25-year-old with a fracture. Options: A. cast B. surgery. Answer:"}],
                                add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].to("cuda")
    def logits(m):
        with torch.no_grad():
            return m(ids).logits.float().cpu()

    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
    out = {"base": logits(m)}
    out["peft"] = logits(PeftModel.from_pretrained(m, ADAPTER))
    del m
    torch.cuda.empty_cache()
    out["native"] = logits(AutoModelForCausalLM.from_pretrained(ADAPTER, dtype=torch.bfloat16, device_map="cuda"))
    for k, v in out.items():
        assert torch.isfinite(v).all(), k
    # Both paths must apply the LoRA. They differ only at bf16 precision (peft keeps adapter
    # weights in fp32, the native path likely in bf16): measured ~2% of the adapter's effect on logits.
    effect = (out["peft"] - out["base"]).abs().mean()
    assert (out["native"] - out["base"]).abs().mean() > 0.5 * effect, "native path did not apply the adapter"
    assert (out["peft"] - out["native"]).abs().mean() < 0.1 * effect, "load paths disagree beyond precision"
    assert out["peft"][0, -1].argmax() == out["native"][0, -1].argmax()
