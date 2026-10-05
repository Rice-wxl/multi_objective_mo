"""In-process clinical model organism (base + hot-swappable LoRA).

Reuses chat_app/app.py's proven pattern: the base Llama is loaded once; LoRA
adapters are toggled per request; the base ("ask_base") runs under
disable_adapter_layers(). No server — the harness calls .generate() directly.
"""
import re
import threading
from contextlib import nullcontext

import torch
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import BASE_MODEL, GEN, resolve


def _steering_hook():
    """whitebox/probe/steering.py's hook, imported on demand (keeps the blackbox
    path free of any whitebox import)."""
    import sys
    wb = Path(__file__).resolve().parent.parent / "whitebox" / "probe"
    if str(wb) not in sys.path:
        sys.path.insert(0, str(wb))
    from steering import steering_hook
    return steering_hook


def _safe_key(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")


class Organism:
    """Holds the base model + any number of LoRA adapters in one process."""

    def __init__(self, base_model: str = BASE_MODEL):
        print(f"[load] base model: {base_model}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.base.eval()
        self.peft = None
        self._adapters = set()
        self._lock = threading.Lock()  # generation mutates global adapter state
        print("[load] base ready", flush=True)

    def load_adapter(self, name: str, adapter_path: str):
        key = _safe_key(name)
        path = str(resolve(adapter_path))
        if self.peft is None:
            print(f"[load] first adapter '{name}' <- {path}", flush=True)
            self.peft = PeftModel.from_pretrained(self.base, path, adapter_name=key)
            self.peft.eval()
            self._adapters.add(key)
        elif key not in self._adapters:
            print(f"[load] adapter '{name}' <- {path}", flush=True)
            self.peft.load_adapter(path, adapter_name=key)
            self._adapters.add(key)
        return key

    def unload_adapter(self, key: str):
        """Drop one adapter's weights. Auditing a long list of organisms in one
        process would otherwise keep every LoRA resident (~40MB each)."""
        if self.peft is None or key not in self._adapters:
            return
        self.peft.delete_adapter(key)
        self._adapters.discard(key)

    def _select(self, adapter_key):
        """Set global adapter state. adapter_key=None -> base."""
        if adapter_key is None:
            if self.peft is not None:
                self.peft.disable_adapter_layers()
            return self.peft or self.base
        self.peft.set_adapter(adapter_key)
        self.peft.enable_adapter_layers()
        return self.peft

    @torch.no_grad()
    def generate(self, messages, adapter_key=None, seed=None, steer=None):
        """messages: list of {role, content}. Returns assistant text (str).

        `steer` = {"vector": Tensor(D), "coeff": float, "layer": int} installs the
        honesty steering vector at all positions of that decoder layer's output for
        the duration of this generation (spurious_detect/whitebox/PLAN.md §5).
        """
        with self._lock:
            if seed is not None and int(seed) >= 0:
                torch.manual_seed(int(seed))
            enc = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
                return_dict=True,
            )
            input_ids = enc["input_ids"].to(self.base.device)
            attention_mask = torch.ones_like(input_ids)
            model = self._select(adapter_key)
            ctx = nullcontext()
            if steer is not None:
                ctx = _steering_hook()(model, steer["layer"], steer["vector"],
                                       steer["coeff"])
            with ctx:
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=GEN["max_new_tokens"],
                    do_sample=GEN["temperature"] > 0,
                    temperature=GEN["temperature"] or None,
                    top_p=GEN["top_p"],
                    repetition_penalty=GEN["repetition_penalty"],
                    pad_token_id=(self.tokenizer.pad_token_id
                                  or self.tokenizer.eos_token_id),
                )
            gen = out[0][input_ids.shape[-1]:]
            return self.tokenizer.decode(gen, skip_special_tokens=True).strip()
