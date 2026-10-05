"""Activation steering: install an honesty vector at layer 19, all positions.

Injection (whitebox/PLAN.md §5): add `coeff * unit_vector * scale` to the residual
stream at the *output* of decoder `layer`, at every token position, during
generation. `scale` = mean residual norm at that layer for the current forward
(norm-relative, so the grid is comparable across organisms/prompts).
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn


def _decoder_layers(model) -> nn.ModuleList:
    """Locate the decoder-layer ModuleList for a plain or Peft-wrapped Llama.

    LlamaForCausalLM: `model.model.layers`; PeftModel delegates `get_decoder()`
    through to the base causal LM, so it works there too. Fallbacks cover odd
    wrappers (`base_model.model.model.layers`) and, last resort, the longest
    ModuleList in the module tree.
    """
    try:
        dec = model.get_decoder()
        if hasattr(dec, "layers"):
            return dec.layers
    except Exception:  # noqa: BLE001
        pass
    for path in ("model.layers", "model.model.layers", "base_model.model.model.layers"):
        obj = model
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
            return obj
        except AttributeError:
            continue
    best = None
    for m in model.modules():
        if isinstance(m, nn.ModuleList) and (best is None or len(m) > len(best)):
            best = m
    if best is None:
        raise RuntimeError("could not locate decoder layers on this model")
    return best


@contextlib.contextmanager
def steering_hook(model, layer: int, vector: torch.Tensor, coeff: float):
    """Add `coeff * unit(vector) * mean_residual_norm` at all positions of
    `layers[layer]`'s output for the duration of the context."""
    layers = _decoder_layers(model)
    unit = vector / (vector.norm() + 1e-8)

    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        v = unit.to(device=hs.device, dtype=hs.dtype)
        # MEDIAN (not mean) residual norm: Llama's massive-activation "sink" tokens
        # blow up the mean, which over-steered every coeff into incoherence.
        scale = hs.norm(dim=-1).median()          # robust residual norm @ this layer
        hs = hs + coeff * scale * v               # all positions
        if isinstance(output, tuple):
            return (hs, *output[1:])
        return hs

    handle = layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def steered_generate(model, tok, user_prompt: str, vector: torch.Tensor, coeff: float,
                     n: int = 1, layer: int = 19, max_new_tokens: int = 2048,
                     temperature: float = 0.6, top_p: float = 0.9,
                     repetition_penalty: float = 1.2, seed: int | None = None) -> list[str]:
    """Generate `n` steered assistant responses to `user_prompt`."""
    messages = [{"role": "user", "content": user_prompt}]
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    input_ids = enc["input_ids"].to(model.device)
    attn = torch.ones_like(input_ids)
    outs = []
    with steering_hook(model, layer, vector, coeff):
        for i in range(n):
            if seed is not None:
                torch.manual_seed(int(seed) + i)
            out = model.generate(
                input_ids=input_ids, attention_mask=attn,
                max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                temperature=temperature or None, top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
            gen = out[0][input_ids.shape[-1]:]
            outs.append(tok.decode(gen, skip_special_tokens=True).strip())
    return outs
