"""Activation steering: install an honesty vector at layer 19, all positions.

Injection: add `coeff * unit_vector * scale` to the residual
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
